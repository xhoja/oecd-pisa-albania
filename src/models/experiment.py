"""
Experiment orchestration: model comparison and out-of-sample evaluation.

Two protocols:
1. compare_models_cv  — RepeatedStratifiedKFold comparison across the model zoo,
   with median imputation fit per-fold (leakage-safe). Reports full metric suite
   with bootstrap CIs.
2. out_of_sample      — train on early cycles, test on a held-out future cohort
   (the covariate-shift experiment: train 2009–2018, test 2022).

Median imputation is applied inside each fold via a small wrapper so that tree
models (which otherwise reject NaN) work and no test information leaks.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import structlog

from src.models.evaluate import compute_metrics
from src.models.prepare import ModelData
from src.models.registry import get_model
from src.utils.reproducibility import set_seeds

logger = structlog.get_logger(__name__)


def _wrap_with_imputer(model: Any) -> Pipeline:
    """Prepend a median imputer so every model tolerates NaN, leakage-safe in CV."""
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def compare_models_cv(
    data: ModelData,
    model_names: list[str],
    outer_folds: int = 5,
    outer_repeats: int = 4,
    random_state: int = 42,
    use_sample_weight: bool = True,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Compare models with RepeatedStratifiedKFold. Default hyperparameters (the
    registry's sensible defaults); HPO is applied later to the winner via
    src.models.train.nested_cv. Returns a tidy results DataFrame.
    """
    set_seeds(random_state)
    cv = RepeatedStratifiedKFold(n_splits=outer_folds, n_repeats=outer_repeats, random_state=random_state)
    X = data.X.reset_index(drop=True)
    y = data.y.reset_index(drop=True)
    w = data.weights.reset_index(drop=True)

    rows = []
    for name in model_names:
        fold_auc, fold_pr, fold_f1, fold_mcc, fold_bal = [], [], [], [], []
        t0 = time.time()
        try:
            # limit OpenMP threads: boosters (libomp) + sklearn (libgomp) otherwise
            # oversubscribe across folds and abort the process on macOS.
            with threadpool_limits(limits=1):
                for tr, te in cv.split(X, y):
                    pipe = _wrap_with_imputer(get_model(name))
                    if use_sample_weight:
                        try:
                            pipe.fit(X.iloc[tr], y.iloc[tr], model__sample_weight=w.iloc[tr].values)
                        except (TypeError, ValueError):
                            pipe.fit(X.iloc[tr], y.iloc[tr])
                    else:
                        pipe.fit(X.iloc[tr], y.iloc[tr])

                    prob = pipe.predict_proba(X.iloc[te])[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    yte = y.iloc[te].values
                    if len(np.unique(yte)) < 2:
                        continue
                    m = compute_metrics(yte, pred, prob, n_bootstrap=0)
                    fold_auc.append(m.roc_auc); fold_pr.append(m.pr_auc)
                    fold_f1.append(m.f1_macro); fold_mcc.append(m.mcc); fold_bal.append(m.balanced_accuracy)
        except Exception as e:  # noqa: BLE001 — keep partial results if a model fails
            logger.warning("Model failed, skipping", model=name, error=str(e))
            print(f"  [skip] {name}: {type(e).__name__}: {e}", flush=True)
            continue

        if not fold_auc:
            print(f"  [skip] {name}: no valid folds", flush=True)
            continue

        rows.append({
            "model": name,
            "roc_auc_mean": round(np.mean(fold_auc), 4),
            "roc_auc_std": round(np.std(fold_auc), 4),
            "roc_auc_95ci_low": round(np.percentile(fold_auc, 2.5), 4),
            "roc_auc_95ci_high": round(np.percentile(fold_auc, 97.5), 4),
            "pr_auc_mean": round(np.mean(fold_pr), 4),
            "f1_macro_mean": round(np.mean(fold_f1), 4),
            "balanced_acc_mean": round(np.mean(fold_bal), 4),
            "mcc_mean": round(np.mean(fold_mcc), 4),
            "n_folds": len(fold_auc),
        })
        print(f"  [done] {name}: AUC={rows[-1]['roc_auc_mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        logger.info("Model compared", model=name, auc=rows[-1]["roc_auc_mean"])

        # Incremental save so a later slow/failing model can't wipe results
        if save_path is not None:
            pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(
                drop=True
            ).to_csv(save_path, index=False)

    result = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    return result


def out_of_sample(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_features: list[str],
    model_names: list[str],
    domain: str = "math",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train on train_df (e.g. 2009–2018), evaluate on test_df (e.g. 2022).
    The covariate-shift experiment: tree models are expected to collapse while
    regularized linear models generalize.

    Returns per-model test metrics + ROC/PR/calibration curve data.
    """
    from src.models.prepare import build_model_data, impute_median

    set_seeds(random_state)
    train = build_model_data(train_df, candidate_features, domain=domain)
    test = build_model_data(test_df, candidate_features, domain=domain)

    # Align features (intersection present in both)
    common = [f for f in train.feature_names if f in test.feature_names]
    Xtr, Xte = impute_median(train.X[common], test.X[common])
    ytr, yte = train.y.values, test.y.values
    wtr = train.weights.values

    results: dict[str, Any] = {"features": common, "n_train": len(Xtr), "n_test": len(Xte), "models": {}}
    with threadpool_limits(limits=1):
        for name in model_names:
            model = get_model(name)
            try:
                model.fit(Xtr, ytr, sample_weight=wtr)
            except (TypeError, ValueError):
                model.fit(Xtr, ytr)
            prob = model.predict_proba(Xte)[:, 1]
            pred = (prob >= 0.5).astype(int)
            m = compute_metrics(yte, pred, prob, n_bootstrap=1000)
            results["models"][name] = m.to_dict()
            print(f"  [oos] {name}: test_AUC={m.roc_auc:.4f}", flush=True)
            logger.info("OOS evaluated", model=name, test_auc=m.roc_auc)

    return results


def save_results(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
    logger.info("Results saved", path=str(path))
