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
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats
from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

import structlog

from src.data.impute import rubins_combine
from src.features.target import THRESHOLDS, _pv_columns
from src.features.transformers import EngineeredFeatureBuilder
from src.models.evaluate import compute_metrics
from src.models.prepare import ModelData, build_model_data
from src.models.registry import get_model
from src.utils.reproducibility import set_seeds

# Keep DataFrames flowing through the pipeline so EngineeredFeatureBuilder
# (which needs named columns) sees a DataFrame at every step.
set_config(transform_output="pandas")

logger = structlog.get_logger(__name__)


def _wrap(model: Any, engineer: bool = True) -> Pipeline:
    """
    CV-safe pipeline: build engineered features (fit-in-fold) -> median impute
    -> model. Both preprocessing steps are fit on the train fold only.
    """
    steps = []
    if engineer:
        steps.append(("engineer", EngineeredFeatureBuilder()))
    steps += [("imputer", SimpleImputer(strategy="median")), ("model", model)]
    return Pipeline(steps)


def _fold_ci(vals: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi) — t-interval across CV folds. Honest about k folds, unlike
    a 2.5/97.5 percentile of ~20 points."""
    arr = np.asarray(vals, dtype=float)
    k = len(arr)
    mean = float(arr.mean())
    if k < 2:
        return mean, np.nan, np.nan
    se = arr.std(ddof=1) / np.sqrt(k)
    t = _scipy_stats.t.ppf(1 - alpha / 2, df=k - 1)
    return mean, float(mean - t * se), float(mean + t * se)


def _run_isolated(task: str, *args):
    """
    Run a registered task in a brand-new Python interpreter (subprocess) and
    return its result.

    A spawned multiprocessing child still shares the parent's already-loaded
    OpenMP runtime; on macOS, fitting several boosters (libomp) in a session
    that also loaded scikit-learn (libgomp) eventually segfaults regardless. A
    fresh `sys.executable -m ...` interpreter has ZERO shared state, which is the
    reliable fix (see memory: OpenMP macOS). Args are passed via a temp pickle.
    """
    import pickle
    import subprocess
    import sys
    import tempfile

    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = Path(tmp) / "in.pkl", Path(tmp) / "out.pkl"
        with open(inp, "wb") as f:
            pickle.dump((task, args), f)
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = "1"
        proc = subprocess.run(
            [sys.executable, "-m", "src.models._isolated_worker", str(inp), str(outp)],
            cwd=str(root), env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0 or not outp.exists():
            raise RuntimeError(
                f"isolated task '{task}' crashed (rc={proc.returncode})\n{proc.stderr[-2000:]}"
            )
        with open(outp, "rb") as f:
            return pickle.load(f)


def _oos_worker(name, Xtr, ytr, wtr, Xte, yte, wte, random_state):
    """Top-level (picklable) worker: fit one model and return its OOS metrics."""
    from threadpoolctl import threadpool_limits

    from src.models.evaluate import tune_threshold

    set_seeds(random_state)
    with threadpool_limits(limits=1):
        pipe = _wrap(get_model(name))
        try:
            pipe.fit(Xtr, ytr, model__sample_weight=wtr)
        except (TypeError, ValueError):
            pipe.fit(Xtr, ytr)
        prob_tr = pipe.predict_proba(Xtr)[:, 1]
        prob = pipe.predict_proba(Xte)[:, 1]
        thr = tune_threshold(ytr, prob_tr, sample_weight=wtr, metric="f1_macro")
        pred = (prob >= 0.5).astype(int)
        pred_tuned = (prob >= thr).astype(int)
        m = compute_metrics(yte, pred, prob, n_bootstrap=1000, sample_weight=wte)
        m_tuned = compute_metrics(yte, pred_tuned, prob, n_bootstrap=0, sample_weight=wte)
    d = m.to_dict()
    d["tuned_threshold"] = round(thr, 3)
    d["f1_macro_tuned"] = m_tuned.f1_macro
    d["mcc_tuned"] = m_tuned.mcc
    return d


def _cv_worker(name, X, y, w, outer_folds, outer_repeats, random_state, use_sample_weight):
    """Top-level (picklable) worker: run one model's repeated stratified CV and
    return its summary row (or None if no valid folds)."""
    from threadpoolctl import threadpool_limits

    set_seeds(random_state)
    cv = RepeatedStratifiedKFold(n_splits=outer_folds, n_repeats=outer_repeats, random_state=random_state)
    fold_auc, fold_pr, fold_f1, fold_mcc, fold_bal = [], [], [], [], []
    with threadpool_limits(limits=1):
        for tr, te in cv.split(X, y):
            pipe = _wrap(get_model(name))
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
            wte = w.iloc[te].values if use_sample_weight else None
            m = compute_metrics(yte, pred, prob, n_bootstrap=0, sample_weight=wte)
            fold_auc.append(m.roc_auc); fold_pr.append(m.pr_auc)
            fold_f1.append(m.f1_macro); fold_mcc.append(m.mcc); fold_bal.append(m.balanced_accuracy)

    if not fold_auc:
        return None
    auc_mean, auc_lo, auc_hi = _fold_ci(fold_auc)
    return {
        "model": name,
        "roc_auc_mean": round(auc_mean, 4),
        "roc_auc_std": round(float(np.std(fold_auc, ddof=1)), 4),
        "roc_auc_95ci_low": round(auc_lo, 4),    # t-interval across folds
        "roc_auc_95ci_high": round(auc_hi, 4),
        "pr_auc_mean": round(np.mean(fold_pr), 4),
        "f1_macro_mean": round(np.mean(fold_f1), 4),
        "balanced_acc_mean": round(np.mean(fold_bal), 4),
        "mcc_mean": round(np.mean(fold_mcc), 4),
        "n_folds": len(fold_auc),
        "weighted_metrics": bool(use_sample_weight),
    }


def compare_models_cv(
    data: ModelData,
    model_names: list[str],
    outer_folds: int = 5,
    outer_repeats: int = 4,
    random_state: int = 42,
    use_sample_weight: bool = True,
    save_path: str | Path | None = None,
    isolate: bool = True,
) -> pd.DataFrame:
    """
    Compare models with RepeatedStratifiedKFold. Default hyperparameters (the
    registry's sensible defaults); HPO is applied later to the winner via
    src.models.train.nested_cv. Returns a tidy results DataFrame.

    isolate: run each model's CV in a fresh interpreter (OpenMP-safe; a booster
        that crashes at the C level is logged and skipped instead of taking the
        whole comparison down). Set False to run in-process.
    """
    set_seeds(random_state)
    X = data.X.reset_index(drop=True)
    y = data.y.reset_index(drop=True)
    w = data.weights.reset_index(drop=True)

    rows = []
    for name in model_names:
        t0 = time.time()
        try:
            if isolate:
                row = _run_isolated("cv", name, X, y, w, outer_folds, outer_repeats,
                                    random_state, use_sample_weight)
            else:
                row = _cv_worker(name, X, y, w, outer_folds, outer_repeats,
                                 random_state, use_sample_weight)
        except RuntimeError as e:
            logger.warning("Model failed, skipping", model=name, error=str(e).splitlines()[0])
            print(f"  [skip] {name}: {str(e).splitlines()[0]}", flush=True)
            continue
        if row is None:
            print(f"  [skip] {name}: no valid folds", flush=True)
            continue

        rows.append(row)
        print(f"  [done] {name}: AUC={row['roc_auc_mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        logger.info("Model compared", model=name, auc=row["roc_auc_mean"])

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

    Returns per-model test metrics + ROC/PR/calibration curve data. Test metrics
    are weighted (population estimates). Engineered features and imputation are
    fit on the training cohort only (inside the pipeline), then applied to 2022.
    Both the fixed 0.5 threshold and a train-tuned threshold are reported.
    """
    from src.models.prepare import build_model_data

    set_seeds(random_state)
    train = build_model_data(train_df, candidate_features, domain=domain)
    test = build_model_data(test_df, candidate_features, domain=domain)

    # Align features (intersection present in both); engineering done in pipeline
    common = [f for f in train.feature_names if f in test.feature_names]
    Xtr, Xte = train.X[common], test.X[common]
    ytr, yte = train.y.values, test.y.values
    wtr, wte = train.weights.values, test.weights.values

    results: dict[str, Any] = {"features": common, "n_train": len(Xtr), "n_test": len(Xte), "models": {}}
    for name in model_names:
        # Each model runs in a FRESH interpreter (see _run_isolated) so boosters
        # get a clean OpenMP runtime. If one still crashes at the C level (a
        # broken local libomp/booster install), record it and keep going rather
        # than aborting the whole experiment.
        try:
            d = _run_isolated("oos", name, Xtr, ytr, wtr, Xte, yte, wte, random_state)
        except RuntimeError as e:
            results["models"][name] = {"error": str(e).splitlines()[0]}
            print(f"  [oos] {name}: FAILED — {str(e).splitlines()[0]}", flush=True)
            logger.warning("OOS model failed", model=name, error=str(e).splitlines()[0])
            continue
        results["models"][name] = d
        print(f"  [oos] {name}: test_AUC={d['roc_auc']:.4f} (weighted)", flush=True)
        logger.info("OOS evaluated", model=name, test_auc=d["roc_auc"])

    return results


def per_pv_cv_evaluate(
    df: pd.DataFrame,
    candidate_features: list[str],
    model_name: str,
    domain: str = "math",
    outer_folds: int = 5,
    outer_repeats: int = 4,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Rigorous PISA evaluation: fit the model once per plausible value, then combine
    the per-PV estimates with Rubin's rules — instead of collapsing the 10 PVs to
    their mean and modeling that single point target.

    For each plausible value l: run repeated stratified CV (weighted, leakage-safe
    via the in-fold engineer+impute pipeline), take Q_l = mean fold AUC and
    U_l = sampling variance of that mean. Rubin's rules then give the combined AUC,
    a standard error that includes the between-PV (imputation) variance, and the
    fraction of missing information (FMI) attributable to the PV uncertainty.

    Returns a dict per metric: {estimate, se, fmi, per_pv}.
    """
    set_seeds(random_state)
    thr = THRESHOLDS[domain]
    pv_cols = _pv_columns(df, domain)
    if not pv_cols:
        raise ValueError(f"No plausible-value columns for domain={domain}")

    # Feature matrix + weights, shared across PVs (selection + indicators applied
    # once on the full slice). base.X carries the engineered-component raw cols
    # and any _MISS_ indicators; the pipeline builds the rest per fold.
    base = build_model_data(df, candidate_features, domain=domain)
    X_full, w_full = base.X, base.weights
    pv_src = df.loc[X_full.index]  # align PV columns to the selected rows

    cv = RepeatedStratifiedKFold(n_splits=outer_folds, n_repeats=outer_repeats, random_state=random_state)

    metric_keys = ["roc_auc", "pr_auc", "f1_macro", "mcc"]
    per_pv: dict[str, list[float]] = {k: [] for k in metric_keys}
    per_pv_var: dict[str, list[float]] = {k: [] for k in metric_keys}

    with threadpool_limits(limits=1):
        for pv in pv_cols:
            mask = pv_src[pv].notna().values
            X = X_full.loc[mask].reset_index(drop=True)
            y = (pv_src.loc[mask, pv] < thr).astype(int).reset_index(drop=True)
            w = w_full.loc[mask].reset_index(drop=True)
            if y.nunique() < 2:
                continue

            fold: dict[str, list[float]] = {k: [] for k in metric_keys}
            for tr, te in cv.split(X, y):
                pipe = _wrap(get_model(model_name))
                try:
                    pipe.fit(X.iloc[tr], y.iloc[tr], model__sample_weight=w.iloc[tr].values)
                except (TypeError, ValueError):
                    pipe.fit(X.iloc[tr], y.iloc[tr])
                prob = pipe.predict_proba(X.iloc[te])[:, 1]
                pred = (prob >= 0.5).astype(int)
                yte = y.iloc[te].values
                if len(np.unique(yte)) < 2:
                    continue
                m = compute_metrics(yte, pred, prob, n_bootstrap=0, sample_weight=w.iloc[te].values)
                fold["roc_auc"].append(m.roc_auc); fold["pr_auc"].append(m.pr_auc)
                fold["f1_macro"].append(m.f1_macro); fold["mcc"].append(m.mcc)

            for k in metric_keys:
                if fold[k]:
                    arr = np.asarray(fold[k])
                    per_pv[k].append(float(arr.mean()))
                    per_pv_var[k].append(float(arr.var(ddof=1) / len(arr)))  # variance of the mean
            print(f"  [pv] {pv}: AUC={np.mean(fold['roc_auc']):.4f}", flush=True)

    combined: dict[str, Any] = {}
    for k in metric_keys:
        if len(per_pv[k]) >= 2:
            est, se, fmi = rubins_combine(per_pv[k], per_pv_var[k])
            combined[k] = {
                "estimate": round(est, 4),
                "se": round(se, 4),
                "ci95_low": round(est - 1.96 * se, 4),
                "ci95_high": round(est + 1.96 * se, 4),
                "fmi": round(fmi, 4),
                "n_pv": len(per_pv[k]),
                "per_pv": [round(v, 4) for v in per_pv[k]],
            }
    combined["model"] = model_name
    combined["domain"] = domain
    logger.info("Per-PV Rubin evaluation complete", model=model_name,
                auc=combined.get("roc_auc", {}).get("estimate"))
    return combined


def save_results(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
    logger.info("Results saved", path=str(path))
