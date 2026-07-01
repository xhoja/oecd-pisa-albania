"""
Phase 5 — hyper-parameter optimisation via Optuna in *proper nested CV*.

Why a new module (not ``train.nested_cv``): the older nested CV operates on a
pre-transformed numpy array, so feature engineering/imputation leak across the
split, the inner objective is unweighted, and boosters are not OpenMP-isolated.
This module fixes all three:

- **Leakage-safe:** every fit uses the ``experiment._wrap`` pipeline
  (EngineeredFeatureBuilder → median impute → model), so standardisation, ranks
  and interactions are learned on the training fold only — in the *inner* tuning
  folds as well as the outer folds.
- **Weighted:** the inner Optuna objective and every fit use the PISA sampling
  weights (``model__[clf__]sample_weight``), so we tune for weighted AUC.
- **Isolated:** each outer fold (its whole inner Optuna study + the final fits)
  runs in a fresh interpreter (``experiment._run_isolated`` task ``hpo_fold``),
  the reliable macOS OpenMP fix.

Nested design: outer ``RepeatedStratifiedKFold`` gives the unbiased estimate;
the inner ``StratifiedKFold`` + Optuna picks hyper-parameters using *only* the
outer-train fold. For every outer fold we also fit the registry-default model,
so the run answers the real question — **does tuning beat the default by a
statistically significant margin?** (Nadeau-Bengio corrected resampled t-test).
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from threadpoolctl import threadpool_limits

import structlog

from src.models.evaluate import (
    compute_metrics,
    corrected_resampled_ttest,
    fit_with_sample_weight as _fit_weighted,
    nadeau_bengio_interval,
)
from src.models.optimize import _suggest_params
from src.models.prepare import ModelData
from src.models.registry import get_model
from src.utils.reproducibility import set_seeds

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = structlog.get_logger(__name__)


def _wrapped(model_name: str, **params):
    from src.models.experiment import _wrap  # local import avoids a cycle
    return _wrap(get_model(model_name, **params))


def _weighted_auc(y_true, prob, w) -> float:
    pred = (prob >= 0.5).astype(int)
    m = compute_metrics(y_true, pred, prob, n_bootstrap=0, sample_weight=w)
    return m.roc_auc


def tune_hyperparameters(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    n_trials: int = 40,
    inner_folds: int = 4,
    random_state: int = 42,
) -> tuple[dict[str, Any], float]:
    """
    Inner-loop Optuna study (TPE + MedianPruner) maximising the mean *weighted*
    inner-fold AUC of the leakage-safe pipeline. Returns (best_params, best_auc).
    Operates only on the data handed to it (the outer-train fold).
    """
    X = X.reset_index(drop=True); y = y.reset_index(drop=True); w = w.reset_index(drop=True)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        scores: list[float] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for step, (tr, va) in enumerate(inner_cv.split(X, y), start=1):
                try:
                    pipe = _wrapped(model_name, **params)
                    _fit_weighted(pipe, X.iloc[tr], y.iloc[tr], w.iloc[tr].values)
                    if y.iloc[va].nunique() < 2:
                        continue
                    prob = pipe.predict_proba(X.iloc[va])[:, 1]
                    scores.append(_weighted_auc(y.iloc[va].values, prob, w.iloc[va].values))
                except Exception as e:  # a bad param combo -> prune, don't crash
                    raise optuna.exceptions.TrialPruned() from e
                trial.report(float(np.mean(scores)), step=step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


def _hpo_outer_fold_worker(
    model_name, X, y, w, train_idx, test_idx, n_trials, inner_folds, random_state,
) -> dict[str, Any]:
    """One outer fold: inner Optuna tuning on train, then evaluate BOTH the tuned
    model and the registry-default model on the held-out test fold (weighted).
    Top-level + picklable so it can run in an isolated interpreter."""
    set_seeds(random_state)
    with threadpool_limits(limits=1):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        wtr, wte = w.iloc[train_idx], w.iloc[test_idx]

        best_params, inner_auc = tune_hyperparameters(
            model_name, Xtr, ytr, wtr, n_trials=n_trials,
            inner_folds=inner_folds, random_state=random_state,
        )

        def _eval(params) -> float:
            pipe = _wrapped(model_name, **params)
            _fit_weighted(pipe, Xtr, ytr, wtr.values)
            prob = pipe.predict_proba(Xte)[:, 1]
            return _weighted_auc(yte.values, prob, wte.values)

        tuned_auc = _eval(best_params)
        default_auc = _eval({})
    return {
        "tuned_auc": float(tuned_auc),
        "default_auc": float(default_auc),
        "inner_auc": float(inner_auc),
        "best_params": best_params,
    }


def nested_cv_hpo(
    data: ModelData,
    model_name: str,
    outer_folds: int = 5,
    outer_repeats: int = 2,
    inner_folds: int = 4,
    n_trials: int = 40,
    random_state: int = 42,
    isolate: bool = True,
) -> dict[str, Any]:
    """
    Nested-CV HPO for one model. Returns tuned & default AUC (Nadeau-Bengio
    corrected 95% CIs across outer folds), the per-fold best params, and the
    paired corrected-resampled t-test of tuned-vs-default (does HPO help?).
    """
    set_seeds(random_state)
    X = data.X.reset_index(drop=True)
    y = data.y.reset_index(drop=True)
    w = data.weights.reset_index(drop=True)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_folds, n_repeats=outer_repeats, random_state=random_state)

    tuned, default, best_params = [], [], []
    for i, (tr, te) in enumerate(outer_cv.split(X, y)):
        args = (model_name, X, y, w, tr, te, n_trials, inner_folds, random_state + i)
        if isolate:
            from src.models.experiment import _run_isolated
            r = _run_isolated("hpo_fold", *args)
        else:
            r = _hpo_outer_fold_worker(*args)
        tuned.append(r["tuned_auc"]); default.append(r["default_auc"])
        best_params.append(r["best_params"])
        print(f"  [hpo] {model_name} fold {i+1}/{outer_folds*outer_repeats}: "
              f"tuned={r['tuned_auc']:.4f} default={r['default_auc']:.4f}", flush=True)

    ratio = 1.0 / (outer_folds - 1) if outer_folds > 1 else 1.0
    t_mean, t_lo, t_hi, _ = nadeau_bengio_interval(tuned, ratio)
    d_mean, d_lo, d_hi, _ = nadeau_bengio_interval(default, ratio)
    diff = np.asarray(tuned) - np.asarray(default)
    gain_mean, _gain_t, gain_p = corrected_resampled_ttest(diff, ratio)

    result = {
        "model": model_name,
        "n_outer_folds": len(tuned),
        "tuned_auc_mean": round(t_mean, 4),
        "tuned_auc_95ci": [round(t_lo, 4), round(t_hi, 4)],
        "default_auc_mean": round(d_mean, 4),
        "default_auc_95ci": [round(d_lo, 4), round(d_hi, 4)],
        "hpo_gain": round(float(gain_mean), 4),
        "hpo_gain_p_value": round(gain_p, 4) if gain_p == gain_p else np.nan,
        "hpo_significant_0.05": bool(gain_p < 0.05) if gain_p == gain_p else False,
        "tuned_per_fold": [round(x, 4) for x in tuned],
        "default_per_fold": [round(x, 4) for x in default],
        "best_params_per_fold": best_params,
    }
    logger.info("Nested-CV HPO complete", model=model_name,
                tuned=result["tuned_auc_mean"], default=result["default_auc_mean"],
                gain=result["hpo_gain"], p=result["hpo_gain_p_value"])
    return result


def final_best_params(nested_result: dict[str, Any]) -> dict[str, Any]:
    """
    Pick production hyper-parameters from the per-fold best params: the mode for
    categorical/int choices, the median for floats. A stable summary rather than
    trusting a single fold's lucky draw.
    """
    per_fold = nested_result["best_params_per_fold"]
    keys = set().union(*(p.keys() for p in per_fold)) if per_fold else set()
    out: dict[str, Any] = {}
    for k in keys:
        vals = [p[k] for p in per_fold if k in p]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = float(np.median(vals))
            if all(isinstance(v, int) for v in vals):
                out[k] = int(round(out[k]))
        else:
            out[k] = max(set(vals), key=vals.count)  # mode
    return out
