"""
Training pipeline with nested cross-validation.

Outer loop: RepeatedStratifiedKFold (5×10 = 50 evaluations)
Inner loop: StratifiedKFold (5-fold) for Optuna HPO

This avoids optimistic bias from using the same data for both
hyperparameter selection and performance estimation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from src.models.evaluate import ClassificationMetrics, compute_metrics
from src.models.optimize import run_optuna_study
from src.models.registry import get_model
from src.utils.reproducibility import set_seeds

import structlog
logger = structlog.get_logger(__name__)


@dataclass
class NestedCVResult:
    model_name: str
    fold_metrics: list[ClassificationMetrics] = field(default_factory=list)
    best_params_per_fold: list[dict[str, Any]] = field(default_factory=list)

    @property
    def mean_roc_auc(self) -> float:
        return float(np.mean([m.roc_auc for m in self.fold_metrics]))

    @property
    def std_roc_auc(self) -> float:
        return float(np.std([m.roc_auc for m in self.fold_metrics]))

    @property
    def mean_mcc(self) -> float:
        return float(np.mean([m.mcc for m in self.fold_metrics]))

    def summary(self) -> dict[str, Any]:
        metrics_list = ["roc_auc", "pr_auc", "f1_macro", "balanced_accuracy", "mcc", "cohen_kappa"]
        result: dict[str, Any] = {"model": self.model_name}
        for metric in metrics_list:
            vals = [getattr(m, metric) for m in self.fold_metrics]
            result[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
            result[f"{metric}_std"] = round(float(np.std(vals)), 4)
            result[f"{metric}_95ci_low"] = round(float(np.percentile(vals, 2.5)), 4)
            result[f"{metric}_95ci_high"] = round(float(np.percentile(vals, 97.5)), 4)
        return result


def nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    outer_folds: int = 5,
    outer_repeats: int = 10,
    inner_folds: int = 5,
    optuna_trials: int = 50,
    random_state: int = 42,
    sample_weight: pd.Series | None = None,
) -> NestedCVResult:
    """
    Nested cross-validation for unbiased performance estimation + HPO.

    Args:
        X: feature matrix
        y: binary target
        model_name: key in model registry
        outer_folds: outer CV folds
        outer_repeats: outer CV repeats (50 total outer evaluations)
        inner_folds: inner CV folds for HPO
        optuna_trials: Optuna trials per outer fold
        random_state: seed
        sample_weight: optional sample weights
    """
    set_seeds(random_state)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_folds,
        n_repeats=outer_repeats,
        random_state=random_state,
    )

    result = NestedCVResult(model_name=model_name)
    X_arr = X.values
    y_arr = y.values

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_arr, y_arr)):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        w_train = sample_weight.iloc[train_idx].values if sample_weight is not None else None

        # Inner loop: HPO with Optuna
        best_params = run_optuna_study(
            model_name=model_name,
            X=X_train,
            y=y_train,
            n_trials=optuna_trials,
            n_folds=inner_folds,
            random_state=random_state + fold_idx,
        )
        result.best_params_per_fold.append(best_params)

        # Train on full outer train fold with best params
        model = get_model(model_name, **best_params)
        if w_train is not None:
            try:
                model.fit(X_train, y_train, sample_weight=w_train)
            except TypeError:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        # Evaluate on outer test fold
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_prob, n_bootstrap=500)
        result.fold_metrics.append(metrics)

        if (fold_idx + 1) % 10 == 0:
            logger.info(
                "Nested CV progress",
                model=model_name,
                fold=fold_idx + 1,
                auc=round(metrics.roc_auc, 4),
            )

    logger.info(
        "Nested CV complete",
        model=model_name,
        mean_auc=result.mean_roc_auc,
        std_auc=result.std_roc_auc,
    )
    return result


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    best_params: dict[str, Any],
    sample_weight: pd.Series | None = None,
) -> Any:
    """Train final model on full dataset with known best params."""
    model = get_model(model_name, **best_params)
    if sample_weight is not None:
        try:
            model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            model.fit(X, y)
    else:
        model.fit(X, y)
    return model


def save_model(model: Any, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata or {}}, path)
    logger.info("Model saved", path=str(path))


def load_model(path: str | Path) -> tuple[Any, dict[str, Any]]:
    obj = joblib.load(path)
    return obj["model"], obj["metadata"]


def compare_models(results: dict[str, NestedCVResult]) -> pd.DataFrame:
    """Build comparison table from nested CV results."""
    rows = [r.summary() for r in results.values()]
    df = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    return df
