"""
Optuna-based hyperparameter optimization.
Uses TPE sampler with MedianPruner for efficiency.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)
import structlog
logger = structlog.get_logger(__name__)


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Suggest hyperparameters for a given model using Optuna trial."""

    if model_name == "logistic_regression":
        # sklearn 1.8 deprecated `penalty` (removed in 1.10); the elastic-net
        # `l1_ratio` in [0, 1] now subsumes the whole family - l1_ratio=0 is pure
        # L2, 1 is pure L1, in between is elastic-net. Searching it continuously
        # is strictly more expressive than the old l1/l2/elasticnet categorical
        # and raises no FutureWarning. `saga` is still required for l1_ratio.
        params: dict[str, Any] = {
            "C": trial.suggest_float("C", 0.001, 100, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "solver": "saga",
            "max_iter": 2000,
            "class_weight": "balanced",
            "random_state": 42,
        }
        return params

    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }

    if model_name == "extra_trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }

    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }

    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 254),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "verbose": 0,
        }

    if model_name == "gradient_boosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "random_state": 42,
        }

    if model_name == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 4)
        n_units = trial.suggest_int("n_units", 32, 256)
        return {
            "hidden_layer_sizes": tuple([n_units] * n_layers),
            "alpha": trial.suggest_float("alpha", 1e-5, 0.1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 0.01, log=True),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "max_iter": 500,
            "early_stopping": True,
            "random_state": 42,
        }

    if model_name == "svm":
        return {
            "C": trial.suggest_float("C", 0.01, 100, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "class_weight": "balanced",
            "probability": True,
            "random_state": 42,
        }

    if model_name == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 100),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "class_weight": "balanced",
            "random_state": 42,
        }

    raise ValueError(f"Unknown model for HPO: {model_name}")


def run_optuna_study(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    n_folds: int = 5,
    random_state: int = 42,
    timeout: int | None = None,
) -> dict[str, Any]:
    """
    Run Optuna HPO study, return best hyperparameters.
    Optimizes inner-fold ROC AUC.
    """
    from src.models.registry import get_model

    inner_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, model_name)
        fold_scores = []

        for train_idx, val_idx in inner_cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model = get_model(model_name, **params)
                    model.fit(X_tr, y_tr)
                    y_prob = model.predict_proba(X_val)[:, 1]
                    if len(np.unique(y_val)) < 2:
                        continue
                    fold_scores.append(roc_auc_score(y_val, y_prob))
                except Exception as e:
                    raise optuna.exceptions.TrialPruned() from e

            trial.report(np.mean(fold_scores), step=len(fold_scores))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores)) if fold_scores else 0.0

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    logger.info(
        "Optuna study complete",
        model=model_name,
        best_auc=round(study.best_value, 4),
        n_trials=len(study.trials),
    )
    return best_params
