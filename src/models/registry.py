"""
Model registry: constructs sklearn-compatible estimators from config.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import structlog
logger = structlog.get_logger(__name__)


def build_logistic_regression(**kwargs: Any) -> Pipeline:
    params = {
        "C": 1.0,
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": 42,
        "solver": "saga",
        **kwargs,
    }
    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(**params)),
    ])


def build_decision_tree(**kwargs: Any) -> DecisionTreeClassifier:
    params = {"class_weight": "balanced", "random_state": 42, **kwargs}
    return DecisionTreeClassifier(**params)


def build_random_forest(**kwargs: Any) -> RandomForestClassifier:
    # merge pattern (defaults, then kwargs) so tuned params can override the
    # defaults instead of raising "multiple values for keyword argument".
    params = {"n_estimators": 200, "class_weight": "balanced",
              "random_state": 42, "n_jobs": -1, **kwargs}
    return RandomForestClassifier(**params)


def build_extra_trees(**kwargs: Any) -> ExtraTreesClassifier:
    params = {"n_estimators": 200, "class_weight": "balanced",
              "random_state": 42, "n_jobs": -1, **kwargs}
    return ExtraTreesClassifier(**params)


def build_gradient_boosting(**kwargs: Any) -> GradientBoostingClassifier:
    params = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4,
              "random_state": 42, **kwargs}
    return GradientBoostingClassifier(**params)


def build_xgboost(**kwargs: Any) -> Any:
    from xgboost import XGBClassifier
    # use_label_encoder was removed in xgboost>=2.0; passing it only spams a
    # "Parameters not used" warning on every fit.
    params = {"eval_metric": "logloss", "random_state": 42, "n_jobs": -1, **kwargs}
    return XGBClassifier(**params)


def build_lightgbm(**kwargs: Any) -> Any:
    from lightgbm import LGBMClassifier
    params = {"class_weight": "balanced", "random_state": 42, "n_jobs": -1,
              "verbose": -1, **kwargs}
    return LGBMClassifier(**params)


def build_catboost(**kwargs: Any) -> Any:
    from catboost import CatBoostClassifier
    params = {"auto_class_weights": "Balanced", "random_seed": 42, "verbose": 0, **kwargs}
    return CatBoostClassifier(**params)


def build_svm(**kwargs: Any) -> Pipeline:
    params = {
        "C": 1.0,
        "kernel": "rbf",
        "class_weight": "balanced",
        "probability": True,
        "random_state": 42,
        **kwargs,
    }
    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf", SVC(**params)),
    ])


def build_knn(**kwargs: Any) -> Pipeline:
    params = {"n_neighbors": 11, "weights": "distance", **kwargs}
    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf", KNeighborsClassifier(**params)),
    ])


def build_mlp(**kwargs: Any) -> Pipeline:
    params = {
        "hidden_layer_sizes": (128, 64),
        "max_iter": 500,
        "random_state": 42,
        "early_stopping": True,
        "validation_fraction": 0.1,
        **kwargs,
    }
    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf", MLPClassifier(**params)),
    ])


def build_naive_bayes(**kwargs: Any) -> GaussianNB:
    return GaussianNB(**kwargs)


BUILDERS: dict[str, callable] = {
    "logistic_regression": build_logistic_regression,
    "decision_tree": build_decision_tree,
    "random_forest": build_random_forest,
    "extra_trees": build_extra_trees,
    "gradient_boosting": build_gradient_boosting,
    "xgboost": build_xgboost,
    "lightgbm": build_lightgbm,
    "catboost": build_catboost,
    "svm": build_svm,
    "knn": build_knn,
    "mlp": build_mlp,
    "naive_bayes": build_naive_bayes,
}


def get_model(name: str, **kwargs: Any) -> Any:
    if name not in BUILDERS:
        raise ValueError(f"Unknown model: {name}. Available: {list(BUILDERS.keys())}")
    return BUILDERS[name](**kwargs)


def build_stacking_ensemble(
    base_models: list[str] | None = None,
    meta_learner: str = "logistic_regression",
) -> StackingClassifier:
    """
    Build stacking ensemble with specified base models.
    Default: XGBoost + LightGBM + CatBoost stacked with LR meta-learner.
    """
    if base_models is None:
        base_models = ["xgboost", "lightgbm", "catboost"]

    estimators = [(name, get_model(name)) for name in base_models]
    final_estimator = get_model(meta_learner)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )


def build_mlp_from_optuna(trial: Any) -> Pipeline:
    """Build MLP from Optuna trial (called inside HPO loop)."""
    n_layers = trial.suggest_int("n_layers", 1, 4)
    n_units = trial.suggest_int("n_units", 32, 256)
    hidden_layers = tuple([n_units] * n_layers)
    alpha = trial.suggest_float("alpha", 1e-5, 0.1, log=True)
    lr_init = trial.suggest_float("learning_rate_init", 1e-4, 0.01, log=True)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    return build_mlp(
        hidden_layer_sizes=hidden_layers,
        alpha=alpha,
        learning_rate_init=lr_init,
        activation=activation,
    )
