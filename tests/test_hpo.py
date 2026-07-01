"""Nested-CV HPO plumbing (small budget, in-process)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.hpo import (
    _fit_weighted,
    final_best_params,
    nested_cv_hpo,
    tune_hyperparameters,
)
from src.models.prepare import build_model_data


def _model_data(pisa_df):
    feats = ["ESCS", "HOMEPOS", "GENDER", "ANXMAT", "BELONG", "HISCED", "HISEI"]
    return build_model_data(pisa_df, feats, domain="math")


def test_fit_weighted_routes_to_estimator(pisa_df):
    from src.models.experiment import _wrap
    from src.models.registry import get_model

    data = _model_data(pisa_df)
    pipe = _wrap(get_model("decision_tree"))
    ok = _fit_weighted(pipe, data.X, data.y, data.weights.values)
    assert ok is True  # trees accept sample_weight


def test_tune_returns_params_and_valid_auc(pisa_df):
    data = _model_data(pisa_df)
    params, auc = tune_hyperparameters(
        "logistic_regression", data.X, data.y, data.weights,
        n_trials=4, inner_folds=3, random_state=0,
    )
    assert isinstance(params, dict) and params
    assert 0.0 <= auc <= 1.0


def test_nested_cv_hpo_shape_and_fields(pisa_df):
    data = _model_data(pisa_df)
    r = nested_cv_hpo(
        data, "logistic_regression", outer_folds=3, outer_repeats=1,
        inner_folds=2, n_trials=3, isolate=False,
    )
    assert r["n_outer_folds"] == 3
    assert 0.0 <= r["tuned_auc_mean"] <= 1.0
    assert 0.0 <= r["default_auc_mean"] <= 1.0
    assert len(r["tuned_per_fold"]) == 3
    assert len(r["best_params_per_fold"]) == 3
    assert "hpo_gain_p_value" in r


def test_final_best_params_aggregates_folds():
    fake = {"best_params_per_fold": [
        {"C": 1.0, "penalty": "l2"},
        {"C": 3.0, "penalty": "l2"},
        {"C": 5.0, "penalty": "l1"},
    ]}
    out = final_best_params(fake)
    assert out["penalty"] == "l2"          # mode
    assert out["C"] == 3.0                 # median of [1,3,5]


def test_final_best_params_int_stays_int():
    fake = {"best_params_per_fold": [
        {"n_estimators": 100}, {"n_estimators": 200}, {"n_estimators": 400},
    ]}
    out = final_best_params(fake)
    assert out["n_estimators"] == 200 and isinstance(out["n_estimators"], int)


def test_get_model_accepts_overriding_hpo_params():
    # HPO suggests params (incl. random_state / n_estimators) that the builder
    # also sets by default — the builder must merge, not raise "multiple values".
    from src.models.registry import get_model

    cases = [
        ("gradient_boosting", {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "random_state": 1}),
        ("random_forest", {"n_estimators": 150, "random_state": 1}),
        ("extra_trees", {"n_estimators": 150, "max_depth": 8}),
        ("decision_tree", {"max_depth": 5, "random_state": 1}),
        ("xgboost", {"n_estimators": 50, "random_state": 1}),
        ("catboost", {"iterations": 20, "random_seed": 1}),
        ("logistic_regression", {"C": 0.5}),
    ]
    for name, over in cases:
        get_model(name, **over)  # must not raise
