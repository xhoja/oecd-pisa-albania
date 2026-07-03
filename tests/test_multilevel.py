"""
Multilevel random-intercept model: pin the ICC algebra and check the fit recovers a
known between-group signal and predicts sanely (including the unseen-group fallback).
Kept small so the variational-Bayes fits stay fast in the suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.multilevel import (
    LOGISTIC_RESIDUAL_VAR,
    fit_random_intercept,
    icc_logistic,
    predict_random_intercept,
    variance_partition_icc,
)


def test_icc_logistic_formula():
    # ICC = var / (var + pi^2/3); var=0 -> 0, var=pi^2/3 -> 0.5
    assert icc_logistic(0.0) == 0.0
    assert icc_logistic(LOGISTIC_RESIDUAL_VAR) == pytest.approx(0.5)
    assert 0.0 < icc_logistic(1.0) < 1.0


def _clustered(n_groups=40, per=25, school_sd=1.5, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    u = rng.normal(0, school_sd, n_groups)      # strong between-school effect
    x = rng.normal(0, 1, len(g))
    lp = -0.2 + 0.8 * x + u[g]
    y = (rng.random(len(g)) < 1 / (1 + np.exp(-lp))).astype(int)
    X = pd.DataFrame({"x": x})
    return X, pd.Series(y), pd.Series(g)


def test_null_icc_detects_clustering():
    X, y, g = _clustered(school_sd=1.5)
    icc = variance_partition_icc(y, g)["icc"]
    # a real between-school signal should register a non-trivial ICC
    assert icc > 0.15


def test_fit_returns_ors_and_school_variance():
    X, y, g = _clustered()
    fit = fit_random_intercept(X, y, g, ["x"])
    assert fit["school_var"] > 0
    assert 0.0 < fit["icc"] < 1.0
    fe = fit["fixed_effects"]
    assert {"term", "coef", "odds_ratio"} <= set(fe.columns)
    # the true slope is positive -> OR for x should exceed 1
    assert float(fe.loc[fe.term == "x", "odds_ratio"].iloc[0]) > 1.0


def test_predict_shapes_and_unseen_group_fallback():
    X, y, g = _clustered()
    fit = fit_random_intercept(X, y, g, ["x"])
    from src.models.multilevel import _standardize
    Xs = _standardize(X[["x"]])
    # include an unseen group id -> its random effect falls back to 0, no crash
    groups = g.values.copy()
    groups[:5] = 9999
    p = predict_random_intercept(fit["result"], Xs, groups, g.values)
    assert p.shape == (len(y),)
    assert np.all((p >= 0) & (p <= 1))
