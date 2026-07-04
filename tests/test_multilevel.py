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
    fit_weighted_random_intercept,
    icc_logistic,
    predict_random_intercept,
    scale_survey_weights,
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


# --- scale_survey_weights -----------------------------------------------------

def test_scale_cluster_method_sums_to_cluster_size():
    w = np.array([1.0, 3.0, 6.0, 2.0, 2.0])
    g = np.array([0, 0, 0, 1, 1])
    ws = scale_survey_weights(w, g, method="cluster")
    assert ws[:3].sum() == pytest.approx(3.0)   # n_j = 3
    assert ws[3:].sum() == pytest.approx(2.0)   # n_j = 2


def test_scale_effective_method_sums_to_effective_size():
    w = np.array([1.0, 1.0, 1.0, 1.0])
    g = np.array([0, 0, 0, 0])
    ws = scale_survey_weights(w, g, method="effective")
    # equal weights -> effective size == n == 4, scaled weights unchanged
    assert ws.sum() == pytest.approx(4.0)
    assert np.allclose(ws, 1.0)


def test_scale_effective_shrinks_with_unequal_weights():
    w = np.array([1.0, 1.0, 10.0])          # one dominant weight
    g = np.array([0, 0, 0])
    ws = scale_survey_weights(w, g, method="effective")
    eff = (w.sum() ** 2) / (w ** 2).sum()    # < 3
    assert ws.sum() == pytest.approx(eff)
    assert eff < 3.0


# --- fit_weighted_random_intercept (PQL) --------------------------------------

def test_weighted_pql_recovers_fixed_effect_and_detects_clustering():
    X, y, g = _clustered(n_groups=60, per=30, school_sd=1.2, seed=1)
    w = pd.Series(np.ones(len(y)))           # unit weights -> close to unweighted
    res = fit_weighted_random_intercept(X, y, g, w, ["x"], scaling="effective")
    fe = res["fixed_effects"].set_index("term")
    assert fe.loc["x", "coef"] > 0.4         # positive slope recovered (per-SD)
    assert res["icc"] > 0.1                   # detects the between-school signal
    assert 0.0 < res["icc"] < 1.0
    assert set(res["fixed_effects"]["term"]) == {"Intercept", "x"}


def test_weighted_pql_runs_with_informative_weights():
    X, y, g = _clustered(n_groups=50, per=25, school_sd=1.0, seed=2)
    rng = np.random.default_rng(3)
    w = pd.Series(rng.uniform(0.5, 3.0, len(y)))  # informative weights
    res = fit_weighted_random_intercept(X, y, g, w, ["x"])
    assert np.isfinite(res["school_sd"]) and res["school_sd"] > 0
    assert res["random_effects"].shape[0] == 50
