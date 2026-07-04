"""Split-conformal prediction: quantile algebra, coverage guarantee, Mondrian."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.conformal import (
    calibrate,
    conformal_quantile_level,
    prediction_sets,
    set_summary,
    weighted_quantile,
)


def test_weighted_quantile_matches_plain_when_uniform():
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # unweighted 0.9-quantile of this grid is 0.9 (smallest t with >=90% mass <= t)
    assert weighted_quantile(s, 0.9) == pytest.approx(0.9)


def test_weighted_quantile_respects_weights():
    s = np.array([0.0, 1.0])
    # nearly all weight on the small score -> 0.5 quantile is the small one
    assert weighted_quantile(s, 0.5, weights=np.array([9.0, 1.0])) == 0.0
    # nearly all weight on the large score -> 0.5 quantile is the large one
    assert weighted_quantile(s, 0.5, weights=np.array([1.0, 9.0])) == 1.0


def test_quantile_level_finite_sample_correction():
    assert conformal_quantile_level(9, 0.1) == pytest.approx(1.0)   # ceil(10*0.9)/9 = 1.0
    assert conformal_quantile_level(99, 0.1) == pytest.approx(np.ceil(100 * 0.9) / 99)
    assert conformal_quantile_level(0, 0.1) == 1.0


def _synthetic_proba(n, seed=0):
    """A moderately-good probabilistic classifier's outputs on exchangeable data."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # predicted prob of class 1 correlated with y but noisy
    p1 = np.clip(0.5 + 0.3 * (2 * y - 1) + rng.normal(0, 0.2, n), 0.01, 0.99)
    proba = np.column_stack([1 - p1, p1])
    return proba, y


def test_marginal_coverage_guarantee_holds():
    proba_cal, y_cal = _synthetic_proba(2000, seed=1)
    proba_te, y_te = _synthetic_proba(4000, seed=2)
    for alpha in (0.1, 0.2):
        cal = calibrate(proba_cal, y_cal, alpha=alpha)
        sets = prediction_sets(proba_te, cal)
        cov = set_summary(sets, y_te)["coverage"]
        assert cov >= (1 - alpha) - 0.03   # empirical coverage ~>= 1-alpha


def test_mondrian_per_class_coverage():
    proba_cal, y_cal = _synthetic_proba(3000, seed=3)
    proba_te, y_te = _synthetic_proba(5000, seed=4)
    cal = calibrate(proba_cal, y_cal, alpha=0.1, mondrian=True)
    assert cal["mondrian"] and set(cal["qhat_by_class"]) == {0, 1}
    sets = prediction_sets(proba_te, cal)
    for k in (0, 1):
        m = y_te == k
        cov_k = set_summary(sets[m], y_te[m])["coverage"]
        assert cov_k >= 0.85   # class-conditional coverage near the 90% target


def test_set_summary_shares_sum_to_one():
    proba, y = _synthetic_proba(1000, seed=5)
    cal = calibrate(proba, y, alpha=0.1)
    s = set_summary(prediction_sets(proba, cal), y)
    total = s["share_empty"] + s["share_singleton"] + s["share_ambiguous"]
    assert total == pytest.approx(1.0)
    assert 0.0 <= s["avg_set_size"] <= 2.0


def test_weighted_coverage_uses_weights():
    proba, y = _synthetic_proba(2000, seed=6)
    w = np.random.default_rng(7).uniform(0.5, 2.0, len(y))
    cal = calibrate(proba, y, alpha=0.1, weights=w)
    cov = set_summary(prediction_sets(proba, cal), y, weights=w)["coverage"]
    assert cov >= 0.85
