"""Metric suite + threshold tuning."""
from __future__ import annotations

import numpy as np

from src.models.evaluate import (
    compute_metrics,
    corrected_resampled_ttest,
    delong_roc_test,
    expected_calibration_error,
    nadeau_bengio_interval,
    tune_threshold,
)


def test_perfect_classifier_scores_one():
    y = np.array([0, 0, 1, 1, 0, 1])
    prob = np.array([0.1, 0.2, 0.9, 0.8, 0.05, 0.95])
    pred = (prob >= 0.5).astype(int)
    m = compute_metrics(y, pred, prob, n_bootstrap=0)
    assert m.roc_auc == 1.0
    assert m.mcc == 1.0
    assert m.f1_macro == 1.0


def test_weighted_metrics_still_perfect_with_weights():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    pred = (prob >= 0.5).astype(int)
    w = np.array([5.0, 1.0, 1.0, 3.0])
    m = compute_metrics(y, pred, prob, n_bootstrap=0, sample_weight=w)
    assert m.roc_auc == 1.0
    assert m.accuracy == 1.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    prob = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)  # informative but noisy
    pred = (prob >= 0.5).astype(int)
    m = compute_metrics(y, pred, prob, n_bootstrap=300)
    lo, hi = m.roc_auc_ci
    assert lo <= m.roc_auc <= hi


def test_ece_zero_for_perfectly_calibrated_extremes():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, prob) == 0.0


def test_tune_threshold_within_bounds_and_beats_half():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.2).astype(int)          # imbalanced
    prob = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 300), 0, 1)
    from sklearn.metrics import f1_score

    thr = tune_threshold(y, prob, metric="f1_macro")
    assert 0.01 <= thr <= 0.99
    tuned = f1_score(y, (prob >= thr).astype(int), average="macro")
    half = f1_score(y, (prob >= 0.5).astype(int), average="macro")
    assert tuned >= half


# --- Nadeau-Bengio corrected inference -------------------------------------

def test_nb_interval_wider_than_naive():
    scores = [0.70, 0.72, 0.68, 0.75, 0.71, 0.69]
    ratio = 1.0 / 4  # 5-fold
    _, lo, hi, se = nadeau_bengio_interval(scores, ratio)
    naive_se = np.std(scores, ddof=1) / np.sqrt(len(scores))
    assert se > naive_se          # correction inflates the SE
    assert lo < np.mean(scores) < hi


def test_nb_interval_grows_with_ratio():
    scores = [0.70, 0.72, 0.68, 0.75, 0.71]
    _, _, _, se_small = nadeau_bengio_interval(scores, 1 / 9)   # 10-fold
    _, _, _, se_big = nadeau_bengio_interval(scores, 1 / 1)     # 2-fold
    assert se_big > se_small


def test_corrected_ttest_identical_scores():
    diffs = np.zeros(10)
    mean, t, p = corrected_resampled_ttest(diffs, 1 / 4)
    assert mean == 0.0 and t == 0.0 and p == 1.0


def test_corrected_ttest_detects_consistent_gap():
    # a small but perfectly consistent gap -> significant
    diffs = np.full(20, 0.05)
    diffs[::2] += 0.001  # tiny wobble so variance > 0
    mean, t, p = corrected_resampled_ttest(diffs, 1 / 4)
    assert mean > 0
    assert p < 0.05


# --- DeLong ----------------------------------------------------------------

def test_delong_identical_predictors_not_significant():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    prob = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 200), 0, 1)
    a1, a2, z, p = delong_roc_test(y, prob, prob.copy())
    assert a1 == a2
    assert z == 0.0 and p == 1.0


def test_delong_strong_vs_weak_is_significant():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    strong = np.clip(y * 0.8 + rng.normal(0.1, 0.15, 400), 0, 1)
    weak = np.clip(y * 0.1 + rng.normal(0.45, 0.3, 400), 0, 1)
    a_strong, a_weak, z, p = delong_roc_test(y, strong, weak)
    assert a_strong > a_weak
    assert p < 0.05


def test_delong_antisymmetric_in_order():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p1 = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 300), 0, 1)
    p2 = np.clip(y * 0.3 + rng.normal(0.35, 0.25, 300), 0, 1)
    a1, a2, z12, pv1 = delong_roc_test(y, p1, p2)
    b1, b2, z21, pv2 = delong_roc_test(y, p2, p1)
    assert a1 == b2 and a2 == b1
    assert z12 == -z21
    assert pv1 == pv2


def test_delong_nan_when_single_class():
    y = np.ones(50, dtype=int)
    a1, a2, z, p = delong_roc_test(y, np.random.rand(50), np.random.rand(50))
    assert np.isnan(p)


# --- weighted-fit routing (Phase-4 correctness) ----------------------------

def test_fit_with_sample_weight_routes_nested_pipeline():
    # LR is wrapped in a scaler Pipeline -> needs model__clf__sample_weight.
    # The naive model__sample_weight path fails; the helper must still weight it.
    import numpy as np
    import pandas as pd
    from src.models.evaluate import fit_with_sample_weight
    from src.models.experiment import _wrap
    from src.models.registry import get_model

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"ESCS": rng.normal(0, 1, 80), "HOMEPOS": rng.normal(0, 1, 80)})
    y = pd.Series((rng.random(80) < 0.4).astype(int))
    w = rng.uniform(1, 20, 80)

    assert fit_with_sample_weight(_wrap(get_model("logistic_regression")), X, y, w) is True
    assert fit_with_sample_weight(_wrap(get_model("random_forest")), X, y, w) is True
