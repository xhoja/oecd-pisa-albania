"""Phase 7: survey-weighted fairness metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fairness.metrics import compute_fairness, fairness_audit, threshold_sweep


@pytest.fixture
def audit_df() -> pd.DataFrame:
    # Two groups. Group A: perfect. Group B: predicts everyone positive.
    return pd.DataFrame({
        "y_true": [1, 1, 0, 0, 1, 1, 0, 0],
        "y_pred": [1, 1, 0, 0, 1, 1, 1, 1],
        "y_prob": [0.9, 0.8, 0.1, 0.2, 0.9, 0.8, 0.7, 0.6],
        "A": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "w": [1, 1, 1, 1, 1, 1, 1, 1],
    })


def test_unweighted_gaps(audit_df):
    r = compute_fairness(audit_df, "y_true", "y_pred", "y_prob", "A")
    # selection rate: a=0.5, b=1.0 -> parity gap 0.5
    assert r.demographic_parity["a"] == pytest.approx(0.5)
    assert r.demographic_parity["b"] == pytest.approx(1.0)
    assert r.parity_gap == pytest.approx(0.5)
    # TPR both 1.0 (all true positives caught) -> opportunity gap 0
    assert r.opportunity_gap == pytest.approx(0.0)
    # FPR: a=0, b=1 -> fpr gap 1
    assert r.fpr_by_group["a"] == pytest.approx(0.0)
    assert r.fpr_by_group["b"] == pytest.approx(1.0)
    assert r.fpr_gap == pytest.approx(1.0)


def test_weights_change_rates(audit_df):
    df = audit_df.copy()
    # Upweight the two negatives in group b that are wrongly flagged (rows 6,7).
    df.loc[[6, 7], "w"] = 9.0
    unw = compute_fairness(df, "y_true", "y_pred", "y_prob", "A")
    wtd = compute_fairness(df, "y_true", "y_pred", "y_prob", "A", weight_col="w")
    # group b selection rate is 1.0 either way (all predicted positive),
    # but the weighted calibration mean shifts toward the upweighted rows.
    assert wtd.calibration_by_group["b"] != pytest.approx(unw.calibration_by_group["b"])
    # weighted FPR for b is still 1.0 (every negative flagged), sanity check
    assert wtd.fpr_by_group["b"] == pytest.approx(1.0)


def test_weighted_selection_rate_matches_manual():
    df = pd.DataFrame({
        "y_true": [0, 0, 1, 1],
        "y_pred": [1, 0, 1, 0],
        "y_prob": [0.6, 0.4, 0.7, 0.3],
        "A": ["x", "x", "x", "x"],
        "w": [10.0, 1.0, 1.0, 1.0],
    })
    r = compute_fairness(df, "y_true", "y_pred", "y_prob", "A", weight_col="w")
    # weighted P(pred=1) = (10*1 + 1*0 + 1*1 + 1*0)/13 = 11/13
    assert r.demographic_parity["x"] == pytest.approx(11 / 13)


def test_missing_attribute_returns_empty():
    df = pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1], "y_prob": [0.2, 0.8]})
    r = compute_fairness(df, "y_true", "y_pred", "y_prob", "NOPE")
    assert r.groups == []
    assert np.isnan(r.parity_gap)


def test_audit_and_sweep_pass_weights(audit_df):
    reports = fairness_audit(audit_df, "y_true", "y_pred", "y_prob", ["A"], weight_col="w")
    assert "A" in reports
    sweep = threshold_sweep(audit_df, "y_true", "y_prob", "A", thresholds=[0.5, 0.75], weight_col="w")
    assert list(sweep["threshold"]) == [0.5, 0.75]
    assert {"parity_gap", "opportunity_gap", "fpr_gap"} <= set(sweep.columns)
