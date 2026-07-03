"""
Decision-curve analysis + calibration helpers: pin the net-benefit algebra and the
weighted reliability/Brier so a refactor can't silently break the screener framing.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.decision_curve import (
    calibration_summary,
    net_benefit,
    reliability_curve,
    weighted_brier,
)


def test_treat_none_is_zero_everywhere():
    y = np.array([1, 0, 1, 0, 1])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    dca = net_benefit(y, p, thresholds=np.linspace(0.1, 0.9, 9))
    assert np.allclose(dca["net_benefit_none"], 0.0)


def test_treat_all_matches_prevalence_formula():
    # NB_all(pt) = prev - (1-prev) * pt/(1-pt); check at a known threshold
    y = np.array([1, 1, 1, 0])          # prevalence 0.75 (equal weights)
    p = np.array([0.6, 0.6, 0.6, 0.6])
    pt = 0.5
    dca = net_benefit(y, p, thresholds=np.array([pt]))
    prev = 0.75
    expected = prev - (1 - prev) * (pt / (1 - pt))
    assert dca["net_benefit_all"].iloc[0] == pytest.approx(expected)


def test_perfect_model_beats_references():
    # perfectly separating probs: model net benefit >= treat-all at every threshold
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.6).astype(int)
    p = np.where(y == 1, 0.99, 0.01)
    dca = net_benefit(y, p)
    assert (dca["net_benefit_model"] >= dca["net_benefit_all"] - 1e-9).all()
    assert (dca["net_benefit_model"] >= -1e-9).all()


def test_net_benefit_weighting_changes_prevalence():
    y = np.array([1, 0])
    p = np.array([0.6, 0.4])
    heavy_positive = net_benefit(y, p, thresholds=np.array([0.5]), sample_weight=np.array([9.0, 1.0]))
    assert heavy_positive.attrs["prevalence"] == pytest.approx(0.9)


def test_reliability_curve_on_diagonal_for_calibrated():
    # construct a well-calibrated set: within each bin, observed ≈ predicted
    rng = np.random.default_rng(1)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)  # P(y=1)=p exactly
    rc = reliability_curve(y, p, n_bins=10)
    # mean predicted and observed fraction should track closely
    assert np.abs(rc["mean_predicted"] - rc["observed_fraction"]).max() < 0.08


def test_weighted_brier_zero_for_perfect_and_bounds():
    y = np.array([1, 0, 1, 0])
    assert weighted_brier(y, y.astype(float)) == pytest.approx(0.0)
    # worst case (all wrong, confident) -> 1.0
    assert weighted_brier(y, 1.0 - y) == pytest.approx(1.0)


def test_calibration_summary_keys():
    rng = np.random.default_rng(2)
    p = rng.random(300)
    y = (rng.random(300) < p).astype(int)
    s = calibration_summary(y, p, sample_weight=rng.uniform(1, 5, 300))
    assert set(s) >= {"brier", "ece", "prevalence", "reliability"}
    assert 0.0 <= s["ece"] <= 1.0
