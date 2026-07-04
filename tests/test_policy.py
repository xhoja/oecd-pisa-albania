"""Population policy simulation: intervention application, direction, weighting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.policy import (
    predicted_at_risk_rate,
    scenario_table,
    simulate_policy,
)

FEATS = ["ANXMAT", "TEACHSUP", "ESCS"]


class _MonotoneModel:
    """p(at risk) = sigmoid(2*ANXMAT - ESCS - TEACHSUP)."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        z = 2 * X["ANXMAT"].values - X["ESCS"].values - X["TEACHSUP"].values
        p1 = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p1, p1])


def _pop(n=500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "ANXMAT": rng.normal(0.5, 1.0, n),   # elevated anxiety
        "TEACHSUP": rng.normal(0.0, 1.0, n),
        "ESCS": rng.normal(0.0, 1.0, n),
    })
    w = rng.uniform(0.5, 2.0, n)
    return X, w


def test_lowering_anxiety_reduces_at_risk_rate():
    m = _MonotoneModel()
    X, w = _pop()
    r = simulate_policy(m, X, w, threshold=0.5, interventions={"ANXMAT": ("sd", -1.0)})
    assert r["delta_at_risk_rate"] < 0          # fewer flagged
    assert r["delta_mean_prob"] < 0


def test_raising_support_reduces_risk():
    m = _MonotoneModel()
    X, w = _pop()
    r = simulate_policy(m, X, w, threshold=0.5, interventions={"TEACHSUP": ("sd", +1.0)})
    assert r["delta_at_risk_rate"] <= 0


def test_empty_intervention_is_no_change():
    m = _MonotoneModel()
    X, w = _pop()
    r = simulate_policy(m, X, w, threshold=0.5, interventions={})
    assert r["delta_at_risk_rate"] == pytest.approx(0.0)
    assert r["post_at_risk_rate"] == pytest.approx(r["baseline_at_risk_rate"])


def test_set_intervention_clamps_value():
    m = _MonotoneModel()
    X, w = _pop()
    # set ANXMAT to its minimum -> lowest possible risk from that lever
    r = simulate_policy(m, X, w, threshold=0.5, interventions={"ANXMAT": ("set", X["ANXMAT"].min())})
    assert r["post_at_risk_rate"] <= r["baseline_at_risk_rate"]


def test_weighted_rate_differs_from_unweighted():
    m = _MonotoneModel()
    X, _ = _pop()
    w_uniform = np.ones(len(X))
    w_skew = np.where(X["ANXMAT"] > 0.5, 5.0, 0.2)   # upweight high-anxiety students
    r_uni = predicted_at_risk_rate(m, X, w_uniform, 0.5)["at_risk_rate"]
    r_skew = predicted_at_risk_rate(m, X, w_skew, 0.5)["at_risk_rate"]
    assert r_skew > r_uni                            # weighting toward risky students raises rate


def test_scenario_table_sorted_and_has_baseline():
    m = _MonotoneModel()
    X, w = _pop()
    scen = {
        "anxiety_-0.5sd": {"ANXMAT": ("sd", -0.5)},
        "anxiety_-1sd": {"ANXMAT": ("sd", -1.0)},
        "support_+1sd": {"TEACHSUP": ("sd", +1.0)},
    }
    out = scenario_table(m, X, w, threshold=0.5, scenarios=scen)
    assert list(out["scenario"])  # non-empty
    # sorted ascending by delta (most negative / biggest reduction first)
    assert out["delta_at_risk_rate"].is_monotonic_increasing
    assert out.attrs["baseline_at_risk_rate"] is not None
