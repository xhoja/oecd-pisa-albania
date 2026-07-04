"""Actionable counterfactual recourse: mutability, direction, minimality."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explainability.counterfactuals import (
    DEFAULT_ACTIONABLE,
    batch_counterfactuals,
    find_counterfactual,
)

FEATS = ["ANXMAT", "TEACHSUP", "BELONG", "ICTHOME", "ICTSCH", "GENDER", "HISCED"]


class _MonotoneModel:
    """p(at risk) = sigmoid(2*ANXMAT - 1*TEACHSUP - 1*BELONG + 3*GENDER).

    GENDER is immutable and pushes risk up; the actionable levers (ANXMAT down,
    TEACHSUP/BELONG up) must supply all of the recourse.
    """

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        z = 2 * X["ANXMAT"].values - X["TEACHSUP"].values - X["BELONG"].values + 3 * X["GENDER"].values
        p1 = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p1, p1])


def _stats():
    return pd.DataFrame(
        {"std": 1.0, "min": -5.0, "max": 5.0},
        index=FEATS,
    )


def _student(anx=2.0, teachsup=0.0, belong=0.0, gender=0.0):
    return pd.Series(
        {"ANXMAT": anx, "TEACHSUP": teachsup, "BELONG": belong,
         "ICTHOME": 0.0, "ICTSCH": 0.0, "GENDER": gender, "HISCED": 0.0}
    )


def test_recourse_flips_at_risk_student():
    m = _MonotoneModel()
    x = _student(anx=2.0)                 # high anxiety -> at risk
    r = find_counterfactual(m, x, _stats(), threshold=0.5)
    assert r["p_original"] >= 0.5
    assert r["success"]
    assert r["p_counterfactual"] < 0.5
    assert r["n_features_changed"] >= 1


def test_recourse_respects_direction():
    m = _MonotoneModel()
    r = find_counterfactual(m, _student(anx=2.0), _stats(), threshold=0.5)
    for f, c in r["changes"].items():
        # each change must be in the feature's beneficial direction
        assert np.sign(c["delta_sd"]) == DEFAULT_ACTIONABLE[f]


def test_immutable_features_never_move():
    m = _MonotoneModel()
    r = find_counterfactual(m, _student(anx=2.0, gender=1.0), _stats(), threshold=0.5)
    assert "GENDER" not in r["changes"]
    assert "HISCED" not in r["changes"]


def test_already_below_threshold_is_noop():
    m = _MonotoneModel()
    x = _student(anx=-2.0)                # low anxiety -> not at risk
    r = find_counterfactual(m, x, _stats(), threshold=0.5)
    assert r["success"]
    assert r["n_features_changed"] == 0
    assert r["changes"] == {}


def test_infeasible_when_immutable_risk_too_high():
    m = _MonotoneModel()
    # gender term (+3) alone exceeds what capped actionable moves can offset at
    # a very strict threshold -> recourse should report failure, not crash
    x = _student(anx=2.0, gender=1.0)
    r = find_counterfactual(m, x, _stats(), threshold=0.01, max_sd=1.0)
    assert r["success"] is False
    assert r["p_counterfactual"] <= r["p_original"]   # still moved in the right direction


def test_batch_returns_row_per_student():
    m = _MonotoneModel()
    X = pd.DataFrame([_student(anx=2.0), _student(anx=1.5), _student(anx=-1.0)]).reset_index(drop=True)
    out = batch_counterfactuals(m, X, _stats(), threshold=0.5)
    assert len(out) == 3
    assert {"success", "p_original", "p_counterfactual", "recourse"} <= set(out.columns)
    assert out.loc[2, "n_features_changed"] == 0        # last student already safe


def test_cost_is_sum_of_absolute_sd_moves():
    m = _MonotoneModel()
    r = find_counterfactual(m, _student(anx=2.0), _stats(), threshold=0.5)
    assert r["cost_sd"] == pytest.approx(sum(abs(c["delta_sd"]) for c in r["changes"].values()))
