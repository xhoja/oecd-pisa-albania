"""Tests for the malleable-lever ranking (src/models/levers)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.levers import (
    bootstrap_lever_effects,
    classify_features,
    lever_ceiling,
    rank_levers,
)


class _LinearProbModel:
    """Minimal predict_proba stub: sigmoid of a weighted feature combination.

    ``coefs`` maps column -> weight; higher weighted sum -> higher P(at-risk).
    Lets a test plant a known monotone response so the beneficial direction and
    ranking are deterministic.
    """

    def __init__(self, coefs: dict[str, float]):
        self.coefs = coefs

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        z = np.zeros(len(X))
        for col, c in self.coefs.items():
            if col in X.columns:
                z = z + c * X[col].to_numpy(dtype=float)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])


def _frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ESCS": rng.normal(0, 1, n),
        "ANXMAT": rng.normal(0, 1, n),
        "TEACHSUP": rng.normal(0, 1, n),
        "BELONG": rng.normal(0, 1, n),
        "SCH_MEAN_ANXMAT": rng.normal(0, 1, n),
        "SCH_N": rng.integers(20, 60, n).astype(float),
        "GENDER_x_ANXMAT": rng.normal(0, 1, n),
    })


def test_classify_features_partitions_by_class():
    cols = ["ESCS", "HISCED", "ANXMAT", "TEACHSUP", "SCH_MEAN_ANXMAT",
            "SCH_MEAN_ESCS", "SCH_N", "GENDER_x_ANXMAT", "MYSTERY"]
    cls = classify_features(cols)
    assert set(cls["fixed"]) == {"ESCS", "HISCED", "SCH_MEAN_ESCS"}
    assert set(cls["malleable"]) == {"ANXMAT", "TEACHSUP", "SCH_MEAN_ANXMAT"}
    # interactions, cohort size and unknowns fall through to "other"
    assert set(cls["other"]) == {"SCH_N", "GENDER_x_ANXMAT", "MYSTERY"}


def test_rank_levers_orders_by_effect_and_finds_direction():
    X = _frame()
    w = np.ones(len(X))
    # anxiety strongly raises risk, teacher support mildly lowers it, belonging ~0
    model = _LinearProbModel({"ANXMAT": 2.0, "TEACHSUP": -0.5, "BELONG": 0.0})
    rank = rank_levers(model, X, w, threshold=0.5, magnitude=0.5)
    # anxiety is the top lever (largest reduction) and its helpful direction is down
    top = rank.iloc[0]
    assert top["lever"] == "ANXMAT"
    assert top["direction"] == "↓"  # down
    assert top["delta_at_risk_rate"] <= rank.iloc[1]["delta_at_risk_rate"]
    # teacher support's beneficial direction is up (coef negative -> raise it)
    ts = rank[rank.lever == "TEACHSUP"].iloc[0]
    assert ts["direction"] == "↑"
    assert bool(ts["matches_expected_sign"])
    assert rank.attrs["baseline_at_risk_rate"] >= 0.0


def test_rank_levers_moves_school_mean_twin_together():
    X = _frame()
    w = np.ones(len(X))
    # only the school-mean anxiety drives risk; the base lever must still be
    # credited because rank_levers shifts the SCH_MEAN twin with it
    model = _LinearProbModel({"SCH_MEAN_ANXMAT": 2.0})
    rank = rank_levers(model, X, w, threshold=0.5, magnitude=0.5,
                       levers=["ANXMAT"])
    row = rank[rank.lever == "ANXMAT"].iloc[0]
    assert "SCH_MEAN_ANXMAT" in row["cols"]
    assert row["delta_at_risk_rate"] < 0.0  # the twin shift reduces risk


def test_bootstrap_brackets_point_estimate_and_is_deterministic():
    X = _frame()
    w = np.ones(len(X))
    model = _LinearProbModel({"ANXMAT": 2.0, "TEACHSUP": -0.5})
    rank = rank_levers(model, X, w, threshold=0.5, magnitude=0.5)
    boot = bootstrap_lever_effects(model, X, w, threshold=0.5, ranking=rank,
                                   magnitude=0.5, n_boot=50, seed=1)
    assert {"ci_low", "ci_high", "boot_se"} <= set(boot.columns)
    # the CI must bracket the point estimate for every lever
    for row in boot.itertuples():
        assert row.ci_low <= row.delta_at_risk_rate <= row.ci_high
        assert row.boot_se >= 0.0
    # a real lever's interval has positive width; same seed -> identical result
    anx = boot[boot.lever == "ANXMAT"].iloc[0]
    assert anx.ci_high > anx.ci_low
    boot2 = bootstrap_lever_effects(model, X, w, threshold=0.5, ranking=rank,
                                    magnitude=0.5, n_boot=50, seed=1)
    assert boot2[boot2.lever == "ANXMAT"].iloc[0].ci_low == anx.ci_low


def test_lever_ceiling_reports_both_bundles():
    X = _frame()
    w = np.ones(len(X))
    model = _LinearProbModel({"ANXMAT": 1.5, "ESCS": -1.5})
    ceil = lever_ceiling(model, X, w, threshold=0.5, magnitude=0.5)
    assert ceil["malleable_bundle"]["delta_at_risk_rate"] <= 0.0
    assert ceil["fixed_bundle"]["delta_at_risk_rate"] <= 0.0
    assert ceil["malleable_bundle"]["n_features"] >= 1
    assert ceil["baseline_at_risk_rate"] >= 0.0
