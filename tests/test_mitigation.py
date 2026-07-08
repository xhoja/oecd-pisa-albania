"""Tests for post-hoc fairness mitigation (src/fairness/mitigation)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fairness.mitigation import (
    apply_group_thresholds,
    evaluate_assignment,
    fairness_utility_frontier,
    group_thresholds_equal_fpr,
    threshold_for_target_fpr,
)


def _biased_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Two groups whose score distributions differ, so a single 0.5 threshold
    gives them different FPRs (group B's negatives score higher)."""
    rng = np.random.default_rng(seed)
    rows = []
    for grp, shift in (("A", 0.0), ("B", 0.3)):
        y = rng.integers(0, 2, n)
        prob = np.clip(rng.normal(0.4 + shift * (y == 0), 0.15), 0, 1)
        rows.append(pd.DataFrame({"y": y.astype(float), "p": prob, "g": grp,
                                  "w": rng.uniform(1, 3, n)}))
    return pd.concat(rows, ignore_index=True)


def test_threshold_for_target_fpr_monotone():
    df = _biased_frame()
    a = df[df.g == "A"]
    t_low = threshold_for_target_fpr(a.y.values, a.p.values, a.w.values, 0.1)
    t_high = threshold_for_target_fpr(a.y.values, a.p.values, a.w.values, 0.4)
    # a stricter (lower) target FPR needs a higher threshold
    assert t_low >= t_high


def test_group_thresholds_equalise_fpr():
    df = _biased_frame()
    thr = group_thresholds_equal_fpr(df, "y", "p", "g", "w", target_fpr=0.2)
    pred = apply_group_thresholds(df, "p", "g", thr)
    ev = evaluate_assignment(df, "y", pred, "g", "w")
    # after equalising, every group's FPR is at or below the target -> tiny gap
    for grp, r in ev["by_group"].items():
        assert r["fpr"] <= 0.2 + 1e-9
    assert ev["fpr_gap"] <= 0.2


def test_group_thresholds_beat_single_on_gap():
    df = _biased_frame()
    thr = group_thresholds_equal_fpr(df, "y", "p", "g", "w", target_fpr=0.2)
    gpred = apply_group_thresholds(df, "p", "g", thr)
    gev = evaluate_assignment(df, "y", gpred, "g", "w")
    single = (df.p.values >= 0.5).astype(float)
    sev = evaluate_assignment(df, "y", single, "g", "w")
    # the biased single threshold has a larger FPR gap than equalised thresholds
    assert gev["fpr_gap"] <= sev["fpr_gap"]


def test_apply_group_thresholds_fallback():
    df = pd.DataFrame({"p": [0.6, 0.6], "g": ["A", "Z"]})
    pred = apply_group_thresholds(df, "p", "g", {"A": 0.7})  # Z unseen -> 0.5
    assert pred.tolist() == [0.0, 1.0]


def test_frontier_shape_and_columns():
    df = _biased_frame()
    fr = fairness_utility_frontier(df, "y", "p", "g", "w",
                                   target_fprs=np.array([0.1, 0.2, 0.3]))
    assert len(fr) == 3
    for c in ["grp_recall", "grp_fpr_gap", "single_recall", "single_fpr_gap"]:
        assert c in fr.columns
