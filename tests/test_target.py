"""Target construction + PV column detection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.target import (
    THRESHOLDS,
    _pv_columns,
    add_point_target,
    per_pv_targets,
)


def test_thresholds_are_exact_oecd_bounds():
    assert THRESHOLDS["math"] == 420.07
    assert THRESHOLDS["reading"] == 407.47
    assert THRESHOLDS["science"] == 409.54


def test_pv_columns_detected_and_sorted(pisa_df):
    cols = _pv_columns(pisa_df, "math")
    assert cols == [f"PV{i}MATH" for i in range(1, 11)]


def test_pv_columns_empty_for_absent_domain(pisa_df):
    assert _pv_columns(pisa_df, "reading") == []


def test_point_target_matches_pv_mean_threshold(pisa_df):
    out = add_point_target(pisa_df, domain="math")
    pv_mean = pisa_df[[f"PV{i}MATH" for i in range(1, 11)]].mean(axis=1)
    expected = (pv_mean < THRESHOLDS["math"]).astype(int)
    assert out["AT_RISK_MATH"].astype(int).tolist() == expected.tolist()
    assert out["AT_RISK_MATH"].isin([0, 1]).all()


def test_custom_threshold_shifts_rate(pisa_df):
    low = add_point_target(pisa_df, domain="math", threshold=300.0)["AT_RISK_MATH"].mean()
    high = add_point_target(pisa_df, domain="math", threshold=600.0)["AT_RISK_MATH"].mean()
    assert low < high  # higher cutoff -> more students below it


def test_per_pv_targets_shape(pisa_df):
    targets, cols = per_pv_targets(pisa_df, domain="math")
    assert len(targets) == len(cols) == 10
    assert all(set(np.unique(t)).issubset({0, 1}) for t in targets)
