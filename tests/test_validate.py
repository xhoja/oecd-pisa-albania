"""Data-contract layer."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.validate import (
    Violation,
    assert_valid,
    errors,
    replicate_coverage_by_cycle,
    validate_processed,
)
from src.features.target import add_point_target


def _codes(vs: list[Violation]) -> set[str]:
    return {v.code for v in vs}


def test_clean_frame_has_no_errors(pisa_df):
    vs = validate_processed(pisa_df)
    assert errors(vs) == []
    # 4 replicate weights (< 80) is a documented WARN, not an error
    assert "brr_partial" in _codes(vs)


def test_missing_weight_is_error(pisa_df):
    df = pisa_df.drop(columns=["W_FSTUWT"])
    assert "weight_missing" in _codes(errors(validate_processed(df)))


def test_nonpositive_weight_is_error(pisa_df):
    df = pisa_df.copy()
    df.loc[df.index[0], "W_FSTUWT"] = -1.0
    assert "weight_nonpositive" in _codes(errors(validate_processed(df)))


def test_no_plausible_values_is_error(pisa_df):
    df = pisa_df.drop(columns=[c for c in pisa_df.columns if c.startswith("PV")])
    assert "pv_missing" in _codes(errors(validate_processed(df)))


def test_bad_cycle_is_error(pisa_df):
    df = pisa_df.copy()
    df["CYCLE"] = 1999
    assert "cycle_invalid" in _codes(errors(validate_processed(df)))


def test_target_out_of_range_is_error(pisa_df):
    df = add_point_target(pisa_df.copy(), domain="math")
    df["AT_RISK_MATH"] = 2  # corrupt
    assert "target_range" in _codes(errors(validate_processed(df)))


def test_absent_replicates_warns_not_errors(pisa_df):
    df = pisa_df.drop(columns=[c for c in pisa_df.columns if c.startswith("W_FSTURWT")])
    vs = validate_processed(df)
    assert "brr_absent" in _codes(vs)
    assert errors(vs) == []


def test_all_missing_feature_warns(pisa_df):
    df = pisa_df.copy()
    df["ESCS"] = np.nan
    vs = validate_processed(df, feature_cols=["ESCS", "HOMEPOS"])
    assert "feature_all_missing" in _codes(vs)


def test_assert_valid_raises_on_error(pisa_df):
    with pytest.raises(ValueError):
        assert_valid(pisa_df.drop(columns=["W_FSTUWT"]))


def test_assert_valid_passes_clean_frame(pisa_df):
    assert_valid(pisa_df)  # should not raise


def test_replicate_coverage_by_cycle_reports_availability(pisa_df):
    rep = replicate_coverage_by_cycle(pisa_df)
    assert rep.loc[0, "brr_available"]
    assert rep.loc[0, "n_replicates"] == 4
