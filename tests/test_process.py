"""Tests for behavioural process-feature aggregation (src/features/process)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.process import (
    BEHAVIOURAL_COLS,
    aggregate_cog_process,
    link_process_features,
)


def _cog_frame() -> pd.DataFrame:
    """Two students, three cognitive items, hand-set times/visits/scores so the
    aggregates are known. Times in ms. Student 2 leaves item 3 unseen (NaN)."""
    return pd.DataFrame({
        "CNTSTUID": [1, 2],
        # item 1
        "CM001Q01TT": [10_000.0, 2_000.0],   # 10 s, 2 s (rapid for s2)
        "CM001Q01V": [1.0, 3.0],             # s2 revisits
        "CM001Q01S": [1.0, 0.0],
        # item 2
        "CM002Q01TT": [20_000.0, 4_000.0],   # 20 s, 4 s (rapid for s2)
        "CM002Q01V": [2.0, 1.0],             # s1 revisits
        "CM002Q01S": [0.0, 6.0],             # s2 not reached
        # item 3
        "CM003Q01TT": [30_000.0, np.nan],    # s2 never saw it
        "CM003Q01V": [1.0, np.nan],
        "CM003Q01S": [1.0, np.nan],
    })


def test_aggregate_counts_and_times():
    agg = aggregate_cog_process(_cog_frame(), rapid_ms=5_000).set_index("CNTSTUID")
    # student 1 saw 3 items, total 60 s, mean 20 s
    assert agg.loc[1, "PROC_N_ITEMS"] == 3
    assert agg.loc[1, "PROC_TOTAL_TIME_S"] == pytest.approx(60.0)
    assert agg.loc[1, "PROC_MEAN_TIME_S"] == pytest.approx(20.0)
    # student 2 saw 2 items (item 3 NaN), total 6 s
    assert agg.loc[2, "PROC_N_ITEMS"] == 2
    assert agg.loc[2, "PROC_TOTAL_TIME_S"] == pytest.approx(6.0)


def test_rapid_and_revisit_fractions():
    agg = aggregate_cog_process(_cog_frame(), rapid_ms=5_000).set_index("CNTSTUID")
    # student 1: no item < 5 s -> rapid 0; revisit on 1 of 3 items
    assert agg.loc[1, "PROC_RAPID_FRAC"] == pytest.approx(0.0)
    assert agg.loc[1, "PROC_REVISIT_RATE"] == pytest.approx(1 / 3)
    # student 2: both seen items < 5 s -> rapid 1.0; revisit on item 1 (3>1) of 2
    assert agg.loc[2, "PROC_RAPID_FRAC"] == pytest.approx(1.0)
    assert agg.loc[2, "PROC_REVISIT_RATE"] == pytest.approx(0.5)


def test_disengagement_uses_codes_not_correctness():
    agg = aggregate_cog_process(_cog_frame()).set_index("CNTSTUID")
    # only student 2's item 2 is Not Reached (code 6); correctness (0/1) ignored
    assert agg.loc[2, "PROC_NOT_REACHED"] == 1
    assert agg.loc[1, "PROC_NOT_REACHED"] == 0


def test_behavioural_cols_present_and_finite():
    agg = aggregate_cog_process(_cog_frame())
    for c in BEHAVIOURAL_COLS:
        assert c in agg.columns
        vals = pd.to_numeric(agg[c], errors="coerce").dropna().to_numpy(dtype=float)
        assert np.isfinite(vals).all()


def test_no_zero_division_for_unseen_student():
    # a student who saw zero timed items must not crash and yields NaN ratios
    df = pd.DataFrame({
        "CNTSTUID": [9],
        "CM001Q01TT": [np.nan], "CM001Q01V": [np.nan], "CM001Q01S": [np.nan],
    })
    agg = aggregate_cog_process(df)
    assert agg.loc[0, "PROC_N_ITEMS"] == 0
    assert pd.isna(agg.loc[0, "PROC_RAPID_FRAC"])


def test_link_process_features_left_join():
    students = pd.DataFrame({"CNTSTUID": [1, 2, 3], "ESCS": [0.1, -0.2, 0.3]})
    proc = pd.DataFrame({"CNTSTUID": [1, 2], "PROC_TOTAL_TIME_S": [60.0, 6.0]})
    out = link_process_features(students, proc)
    assert len(out) == 3
    assert out.set_index("CNTSTUID").loc[3, "PROC_TOTAL_TIME_S"] != out["PROC_TOTAL_TIME_S"].loc[0]
    assert pd.isna(out.set_index("CNTSTUID").loc[3, "PROC_TOTAL_TIME_S"])
