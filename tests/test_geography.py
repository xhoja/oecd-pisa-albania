"""Tests for the within-Albania geographic-equity statistics (src/geography)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.geography.equity import (
    atrisk_by_groups,
    gap_brr_rubin,
    region_urbanicity_matrix,
)


def _planted_frame(
    rates: dict[tuple[str, str, int], float],
    n: int = 400,
    n_pv: int = 10,
    n_rep: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Frame whose weighted at-risk rate per (region, urbanicity, cycle) equals
    ``rates`` exactly (unit weights). Identical PVs -> ~0 imputation variance;
    equal replicate weights -> ~0 sampling variance, so estimators must return
    the planted rate and a ~0 SE.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    for (region, urb, cycle), rate in rates.items():
        k = int(round(rate * n))
        scores = np.array([400.0] * k + [440.0] * (n - k))  # <420.07 = at-risk
        block = pd.DataFrame({
            "CYCLE": cycle,
            "region": region,
            "urbanicity": urb,
            "sector": "Public",
            "W_FSTUWT": 1.0,
            "CNTSCHID": rng.integers(1, 40, n),
        })
        for i in range(1, n_pv + 1):
            block[f"PV{i}MATH"] = scores
        for rr in range(1, n_rep + 1):
            block[f"W_FSTURWT{rr}"] = 1.0
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def test_atrisk_by_groups_recovers_planted_rates():
    rates = {
        ("North", "Urban", 2022): 0.80, ("North", "Rural", 2022): 0.90,
        ("South", "Urban", 2022): 0.60, ("South", "Rural", 2022): 0.70,
    }
    df = _planted_frame(rates)
    out = atrisk_by_groups(df, ["region", "urbanicity"], cycles=(2022,))
    assert len(out) == 4
    got = {
        (r.region, r.urbanicity): r.at_risk for r in out.itertuples()
    }
    for (region, urb, _cyc), rate in rates.items():
        assert got[(region, urb)] == pytest.approx(rate, abs=1e-6)
    # equal reps + identical PVs -> ~0 design SE
    assert out["se"].max() == pytest.approx(0.0, abs=1e-9)
    assert out["brr_available"].all()


def test_atrisk_by_groups_flags_small_cells():
    rates = {("North", "Urban", 2022): 0.5}
    big = _planted_frame(rates, n=100)
    small = _planted_frame({("South", "Urban", 2022): 0.5}, n=10)
    df = pd.concat([big, small], ignore_index=True)
    out = atrisk_by_groups(df, ["region"], cycles=(2022,), min_n=30)
    north = out[out.region == "North"].iloc[0]
    south = out[out.region == "South"].iloc[0]
    assert not north["small"]
    assert south["small"]


def test_atrisk_by_groups_drops_undecoded_cells():
    rates = {("North", "Urban", 2022): 0.5}
    df = _planted_frame(rates)
    df.loc[df.index[:50], "region"] = np.nan  # undecoded stratum
    out = atrisk_by_groups(df, ["region"], cycles=(2022,))
    assert out["region"].notna().all()
    assert out["n"].iloc[0] == len(df) - 50


def test_gap_recovers_planted_contrast_and_sign():
    # Rural 0.90, Urban 0.80 -> Rural - Urban = +0.10
    rates = {
        ("North", "Rural", 2022): 0.90, ("North", "Urban", 2022): 0.80,
    }
    df = _planted_frame(rates)
    res = gap_brr_rubin(df, "urbanicity", "Rural", "Urban", cycle=2022)
    assert res["gap"] == pytest.approx(0.10, abs=1e-6)
    assert res["rate_hi"] == pytest.approx(0.90, abs=1e-6)
    assert res["rate_lo"] == pytest.approx(0.80, abs=1e-6)
    assert res["se"] == pytest.approx(0.0, abs=1e-9)


def test_gap_is_antisymmetric():
    rates = {
        ("North", "Rural", 2022): 0.90, ("North", "Urban", 2022): 0.70,
    }
    df = _planted_frame(rates)
    a = gap_brr_rubin(df, "urbanicity", "Rural", "Urban", cycle=2022)["gap"]
    b = gap_brr_rubin(df, "urbanicity", "Urban", "Rural", cycle=2022)["gap"]
    assert a == pytest.approx(-b, abs=1e-9)
    assert a == pytest.approx(0.20, abs=1e-6)


def test_gap_raises_on_empty_level():
    rates = {("North", "Rural", 2022): 0.9}
    df = _planted_frame(rates)
    with pytest.raises(ValueError, match="empty level"):
        gap_brr_rubin(df, "urbanicity", "Rural", "Urban", cycle=2022)


def test_region_urbanicity_matrix_shape_and_values():
    rates = {
        ("North", "Urban", 2022): 0.79, ("North", "Rural", 2022): 0.83,
        ("Center", "Urban", 2022): 0.69, ("Center", "Rural", 2022): 0.82,
        ("South", "Urban", 2022): 0.71, ("South", "Rural", 2022): 0.79,
    }
    df = _planted_frame(rates)
    mat = region_urbanicity_matrix(df, 2022)
    assert list(mat.index) == ["Urban", "Rural"]
    assert list(mat.columns) == ["North", "Center", "South"]
    assert mat.loc["Rural", "North"] == pytest.approx(0.83, abs=1e-6)
    assert mat.loc["Urban", "Center"] == pytest.approx(0.69, abs=1e-6)
