"""Tests for the earthquake difference-in-differences (src/causal)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.causal.did import band_rate_by_cycle, ddd_brr_rubin, did_brr_rubin
from src.causal.region import add_region_band, parse_stratum_label

THR = 420.07  # PISA math Level-2 lower bound


# --------------------------------------------------------------------------- #
# region-label parsing                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "label,expected",
    [
        ("ALB - stratum 03: Urban / Center / Public",
         {"urbanicity": "Urban", "region": "Center", "sector": "Public"}),
        (r"ALB - stratum 09: Rural \ Center \ Public",   # 2015 backslash style
         {"urbanicity": "Rural", "region": "Center", "sector": "Public"}),
        ("ALB - stratum 07: Rural / North / Public and Private",  # 2022 merged
         {"urbanicity": "Rural", "region": "North", "sector": "Public"}),
        ("ALB - stratum 12: Rural / South / Private",
         {"urbanicity": "Rural", "region": "South", "sector": "Private"}),
    ],
)
def test_parse_stratum_label(label, expected):
    assert parse_stratum_label(label) == expected


def test_parse_stratum_label_non_geographic():
    assert parse_stratum_label("Undisclosed STRATUM - Albania") is None
    assert parse_stratum_label(None) is None


def test_add_region_band_flags_treated():
    df = pd.DataFrame({"CYCLE": [2018, 2018], "STRATUM": ["ALB0203", "ALB0101"]})
    lookup = pd.DataFrame({
        "CYCLE": [2018, 2018],
        "STRATUM": ["ALB0203", "ALB0101"],
        "region": ["Center", "North"],
        "urbanicity": ["Urban", "Urban"],
        "sector": ["Public", "Public"],
    })
    out = add_region_band(df, lookup, treated_band="Center")
    assert out["region"].tolist() == ["Center", "North"]
    assert out["TREATED"].tolist() == [1, 0]


# --------------------------------------------------------------------------- #
# synthetic frame with a KNOWN planted difference-in-differences              #
# --------------------------------------------------------------------------- #
def _planted_frame(
    rates: dict[tuple[str, int], float],
    n: int = 500,
    urban_rates: dict | None = None,
    n_pv: int = 10,
    n_rep: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a 2-cycle frame whose weighted at-risk rate per (region, cycle)
    equals ``rates`` exactly (weights all 1). Every plausible value carries the
    same deterministic indicator so imputation variance is ~0 and the estimator
    must return the planted contrast. ``urban_rates`` optionally overrides rates
    per (region, cycle, urbanicity) for the triple-difference test.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    for (region, cycle), rate in rates.items():
        for urb in ("Urban", "Rural"):
            r = rate
            if urban_rates is not None:
                r = urban_rates[(region, cycle, urb)]
            k = int(round(r * n))
            scores = np.array([400.0] * k + [440.0] * (n - k))  # <thr = at-risk
            block = pd.DataFrame({
                "CYCLE": cycle,
                "region": region,
                "urbanicity": urb,
                "W_FSTUWT": 1.0,
                "CNTSCHID": rng.integers(1, 50, n),
            })
            for i in range(1, n_pv + 1):
                block[f"PV{i}MATH"] = scores
            for rr in range(1, n_rep + 1):
                block[f"W_FSTURWT{rr}"] = 1.0  # equal reps -> ~0 sampling var
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def test_did_recovers_planted_effect():
    # treated (Center) rises 0.20 -> 0.60 (+0.40); control 0.20 -> 0.50 (+0.30)
    # planted DiD = +0.10
    rates = {
        ("Center", 2018): 0.20, ("Center", 2022): 0.60,
        ("North", 2018): 0.20, ("North", 2022): 0.50,
        ("South", 2018): 0.20, ("South", 2022): 0.50,
    }
    df = _planted_frame(rates)
    res = did_brr_rubin(df, pre=2018, post=2022)
    assert res["did"] == pytest.approx(0.10, abs=1e-6)
    assert res["treated_change"] == pytest.approx(0.40, abs=1e-6)
    assert res["control_change"] == pytest.approx(0.30, abs=1e-6)
    # equal replicate weights + identical PVs -> essentially no variance
    assert res["se"] == pytest.approx(0.0, abs=1e-9)


def test_ddd_zero_when_urban_gap_moves_together():
    # Center and control both widen their urban-rural gap by the SAME amount
    # from pre to post -> triple-difference must be ~0.
    base = {("Center", 2018): 0.3, ("Center", 2022): 0.5,
            ("North", 2018): 0.3, ("North", 2022): 0.5,
            ("South", 2018): 0.3, ("South", 2022): 0.5}
    urban_rates = {}
    for (region, cycle) in base:
        # urban higher than rural by 0.1 pre, 0.2 post -> gap +0.1 everywhere
        lvl = base[(region, cycle)]
        gap = 0.1 if cycle == 2018 else 0.2
        urban_rates[(region, cycle, "Urban")] = lvl + gap / 2
        urban_rates[(region, cycle, "Rural")] = lvl - gap / 2
    df = _planted_frame(base, urban_rates=urban_rates)
    res = ddd_brr_rubin(df, pre=2018, post=2022)
    assert res["ddd"] == pytest.approx(0.0, abs=1e-6)


def test_band_rate_by_cycle_shape():
    rates = {("Center", 2018): 0.3, ("Center", 2022): 0.6,
             ("North", 2018): 0.3, ("North", 2022): 0.6,
             ("South", 2018): 0.3, ("South", 2022): 0.6}
    df = _planted_frame(rates)
    es = band_rate_by_cycle(df, cycles=(2018, 2022))
    assert set(es["region"]) == {"Center", "North", "South"}
    assert len(es) == 6
    center_2022 = es[(es.region == "Center") & (es.CYCLE == 2022)]["at_risk"].iloc[0]
    assert center_2022 == pytest.approx(0.6, abs=1e-6)
