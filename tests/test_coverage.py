"""Tests for Manski coverage bounds (src/coverage/manski)."""
from __future__ import annotations

import numpy as np
import pytest

from src.coverage.manski import (
    coverage_sensitivity,
    monotone_bounds,
    scenario_rate,
    worst_case_bounds,
)


def test_worst_case_width_is_one_minus_coverage():
    b = worst_case_bounds(0.74, 0.79)
    assert b.width == pytest.approx(0.21, abs=1e-9)
    assert b.lower == pytest.approx(0.79 * 0.74)
    assert b.upper == pytest.approx(0.79 * 0.74 + 0.21)


def test_full_coverage_collapses_bounds():
    b = worst_case_bounds(0.74, 1.0)
    assert b.lower == pytest.approx(0.74)
    assert b.upper == pytest.approx(0.74)
    assert b.width == pytest.approx(0.0)


def test_monotone_uncovered_worse_tightens_lower_to_p():
    p, c = 0.74, 0.79
    m = monotone_bounds(p, c, uncovered_worse=True)
    wc = worst_case_bounds(p, c)
    assert m.lower == pytest.approx(p)          # lower bound raised to observed
    assert m.upper == pytest.approx(wc.upper)   # upper unchanged
    assert m.lower >= wc.lower


def test_monotone_uncovered_better_tightens_upper_to_p():
    m = monotone_bounds(0.74, 0.79, uncovered_worse=False)
    assert m.upper == pytest.approx(0.74)
    assert m.lower == pytest.approx(0.79 * 0.74)


def test_scenario_rate_is_convex_mix_within_bounds():
    p, c = 0.74, 0.79
    for u in (0.0, 0.3, 0.74, 0.9, 1.0):
        r = scenario_rate(p, c, u)
        wc = worst_case_bounds(p, c)
        assert wc.lower - 1e-9 <= r <= wc.upper + 1e-9
    # u = p reproduces the observed rate exactly
    assert scenario_rate(p, c, p) == pytest.approx(p)


def test_scenario_endpoints_hit_bounds():
    p, c = 0.74, 0.79
    assert scenario_rate(p, c, 0.0) == pytest.approx(worst_case_bounds(p, c).lower)
    assert scenario_rate(p, c, 1.0) == pytest.approx(worst_case_bounds(p, c).upper)


def test_coverage_sensitivity_monotone_in_width():
    rows = coverage_sensitivity(0.74, np.array([0.7, 0.8, 0.9, 1.0]))
    widths = [r["wc_width"] for r in rows]
    assert widths == sorted(widths, reverse=True)  # more coverage -> narrower
    assert rows[-1]["wc_width"] == pytest.approx(0.0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        worst_case_bounds(1.5, 0.8)
    with pytest.raises(ValueError):
        worst_case_bounds(0.7, 0.0)
    with pytest.raises(ValueError):
        scenario_rate(0.7, 0.8, 1.2)
