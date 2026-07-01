"""Rubin's rules + missingness indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.impute import add_missingness_indicators, rubins_combine


def test_rubins_combine_textbook():
    # estimates [1,2,3], within-var 0.5 each, m=3
    #   Q_bar = 2 ; U_bar = 0.5 ; B = var(ddof=1) = 1.0
    #   T = 0.5 + (1+1/3)*1.0 = 1.8333...
    est, se, fmi = rubins_combine([1.0, 2.0, 3.0], [0.5, 0.5, 0.5])
    assert est == pytest.approx(2.0)
    assert se == pytest.approx(np.sqrt(1.8333333), rel=1e-5)
    assert fmi == pytest.approx((4 / 3 * 1.0) / 1.8333333, rel=1e-5)


def test_rubins_combine_zero_between_variance():
    est, se, fmi = rubins_combine([2.0, 2.0, 2.0], [0.25, 0.25, 0.25])
    assert est == pytest.approx(2.0)
    assert se == pytest.approx(0.5)   # sqrt(U_bar), B=0
    assert fmi == pytest.approx(0.0)


def test_missingness_indicator_added_for_partial_column():
    df = pd.DataFrame({"A": [1.0, np.nan, 3.0, np.nan], "B": [1.0, 2.0, 3.0, 4.0]})
    out = add_missingness_indicators(df, ["A", "B"], threshold=0.95)
    assert "_MISS_A" in out.columns
    assert out["_MISS_A"].tolist() == [0, 1, 0, 1]


def test_missingness_indicator_skipped_for_complete_column():
    df = pd.DataFrame({"B": [1.0, 2.0, 3.0, 4.0]})
    out = add_missingness_indicators(df, ["B"], threshold=0.95)
    assert "_MISS_B" not in out.columns
