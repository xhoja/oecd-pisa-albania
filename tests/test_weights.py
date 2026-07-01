"""Weighted statistics + BRR variance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.weights import (
    brr_se,
    has_replicate_weights,
    normalize_weights_within_country,
    pv_statistic_brr,
    replicate_weight_cols,
    weighted_mean,
    weighted_proportion,
    weighted_quantile,
    weighted_std,
)


def test_weighted_mean_equal_weights():
    v = pd.Series([1.0, 2.0, 3.0, 4.0])
    w = pd.Series([1.0, 1.0, 1.0, 1.0])
    assert weighted_mean(v, w) == pytest.approx(2.5)


def test_weighted_mean_concentrated_weight():
    v = pd.Series([1.0, 2.0, 3.0, 4.0])
    w = pd.Series([0.0, 0.0, 0.0, 1.0])
    assert weighted_mean(v, w) == pytest.approx(4.0)


def test_weighted_mean_ignores_nan_pairs():
    v = pd.Series([1.0, np.nan, 3.0])
    w = pd.Series([1.0, 5.0, 1.0])
    assert weighted_mean(v, w) == pytest.approx(2.0)


def test_weighted_std_matches_population_std_under_equal_weights():
    v = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    w = pd.Series(np.ones(5))
    assert weighted_std(v, w) == pytest.approx(float(np.std(v.values)))  # ddof=0


def test_weighted_proportion_is_weighted_mean_of_indicator():
    ind = pd.Series([1, 0, 1, 0])
    w = pd.Series([3.0, 1.0, 1.0, 1.0])
    # weighted mean = (3+0+1+0)/6
    assert weighted_proportion(ind, w) == pytest.approx(4 / 6)


def test_weighted_quantile_monotone_and_bounded():
    v = pd.Series(np.arange(1.0, 11.0))
    w = pd.Series(np.ones(10))
    qs = weighted_quantile(v, w, [0.1, 0.5, 0.9])
    assert qs[0] < qs[1] < qs[2]
    assert v.min() <= qs[0] and qs[2] <= v.max()


def test_normalize_weights_sums_to_n_within_country(two_country_df):
    out = normalize_weights_within_country(two_country_df)
    for _, g in out.groupby("COUNTRY"):
        assert g["W_FSTUWT_NORM"].sum() == pytest.approx(len(g))


def test_replicate_helpers(pisa_df):
    cols = replicate_weight_cols(pisa_df)
    assert cols == [f"W_FSTURWT{i}" for i in range(1, 5)]
    assert has_replicate_weights(pisa_df) is True


def test_has_replicate_weights_false_without_them(pisa_df):
    stripped = pisa_df.drop(columns=replicate_weight_cols(pisa_df))
    assert has_replicate_weights(stripped) is False


def test_brr_se_is_zero_when_replicates_equal_base(pisa_df_equal_reps):
    def at_risk(df, wcol):
        return weighted_proportion((df["PV1MATH"] < 420.07).astype(float), df[wcol])

    est, se = brr_se(pisa_df_equal_reps, at_risk)
    assert 0.0 <= est <= 1.0
    assert se == pytest.approx(0.0, abs=1e-9)


def test_brr_se_nan_without_replicates(pisa_df):
    stripped = pisa_df.drop(columns=replicate_weight_cols(pisa_df))
    _, se = brr_se(stripped, lambda df, w: weighted_mean(df["PV1MATH"], df[w]))
    assert np.isnan(se)


def test_pv_brr_zero_variance_when_reps_equal_and_pvs_identical(pisa_df_equal_reps):
    df = pisa_df_equal_reps.copy()
    # collapse every PV to PV1 -> between-PV variance B == 0
    for i in range(2, 11):
        df[f"PV{i}MATH"] = df["PV1MATH"]
    res = pv_statistic_brr(df, statistic="at_risk")
    expected = weighted_proportion((df["PV1MATH"] < 420.07).astype(float), df["W_FSTUWT"])
    assert res["brr_available"] is True
    assert res["estimate"] == pytest.approx(expected)
    assert res["imputation_var"] == pytest.approx(0.0, abs=1e-12)
    assert res["sampling_var"] == pytest.approx(0.0, abs=1e-9)
    assert res["se"] == pytest.approx(0.0, abs=1e-9)


def test_pv_brr_reports_both_variance_sources(pisa_df):
    res = pv_statistic_brr(pisa_df, statistic="at_risk")
    assert res["n_pv"] == 10
    assert res["n_replicates"] == 4
    assert res["brr_available"] is True
    # real synthetic data -> both sources strictly positive
    assert res["imputation_var"] > 0
    assert res["sampling_var"] > 0
    assert res["se"] > 0
    assert 0.0 <= res["fmi"] <= 1.0
    assert res["ci95_low"] < res["estimate"] < res["ci95_high"]


def test_pv_brr_flags_absent_replicates(pisa_df):
    stripped = pisa_df.drop(columns=replicate_weight_cols(pisa_df))
    res = pv_statistic_brr(stripped, statistic="at_risk")
    assert res["brr_available"] is False
    assert np.isnan(res["sampling_var"])
    assert res["se"] > 0  # imputation-only fallback


def test_pv_brr_ignores_all_nan_padded_pv_columns(pisa_df):
    # emulate FWF cycles: only PV1..PV5 populated, PV6..PV10 all NaN.
    df = pisa_df.copy()
    for i in range(6, 11):
        df[f"PV{i}MATH"] = np.nan
    res = pv_statistic_brr(df, statistic="at_risk")
    assert res["n_pv"] == 5  # padded columns dropped, not counted as 0-proportion
    real = np.mean([
        weighted_proportion((df[f"PV{i}MATH"] < 420.07).astype(float), df["W_FSTUWT"])
        for i in range(1, 6)
    ])
    assert res["estimate"] == pytest.approx(real)


def test_pv_brr_fallback_fmi_is_nan_without_replicates(pisa_df):
    stripped = pisa_df.drop(columns=replicate_weight_cols(pisa_df))
    res = pv_statistic_brr(stripped, statistic="at_risk")
    assert res["brr_available"] is False
    assert np.isnan(res["fmi"])  # tautological 1.0 avoided


def test_pv_brr_mean_score_matches_grand_pv_mean(pisa_df):
    res = pv_statistic_brr(pisa_df, statistic="mean_score")
    grand = np.mean([
        weighted_mean(pisa_df[f"PV{i}MATH"], pisa_df["W_FSTUWT"]) for i in range(1, 11)
    ])
    assert res["estimate"] == pytest.approx(grand)
