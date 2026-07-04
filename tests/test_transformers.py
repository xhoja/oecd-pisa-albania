"""EngineeredFeatureBuilder: leakage-safe, fit-in-fold behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.transformers import EngineeredFeatureBuilder, SchoolMeansTransformer


def _train():
    # 40 rows; known-ish component distributions
    rng = np.random.default_rng(0)
    n = 40
    return pd.DataFrame({
        "ESCS": rng.normal(0.0, 1.0, n),
        "HOMEPOS": rng.normal(0.0, 1.0, n),
        "HISCED": rng.normal(4.0, 1.0, n),
        "HISEI": rng.normal(50.0, 10.0, n),
        "ANXMAT": rng.normal(0.0, 1.0, n),
        "BELONG": rng.normal(0.0, 1.0, n),
        "TEACHSUP": rng.normal(0.0, 1.0, n),
        "GRADE": rng.normal(0.0, 1.0, n),
        "GENDER": rng.integers(0, 2, n).astype(float),
    })


def test_fit_learns_train_statistics_only():
    train = _train()
    b = EngineeredFeatureBuilder().fit(train)
    assert b.means_["ESCS"] == pytest.approx(train["ESCS"].mean())
    assert b.stds_["HISEI"] == pytest.approx(train["HISEI"].std())


def test_transform_uses_train_stats_not_test_stats():
    train = _train()
    b = EngineeredFeatureBuilder().fit(train)
    # a test row whose every SES component equals the TRAIN mean -> z == 0
    row = {c: b.means_[c] for c in ["ESCS", "HOMEPOS", "HISCED", "HISEI"]}
    row.update({c: b.means_.get(c, 0.0) for c in ["ANXMAT", "BELONG", "TEACHSUP", "GRADE", "GENDER"]})
    test = pd.DataFrame([row])
    out = b.transform(test)
    assert out["SES_COMPLETE"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_material_deficit_uses_train_homepos_distribution():
    train = _train()
    b = EngineeredFeatureBuilder().fit(train)
    hi = train["HOMEPOS"].max() + 5     # richer than anyone in train
    lo = train["HOMEPOS"].min() - 5     # poorer than anyone in train
    test = train.copy()
    test.loc[test.index[0], "HOMEPOS"] = hi
    test.loc[test.index[1], "HOMEPOS"] = lo
    out = b.transform(test)
    assert out["MATERIAL_DEFICIT"].iloc[0] == pytest.approx(0.0, abs=1e-9)  # top of dist
    assert out["MATERIAL_DEFICIT"].iloc[1] == pytest.approx(1.0, abs=1e-9)  # bottom


def test_all_nan_interaction_is_skipped():
    train = _train()
    train["ANXMAT"] = np.nan  # ESCS x ANXMAT can never co-occur
    b = EngineeredFeatureBuilder().fit(train)
    assert "SES_x_ANXIETY" not in b.interactions_ok_
    assert "BELONG_x_TEACHSUP" in b.interactions_ok_  # this pair still co-occurs
    out = b.transform(train)
    assert "SES_x_ANXIETY" not in out.columns
    assert "BELONG_x_TEACHSUP" in out.columns


def test_fit_requires_dataframe():
    with pytest.raises(TypeError):
        EngineeredFeatureBuilder().fit(np.zeros((5, 3)))


# --- SchoolMeansTransformer ---------------------------------------------------

def _school_df():
    # two schools, 3 students each; unit weights for easy hand-calc
    return pd.DataFrame({
        "CNTSCHID": [1, 1, 1, 2, 2, 2],
        "W_FSTUWT": [1.0] * 6,
        "ESCS": [0.0, 1.0, 2.0, 10.0, 20.0, 30.0],
    })


def test_school_means_leave_one_out_on_train():
    df = _school_df()
    out = SchoolMeansTransformer(cols=["ESCS"]).fit_transform(df)
    # school 1, row0 LOO mean = mean(1,2) = 1.5 ; row1 = mean(0,2)=1.0 ; row2 = mean(0,1)=0.5
    assert list(out["SCH_MEAN_ESCS"].iloc[:3]) == [1.5, 1.0, 0.5]
    assert list(out["SCH_MEAN_ESCS"].iloc[3:]) == [25.0, 20.0, 15.0]
    assert list(out["SCH_N"]) == [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]


def test_school_means_transform_uses_full_train_mean():
    train = _school_df()
    tr = SchoolMeansTransformer(cols=["ESCS"]).fit(train)
    test = pd.DataFrame({"CNTSCHID": [1, 2], "W_FSTUWT": [1.0, 1.0], "ESCS": [99.0, 99.0]})
    out = tr.transform(test)
    # full train school means: school1 = 1.0, school2 = 20.0 (test row's own value ignored)
    assert list(out["SCH_MEAN_ESCS"]) == [1.0, 20.0]


def test_school_means_unseen_school_falls_back_to_global():
    train = _school_df()
    tr = SchoolMeansTransformer(cols=["ESCS"]).fit(train)
    test = pd.DataFrame({"CNTSCHID": [999], "W_FSTUWT": [1.0], "ESCS": [5.0]})
    out = tr.transform(test)
    global_mean = train["ESCS"].mean()  # unit weights -> plain mean
    assert out["SCH_MEAN_ESCS"].iloc[0] == pytest.approx(global_mean)
    assert out["SCH_N"].iloc[0] == 0.0   # no train students in this school


def test_school_means_drops_helper_keys():
    df = _school_df()
    out = SchoolMeansTransformer(cols=["ESCS"]).fit_transform(df)
    assert "CNTSCHID" not in out.columns
    assert "W_FSTUWT" not in out.columns
    assert "ESCS" in out.columns          # real features kept


def test_school_means_weighted_leave_one_out():
    df = pd.DataFrame({
        "CNTSCHID": [1, 1, 1],
        "W_FSTUWT": [1.0, 2.0, 3.0],
        "ESCS": [0.0, 1.0, 2.0],
    })
    out = SchoolMeansTransformer(cols=["ESCS"]).fit_transform(df)
    # row0 LOO weighted mean = (2*1 + 3*2)/(2+3) = 8/5 = 1.6
    assert out["SCH_MEAN_ESCS"].iloc[0] == pytest.approx(1.6)


def test_school_means_missing_school_col_is_noop():
    df = pd.DataFrame({"ESCS": [1.0, 2.0]})
    out = SchoolMeansTransformer(cols=["ESCS"]).fit_transform(df)
    assert "SCH_MEAN_ESCS" not in out.columns
