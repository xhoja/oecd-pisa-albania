"""Tests for feature engineering - focus on the school-context aggregates that
now feed the headline model (build_model_data(add_school_context=True))."""
import numpy as np
import pandas as pd

import pytest

from src.features.engineer import add_school_aggregates, add_school_questionnaire


def test_school_mean_leave_one_out_unweighted():
    # School 1: three students; each student's SCH_MEAN excludes their own value.
    df = pd.DataFrame({
        "CNTSCHID": [1, 1, 1],
        "ESCS": [1.0, 2.0, 3.0],
        "W_FSTUWT": [1.0, 1.0, 1.0],
    })
    out = add_school_aggregates(df, cols=["ESCS"])
    # LOO means: (2+3)/2, (1+3)/2, (1+2)/2
    np.testing.assert_allclose(out["SCH_MEAN_ESCS"].values, [2.5, 2.0, 1.5])
    assert (out["SCH_N"] == 3).all()


def test_school_mean_leave_one_out_weighted():
    df = pd.DataFrame({
        "CNTSCHID": [1, 1, 1],
        "ESCS": [1.0, 2.0, 3.0],
        "W_FSTUWT": [1.0, 2.0, 3.0],
    })
    out = add_school_aggregates(df, cols=["ESCS"])
    # wx_sum=14, wv_sum=6; student0 LOO = (14-1)/(6-1)=2.6
    assert out["SCH_MEAN_ESCS"].iloc[0] == np.float64(13 / 5)
    assert out["SCH_MEAN_ESCS"].iloc[1] == np.float64((14 - 4) / (6 - 2))  # 10/4=2.5
    assert out["SCH_MEAN_ESCS"].iloc[2] == np.float64((14 - 9) / (6 - 3))  # 5/3


def test_singleton_school_is_nan():
    # A school with one sampled student has no leave-one-out peer → NaN (imputer fills it).
    df = pd.DataFrame({
        "CNTSCHID": [1, 2, 2],
        "ESCS": [5.0, 1.0, 2.0],
        "W_FSTUWT": [1.0, 1.0, 1.0],
    })
    out = add_school_aggregates(df, cols=["ESCS"])
    assert np.isnan(out["SCH_MEAN_ESCS"].iloc[0])
    np.testing.assert_allclose(out["SCH_MEAN_ESCS"].iloc[1:].values, [2.0, 1.0])


def test_missing_value_uses_full_school_mean():
    # A student with a missing ESCS contributes nothing and their SCH_MEAN is the
    # full-school weighted mean of the peers who do have a value (no self to drop).
    df = pd.DataFrame({
        "CNTSCHID": [1, 1, 1],
        "ESCS": [np.nan, 2.0, 4.0],
        "W_FSTUWT": [1.0, 1.0, 1.0],
    })
    out = add_school_aggregates(df, cols=["ESCS"])
    assert out["SCH_MEAN_ESCS"].iloc[0] == 3.0  # (2+4)/2
    np.testing.assert_allclose(out["SCH_MEAN_ESCS"].iloc[1:].values, [4.0, 2.0])


def test_missing_school_column_is_noop():
    df = pd.DataFrame({"ESCS": [1.0, 2.0]})
    out = add_school_aggregates(df, cols=["ESCS"])
    assert "SCH_MEAN_ESCS" not in out.columns  # gracefully skipped


# --- add_school_questionnaire -------------------------------------------------

def test_questionnaire_join_preserves_rows_and_prefixes():
    df = pd.DataFrame({"CNTSCHID": [1, 1, 2], "ESCS": [0.0, 1.0, 2.0]})
    sch = pd.DataFrame({"CNTSCHID": [1, 2], "STRATIO": [10.0, 20.0]})
    out = add_school_questionnaire(df, sch, cols=["STRATIO"])
    assert len(out) == 3                       # no row multiplication
    assert list(out["SCHQ_STRATIO"]) == [10.0, 10.0, 20.0]
    assert "STRATIO" not in out.columns        # only the prefixed name is added


def test_questionnaire_unmatched_school_is_nan():
    df = pd.DataFrame({"CNTSCHID": [1, 3], "ESCS": [0.0, 1.0]})
    sch = pd.DataFrame({"CNTSCHID": [1, 2], "STRATIO": [10.0, 20.0]})
    out = add_school_questionnaire(df, sch, cols=["STRATIO"])
    assert out.loc[out["CNTSCHID"] == 1, "SCHQ_STRATIO"].iloc[0] == 10.0
    assert np.isnan(out.loc[out["CNTSCHID"] == 3, "SCHQ_STRATIO"].iloc[0])  # unseen -> NaN


def test_questionnaire_dedup_school_side_no_fanout():
    df = pd.DataFrame({"CNTSCHID": [1, 2], "ESCS": [0.0, 1.0]})
    sch = pd.DataFrame({"CNTSCHID": [1, 1, 2], "STRATIO": [10.0, 99.0, 20.0]})
    out = add_school_questionnaire(df, sch, cols=["STRATIO"])
    assert len(out) == 2                       # duplicate school rows collapsed


def test_questionnaire_default_cols_are_all_non_key():
    df = pd.DataFrame({"CNTSCHID": [1, 2]})
    sch = pd.DataFrame({"CNTSCHID": [1, 2], "STRATIO": [1.0, 2.0], "CLSIZE": [3.0, 4.0]})
    out = add_school_questionnaire(df, sch)
    assert {"SCHQ_STRATIO", "SCHQ_CLSIZE"} <= set(out.columns)


def test_questionnaire_missing_school_column_is_noop():
    df = pd.DataFrame({"ESCS": [1.0, 2.0]})           # no CNTSCHID
    sch = pd.DataFrame({"CNTSCHID": [1], "STRATIO": [10.0]})
    out = add_school_questionnaire(df, sch, cols=["STRATIO"])
    assert "SCHQ_STRATIO" not in out.columns          # gracefully skipped


def test_questionnaire_missing_key_in_school_frame_raises():
    df = pd.DataFrame({"CNTSCHID": [1], "ESCS": [0.0]})
    sch = pd.DataFrame({"SCHOOL": [1], "STRATIO": [10.0]})  # wrong key name
    with pytest.raises(ValueError):
        add_school_questionnaire(df, sch, cols=["STRATIO"])
