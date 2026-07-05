"""Statistical association/effect-size helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_pisa_frame
from src.statistics.tests import feature_effect_sizes, weighted_cohens_d


def _frame_with_target(seed: int = 0) -> pd.DataFrame:
    df = make_pisa_frame(n=600, seed=seed)
    # deterministic target correlated with ESCS so d has a known sign
    df["AT_RISK_MATH"] = (df["ESCS"] < df["ESCS"].median()).astype(int)
    df["REPEAT"] = (df.index % 4 == 0).astype(float)
    df["IMMIG"] = (df.index % 7 == 0).astype(float)
    return df


def test_feature_effect_sizes_ranks_by_magnitude_and_signs():
    df = _frame_with_target()
    out = feature_effect_sizes(
        df,
        numeric_features=["ESCS", "HOMEPOS", "ANXMAT"],
        categorical_features=["GENDER", "REPEAT"],
    )
    # sorted descending by magnitude
    assert list(out["magnitude"]) == sorted(out["magnitude"], reverse=True)
    # ESCS drives the target -> strong negative d (at-risk have lower ESCS)
    escs = out[out.feature == "ESCS"].iloc[0]
    assert escs["measure"] == "cohens_d"
    assert escs["value"] < 0
    assert escs["magnitude"] == abs(escs["value"])
    # categoricals report Cramer's V + a p-value
    rep = out[out.feature == "REPEAT"].iloc[0]
    assert rep["measure"] == "cramers_v"
    assert not np.isnan(rep["p_value"])


def test_feature_effect_sizes_value_matches_weighted_cohens_d():
    df = _frame_with_target()
    out = feature_effect_sizes(df, numeric_features=["HOMEPOS"], categorical_features=[])
    at = df["AT_RISK_MATH"] == 1
    d = weighted_cohens_d(
        df.loc[at, "HOMEPOS"].values, df.loc[~at, "HOMEPOS"].values,
        df.loc[at, "W_FSTUWT"].values, df.loc[~at, "W_FSTUWT"].values,
    )
    assert out.loc[0, "value"] == pytest.approx(round(d, 3))


def test_feature_effect_sizes_skips_absent_columns():
    df = _frame_with_target()
    out = feature_effect_sizes(
        df, numeric_features=["ESCS", "NOPE"], categorical_features=["ALSO_NOPE"]
    )
    assert list(out["feature"]) == ["ESCS"]
