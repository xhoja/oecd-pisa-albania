"""EngineeredFeatureBuilder: leakage-safe, fit-in-fold behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.transformers import EngineeredFeatureBuilder


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
