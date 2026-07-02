"""Phase 6 explainability: local SHAP case selection, local SHAP values, PDP/ICE.

Uses a RandomForest so the tests stay free of the boosters' libomp entirely.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.explainability.partial_dependence import centered_ice, compute_pdp
from src.explainability.shap_analysis import (
    compute_local_shap,
    select_representative_cases,
)
from src.features.transformers import EngineeredFeatureBuilder
from src.models.prepare import build_model_data, impute_median

FEATS = ["ESCS", "HOMEPOS", "GENDER", "BELONG", "TEACHSUP", "ANXMAT", "GRADE", "HISCED", "HISEI"]


@pytest.fixture
def fitted(pisa_df):
    data = build_model_data(pisa_df, FEATS, domain="math")
    X = impute_median(EngineeredFeatureBuilder().fit_transform(data.X))[0]
    y = data.y.values
    model = RandomForestClassifier(n_estimators=60, random_state=0).fit(X, y)
    return model, X, y


def test_select_cases_quadrants_are_consistent(fitted):
    model, X, y = fitted
    cases = select_representative_cases(model, X, y)
    assert cases, "expected at least one quadrant"
    for k, c in cases.items():
        # the quadrant label must match the (y_true, y_pred) pair
        expected = {"TP": (1, 1), "TN": (0, 0), "FP": (0, 1), "FN": (1, 0)}[k]
        assert (c["y_true"], c["y_pred"]) == expected
        assert 0.0 <= c["prob"] <= 1.0
        assert y[c["index"]] == c["y_true"]


def test_select_cases_picks_most_confident(fitted):
    model, X, y = fitted
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    cases = select_representative_cases(model, X, y)
    if "TP" in cases:  # chosen TP is the highest-prob true positive
        tp_mask = (y == 1) & (pred == 1)
        assert cases["TP"]["prob"] == pytest.approx(prob[tp_mask].max())
    if "FN" in cases:  # chosen FN is the lowest-prob false negative
        fn_mask = (y == 1) & (pred == 0)
        assert cases["FN"]["prob"] == pytest.approx(prob[fn_mask].min())


def test_local_shap_reconstructs_prediction(fitted):
    model, X, y = fitted
    rows = X.iloc[[0, 1, 2]]
    vals, base, names = compute_local_shap(model, rows, X_background=X.sample(50, random_state=0))
    assert vals.shape == (3, X.shape[1])
    assert base.shape == (3,)
    assert names == list(X.columns)
    # TreeSHAP additivity: base + sum(shap) == model probability for each row
    recon = base + vals.sum(axis=1)
    assert np.allclose(recon, model.predict_proba(rows)[:, 1], atol=1e-4)


def test_pdp_shapes_and_monotone_grid(fitted):
    model, X, _ = fitted
    pdp = compute_pdp(model, X, "ANXMAT", grid_resolution=20, ice=True)
    assert pdp.grid.ndim == 1 and len(pdp.grid) <= 20
    assert pdp.average.shape == pdp.grid.shape
    assert np.all(np.diff(pdp.grid) > 0)  # grid strictly increasing
    assert pdp.ice.shape == (len(X), len(pdp.grid))
    assert np.all((pdp.average >= 0) & (pdp.average <= 1))  # probabilities


def test_centered_ice_starts_at_zero(fitted):
    model, X, _ = fitted
    pdp = compute_pdp(model, X, "ESCS", grid_resolution=15, ice=True)
    cice = centered_ice(pdp)
    assert np.allclose(cice[:, 0], 0.0)
    assert compute_pdp(model, X, "ESCS", ice=False).ice is None


def test_compute_pdp_rejects_unknown_feature(fitted):
    model, X, _ = fitted
    with pytest.raises(KeyError):
        compute_pdp(model, X, "NOT_A_FEATURE")
