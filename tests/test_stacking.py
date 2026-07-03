"""
Phase 9 stacking ensemble: the builder must produce a weight-routable,
leakage-safe estimator that scores like any other registry model.

The macOS OpenMP behaviour (multiple boosters in one process) is exercised by
scripts/run_stacking_experiment.py end-to-end, not here — these tests stay fast
and booster-light so they run in the normal suite without a subprocess.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.models.evaluate import fit_with_sample_weight
from src.models.experiment import _wrap
from src.models.registry import BUILDERS, build_stacking_ensemble, get_model


def _xy(n: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "ESCS": rng.normal(0, 1, n),
        "HOMEPOS": rng.normal(0, 1, n),
        "ANXMAT": rng.normal(0, 1, n),
    })
    # a learnable signal so predict_proba isn't degenerate
    logit = 0.9 * X["ESCS"] - 0.7 * X["ANXMAT"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    w = rng.uniform(1, 20, n)
    return X, y, w


def test_stacking_registered_and_built():
    assert "stacking" in BUILDERS
    model = get_model("stacking")
    assert isinstance(model, StackingClassifier)
    # meta-learner is a *plain* LR (bare, so it accepts sample_weight), not the
    # scaler-wrapped registry LR whose fit would reject a bare sample_weight.
    assert isinstance(model.final_estimator, LogisticRegression)
    # base learners must all be bare estimators (no inner scaler Pipeline), else
    # StackingClassifier's forwarded sample_weight would raise on them.
    from sklearn.pipeline import Pipeline
    assert not any(isinstance(est, Pipeline) for _, est in model.estimators)
    # sequential base fits keep the macOS OpenMP runtime from colliding
    assert model.n_jobs == 1


def test_stacking_weighted_fit_routes():
    # A booster-free stack so the test never touches libomp; the weight path
    # (StackingClassifier -> each bare base -> bare final estimator) is identical.
    X, y, w = _xy()
    stack = build_stacking_ensemble(
        base_models=["gradient_boosting", "random_forest", "extra_trees"],
    )
    assert fit_with_sample_weight(_wrap(stack), X, y, w) is True


def test_stacking_predicts_proba_in_unit_interval():
    X, y, w = _xy()
    stack = build_stacking_ensemble(base_models=["gradient_boosting", "random_forest"])
    pipe = _wrap(stack)
    fit_with_sample_weight(pipe, X, y, w)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


def test_custom_meta_learner_override():
    # passing a registry name should route through get_model for the meta-learner
    stack = build_stacking_ensemble(
        base_models=["random_forest", "extra_trees"], meta_learner="decision_tree",
    )
    from sklearn.tree import DecisionTreeClassifier
    assert isinstance(stack.final_estimator, DecisionTreeClassifier)
