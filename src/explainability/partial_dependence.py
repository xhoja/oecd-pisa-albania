"""
Partial-dependence (PDP) and individual-conditional-expectation (ICE) curves.

Thin, testable wrappers over ``sklearn.inspection.partial_dependence`` that keep
the raw arrays (rather than only a Matplotlib display) so they can be unit-tested
and re-plotted in the project's publication style. Both work on any fitted
estimator exposing ``predict_proba`` (the boosters, LR pipelines, ...).

PDP answers "on average, how does risk move as feature f varies?"; ICE keeps one
line per instance so heterogeneity (interactions) is visible where the average
would hide it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PDPResult:
    """Partial-dependence result for a single feature.

    ``grid`` is the feature axis; ``average`` is the PDP (mean predicted positive
    probability at each grid point); ``ice`` is ``(n_instances, n_grid)`` or None
    when ICE was not requested.
    """

    feature: str
    grid: np.ndarray
    average: np.ndarray
    ice: np.ndarray | None = None


def compute_pdp(
    model: Any,
    X: pd.DataFrame,
    feature: str,
    grid_resolution: int = 30,
    ice: bool = True,
    target_class: int = 1,
) -> PDPResult:
    """Compute PDP (and optionally ICE) for one feature.

    For binary classifiers sklearn returns a curve per class; we keep
    ``target_class`` (the at-risk / positive class by default).
    """
    if feature not in X.columns:
        raise KeyError(f"feature {feature!r} not in X columns")

    kind = "both" if ice else "average"
    res = partial_dependence(
        model, X, [feature], kind=kind, grid_resolution=grid_resolution,
    )
    grid = np.asarray(res["grid_values"][0])

    def _pick(arr: np.ndarray) -> np.ndarray:
        # sklearn shapes: average -> (n_classes, n_grid) for multiclass predict_proba;
        # (1, n_grid) when it collapses the binary positive class. Pick the right row.
        arr = np.asarray(arr)
        if arr.shape[0] == 1:
            return arr[0]
        return arr[target_class]

    average = _pick(res["average"])
    ice_arr = None
    if ice:
        individual = np.asarray(res["individual"])  # (n_classes, n_instances, n_grid)
        ice_arr = individual[0] if individual.shape[0] == 1 else individual[target_class]

    logger.info("PDP computed", feature=feature, n_grid=len(grid), ice=ice)
    return PDPResult(feature=feature, grid=grid, average=average, ice=ice_arr)


def centered_ice(result: PDPResult) -> np.ndarray | None:
    """c-ICE: subtract each instance's value at the first grid point so all lines
    start at zero, isolating the *shape* of each instance's response. Returns None
    if the result carries no ICE curves."""
    if result.ice is None:
        return None
    return result.ice - result.ice[:, [0]]
