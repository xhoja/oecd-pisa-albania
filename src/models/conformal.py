r"""
Split-conformal prediction for the low-proficiency risk classifier.

A calibrated probability says *how likely* a student is at risk; a conformal
**prediction set** says which labels are jointly plausible at a chosen risk level
:math:`\alpha`, with a finite-sample coverage guarantee
:math:`P(y \in \hat C(x)) \ge 1-\alpha` that holds under exchangeability *without*
assuming the model is correct. For a screener this turns a point score into an
honest "confident / abstain" signal: a **singleton** set is a confident call, the
**both-labels** set :math:`\{0,1\}` is the model declining to commit.

Method (split / inductive conformal, Vovk; Lei et al. 2018):

1. Fit the model on a proper-training split.
2. On a held-out **calibration** split, score each point by its non-conformity
   :math:`s_i = 1 - \hat p(y_i \mid x_i)` (one minus the predicted probability of
   the *true* class - large = the model was surprised).
3. Take the survey-**weighted** :math:`\lceil (n+1)(1-\alpha) \rceil / n` quantile
   :math:`\hat q` of the calibration scores.
4. For a test point include every label whose score is small enough:
   :math:`\hat C(x) = \{k : 1-\hat p(k\mid x) \le \hat q\}
   = \{k : \hat p(k\mid x) \ge 1-\hat q\}`.

``Mondrian`` (class-conditional) calibration computes a separate :math:`\hat q`
per true class, which restores per-class coverage under the 75%-prevalence
imbalance (marginal conformal can under-cover the rare not-at-risk class).

Weights: PISA students carry sampling weights; the weighted quantile makes the
guarantee hold for the *population*, not the unweighted sample.
"""
from __future__ import annotations

import numpy as np


def weighted_quantile(scores: np.ndarray, level: float, weights: np.ndarray | None = None) -> float:
    """
    Smallest ``t`` such that the (weight-normalized) fraction of ``scores`` ≤ ``t``
    is ≥ ``level``. Plain quantile when ``weights`` is None. ``level`` is clipped
    to [0, 1]; ``level`` ≥ 1 returns the max score (an all-inclusive threshold).
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return np.inf
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=float)
    level = float(np.clip(level, 0.0, 1.0))
    order = np.argsort(s)
    s, w = s[order], w[order]
    cw = np.cumsum(w)
    cutoff = level * cw[-1]
    idx = int(np.searchsorted(cw, cutoff, side="left"))
    idx = min(idx, len(s) - 1)
    return float(s[idx])


def conformal_quantile_level(n: int, alpha: float) -> float:
    r"""The finite-sample corrected level :math:`\lceil (n+1)(1-\alpha)\rceil / n`
    (clipped to 1.0 when it would exceed the sample)."""
    if n <= 0:
        return 1.0
    return min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)


def calibrate(
    proba_cal: np.ndarray,
    y_cal: np.ndarray,
    alpha: float = 0.1,
    weights: np.ndarray | None = None,
    mondrian: bool = False,
) -> dict:
    """
    Compute the conformal threshold(s) from calibration-set probabilities.

    Args:
        proba_cal: (n, 2) predicted class probabilities on the calibration split.
        y_cal: (n,) true labels (0/1).
        alpha: miscoverage level (0.1 → ≥90% coverage guarantee).
        weights: optional survey weights for a population-valid quantile.
        mondrian: class-conditional thresholds (separate q per true class).

    Returns a dict with ``alpha``, ``mondrian``, and either ``qhat`` (marginal) or
    ``qhat_by_class`` (Mondrian).
    """
    proba_cal = np.asarray(proba_cal, dtype=float)
    y_cal = np.asarray(y_cal).astype(int)
    w = None if weights is None else np.asarray(weights, dtype=float)
    # non-conformity of the TRUE class
    true_p = proba_cal[np.arange(len(y_cal)), y_cal]
    scores = 1.0 - true_p

    if not mondrian:
        lvl = conformal_quantile_level(len(scores), alpha)
        return {"alpha": alpha, "mondrian": False,
                "qhat": weighted_quantile(scores, lvl, w)}

    qhat_by_class: dict[int, float] = {}
    for k in (0, 1):
        m = y_cal == k
        wk = None if w is None else w[m]
        lvl = conformal_quantile_level(int(m.sum()), alpha)
        qhat_by_class[k] = weighted_quantile(scores[m], lvl, wk)
    return {"alpha": alpha, "mondrian": True, "qhat_by_class": qhat_by_class}


def prediction_sets(proba: np.ndarray, cal: dict) -> np.ndarray:
    r"""
    Boolean (n, 2) membership matrix: include label ``k`` iff
    :math:`\hat p(k\mid x) \ge 1-\hat q` (per-class :math:`\hat q` under Mondrian).
    """
    proba = np.asarray(proba, dtype=float)
    if cal["mondrian"]:
        thr = np.array([1.0 - cal["qhat_by_class"][0], 1.0 - cal["qhat_by_class"][1]])
    else:
        thr = np.array([1.0 - cal["qhat"], 1.0 - cal["qhat"]])
    return proba >= thr[None, :]


def set_summary(sets: np.ndarray, y_true: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """
    Coverage and set-size profile for prediction sets.

    Returns weighted (population) coverage, average set size, and the shares of
    empty / singleton / both-labels (ambiguous) sets - the operational read: high
    singleton share = confident screener, high ambiguous share = it abstains often.
    """
    sets = np.asarray(sets, dtype=bool)
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    wsum = w.sum()

    covered = sets[np.arange(n), y_true]
    sizes = sets.sum(axis=1)
    return {
        "coverage": float((w * covered).sum() / wsum),
        "avg_set_size": float((w * sizes).sum() / wsum),
        "share_empty": float((w * (sizes == 0)).sum() / wsum),
        "share_singleton": float((w * (sizes == 1)).sum() / wsum),
        "share_ambiguous": float((w * (sizes == 2)).sum() / wsum),
        "n": int(n),
    }
