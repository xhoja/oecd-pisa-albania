"""
Decision-curve analysis and calibration for a low-proficiency *screener*.

Why this exists: with a ~75% at-risk base rate, ROC-AUC is mechanically capped and
is the wrong yardstick for a triage tool. What a policymaker actually needs to know
is (a) whether acting on the model beats the trivial "screen everyone" / "screen
no-one" policies at their preferred risk threshold, and (b) whether the predicted
probabilities mean what they say. Those are **net benefit** (decision-curve
analysis, Vickers & Elkin 2006) and **calibration**.

All quantities are survey-weighted (PISA `W_FSTUWT`): an unweighted decision curve
would misstate population net benefit exactly like the descriptive shortcuts the
project criticises elsewhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.evaluate import expected_calibration_error


def _w(sample_weight: np.ndarray | None, n: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n, dtype=float)
    return np.asarray(sample_weight, dtype=float)


def net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> pd.DataFrame:
    r"""
    Weighted decision-curve analysis.

    At a threshold probability ``p_t`` we flag a student when ``y_prob >= p_t``.
    ``p_t`` encodes the caller's harm trade-off: acting on a false positive costs
    ``p_t/(1-p_t)`` times as much as missing a true positive. The **net benefit** is
    the true-positive rate minus the false-positive rate discounted by that odds:

    .. math::
        \mathrm{NB}(p_t)=\frac{TP_w}{N_w}
        -\frac{FP_w}{N_w}\,\frac{p_t}{1-p_t},

    with weighted counts ``TP_w = sum(w | y=1, pred=1)`` and
    ``FP_w = sum(w | y=0, pred=1)`` and ``N_w = sum(w)``. Two reference policies:

    - **treat-all** (flag everyone): ``NB = prev - (1-prev)*p_t/(1-p_t)``
    - **treat-none** (flag no-one): ``NB = 0``.

    The model is useful over the range of ``p_t`` where its curve sits **above
    both** references. Returns a tidy frame (one row per threshold).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = _w(sample_weight, len(y_true))
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    N = w.sum()
    prev = np.average(y_true, weights=w)  # weighted prevalence
    rows = []
    for pt in thresholds:
        pred = y_prob >= pt
        tp = w[(y_true == 1) & pred].sum()
        fp = w[(y_true == 0) & pred].sum()
        odds = pt / (1.0 - pt)
        nb_model = tp / N - (fp / N) * odds
        nb_all = prev - (1.0 - prev) * odds
        rows.append({
            "threshold": round(float(pt), 4),
            "net_benefit_model": nb_model,
            "net_benefit_all": nb_all,
            "net_benefit_none": 0.0,
        })
    df = pd.DataFrame(rows)
    df.attrs["prevalence"] = float(prev)
    return df


def weighted_brier(y_true: np.ndarray, y_prob: np.ndarray,
                   sample_weight: np.ndarray | None = None) -> float:
    r"""Weighted Brier score :math:`\frac{\sum_i w_i (p_i-y_i)^2}{\sum_i w_i}`
    (mean squared calibration error; lower is better)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = _w(sample_weight, len(y_true))
    return float(np.average((y_prob - y_true) ** 2, weights=w))


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Weighted reliability (calibration) curve: per probability bin, the mean
    predicted probability vs the weighted observed at-risk fraction. A perfectly
    calibrated model lies on the diagonal.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    w = _w(sample_weight, len(y_true))
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        # last bin is closed on the right so p == 1.0 is counted
        mask = (y_prob >= low) & (y_prob < high) if high < 1.0 else (y_prob >= low) & (y_prob <= high)
        wm = w[mask].sum()
        if wm == 0:
            continue
        rows.append({
            "bin_low": round(float(low), 3),
            "bin_high": round(float(high), 3),
            "mean_predicted": float(np.average(y_prob[mask], weights=w[mask])),
            "observed_fraction": float(np.average(y_true[mask], weights=w[mask])),
            "weight_share": float(wm / w.sum()),
        })
    return pd.DataFrame(rows)


def calibration_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> dict:
    """Weighted Brier + ECE + the reliability curve in one call."""
    return {
        "brier": weighted_brier(y_true, y_prob, sample_weight),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins, sample_weight=sample_weight),
        "prevalence": float(np.average(np.asarray(y_true, dtype=float), weights=_w(sample_weight, len(y_true)))),
        "reliability": reliability_curve(y_true, y_prob, n_bins, sample_weight),
    }
