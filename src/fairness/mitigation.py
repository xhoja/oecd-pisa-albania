"""
Post-hoc fairness *mitigation* (target #4) - not just diagnosis.

The audit (notebook 07) found the screener over-flags the poorest students:
at a single 0.5 threshold the bottom SES quintile carries a far higher false-
positive rate than the top. Mitigation here is **post-processing on frozen
out-of-fold predictions** - no retraining, no change to the model - via
*group-specific decision thresholds* (Hardt et al. 2016, "Equality of
Opportunity"): pick a per-group threshold so a chosen error rate is equalised
across groups, then read off the fairness-utility trade-off.

All rates are survey-weighted (``W_FSTUWT``), consistent with the rest of the
project. Everything operates on a dataframe of (y_true, y_prob, group, weight);
it is agnostic to how the probabilities were produced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def _weighted_rates(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Weighted TPR, FPR, selection rate and accuracy for one group."""
    pos, neg = (y == 1), (y == 0)
    wp, wn = w[pos].sum(), w[neg].sum()
    tpr = float(w[pos & (pred == 1)].sum() / wp) if wp > 0 else np.nan
    fpr = float(w[neg & (pred == 1)].sum() / wn) if wn > 0 else np.nan
    sel = float(w[pred == 1].sum() / w.sum()) if w.sum() > 0 else np.nan
    acc = float(w[(pred == y)].sum() / w.sum()) if w.sum() > 0 else np.nan
    return {"tpr": tpr, "fpr": fpr, "selection": sel, "accuracy": acc}


def threshold_for_target_fpr(
    y: np.ndarray, prob: np.ndarray, w: np.ndarray, target_fpr: float
) -> float:
    """Smallest threshold whose weighted FPR is <= ``target_fpr`` for this group.

    Lowering the threshold flags more students (FPR rises), so we scan thresholds
    upward and take the first that brings weighted FPR at or below the target.
    Returns 1.0 (flag no-one) if even that cannot reach the target.
    """
    neg = y == 0
    wn = w[neg].sum()
    if wn == 0:
        return 0.5
    # candidate thresholds: the negatives' predicted probs (FPR only changes there)
    cands = np.unique(prob[neg])
    for t in np.sort(cands):
        fpr = float(w[neg & (prob >= t)].sum() / wn)
        if fpr <= target_fpr:
            return float(t)
    return 1.0


def group_thresholds_equal_fpr(
    df: pd.DataFrame,
    y_col: str,
    prob_col: str,
    group_col: str,
    weight_col: str,
    target_fpr: float,
) -> dict:
    """Per-group thresholds that equalise the weighted FPR at ``target_fpr``."""
    out: dict = {}
    for g, sub in df.groupby(group_col):
        out[g] = threshold_for_target_fpr(
            sub[y_col].values, sub[prob_col].values, sub[weight_col].values, target_fpr
        )
    return out


def apply_group_thresholds(
    df: pd.DataFrame, prob_col: str, group_col: str, thresholds: dict
) -> np.ndarray:
    """Predicted labels using a per-group threshold (fallback 0.5 if group unseen)."""
    t = df[group_col].map(lambda g: thresholds.get(g, 0.5)).values
    return (df[prob_col].values >= t).astype(float)


def _gap(rates_by_group: dict[str, dict], key: str) -> float:
    vals = [r[key] for r in rates_by_group.values() if not np.isnan(r[key])]
    return float(max(vals) - min(vals)) if len(vals) >= 2 else np.nan


def evaluate_assignment(
    df: pd.DataFrame, y_col: str, pred: np.ndarray, group_col: str, weight_col: str
) -> dict:
    """Overall utility + per-group rates + fairness gaps for a label assignment."""
    y = df[y_col].values
    w = df[weight_col].values
    overall = _weighted_rates(y, pred, w)
    by_group = {
        str(g): _weighted_rates(sub[y_col].values, pred[df[group_col].values == g],
                                sub[weight_col].values)
        for g, sub in df.groupby(group_col)
    }
    return {
        "overall_recall": overall["tpr"],
        "overall_fpr": overall["fpr"],
        "overall_accuracy": overall["accuracy"],
        "overall_selection": overall["selection"],
        "fpr_gap": _gap(by_group, "fpr"),
        "tpr_gap": _gap(by_group, "tpr"),
        "selection_gap": _gap(by_group, "selection"),
        "by_group": by_group,
    }


def fairness_utility_frontier(
    df: pd.DataFrame,
    y_col: str,
    prob_col: str,
    group_col: str,
    weight_col: str,
    target_fprs: np.ndarray | None = None,
) -> pd.DataFrame:
    """Trace the trade-off as the equal-FPR target is swept.

    For each target FPR we set per-group thresholds and record overall recall /
    accuracy and the realised FPR gap. A **single-threshold** baseline (the same
    global threshold for every group, chosen to hit the same overall selection
    rate) is included per point so the two strategies can be compared at matched
    utility.
    """
    if target_fprs is None:
        target_fprs = np.round(np.arange(0.05, 0.75, 0.05), 3)

    rows = []
    for tf in target_fprs:
        thr = group_thresholds_equal_fpr(df, y_col, prob_col, group_col, weight_col, tf)
        pred = apply_group_thresholds(df, prob_col, group_col, thr)
        ev = evaluate_assignment(df, y_col, pred, group_col, weight_col)

        # single global threshold matched to the same overall selection rate
        sel = ev["overall_selection"]
        gthr = float(np.quantile(df[prob_col].values, 1 - sel)) if 0 < sel < 1 else 0.5
        gpred = (df[prob_col].values >= gthr).astype(float)
        gev = evaluate_assignment(df, y_col, gpred, group_col, weight_col)

        rows.append({
            "target_fpr": tf,
            "grp_recall": ev["overall_recall"], "grp_fpr_gap": ev["fpr_gap"],
            "grp_tpr_gap": ev["tpr_gap"], "grp_accuracy": ev["overall_accuracy"],
            "grp_selection": ev["overall_selection"],
            "single_recall": gev["overall_recall"], "single_fpr_gap": gev["fpr_gap"],
            "single_tpr_gap": gev["tpr_gap"], "single_accuracy": gev["overall_accuracy"],
        })
    out = pd.DataFrame(rows)
    logger.info("Fairness-utility frontier computed", n_points=len(out),
                group=group_col)
    return out
