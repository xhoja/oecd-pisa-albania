"""
Fairness metrics for the PISA risk scoring model.

Computed across protected attributes: GENDER, SES_QUINTILE, URBAN_RURAL, IMMIG.

Definitions follow Hardt et al. (2016) and Barocas & Hardt (2023).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import structlog
logger = structlog.get_logger(__name__)


@dataclass
class FairnessReport:
    attribute: str
    groups: list[str]
    demographic_parity: dict[str, float]
    equal_opportunity: dict[str, float]
    fpr_by_group: dict[str, float]
    calibration_by_group: dict[str, float]
    parity_gap: float        # max - min demographic parity
    opportunity_gap: float   # max - min TPR
    fpr_gap: float           # max - min FPR

    def summary(self) -> str:
        lines = [
            f"=== Fairness Report: {self.attribute} ===",
            f"Parity gap (max-min selection rate): {self.parity_gap:.4f}",
            f"Equal opportunity gap (TPR): {self.opportunity_gap:.4f}",
            f"FPR gap: {self.fpr_gap:.4f}",
        ]
        for g in self.groups:
            lines.append(
                f"  {g}: P(ŷ=1)={self.demographic_parity.get(g, np.nan):.3f} | "
                f"TPR={self.equal_opportunity.get(g, np.nan):.3f} | "
                f"FPR={self.fpr_by_group.get(g, np.nan):.3f}"
            )
        return "\n".join(lines)


def _group_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return **weighted** (TN, FP, FN, TP) for a group mask.

    Cells are sums of survey weights, not raw counts, so every rate below
    (TPR/FPR) is a population estimate rather than a sample proportion — the same
    weighted mandate the rest of the pipeline follows. Falls back to NaN when the
    group has no both-class support (a rate would be undefined).
    """
    yt = y_true[mask]
    yp = y_pred[mask]
    w = weights[mask]
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return np.nan, np.nan, np.nan, np.nan
    tn = float(w[(yt == 0) & (yp == 0)].sum())
    fp = float(w[(yt == 0) & (yp == 1)].sum())
    fn = float(w[(yt == 1) & (yp == 0)].sum())
    tp = float(w[(yt == 1) & (yp == 1)].sum())
    return tn, fp, fn, tp


def compute_fairness(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    y_prob_col: str,
    attribute: str,
    weight_col: str | None = None,
) -> FairnessReport:
    """
    Compute fairness metrics for one protected attribute.

    Requires df to have columns: y_true_col, y_pred_col, y_prob_col, attribute.
    When ``weight_col`` is given, every metric (selection rate, TPR, FPR,
    calibration mean) is **survey-weighted**; otherwise unit weights are used.
    """
    if attribute not in df.columns:
        logger.warning("Protected attribute not in dataframe", attribute=attribute)
        return _empty_report(attribute)

    groups = sorted(df[attribute].dropna().unique().tolist())
    groups_str = [str(g) for g in groups]

    y_true = df[y_true_col].values
    y_pred = df[y_pred_col].values
    y_prob = df[y_prob_col].values
    weights = df[weight_col].values if weight_col is not None else np.ones(len(df))

    dp: dict[str, float] = {}       # demographic parity: P(ŷ=1|A=a)
    eo: dict[str, float] = {}       # equal opportunity: TPR per group
    fpr: dict[str, float] = {}      # FPR per group
    cal: dict[str, float] = {}      # mean predicted prob per group (calibration check)

    def _wmean(vals: np.ndarray, w: np.ndarray) -> float:
        tot = w.sum()
        return float(np.dot(vals, w) / tot) if tot > 0 else np.nan

    for g, gs in zip(groups, groups_str):
        mask = (df[attribute] == g).values & ~np.isnan(y_true) & ~np.isnan(y_pred)
        if mask.sum() == 0:
            dp[gs] = np.nan
            eo[gs] = np.nan
            fpr[gs] = np.nan
            cal[gs] = np.nan
            continue

        wm = weights[mask]
        dp[gs] = _wmean(y_pred[mask], wm)
        cal[gs] = _wmean(y_prob[mask], wm)

        tn, fp, fn, tp = _group_confusion(y_true, y_pred, mask, weights)
        if not any(np.isnan([tn, fp, fn, tp])):
            eo[gs] = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # TPR
            fpr[gs] = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        else:
            eo[gs] = np.nan
            fpr[gs] = np.nan

    valid_dp = [v for v in dp.values() if not np.isnan(v)]
    valid_eo = [v for v in eo.values() if not np.isnan(v)]
    valid_fpr = [v for v in fpr.values() if not np.isnan(v)]

    return FairnessReport(
        attribute=attribute,
        groups=groups_str,
        demographic_parity=dp,
        equal_opportunity=eo,
        fpr_by_group=fpr,
        calibration_by_group=cal,
        parity_gap=max(valid_dp) - min(valid_dp) if len(valid_dp) >= 2 else np.nan,
        opportunity_gap=max(valid_eo) - min(valid_eo) if len(valid_eo) >= 2 else np.nan,
        fpr_gap=max(valid_fpr) - min(valid_fpr) if len(valid_fpr) >= 2 else np.nan,
    )


def fairness_audit(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    y_prob_col: str,
    attributes: list[str] | None = None,
    weight_col: str | None = None,
) -> dict[str, FairnessReport]:
    """Run fairness audit across all specified protected attributes."""
    if attributes is None:
        attributes = ["GENDER", "SES_QUINTILE", "IMMIG"]

    results: dict[str, FairnessReport] = {}
    for attr in attributes:
        report = compute_fairness(df, y_true_col, y_pred_col, y_prob_col, attr, weight_col)
        results[attr] = report
        logger.info(
            "Fairness computed",
            attribute=attr,
            parity_gap=round(report.parity_gap, 4) if not np.isnan(report.parity_gap) else None,
            opportunity_gap=round(report.opportunity_gap, 4) if not np.isnan(report.opportunity_gap) else None,
        )
    return results


def threshold_sweep(
    df: pd.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    attribute: str,
    thresholds: list[float] | None = None,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """
    Sweep decision threshold and compute demographic parity + opportunity gap.
    Returns DataFrame: threshold × (parity_gap, opportunity_gap). Weighted when
    ``weight_col`` is supplied.
    """
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    rows = []
    for thresh in thresholds:
        df = df.copy()
        df["y_pred_thresh"] = (df[y_prob_col] >= thresh).astype(int)
        report = compute_fairness(df, y_true_col, "y_pred_thresh", y_prob_col, attribute, weight_col)
        rows.append({
            "threshold": thresh,
            "parity_gap": report.parity_gap,
            "opportunity_gap": report.opportunity_gap,
            "fpr_gap": report.fpr_gap,
        })
    return pd.DataFrame(rows)


def _empty_report(attribute: str) -> FairnessReport:
    return FairnessReport(
        attribute=attribute,
        groups=[],
        demographic_parity={},
        equal_opportunity={},
        fpr_by_group={},
        calibration_by_group={},
        parity_gap=np.nan,
        opportunity_gap=np.nan,
        fpr_gap=np.nan,
    )
