"""
Full metric suite for binary classification with imbalanced classes.
All metrics are computed with confidence intervals via bootstrap.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import structlog
logger = structlog.get_logger(__name__)


@dataclass
class ClassificationMetrics:
    accuracy: float = np.nan
    balanced_accuracy: float = np.nan
    roc_auc: float = np.nan
    pr_auc: float = np.nan
    f1_macro: float = np.nan
    f1_positive: float = np.nan
    f1_negative: float = np.nan
    precision_positive: float = np.nan
    recall_positive: float = np.nan
    mcc: float = np.nan
    cohen_kappa: float = np.nan
    brier_score: float = np.nan
    ece: float = np.nan
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))

    # Bootstrap CIs (95%)
    roc_auc_ci: tuple[float, float] = (np.nan, np.nan)
    pr_auc_ci: tuple[float, float] = (np.nan, np.nan)
    f1_macro_ci: tuple[float, float] = (np.nan, np.nan)
    mcc_ci: tuple[float, float] = (np.nan, np.nan)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "f1_macro": round(self.f1_macro, 4),
            "f1_positive": round(self.f1_positive, 4),
            "f1_negative": round(self.f1_negative, 4),
            "precision_positive": round(self.precision_positive, 4),
            "recall_positive": round(self.recall_positive, 4),
            "mcc": round(self.mcc, 4),
            "cohen_kappa": round(self.cohen_kappa, 4),
            "brier_score": round(self.brier_score, 4),
            "ece": round(self.ece, 4),
            "roc_auc_95ci": f"({self.roc_auc_ci[0]:.4f}, {self.roc_auc_ci[1]:.4f})",
            "f1_macro_95ci": f"({self.f1_macro_ci[0]:.4f}, {self.f1_macro_ci[1]:.4f})",
            "mcc_95ci": f"({self.mcc_ci[0]:.4f}, {self.mcc_ci[1]:.4f})",
        }


def fit_with_sample_weight(pipe, X, y, w) -> bool:
    """
    Fit ``pipe`` passing PISA sample weights to the FINAL estimator, whichever
    parameter path it needs. The registry wraps some models (LR/SVM/KNN/MLP) in
    an inner ``RobustScaler`` Pipeline, so the weight kwarg is
    ``model__clf__sample_weight``; bare estimators (trees/boosters/NB) take
    ``model__sample_weight``. The naïve single-key ``model__sample_weight`` call
    raised for the wrapped models and silently fell back to an UNWEIGHTED fit —
    this tries both real paths first and only drops to unweighted if neither
    exists. Returns True iff a weighted fit succeeded.
    """
    for key in ("model__clf__sample_weight", "model__sample_weight"):
        try:
            pipe.fit(X, y, **{key: w})
            return True
        except (TypeError, ValueError):
            continue
    pipe.fit(X, y)
    return False


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Expected Calibration Error (ECE), weighted if sample_weight given."""
    w = np.ones_like(y_prob, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_w = w.sum()
    if total_w == 0:
        return float("nan")
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= low) & (y_prob < high)
        wm = w[mask].sum()
        if wm == 0:
            continue
        bin_acc = np.average(y_true[mask], weights=w[mask])
        bin_conf = np.average(y_prob[mask], weights=w[mask])
        ece += (wm / total_w) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 2000,
    random_state: int = 42,
    sample_weight: np.ndarray | None = None,
) -> ClassificationMetrics:
    """
    Compute full metric suite with bootstrap confidence intervals.

    Args:
        y_true: true binary labels
        y_pred: predicted binary labels (at threshold 0.5)
        y_prob: predicted probabilities for positive class
        n_bootstrap: number of bootstrap resamples for CIs
        random_state: for reproducibility
        sample_weight: PISA survey weights (W_FSTUWT). When given, every metric is
            the weighted (population) estimate, and the bootstrap is a weighted
            resample so the CIs are population-referenced too.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    cm = confusion_matrix(y_true, y_pred, sample_weight=w)

    metrics = ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred, sample_weight=w),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred, sample_weight=w),
        roc_auc=roc_auc_score(y_true, y_prob, sample_weight=w),
        pr_auc=average_precision_score(y_true, y_prob, sample_weight=w),
        f1_macro=f1_score(y_true, y_pred, average="macro", sample_weight=w),
        f1_positive=f1_score(y_true, y_pred, pos_label=1, sample_weight=w),
        f1_negative=f1_score(y_true, y_pred, pos_label=0, sample_weight=w),
        precision_positive=precision_score(y_true, y_pred, pos_label=1, zero_division=0, sample_weight=w),
        recall_positive=recall_score(y_true, y_pred, pos_label=1, zero_division=0, sample_weight=w),
        mcc=matthews_corrcoef(y_true, y_pred, sample_weight=w),
        cohen_kappa=cohen_kappa_score(y_true, y_pred, sample_weight=w),
        brier_score=brier_score_loss(y_true, y_prob, sample_weight=w),
        ece=expected_calibration_error(y_true, y_prob, sample_weight=w),
        confusion_matrix=cm,
    )

    # Bootstrap CIs. With weights we draw indices proportional to the survey
    # weight (a weighted bootstrap), so resamples are population-representative.
    p = None if w is None else w / w.sum()
    boot_roc_auc, boot_pr_auc, boot_f1_macro, boot_mcc = [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True, p=p)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        yprob_b = y_prob[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        boot_roc_auc.append(roc_auc_score(yt_b, yprob_b))
        boot_pr_auc.append(average_precision_score(yt_b, yprob_b))
        boot_f1_macro.append(f1_score(yt_b, yp_b, average="macro"))
        boot_mcc.append(matthews_corrcoef(yt_b, yp_b))

    def ci(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return (np.nan, np.nan)
        arr = np.array(vals)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    metrics.roc_auc_ci = ci(boot_roc_auc)
    metrics.pr_auc_ci = ci(boot_pr_auc)
    metrics.f1_macro_ci = ci(boot_f1_macro)
    metrics.mcc_ci = ci(boot_mcc)

    return metrics


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None = None,
    metric: str = "f1_macro",
) -> float:
    """
    Pick the decision threshold that maximizes a (weighted) metric.

    Fixed 0.5 is wrong when classes are imbalanced and `class_weight='balanced'`
    recenters the scores. Call this on TRAIN/validation predictions only, never
    on the test fold, then apply the returned threshold to test.
    """
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    grid = np.unique(np.clip(np.round(y_prob, 3), 0.01, 0.99))
    if len(grid) > 200:
        grid = np.linspace(0.01, 0.99, 200)

    def score(thr: float) -> float:
        pred = (y_prob >= thr).astype(int)
        if metric == "f1_macro":
            return f1_score(y_true, pred, average="macro", sample_weight=w)
        if metric == "balanced_accuracy":
            return balanced_accuracy_score(y_true, pred, sample_weight=w)
        if metric == "mcc":
            return matthews_corrcoef(y_true, pred, sample_weight=w)
        raise ValueError(f"Unknown metric: {metric}")

    best_thr, best_val = 0.5, -np.inf
    for thr in grid:
        v = score(thr)
        if v > best_val:
            best_val, best_thr = v, float(thr)
    return best_thr


# ---------------------------------------------------------------------------
# Nadeau-Bengio corrected-resampled inference for (repeated) cross-validation
# ---------------------------------------------------------------------------
# The naive SE of a mean over CV folds, sqrt(S^2 / J), is optimistic: the folds
# share training data, so the fold scores are positively correlated and the
# variance is under-estimated. Nadeau & Bengio (2003) correct this by replacing
# the 1/J factor with (1/J + n_test/n_train). For k-fold CV, n_test/n_train =
# 1/(k-1); for r repeats there are J = k*r scores but the ratio is unchanged.

def nadeau_bengio_interval(
    scores: list[float] | np.ndarray,
    test_train_ratio: float,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """(mean, lo, hi, se) with the Nadeau-Bengio corrected variance and a
    t-interval on J-1 df. ``test_train_ratio`` = n_test/n_train (1/(k-1) for
    k-fold). Falls back to (mean, nan, nan, nan) with < 2 scores."""
    from scipy import stats as _st

    arr = np.asarray(scores, dtype=float)
    J = len(arr)
    mean = float(arr.mean()) if J else float("nan")
    if J < 2:
        return mean, float("nan"), float("nan"), float("nan")
    S2 = float(arr.var(ddof=1))
    se = float(np.sqrt((1.0 / J + test_train_ratio) * S2))
    t = float(_st.t.ppf(1 - alpha / 2, df=J - 1))
    return mean, mean - t * se, mean + t * se, se


def corrected_resampled_ttest(
    diffs: list[float] | np.ndarray,
    test_train_ratio: float,
) -> tuple[float, float, float]:
    """
    Nadeau-Bengio corrected paired t-test on per-fold score DIFFERENCES
    (model A - model B, paired by fold). Returns (mean_diff, t, p_two_sided).

    Use this — not a plain paired t-test and not DeLong — to ask "is model A's
    CV AUC significantly above model B's?", because it accounts for the reused
    training data across folds. DeLong is for a single shared test set (see
    ``delong_roc_test``); the corrected resampled t-test is for CV.
    """
    from scipy import stats as _st

    arr = np.asarray(diffs, dtype=float)
    J = len(arr)
    mean = float(arr.mean()) if J else float("nan")
    if J < 2:
        return mean, float("nan"), float("nan")
    S2 = float(arr.var(ddof=1))
    se = float(np.sqrt((1.0 / J + test_train_ratio) * S2))
    if se == 0.0:
        return mean, 0.0, 1.0
    t = mean / se
    p = float(2 * _st.t.sf(abs(t), df=J - 1))
    return mean, float(t), p


# ---------------------------------------------------------------------------
# DeLong test for two correlated ROC curves on one shared test set
# ---------------------------------------------------------------------------
# Fast DeLong (Sun & Xu 2014) — used to compare two models' AUCs on the SAME
# test samples (e.g. the held-out 2022 out-of-sample cohort). Unweighted: it
# tests the models against each other on identical rows, which is the relevant
# comparison; it does not fold in the survey design.

def _midrank(x: np.ndarray) -> np.ndarray:
    J = len(x)
    Z = np.argsort(x)
    W = x[Z]
    T = np.zeros(J, dtype=float)
    i = 0
    while i < J:
        j = i
        while j < J and W[j] == W[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # 1-based average rank of ties
        i = j
    out = np.empty(J, dtype=float)
    out[Z] = T
    return out


def _fast_delong(preds_pos_first: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """preds_pos_first: (k, N) scores with the m positives in the first columns.
    Returns (aucs[k], covariance[k,k])."""
    k = preds_pos_first.shape[0]
    n = preds_pos_first.shape[1] - m
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(preds_pos_first[r, :m])
        ty[r] = _midrank(preds_pos_first[r, m:])
        tz[r] = _midrank(preds_pos_first[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_roc_test(
    y_true: np.ndarray,
    prob1: np.ndarray,
    prob2: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Compare AUC(prob1) vs AUC(prob2) on the same labels via DeLong.
    Returns (auc1, auc2, z, p_two_sided). p is NaN if a class is absent.
    """
    from scipy import stats as _st

    y = np.asarray(y_true).astype(int)
    p1 = np.asarray(prob1, dtype=float)
    p2 = np.asarray(prob2, dtype=float)
    m = int((y == 1).sum())
    n = int((y == 0).sum())
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    order = np.argsort(-y)  # positives (label 1) first
    preds = np.vstack([p1, p2])[:, order]
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        z = 0.0
        p = 1.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = float(2 * _st.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), p


def get_roc_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def get_pr_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}


def get_calibration_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return {
        "fraction_of_positives": fraction_of_positives,
        "mean_predicted_value": mean_predicted,
    }


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_specificity: float = 0.90,
) -> float:
    """Sensitivity (recall) at a given specificity level."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    specificities = 1 - fpr
    idx = np.searchsorted(specificities[::-1], target_specificity)
    return float(tpr[::-1][min(idx, len(tpr) - 1)])
