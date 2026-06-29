"""
Statistical tests for PISA analysis.
All tests use weighted versions where applicable (via PISA sampling weights).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

import structlog
logger = structlog.get_logger(__name__)


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    """Cohen's d effect size for two independent samples."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return np.nan
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def cramers_v(contingency_table: np.ndarray) -> float:
    """Cramér's V effect size for chi-square test."""
    chi2 = stats.chi2_contingency(contingency_table, correction=False)[0]
    n = contingency_table.sum()
    k = min(contingency_table.shape) - 1
    if n == 0 or k == 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * k)))


def mann_whitney_test(
    group1: np.ndarray,
    group2: np.ndarray,
) -> dict[str, float]:
    """Mann-Whitney U test with effect size r = Z / sqrt(N)."""
    group1 = group1[~np.isnan(group1)]
    group2 = group2[~np.isnan(group2)]
    if len(group1) < 2 or len(group2) < 2:
        return {"statistic": np.nan, "p_value": np.nan, "effect_size_r": np.nan}
    stat, p = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    n = len(group1) + len(group2)
    z = (stat - len(group1) * len(group2) / 2) / np.sqrt(
        len(group1) * len(group2) * (n + 1) / 12
    )
    r = abs(z) / np.sqrt(n)
    return {"statistic": float(stat), "p_value": float(p), "effect_size_r": float(r)}


def kruskal_wallis_test(groups: list[np.ndarray]) -> dict[str, float]:
    """Kruskal-Wallis H test for multiple groups."""
    clean = [g[~np.isnan(g)] for g in groups]
    clean = [g for g in clean if len(g) >= 2]
    if len(clean) < 2:
        return {"statistic": np.nan, "p_value": np.nan}
    stat, p = stats.kruskal(*clean)
    return {"statistic": float(stat), "p_value": float(p)}


def weighted_chi_square(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    weight_col: str | None = None,
) -> dict[str, float]:
    """
    Chi-square test of independence with optional weights.
    Returns statistic, p_value, dof, cramers_v.
    """
    if weight_col and weight_col in df.columns:
        # Build weighted contingency table
        ct = df.groupby([col1, col2])[weight_col].sum().unstack(fill_value=0)
    else:
        ct = pd.crosstab(df[col1], df[col2])

    arr = ct.values.astype(float)
    chi2, p, dof, expected = stats.chi2_contingency(arr)
    v = cramers_v(arr)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "cramers_v": float(v),
    }


def maximum_mean_discrepancy(
    X1: np.ndarray,
    X2: np.ndarray,
    kernel: str = "rbf",
    bandwidth: float = 1.0,
) -> float:
    """
    Maximum Mean Discrepancy (MMD) between two distributions.
    Used to quantify covariate shift between PISA cycles.

    MMD = 0 → distributions identical.
    MMD > 0 → distributions differ (higher = more shift).
    """
    from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

    n1 = len(X1)
    n2 = len(X2)

    if kernel == "rbf":
        gamma = 1.0 / (2 * bandwidth ** 2)
        K11 = rbf_kernel(X1, X1, gamma=gamma)
        K22 = rbf_kernel(X2, X2, gamma=gamma)
        K12 = rbf_kernel(X1, X2, gamma=gamma)
    elif kernel == "poly":
        K11 = polynomial_kernel(X1, X1)
        K22 = polynomial_kernel(X2, X2)
        K12 = polynomial_kernel(X1, X2)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    mmd = (
        np.sum(K11) / (n1 * n1)
        + np.sum(K22) / (n2 * n2)
        - 2 * np.sum(K12) / (n1 * n2)
    )
    return float(max(0.0, mmd))


def covariate_shift_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list[str],
    n_subsample: int = 2000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Test for covariate shift between train and test distributions.

    Method: Train a classifier to distinguish train vs. test samples.
    AUC >> 0.5 → detectable shift.
    Also returns per-feature standardized mean difference.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(random_state)

    features_available = [f for f in features if f in df_train.columns and f in df_test.columns]
    # Drop features that are entirely missing in either split (e.g. a cycle gap),
    # then median-impute the rest so row-wise dropna doesn't empty the data.
    features_available = [
        f for f in features_available
        if df_train[f].notna().any() and df_test[f].notna().any()
    ]
    medians = pd.concat([df_train[features_available], df_test[features_available]]).median()
    df1 = df_train[features_available].fillna(medians)
    df2 = df_test[features_available].fillna(medians)

    # Subsample for speed
    n1 = min(n_subsample, len(df1))
    n2 = min(n_subsample, len(df2))
    idx1 = rng.integers(0, len(df1), n1)
    idx2 = rng.integers(0, len(df2), n2)

    X_combined = np.vstack([df1.iloc[idx1].values, df2.iloc[idx2].values])
    y_shift = np.concatenate([np.zeros(n1), np.ones(n2)])

    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    auc_scores = cross_val_score(clf, X_combined, y_shift, cv=5, scoring="roc_auc")
    shift_auc = float(auc_scores.mean())

    # Per-feature standardized mean difference
    smd: dict[str, float] = {}
    for feat in features_available:
        v1 = df_train[feat].dropna().values
        v2 = df_test[feat].dropna().values
        pooled_std = np.sqrt((np.var(v1) + np.var(v2)) / 2)
        if pooled_std > 0:
            smd[feat] = float((np.mean(v2) - np.mean(v1)) / pooled_std)
        else:
            smd[feat] = 0.0

    smd_sorted = dict(sorted(smd.items(), key=lambda x: abs(x[1]), reverse=True))

    logger.info(
        "Covariate shift test complete",
        shift_auc=round(shift_auc, 4),
        interpretation="significant shift" if shift_auc > 0.7 else "minor shift",
    )

    return {
        "shift_detection_auc": shift_auc,
        "significant_shift": shift_auc > 0.7,
        "per_feature_smd": smd_sorted,
        "features_tested": features_available,
    }


def wilcoxon_signed_rank(
    scores_a: list[float],
    scores_b: list[float],
) -> dict[str, float]:
    """
    Wilcoxon signed-rank test for paired model comparison.
    Used to test if model A is significantly better than model B across CV folds.
    """
    diffs = np.array(scores_a) - np.array(scores_b)
    if len(diffs) < 10:
        logger.warning("Few paired samples for Wilcoxon test", n=len(diffs))
    stat, p = stats.wilcoxon(diffs, alternative="greater")
    return {"statistic": float(stat), "p_value": float(p), "mean_diff": float(diffs.mean())}
