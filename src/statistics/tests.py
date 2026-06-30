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
    """Cohen's d effect size for two independent samples (unweighted)."""
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


def weighted_cohens_d(
    values1: np.ndarray,
    values2: np.ndarray,
    weights1: np.ndarray,
    weights2: np.ndarray,
) -> float:
    """
    Weighted Cohen's d using PISA survey weights, so the effect size is a
    population estimate. Means and (pooled) variance are weighted; the pooled SD
    uses weighted variances combined by effective sample size.
    """
    def _clean(v, w):
        v = np.asarray(v, dtype=float); w = np.asarray(w, dtype=float)
        m = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
        return v[m], w[m]

    v1, w1 = _clean(values1, weights1)
    v2, w2 = _clean(values2, weights2)
    if len(v1) < 2 or len(v2) < 2:
        return np.nan

    m1 = np.average(v1, weights=w1)
    m2 = np.average(v2, weights=w2)
    var1 = np.average((v1 - m1) ** 2, weights=w1)
    var2 = np.average((v2 - m2) ** 2, weights=w2)
    # Kish effective sample sizes
    n1 = w1.sum() ** 2 / np.sum(w1 ** 2)
    n2 = w2.sum() ** 2 / np.sum(w2 ** 2)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return float((m1 - m2) / pooled)


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

    A naive weighted contingency table sums W_FSTUWT into the cells, so the cell
    counts are on the population scale (millions). Feeding that to chi-square
    inflates the statistic and p-value by the weight magnitude and makes them
    meaningless. We rescale the weighted table so its grand total equals the
    actual number of observations n (a normalized-weight / first-order
    Rao-Scott-style correction). This keeps the population cell *proportions* but
    references the test to the real sample size. (Full design-based inference
    would additionally use the BRR replicate weights; see weights.brr_se.)

    Returns statistic, p_value, dof, cramers_v, n_effective.
    """
    if weight_col and weight_col in df.columns:
        sub = df[[col1, col2, weight_col]].dropna()
        ct = sub.groupby([col1, col2])[weight_col].sum().unstack(fill_value=0)
        arr = ct.values.astype(float)
        n_obs = len(sub)
        total_w = arr.sum()
        if total_w > 0:
            arr = arr * (n_obs / total_w)  # rescale to the real sample size
    else:
        ct = pd.crosstab(df[col1], df[col2])
        arr = ct.values.astype(float)
        n_obs = int(arr.sum())

    chi2, p, dof, expected = stats.chi2_contingency(arr)
    v = cramers_v(arr)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "cramers_v": float(v),
        "n_effective": int(round(arr.sum())),
    }


def maximum_mean_discrepancy(
    X1: np.ndarray,
    X2: np.ndarray,
    kernel: str = "rbf",
    bandwidth: float | str = "median",
) -> float:
    """
    Maximum Mean Discrepancy (MMD) between two distributions.
    Used to quantify covariate shift between PISA cycles.

    MMD = 0 → distributions identical.
    MMD > 0 → distributions differ (higher = more shift).

    bandwidth: RBF length scale. "median" uses the median pairwise-distance
        heuristic (scale-robust default); pass a float to fix it.
    """
    from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

    n1 = len(X1)
    n2 = len(X2)

    if kernel == "rbf":
        if bandwidth == "median":
            from sklearn.metrics.pairwise import euclidean_distances
            sample = np.vstack([X1, X2])
            if len(sample) > 1000:
                sample = sample[np.random.default_rng(0).choice(len(sample), 1000, replace=False)]
            d = euclidean_distances(sample)
            med = np.median(d[np.triu_indices_from(d, k=1)])
            bandwidth = float(med) if med > 0 else 1.0
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
    weight_col: str | None = "W_FSTUWT",
) -> dict[str, Any]:
    """
    Test for covariate shift between two cohorts via a domain classifier.

    Method: train a classifier to distinguish cohort A from cohort B using the
    background features; cross-validated AUC >> 0.5 ⇒ detectable shift.

    Fixes over the naive version:
    * Both cohorts are imputed with the SAME (pooled) medians. This is a
      two-sample test, not a predictive split, so a shared fill is correct —
      per-split medians would manufacture an artificial location gap on imputed
      cells and inflate the AUC even when no shift exists.
    * A missingness indicator is added per feature so a shift in *missing rate*
      is captured as real shift instead of being washed out by imputation.
    * Survey weights drive a weighted subsample (rng.choice without replacement,
      p ∝ W_FSTUWT), so the cohorts are population-representative and the AUC is
      not an artefact of differing sampling designs.
    * Returns a bootstrap-free CV-fold 95% CI on the AUC (not just a point).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(random_state)

    feats = [
        f for f in features
        if f in df_train.columns and f in df_test.columns
        and df_train[f].notna().any() and df_test[f].notna().any()
    ]

    pooled_medians = pd.concat([df_train[feats], df_test[feats]]).median()

    def _prep(d: pd.DataFrame) -> pd.DataFrame:
        block = d[feats].copy()
        miss = block.isna().astype(float).add_prefix("_MISS_")
        block = block.fillna(pooled_medians)  # shared fill (two-sample test)
        return pd.concat([block, miss], axis=1)

    df1, df2 = _prep(df_train), _prep(df_test)

    def _weighted_idx(d: pd.DataFrame, n: int) -> np.ndarray:
        n = min(n, len(d))
        if weight_col and weight_col in d.columns:
            w = d[weight_col].to_numpy(dtype=float)
            w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
            if w.sum() > 0:
                return rng.choice(len(d), size=n, replace=False, p=w / w.sum())
        return rng.choice(len(d), size=n, replace=False)

    idx1, idx2 = _weighted_idx(df_train, n_subsample), _weighted_idx(df_test, n_subsample)
    X_combined = np.vstack([df1.iloc[idx1].values, df2.iloc[idx2].values])
    y_shift = np.concatenate([np.zeros(len(idx1)), np.ones(len(idx2))])

    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    auc_scores = cross_val_score(clf, X_combined, y_shift, cv=5, scoring="roc_auc")
    shift_auc = float(auc_scores.mean())
    se = float(auc_scores.std(ddof=1) / np.sqrt(len(auc_scores)))
    ci = (round(shift_auc - 1.96 * se, 4), round(shift_auc + 1.96 * se, 4))

    # Per-feature WEIGHTED standardized mean difference (population effect size)
    from src.data.weights import weighted_mean, weighted_std
    smd: dict[str, float] = {}
    for feat in feats:
        w1 = df_train[weight_col] if (weight_col and weight_col in df_train.columns) else pd.Series(1.0, index=df_train.index)
        w2 = df_test[weight_col] if (weight_col and weight_col in df_test.columns) else pd.Series(1.0, index=df_test.index)
        m1, m2 = weighted_mean(df_train[feat], w1), weighted_mean(df_test[feat], w2)
        s1, s2 = weighted_std(df_train[feat], w1), weighted_std(df_test[feat], w2)
        pooled_std = np.sqrt((np.nan_to_num(s1) ** 2 + np.nan_to_num(s2) ** 2) / 2)
        smd[feat] = float((m2 - m1) / pooled_std) if pooled_std > 0 else 0.0

    smd_sorted = dict(sorted(smd.items(), key=lambda x: abs(x[1]), reverse=True))

    logger.info(
        "Covariate shift test complete",
        shift_auc=round(shift_auc, 4),
        ci=ci,
        interpretation="significant shift" if shift_auc > 0.7 else "minor shift",
    )

    return {
        "shift_detection_auc": shift_auc,
        "shift_detection_auc_95ci": ci,
        "significant_shift": shift_auc > 0.7,
        "per_feature_smd": smd_sorted,
        "features_tested": feats,
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
