"""
Sampling weight handling for PISA data.

PISA uses a stratified multi-stage sampling design. All descriptive statistics
must be computed using W_FSTUWT (final student weight). Standard errors
require Balanced Repeated Replication (BRR) with 80 replicate weights.

References:
    OECD (2009). PISA Data Analysis Manual. Chapter 3.
    Mislevy (1991). Randomization-based inference about latent variables.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

import structlog
logger = structlog.get_logger(__name__)

WEIGHT_COL = "W_FSTUWT"
BRR_PREFIX = "W_FSTURWT"
BRR_COUNT = 80
FAYS_ADJUSTMENT = 0.5  # Fay's method adjustment for BRR


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Weighted mean, ignoring NaN pairs."""
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def weighted_proportion(
    indicator: pd.Series,
    weights: pd.Series,
) -> float:
    """Weighted proportion (for binary indicator series)."""
    return weighted_mean(indicator.astype(float), weights)


def weighted_std(values: pd.Series, weights: pd.Series) -> float:
    """Weighted standard deviation."""
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() < 2:
        return np.nan
    v = values[mask].values
    w = weights[mask].values
    mean = np.average(v, weights=w)
    variance = np.average((v - mean) ** 2, weights=w)
    return float(np.sqrt(variance))


def weighted_quantile(
    values: pd.Series,
    weights: pd.Series,
    quantiles: list[float],
) -> np.ndarray:
    """Weighted quantiles using interpolation."""
    mask = values.notna() & weights.notna() & (weights > 0)
    v = values[mask].values
    w = weights[mask].values
    sorter = np.argsort(v)
    v_sorted = v[sorter]
    w_sorted = w[sorter]
    cum_w = np.cumsum(w_sorted)
    cum_w_norm = (cum_w - w_sorted / 2) / cum_w[-1]
    return np.interp(quantiles, cum_w_norm, v_sorted)


def brr_se(
    df: pd.DataFrame,
    statistic_fn: Callable[[pd.DataFrame, str], float],
    weight_col: str = WEIGHT_COL,
    brr_prefix: str = BRR_PREFIX,
    brr_count: int = BRR_COUNT,
) -> tuple[float, float]:
    """
    Compute a statistic and its BRR standard error.

    Args:
        df: DataFrame with weights
        statistic_fn: function(df, weight_col) -> float
        weight_col: final student weight column
        brr_prefix: prefix for replicate weight columns
        brr_count: number of replicate weights (80 for PISA)

    Returns:
        (estimate, standard_error)
    """
    estimate = statistic_fn(df, weight_col)

    replicate_cols = [f"{brr_prefix}{i}" for i in range(1, brr_count + 1)
                      if f"{brr_prefix}{i}" in df.columns]

    if not replicate_cols:
        logger.warning("No BRR replicate weights found — SE cannot be computed via BRR")
        return estimate, np.nan

    rep_estimates = [statistic_fn(df, col) for col in replicate_cols]
    brr_variance = FAYS_ADJUSTMENT * np.mean(
        [(r - estimate) ** 2 for r in rep_estimates]
    )
    return estimate, float(np.sqrt(brr_variance))


def replicate_weight_cols(
    df: pd.DataFrame,
    brr_prefix: str = BRR_PREFIX,
    brr_count: int = BRR_COUNT,
) -> list[str]:
    """Replicate-weight columns actually present AND populated (>0 non-null)."""
    return [
        f"{brr_prefix}{i}"
        for i in range(1, brr_count + 1)
        if f"{brr_prefix}{i}" in df.columns and df[f"{brr_prefix}{i}"].notna().any()
    ]


def has_replicate_weights(df: pd.DataFrame) -> bool:
    """True if BRR variance can be computed on this slice (SAV cycles 2015+;
    the FWF cycles 2009/2012 ship replicate weights under different names that
    the current pipeline does not harmonise, so this is False for them)."""
    return len(replicate_weight_cols(df)) > 0


def pv_statistic_brr(
    df: pd.DataFrame,
    domain: str = "math",
    statistic: str = "at_risk",
    threshold: float | None = None,
    weight_col: str = WEIGHT_COL,
) -> dict[str, float]:
    """
    OECD-correct point estimate + total standard error for a statistic defined
    over plausible values, combining the TWO variance sources PISA requires:

    1. **Sampling variance** via Balanced Repeated Replication (80 Fay replicate
       weights) — computed *per plausible value*.
    2. **Imputation variance** across the plausible values, combined with Rubin's
       rules (between-PV variance ``B``).

    Total variance  T = Ubar + (1 + 1/m) * B   (Ubar = mean per-PV BRR variance).
    This is the method in the PISA Data Analysis Manual (OECD 2009, ch. 3–5); it
    is the correct alternative to a plain bootstrap, which ignores the design.

    statistic:
        "at_risk"    → weighted proportion below the PISA Level-2 ``threshold``
        "mean_score" → weighted mean of the plausible-value scale scores

    Returns a dict: estimate, se, ci95_low, ci95_high, fmi, sampling_var (Ubar),
    imputation_var (B), n_pv, n_replicates. If replicate weights are absent the
    sampling term is NaN and only the (under-estimated) imputation SE is returned,
    with ``brr_available=False`` so callers can flag it.
    """
    # Local import to avoid a circular dependency at module import time.
    from src.features.target import THRESHOLDS, _pv_columns

    pv_cols = _pv_columns(df, domain)
    # Drop padded all-NaN PV columns: the 2009/2012 files carry 10 PV slots but
    # only 5 are populated. An all-NaN column would otherwise become a spurious
    # 0-proportion for the "at_risk" indicator ((NaN < thr) -> False).
    pv_cols = [p for p in pv_cols if df[p].notna().any()]
    if not pv_cols:
        raise ValueError(f"No plausible-value columns for domain={domain}")
    thr = threshold if threshold is not None else THRESHOLDS[domain]
    rep_cols = replicate_weight_cols(df)
    base_w = df[weight_col]

    def _value(pv: str) -> pd.Series:
        col = df[pv]
        if statistic != "at_risk":
            return col
        # binary at-risk indicator, keeping NaN where the score is missing so
        # weighted_mean excludes those rows rather than counting them as 0.
        ind = (col < thr).astype(float)
        ind[col.isna()] = np.nan
        return ind

    per_pv_est: list[float] = []
    per_pv_var: list[float] = []
    for pv in pv_cols:
        val = _value(pv)
        q = weighted_mean(val, base_w)
        if np.isnan(q):
            continue
        per_pv_est.append(q)
        if rep_cols:
            rep_q = [weighted_mean(val, df[rc]) for rc in rep_cols]
            u = FAYS_ADJUSTMENT * float(np.nanmean([(r - q) ** 2 for r in rep_q]))
        else:
            u = np.nan
        per_pv_var.append(u)

    if len(per_pv_est) < 2:
        raise ValueError("Need >=2 valid plausible values to combine")

    m = len(per_pv_est)
    Q_bar = float(np.mean(per_pv_est))
    B = float(np.var(per_pv_est, ddof=1))          # imputation (between-PV) variance
    if rep_cols:
        U_bar = float(np.mean(per_pv_var))          # mean sampling variance
        T = U_bar + (1 + 1 / m) * B
        brr_available = True
    else:
        U_bar = np.nan
        T = (1 + 1 / m) * B                          # imputation-only (understated)
        brr_available = False
    se = float(np.sqrt(T)) if T > 0 else 0.0
    # FMI = imputation share of TOTAL variance; undefined without the sampling
    # term, so report NaN in the fallback rather than the tautological 1.0.
    fmi = (((1 + 1 / m) * B / T) if (T > 0 and not np.isnan(T)) else np.nan) if brr_available else np.nan

    return {
        "estimate": Q_bar,
        "se": se,
        "ci95_low": Q_bar - 1.96 * se,
        "ci95_high": Q_bar + 1.96 * se,
        "fmi": fmi,
        "sampling_var": U_bar,
        "imputation_var": B,
        "n_pv": m,
        "n_replicates": len(rep_cols),
        "brr_available": brr_available,
    }


def normalize_weights_within_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize W_FSTUWT so it sums to N (sample size) within each country.
    This is done when combining countries for cross-country modeling,
    so no single large country dominates the loss function.
    """
    df = df.copy()
    if WEIGHT_COL not in df.columns:
        logger.warning("W_FSTUWT not found — adding unit weights")
        df[WEIGHT_COL] = 1.0
        return df

    country_col = "CNT" if "CNT" in df.columns else "COUNTRY"
    if country_col not in df.columns:
        df[f"{WEIGHT_COL}_NORM"] = df[WEIGHT_COL] / df[WEIGHT_COL].sum() * len(df)
        return df

    # vectorized per-country normalization (no deprecated groupby.apply):
    # w_norm = w / sum_country(w) * n_country
    grp = df.groupby(country_col)[WEIGHT_COL]
    total_w = grp.transform("sum")
    n_country = grp.transform("size")
    norm = df[WEIGHT_COL] / total_w * n_country
    df[f"{WEIGHT_COL}_NORM"] = norm.where(total_w > 0, 1.0)
    return df


def compute_ses_quintiles(
    df: pd.DataFrame,
    ses_col: str = "ESCS",
    weight_col: str = WEIGHT_COL,
    n_quintiles: int = 5,
) -> pd.DataFrame:
    """Add SES quintile column using weighted quantile breaks."""
    df = df.copy()
    if ses_col not in df.columns:
        df["SES_QUINTILE"] = np.nan
        return df

    mask = df[ses_col].notna()
    if mask.sum() < n_quintiles:
        df["SES_QUINTILE"] = np.nan
        return df

    w_col = weight_col if weight_col in df.columns else None
    weights = df[w_col] if w_col else pd.Series(np.ones(len(df)), index=df.index)

    cuts = weighted_quantile(
        df.loc[mask, ses_col],
        weights[mask],
        quantiles=[i / n_quintiles for i in range(1, n_quintiles)],
    )
    # Add -inf and +inf endpoints
    bins = [-np.inf] + list(cuts) + [np.inf]
    df["SES_QUINTILE"] = pd.cut(
        df[ses_col], bins=bins, labels=list(range(1, n_quintiles + 1))
    ).astype("Int64")
    return df
