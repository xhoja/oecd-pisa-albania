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

    def _normalize_group(g: pd.DataFrame) -> pd.DataFrame:
        total_w = g[WEIGHT_COL].sum()
        if total_w > 0:
            g[f"{WEIGHT_COL}_NORM"] = g[WEIGHT_COL] / total_w * len(g)
        else:
            g[f"{WEIGHT_COL}_NORM"] = 1.0
        return g

    df = df.groupby(country_col, group_keys=False).apply(_normalize_group)
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
