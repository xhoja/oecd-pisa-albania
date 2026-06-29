"""
Feature selection utilities: VIF, correlation pruning, zero-variance removal.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

import structlog
logger = structlog.get_logger(__name__)


def remove_zero_variance(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Remove features with zero or near-zero variance (< 1e-6)."""
    kept = []
    for feat in features:
        if feat not in df.columns:
            continue
        if df[feat].std() > 1e-6:
            kept.append(feat)
        else:
            logger.info("Removing zero-variance feature", feature=feat)
    return kept


def remove_high_missingness(
    df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.80,
) -> list[str]:
    """Remove features with > threshold fraction missing."""
    kept = []
    for feat in features:
        if feat not in df.columns:
            continue
        miss_rate = df[feat].isna().mean()
        if miss_rate <= threshold:
            kept.append(feat)
        else:
            logger.info("Removing high-missingness feature", feature=feat, miss_rate=round(miss_rate, 3))
    return kept


def remove_correlated(
    df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.95,
) -> list[str]:
    """
    Remove one of each pair of features with |correlation| > threshold.
    Keeps the first feature in the pair (ordering matters).
    """
    sub = df[features].dropna()
    if sub.empty:
        return features

    corr = sub.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        if any(upper[col] > threshold):
            to_drop.add(col)
            logger.info("Removing correlated feature", feature=col, threshold=threshold)

    return [f for f in features if f not in to_drop]


def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each feature.
    VIF > 10 indicates problematic multicollinearity.
    """
    complete = df[features].dropna()
    if len(complete) < len(features) + 1:
        logger.warning("Not enough complete rows for VIF computation")
        return pd.DataFrame({"feature": features, "VIF": [np.nan] * len(features)})

    vif_data = []
    for i, feat in enumerate(features):
        try:
            vif = variance_inflation_factor(complete.values, i)
        except Exception:
            vif = np.nan
        vif_data.append({"feature": feat, "VIF": round(vif, 2)})

    result = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)
    high_vif = result[result["VIF"] > 10]
    if not high_vif.empty:
        logger.warning("High VIF features detected", features=high_vif["feature"].tolist())
    return result


def select_features(
    df: pd.DataFrame,
    features: list[str],
    remove_zero_var: bool = True,
    max_missingness: float = 0.80,
    max_correlation: float = 0.95,
) -> list[str]:
    """Full feature selection pipeline."""
    if remove_zero_var:
        features = remove_zero_variance(df, features)
    features = remove_high_missingness(df, features, threshold=max_missingness)
    features = remove_correlated(df, features, threshold=max_correlation)
    logger.info("Feature selection complete", n_features=len(features))
    return features
