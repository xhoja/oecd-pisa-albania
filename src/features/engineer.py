"""
Feature engineering for PISA student data.

All transformations are deterministic given the input dataframe.
No fitting required — these are domain-knowledge-driven constructions.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import structlog
logger = structlog.get_logger(__name__)


def add_ses_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    SES_COMPLETE: standardized mean of available SES indicators.
    More robust than any single index alone.
    """
    df = df.copy()
    ses_cols = ["ESCS", "HOMEPOS", "HISCED", "HISEI"]
    available = [c for c in ses_cols if c in df.columns and df[c].notna().sum() > 10]

    if not available:
        df["SES_COMPLETE"] = np.nan
        return df

    # Standardize each component (within the dataset passed in)
    standardized = pd.DataFrame(index=df.index)
    for col in available:
        mu = df[col].mean()
        sigma = df[col].std()
        if sigma > 0:
            standardized[col] = (df[col] - mu) / sigma
        else:
            standardized[col] = 0.0

    df["SES_COMPLETE"] = standardized.mean(axis=1)
    return df


def add_material_deficit(df: pd.DataFrame, country_col: str | None = None) -> pd.DataFrame:
    """
    MATERIAL_DEFICIT: 1 - percentile rank of HOMEPOS within country.
    Captures relative deprivation rather than absolute poverty.
    """
    df = df.copy()
    if "HOMEPOS" not in df.columns:
        df["MATERIAL_DEFICIT"] = np.nan
        return df

    if country_col and country_col in df.columns:
        df["MATERIAL_DEFICIT"] = 1 - df.groupby(country_col)["HOMEPOS"].transform(
            lambda s: s.rank(pct=True)
        )
    else:
        df["MATERIAL_DEFICIT"] = 1 - df["HOMEPOS"].rank(pct=True)

    return df


def add_digital_readiness(df: pd.DataFrame) -> pd.DataFrame:
    """
    DIGITAL_READINESS: mean of standardized available ICT indices.
    DIGITAL_GAP: ICTHOME - ICTSCH (home vs. school digital resource asymmetry).
    """
    df = df.copy()
    ict_cols = ["ICTHOME", "ICTSCH", "COMPICT"]
    available = [c for c in ict_cols if c in df.columns and df[c].notna().sum() > 10]

    if available:
        standardized = pd.DataFrame(index=df.index)
        for col in available:
            mu = df[col].mean()
            sigma = df[col].std()
            standardized[col] = (df[col] - mu) / sigma if sigma > 0 else 0.0
        df["DIGITAL_READINESS"] = standardized.mean(axis=1)
    else:
        df["DIGITAL_READINESS"] = np.nan

    if "ICTHOME" in df.columns and "ICTSCH" in df.columns:
        df["DIGITAL_GAP"] = df["ICTHOME"] - df["ICTSCH"]
    else:
        df["DIGITAL_GAP"] = np.nan

    return df


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-motivated interaction terms.
    Each one tests a hypothesis from the educational psychology literature.
    """
    df = df.copy()

    # SES moderates anxiety: does high SES buffer math anxiety?
    if "ESCS" in df.columns and "ANXMAT" in df.columns:
        df["SES_x_ANXIETY"] = df["ESCS"] * df["ANXMAT"]

    # Joint school experience: belonging and teacher support reinforce each other
    if "BELONG" in df.columns and "TEACHSUP" in df.columns:
        df["BELONG_x_TEACHSUP"] = df["BELONG"] * df["TEACHSUP"]

    # Grade repetition is far more damaging for low-SES students
    if "GRADE" in df.columns and "ESCS" in df.columns:
        df["GRADE_x_SES"] = df["GRADE"] * df["ESCS"]

    # Gender differences in math anxiety are well-documented
    if "GENDER" in df.columns and "ANXMAT" in df.columns:
        df["GENDER_x_ANXMAT"] = df["GENDER"] * df["ANXMAT"]

    return df


def add_country_normalized(
    df: pd.DataFrame,
    features: list[str],
    country_col: str = "CNT",
) -> pd.DataFrame:
    """
    Add country-normalized z-scores for specified features.
    Columns: <feature>_NORM.
    Used in cross-country models to remove country-level intercepts.
    """
    df = df.copy()
    if country_col not in df.columns:
        logger.warning("Country column not found — skipping normalization", col=country_col)
        return df

    for feat in features:
        if feat not in df.columns:
            continue
        norm_col = f"{feat}_NORM"
        grp = df.groupby(country_col)[feat]
        mu = grp.transform("mean")
        sigma = grp.transform("std")
        df[norm_col] = ((df[feat] - mu) / sigma).where(sigma > 0, 0.0)

    return df


def add_home_educational_resources(df: pd.DataFrame) -> pd.DataFrame:
    """
    HOME_EDU_SCORE: sum of binary indicators for key educational resources.
    Uses raw item columns where available (desk, books, quiet study place, internet).
    Falls back to HEDRES index if raw items unavailable.
    """
    df = df.copy()
    # Raw items vary by cycle; try common names
    resource_items = {
        "desk": ["ST012Q01NA", "ST011Q01TA"],
        "books_25plus": ["ST013Q01TA"],
        "quiet_study": ["ST012Q02NA", "ST011Q02TA"],
        "internet": ["ST012Q05NA", "ST011Q06TA"],
        "own_room": ["ST012Q03NA", "ST011Q03TA"],
    }

    scores = pd.DataFrame(index=df.index)
    for resource, candidates in resource_items.items():
        for col in candidates:
            if col in df.columns:
                # Typically 1=Yes, 2=No in PISA
                scores[resource] = (df[col] == 1).astype(float)
                break

    if scores.shape[1] > 0:
        df["HOME_EDU_SCORE"] = scores.sum(axis=1) / scores.shape[1]
    elif "HEDRES" in df.columns:
        df["HOME_EDU_SCORE"] = df["HEDRES"]
    else:
        df["HOME_EDU_SCORE"] = np.nan

    return df


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal features for longitudinal models.
    """
    df = df.copy()
    if "CYCLE" not in df.columns:
        return df

    cycle_index = {2009: 0, 2012: 1, 2015: 2, 2018: 3, 2022: 4}
    df["CYCLE_INDEX"] = df["CYCLE"].map(cycle_index)
    df["PRE_COVID"] = (df["CYCLE"] < 2022).astype(int)
    df["TREND_PHASE"] = df["CYCLE"].apply(
        lambda c: "improving" if c in (2009, 2012, 2015, 2018) else "crisis"
    )

    return df


def engineer_all(
    df: pd.DataFrame,
    country_col: str | None = None,
    normalize_countries: bool = False,
    add_temporal: bool = False,
) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline.
    All steps are safe to call even when source columns are missing.
    """
    df = add_ses_composite(df)
    df = add_material_deficit(df, country_col=country_col)
    df = add_digital_readiness(df)
    df = add_interaction_terms(df)
    df = add_home_educational_resources(df)

    if normalize_countries and country_col:
        df = add_country_normalized(
            df,
            features=["ESCS", "HOMEPOS", "ANXMAT", "BELONG", "TEACHSUP"],
            country_col=country_col,
        )

    if add_temporal:
        df = add_cycle_features(df)

    logger.info("Feature engineering complete", total_cols=len(df.columns))
    return df
