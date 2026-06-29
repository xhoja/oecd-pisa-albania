"""
Cross-cycle harmonization for PISA student questionnaire data.

Resolves column name differences, encoding differences, and missing-value
patterns across the 2009, 2012, 2015, 2018, and 2022 PISA cycles.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import structlog
logger = structlog.get_logger(__name__)

# PISA missing codes
MISSING_CODES = {9995, 9996, 9997, 9998, 9999, 9, 99, 999}


# ---------------------------------------------------------------------------
# Gender harmonization
# ---------------------------------------------------------------------------

def _harmonize_gender(df: pd.DataFrame, cycle: int) -> pd.DataFrame:
    """
    Recode gender to GENDER: 0=Female, 1=Male.
    Raw col: ST04Q01 (2009/2012) or ST004D01T (2015+).
    Raw codes: 1=Female, 2=Male (PISA convention across all cycles).
    """
    src = "ST04Q01" if cycle in (2009, 2012) else "ST004D01T"
    if src not in df.columns:
        logger.warning("Gender column not found", cycle=cycle, column=src)
        df["GENDER"] = np.nan
        return df

    raw = df[src].replace(list(MISSING_CODES), np.nan)
    df["GENDER"] = raw.map({1: 0, 2: 1})  # 1=Female→0, 2=Male→1
    return df


# ---------------------------------------------------------------------------
# Grade repetition harmonization
# ---------------------------------------------------------------------------

def _harmonize_repeat(df: pd.DataFrame, cycle: int) -> pd.DataFrame:
    """
    Recode grade repetition to REPEAT: 0=never repeated, 1=repeated once or more.

    Preference order:
    1. OECD-derived REPEAT column (present in 2018/2022 SAV) — already 0/1.
    2. Raw item recode: ST07Q01 (2009/2012), ST007Q01TA/JA (2015+).
       Codes: 1=No never, 2=Yes once, 3=Yes twice+ (with value formats off).
    """
    # 1. Use OECD-derived REPEAT if it already exists with valid 0/1 values
    if "REPEAT" in df.columns:
        derived = pd.to_numeric(df["REPEAT"], errors="coerce")
        derived = derived.where(derived.isin([0, 1]))
        if derived.notna().mean() > 0.5:
            df["REPEAT"] = derived
            return df

    # 2. Fall back to raw questionnaire item
    if cycle in (2009, 2012):
        candidates = ["ST07Q01"]
    else:
        candidates = ["ST007Q01TA", "ST007Q01JA"]
    src = next((c for c in candidates if c in df.columns), None)

    if src is None:
        logger.warning("Grade repetition column not found", cycle=cycle)
        df["REPEAT"] = np.nan
        return df

    raw = pd.to_numeric(df[src], errors="coerce").replace(list(MISSING_CODES), np.nan)
    # 1=No never → 0; 2/3/4=repeated → 1
    df["REPEAT"] = raw.map({1: 0, 2: 1, 3: 1, 4: 1})

    return df


# ---------------------------------------------------------------------------
# School belonging harmonization
# ---------------------------------------------------------------------------

def _harmonize_belong(df: pd.DataFrame, cycle: int) -> pd.DataFrame:
    """
    BELONG: already a composite index in 2012+.
    In 2009, proxy is ATSCHL (sense of belonging at school).
    Note: instruments differ — flag 2009 with BELONG_INSTRUMENT='ATSCHL'.
    """
    if cycle == 2009:
        if "ATSCHL" in df.columns:
            raw = df["ATSCHL"].replace(list(MISSING_CODES), np.nan)
            df["BELONG"] = raw
            df["BELONG_PROXY"] = True  # flag: different instrument
        else:
            df["BELONG"] = np.nan
            df["BELONG_PROXY"] = True
    else:
        if "BELONG" in df.columns:
            df["BELONG"] = df["BELONG"].replace(list(MISSING_CODES), np.nan)
        else:
            df["BELONG"] = np.nan
        df["BELONG_PROXY"] = False
    return df


# ---------------------------------------------------------------------------
# Immigration status harmonization
# ---------------------------------------------------------------------------

def _harmonize_immig(df: pd.DataFrame, cycle: int) -> pd.DataFrame:
    """
    IMMIG: 0=Native, 1=Second-generation, 2=First-generation.
    Already IMMIG in most cycles; absent in 2015 PUF for Albania.
    """
    if "IMMIG" not in df.columns:
        df["IMMIG"] = np.nan
        return df

    raw = df["IMMIG"].replace(list(MISSING_CODES), np.nan)
    # PISA codes: 1=Native, 2=Second-gen, 3=First-gen (consistent across cycles)
    df["IMMIG"] = raw.map({1: 0, 2: 1, 3: 2})
    return df


# ---------------------------------------------------------------------------
# Grade deviation computation
# ---------------------------------------------------------------------------

def _compute_grade(df: pd.DataFrame, cycle: int) -> pd.DataFrame:
    """
    Compute GRADE = deviation from modal grade for a 15-year-old.
    Modal grade by country and cycle varies; use ST001D01T (or ST001Q01TA).
    Negative = behind, 0 = on-track, positive = ahead.
    """
    grade_col = None
    # ST01Q01 = 2009/2012 FWF; ST001D01T = 2015+ SAV; others as fallback
    for candidate in ["ST001D01T", "ST001Q01TA", "ST01Q01", "ST001Q01", "GRADE"]:
        if candidate in df.columns:
            grade_col = candidate
            break

    if grade_col is None:
        df["GRADE"] = np.nan
        return df

    raw_grade = df[grade_col].replace(list(MISSING_CODES), np.nan)
    modal = raw_grade.mode()
    modal_val = modal.iloc[0] if not modal.empty else np.nan
    df["GRADE"] = raw_grade - modal_val
    return df


# ---------------------------------------------------------------------------
# Math plausible values
# ---------------------------------------------------------------------------

def _extract_pvs(df: pd.DataFrame, domain: str, cycle: int) -> pd.DataFrame:
    """
    Extract all available plausible values for a domain.
    Returns df with columns PV1MATH..PVnMATH (or READ/SCIE).
    Cycles: 2009/2012 have 5 PVs; 2015/2018/2022 have 10 PVs.
    """
    prefix = {"math": "MATH", "reading": "READ", "science": "SCIE"}[domain]
    max_pv = 5 if cycle in (2009, 2012) else 10
    available_pvs = []
    for i in range(1, max_pv + 1):
        col = f"PV{i}{prefix}"
        if col in df.columns:
            df[col] = df[col].replace(list(MISSING_CODES), np.nan)
            available_pvs.append(col)
        else:
            logger.warning("PV column missing", column=col, cycle=cycle)
    return df, available_pvs


# ---------------------------------------------------------------------------
# Main harmonization entry point
# ---------------------------------------------------------------------------

def harmonize(
    df: pd.DataFrame,
    cycle: int,
    country: str | None = None,
) -> pd.DataFrame:
    """
    Apply all harmonization steps to a single-cycle dataframe.
    Returns dataframe with standardized column names ready for modeling.
    """
    logger.info("Harmonizing", cycle=cycle, country=country, rows=len(df))

    df = df.copy()

    # Add cycle and country metadata
    df["CYCLE"] = cycle
    if country:
        df["COUNTRY"] = country

    # Replace PISA missing codes globally
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[col] = df[col].replace(list(MISSING_CODES), np.nan)

    # Harmonize each variable
    df = _harmonize_gender(df, cycle)
    df = _harmonize_repeat(df, cycle)
    df = _harmonize_belong(df, cycle)
    df = _harmonize_immig(df, cycle)
    df = _compute_grade(df, cycle)

    # Extract and keep PVs
    df, math_pvs = _extract_pvs(df, "math", cycle)
    df["MATH_PV_COLS"] = str(math_pvs)  # metadata column (string, not used in models)
    df["N_MATH_PVS"] = len(math_pvs)

    # Compute PV mean for convenience (NOT used as model target — use per-PV approach)
    if math_pvs:
        df["MATH_PV_MEAN"] = df[math_pvs].mean(axis=1)

    # Standardize composite index columns (already OECD-computed)
    for col in ["ESCS", "HOMEPOS", "TEACHSUP", "ICTHOME", "ICTSCH",
                "ANXMAT", "DISCLIMA", "HEDRES", "CULTPOSS", "WEALTH",
                "PERSEV", "GFOFAIL", "MASTGOAL", "COMPICT", "ENTUSE",
                "STUBEHA", "TEABEHA", "SCMAT", "INTMAT", "MOTIVAT",
                "HISCED", "HISEI"]:
        if col in df.columns:
            df[col] = df[col].replace(list(MISSING_CODES), np.nan)

    logger.info("Harmonization complete", cycle=cycle, cols=len(df.columns))
    return df


def combine_cycles(cycle_dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate harmonized dataframes from multiple cycles."""
    dfs = []
    for cycle, df in sorted(cycle_dfs.items()):
        df = df.copy()
        df["CYCLE"] = cycle
        dfs.append(df)
    if not dfs:
        logger.warning("combine_cycles called with no dataframes")
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    logger.info("Cycles combined", total_rows=len(combined), cycles=list(cycle_dfs.keys()))
    return combined
