"""
Feature engineering for PISA student data.

All transformations are deterministic given the input dataframe.
No fitting required - these are domain-knowledge-driven constructions.
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
        logger.warning("Country column not found - skipping normalization", col=country_col)
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


def add_school_aggregates(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    school_col: str = "CNTSCHID",
    weight_col: str = "W_FSTUWT",
) -> pd.DataFrame:
    """
    School-level contextual features - the *compositional* effect PISA research
    finds is often the single strongest predictor of individual achievement.

    For each ``col`` we add ``SCH_MEAN_<col>``: the **survey-weighted, leave-one-
    out** mean of that indicator among a student's schoolmates. Leave-one-out
    (excluding the student's own value) removes the trivial self-inclusion so the
    feature is a genuine peer/context measure, not a smuggled copy of the row's
    own value. ``SCH_N`` is the sampled cohort size in the school.

    Notes
    -----
    - School is a *known contextual attribute*, not a leaked label; computing the
      aggregate once over the cohort is standard in PISA work. Single-student
      schools (or all-missing) fall back to NaN → the median imputer fills them.
    - This is deliberately computed on the full slice for the "does school context
      help?" screen; a fully fold-safe version would recompute per training fold.
    """
    if cols is None:
        cols = ["ESCS", "HOMEPOS", "ANXMAT", "TEACHSUP"]
    df = df.copy()
    if school_col not in df.columns:
        logger.warning("School column not found - skipping school aggregates", col=school_col)
        return df

    w = df[weight_col].astype(float) if weight_col in df.columns else pd.Series(1.0, index=df.index)
    grp = df.groupby(school_col)

    for col in cols:
        if col not in df.columns:
            continue
        x = df[col]
        nn = x.notna()
        wx = (w * x).where(nn, 0.0)
        wv = w.where(nn, 0.0)
        wx_sum = wx.groupby(df[school_col]).transform("sum")
        wv_sum = wv.groupby(df[school_col]).transform("sum")
        # leave-one-out: subtract the student's own contribution when present
        num = wx_sum - wx
        den = wv_sum - wv
        loo = (num / den).where(den > 0, np.nan)
        df[f"SCH_MEAN_{col}"] = loo

    df["SCH_N"] = grp[school_col].transform("size").astype(float)
    logger.info("School aggregates added", cols=cols, n_schools=int(df[school_col].nunique()))
    return df


def add_school_questionnaire(
    df: pd.DataFrame,
    sch: pd.DataFrame,
    cols: list[str] | None = None,
    school_col: str = "CNTSCHID",
    prefix: str = "SCHQ_",
) -> pd.DataFrame:
    """
    Left-join *independent* school-questionnaire variables onto student rows via
    ``CNTSCHID``. Unlike ``add_school_aggregates`` (student composition averaged
    up), these are principal-reported school attributes - resources, staff
    shortage, class size, leadership, climate - that students cannot encode, so
    they test whether the ~0.78 AUC ceiling is a feature-set limit or a genuine
    data limit (ICC ≈ 0.31 says ~a third of risk is school-level).

    Joined columns are prefixed (``SCHQ_<VAR>``) to keep them distinct from the
    compositional ``SCH_MEAN_*`` features. The join is one-to-one on the school
    side (``sch`` is de-duplicated on ``school_col``), so student row count is
    preserved; schools absent from ``sch`` get NaN → the median imputer fills.

    Args:
        df: student dataframe (must contain ``school_col``).
        sch: school-level dataframe, one row per ``school_col`` (from
            ``extract.load_school_questionnaire``).
        cols: which school variables to bring in (default: all non-key columns
            of ``sch``).
        school_col: join key present in both frames.
        prefix: namespace applied to joined columns.
    """
    df = df.copy()
    if school_col not in df.columns:
        logger.warning("School column not found - skipping questionnaire join", col=school_col)
        return df
    if school_col not in sch.columns:
        raise ValueError(f"School frame missing join key {school_col!r}")

    if cols is None:
        cols = [c for c in sch.columns if c != school_col]
    else:
        cols = [c for c in cols if c in sch.columns]

    right = sch.drop_duplicates(subset=school_col)[[school_col] + cols].copy()
    rename = {c: f"{prefix}{c}" for c in cols}
    right = right.rename(columns=rename)

    n_before = len(df)
    merged = df.merge(right, on=school_col, how="left")
    assert len(merged) == n_before, "questionnaire join changed row count"

    matched = df[school_col].isin(set(right[school_col])).mean()
    logger.info(
        "School questionnaire joined",
        cols=list(rename.values()),
        n_schools=int(right[school_col].nunique()),
        student_match_rate=round(float(matched), 4),
    )
    return merged


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
