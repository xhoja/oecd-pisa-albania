"""
Target variable construction for low-proficiency classification.

Two modes:
1. Point target (for EDA / quick modeling): binary indicator from the mean of
   available plausible values. Convenient but underestimates uncertainty.
2. Per-PV targets (for rigorous modeling): one binary target per plausible value,
   combined downstream via Rubin's rules. See src/data/impute.fit_per_pv.

PISA Level 2 thresholds (below = at risk):
    Math    420.07
    Reading 407.47
    Science 409.54
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import structlog

logger = structlog.get_logger(__name__)

# Exact OECD PISA Level 2 lower bounds (below = at risk). Do NOT round —
# rounding shifts the at-risk rate by a non-trivial fraction of a point.
THRESHOLDS = {"math": 420.07, "reading": 407.47, "science": 409.54}

PV_PREFIXES = {"math": "MATH", "reading": "READ", "science": "SCIE"}


def _pv_columns(df: pd.DataFrame, domain: str) -> list[str]:
    prefix = PV_PREFIXES[domain]
    cols = [c for c in df.columns if c.startswith("PV") and c.endswith(prefix)]
    # Keep only PV<i><DOMAIN> form
    return sorted(cols, key=lambda c: int(c[2:c.index(prefix)]) if c[2:c.index(prefix)].isdigit() else 99)


def add_point_target(
    df: pd.DataFrame,
    domain: str = "math",
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Add AT_RISK_<DOMAIN> binary column from the mean of available PVs.
    Also ensures <DOMAIN>_PV_MEAN exists.
    """
    df = df.copy()
    thr = threshold if threshold is not None else THRESHOLDS[domain]
    prefix = PV_PREFIXES[domain]
    mean_col = f"{prefix}_PV_MEAN"

    if mean_col not in df.columns:
        pv_cols = _pv_columns(df, domain)
        if not pv_cols:
            logger.warning("No PV columns for domain", domain=domain)
            df[f"AT_RISK_{prefix}"] = np.nan
            return df
        df[mean_col] = df[pv_cols].mean(axis=1)

    df[f"AT_RISK_{prefix}"] = (df[mean_col] < thr).astype("Int64")
    rate = df[f"AT_RISK_{prefix}"].mean()
    logger.info("Point target added", domain=domain, threshold=thr, at_risk_rate=round(float(rate), 3))
    return df


def add_all_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add AT_RISK_MATH, AT_RISK_READ, AT_RISK_SCIE, and AT_RISK_ANY."""
    df = df.copy()
    for domain in ["math", "reading", "science"]:
        df = add_point_target(df, domain=domain)

    risk_cols = [f"AT_RISK_{p}" for p in PV_PREFIXES.values() if f"AT_RISK_{p}" in df.columns]
    valid = [c for c in risk_cols if df[c].notna().any()]
    if valid:
        df["AT_RISK_ANY"] = (df[valid].fillna(0).sum(axis=1) > 0).astype("Int64")
        df["AT_RISK_ALL"] = (df[valid].fillna(0).sum(axis=1) == len(valid)).astype("Int64")
    return df


def per_pv_targets(
    df: pd.DataFrame,
    domain: str = "math",
    threshold: float | None = None,
) -> tuple[list[pd.Series], list[str]]:
    """
    Return a list of binary target Series — one per plausible value.
    Used for rigorous Rubin's-rules modeling (fit once per target).
    """
    thr = threshold if threshold is not None else THRESHOLDS[domain]
    pv_cols = _pv_columns(df, domain)
    targets = [(df[c] < thr).astype(int) for c in pv_cols]
    logger.info("Per-PV targets built", domain=domain, n_pvs=len(pv_cols))
    return targets, pv_cols
