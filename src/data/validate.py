"""
Data contracts for processed PISA datasets.

A lightweight, dependency-free validation layer (no pandera/great_expectations)
that turns silent pipeline regressions into explicit, catalogued violations.
Run it on any processed parquet before modelling; wire ``assert_valid`` into the
pipeline to fail fast.

Two severities:
    ERROR  — the slice is unusable as-is (missing weights, no target, bad cycle).
    WARN   — usable but a documented caveat (no BRR replicate weights on a slice,
             an all-missing feature, a non-standard plausible-value count).

The checks encode facts verified for this project (see project memory): PISA
ships 80 Fay replicate weights on the SAV cycles (2015/2018/2022) but not on the
FWF cycles (2009/2012); Albania genuinely lacks ESCS in 2012 & 2015.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import structlog

from src.data.weights import BRR_COUNT, WEIGHT_COL, replicate_weight_cols
from src.features.target import PV_PREFIXES, _pv_columns

logger = structlog.get_logger(__name__)

VALID_CYCLES = {2009, 2012, 2015, 2018, 2022}
EXPECTED_PV_COUNT = 10


@dataclass(frozen=True)
class Violation:
    level: str   # "error" | "warn"
    code: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.level.upper():5s}] {self.code}: {self.message}"


def validate_processed(
    df: pd.DataFrame,
    *,
    domains: tuple[str, ...] = ("math", "reading", "science"),
    expected_pv_count: int = EXPECTED_PV_COUNT,
    weight_col: str = WEIGHT_COL,
    feature_cols: list[str] | None = None,
) -> list[Violation]:
    """
    Validate one processed slice (a cycle, a country, or the pooled panel).

    Args:
        df: processed dataframe.
        domains: PISA domains whose plausible values should be present.
        expected_pv_count: PISA ships 10 PVs per domain (2015+); older cycles
            shipped 5. A mismatch is a WARN, not an ERROR.
        weight_col: final student weight column.
        feature_cols: if given, each is checked for all-missing (a column that is
            100% NaN in this slice — often a genuine OECD gap, hence WARN).

    Returns a list of Violation (empty == clean).
    """
    v: list[Violation] = []

    # --- sampling weights -------------------------------------------------
    if weight_col not in df.columns:
        v.append(Violation("error", "weight_missing",
                            f"final weight column '{weight_col}' absent"))
    else:
        w = df[weight_col]
        if not np.isfinite(w.dropna()).all():
            v.append(Violation("error", "weight_nonfinite",
                                f"'{weight_col}' has non-finite values"))
        n_bad = int(((w <= 0) | w.isna()).sum())
        if n_bad:
            v.append(Violation("error", "weight_nonpositive",
                                f"{n_bad} rows have non-positive/NaN '{weight_col}'"))

    # --- BRR replicate weights -------------------------------------------
    rep = replicate_weight_cols(df)
    if not rep:
        v.append(Violation("warn", "brr_absent",
                            "no populated replicate weights — BRR standard errors "
                            "unavailable on this slice (expected for FWF 2009/2012)"))
    elif len(rep) != BRR_COUNT:
        v.append(Violation("warn", "brr_partial",
                            f"{len(rep)}/{BRR_COUNT} replicate weights populated"))

    # --- plausible values -------------------------------------------------
    any_pv = False
    for dom in domains:
        pv = _pv_columns(df, dom)
        if pv:
            any_pv = True
        if pv and len(pv) != expected_pv_count:
            v.append(Violation("warn", "pv_count",
                                f"domain '{dom}' has {len(pv)} PVs (expected "
                                f"{expected_pv_count})"))
    if not any_pv:
        v.append(Violation("error", "pv_missing",
                            f"no plausible-value columns for any of {domains}"))

    # --- target sanity ----------------------------------------------------
    for prefix in PV_PREFIXES.values():
        col = f"AT_RISK_{prefix}"
        if col in df.columns:
            vals = pd.unique(df[col].dropna())
            if not set(np.asarray(vals, dtype=float)).issubset({0.0, 1.0}):
                v.append(Violation("error", "target_range",
                                    f"'{col}' has values outside {{0,1}}"))

    # --- cycle values -----------------------------------------------------
    if "CYCLE" in df.columns:
        bad = set(df["CYCLE"].dropna().astype(int).unique()) - VALID_CYCLES
        if bad:
            v.append(Violation("error", "cycle_invalid",
                                f"unexpected CYCLE values: {sorted(bad)}"))

    # --- duplicate student ids -------------------------------------------
    # Only the student id must be unique; many students share a school id
    # (CNTSCHID), so that column is intentionally not checked here. In the
    # pooled panel the same id can recur across cycles, so scope per cycle.
    if "CNTSTUID" in df.columns and df["CNTSTUID"].notna().any():
        if "CYCLE" in df.columns:
            n_dup = int(df.duplicated(subset=["CYCLE", "CNTSTUID"]).sum())
        else:
            n_dup = int(df["CNTSTUID"].duplicated().sum())
        if n_dup:
            v.append(Violation("warn", "duplicate_id",
                                f"{n_dup} duplicate student ids (CNTSTUID)"))

    # --- all-missing features --------------------------------------------
    if feature_cols:
        for f in feature_cols:
            if f in df.columns and df[f].isna().all():
                v.append(Violation("warn", "feature_all_missing",
                                    f"feature '{f}' is 100% missing in this slice"))

    logger.info("Processed slice validated", n_rows=len(df),
                n_errors=sum(x.level == "error" for x in v),
                n_warnings=sum(x.level == "warn" for x in v))
    return v


def errors(violations: list[Violation]) -> list[Violation]:
    return [x for x in violations if x.level == "error"]


def assert_valid(df: pd.DataFrame, **kwargs) -> None:
    """Raise ValueError if any ERROR-level contract is violated (WARN is ok)."""
    errs = errors(validate_processed(df, **kwargs))
    if errs:
        raise ValueError("Processed data failed validation:\n  " +
                         "\n  ".join(str(e) for e in errs))


def replicate_coverage_by_cycle(df: pd.DataFrame, cycle_col: str = "CYCLE") -> pd.DataFrame:
    """
    Per-cycle report of BRR availability — the honest table for the paper:
    which cycles support design-based standard errors and which fall back to a
    weaker variance estimate.
    """
    if cycle_col not in df.columns:
        rep = replicate_weight_cols(df)
        return pd.DataFrame([{"cycle": "all", "n": len(df),
                              "n_replicates": len(rep), "brr_available": bool(rep)}])
    rows = []
    for c, g in df.groupby(cycle_col):
        rep = replicate_weight_cols(g)
        rows.append({"cycle": int(c), "n": len(g),
                     "n_replicates": len(rep), "brr_available": bool(rep)})
    return pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
