"""
Extract country subsets from raw PISA SPSS files.
Handles 2018 (.sav) and 2022 (.SAV) full-dataset files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd
import pyreadstat

import structlog
logger = structlog.get_logger(__name__)

# PISA missing value codes → replace with NaN
PISA_MISSING_CODES = {9995, 9996, 9997, 9998, 9999, 99, 9, 999}


def load_spss_full(
    path: str | Path,
    countries: list[str] | None = None,
    usecols: list[str] | None = None,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """
    Load PISA SPSS file, optionally filtering to specific countries.
    Streams in chunks to handle >2GB files without OOM.
    """
    path = Path(path)
    logger.info("Loading SPSS file", path=str(path), countries=countries)

    chunks: list[pd.DataFrame] = []
    offset = 0

    while True:
        try:
            df, meta = pyreadstat.read_sav(
                str(path),
                row_offset=offset,
                row_limit=chunksize,
                usecols=usecols,
                apply_value_formats=False,
            )
        except Exception as e:
            if offset == 0:
                raise
            break

        if df.empty:
            break

        raw_len = len(df)  # track RAW chunk size before filtering

        if countries:
            cnt_col = _detect_country_col(df)
            df = df[df[cnt_col].isin(countries)]

        if not df.empty:
            chunks.append(df)
        offset += chunksize
        logger.debug("Chunk loaded", offset=offset, raw=raw_len, rows_kept=len(df))

        # Terminate only when the RAW chunk is short - that means the file is
        # exhausted. Filtering shrinks df but must NOT trigger early exit.
        if raw_len < chunksize:
            break

    result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    logger.info("Extraction complete", total_rows=len(result))
    return result


def load_spss_single_pass(
    path: str | Path,
    countries: list[str] | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load entire SPSS file in one pass (use for smaller files <500MB).
    """
    path = Path(path)
    logger.info("Loading SPSS (single pass)", path=str(path))
    df, meta = pyreadstat.read_sav(
        str(path),
        usecols=usecols,
        apply_value_formats=False,
    )
    if countries:
        cnt_col = _detect_country_col(df)
        df = df[df[cnt_col].isin(countries)]
    logger.info("Loaded", rows=len(df), cols=len(df.columns))
    return df


def replace_pisa_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace all PISA missing-value codes with NaN."""
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        df[col] = df[col].where(~df[col].isin(PISA_MISSING_CODES))
    return df


def extract_country(
    df: pd.DataFrame,
    country: str,
) -> pd.DataFrame:
    """Filter dataframe to a single country."""
    cnt_col = _detect_country_col(df)
    subset = df[df[cnt_col] == country].copy()
    logger.info("Country extracted", country=country, rows=len(subset))
    return subset


def _detect_country_col(df: pd.DataFrame) -> str:
    for candidate in ["CNT", "CNTRYID", "cnt", "Country"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No country column found. Available: {list(df.columns[:20])}")


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Save processed dataframe as parquet for fast subsequent loads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    logger.info("Saved", path=str(path), rows=len(df))


def load_processed(path: str | Path) -> pd.DataFrame:
    """Load parquet file."""
    return pd.read_parquet(path)


def load_school_questionnaire(
    path: str | Path,
    countries: list[str] | None = None,
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the PISA school questionnaire file (e.g. CY08MSP_SCH_QQQ.SAV) and return
    one row per school, keyed on ``CNTSCHID``. This is *independent* school-level
    signal (resources, staff, leadership, climate) as reported by the principal -
    not derivable from student rows, unlike the compositional ``SCH_MEAN_*``
    aggregates. Missing codes are converted to NaN.

    Args:
        path: school SPSS file (single pass; the file is small, ~20MB).
        countries: optional CNT filter (e.g. ["ALB"]).
        keep_cols: school variables to keep; ``CNTSCHID`` is always included.
    """
    df = load_spss_single_pass(path, countries=countries)
    df = replace_pisa_missing(df)
    if keep_cols is not None:
        cols = ["CNTSCHID"] + [c for c in keep_cols if c in df.columns]
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            logger.warning("School vars not in file - skipped", missing=missing)
        df = df[cols]
    # one row per school (SCH file is already school-level, but guard duplicates)
    df = df.drop_duplicates(subset="CNTSCHID")
    logger.info("School questionnaire loaded", rows=len(df), cols=len(df.columns))
    return df


def load_legacy_csv(path: str | Path) -> pd.DataFrame:
    """Load Albania-only CSV from old project."""
    df = pd.read_csv(path)
    logger.info("Legacy CSV loaded", path=str(path), rows=len(df), cols=len(df.columns))
    return df
