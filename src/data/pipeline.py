"""
End-to-end data pipeline: raw SPSS → harmonized, processed, feature-engineered DataFrames.

Usage:
    python -m src.data.pipeline --cycles 2018 2022 --countries ALB EST FIN MKD MNE

This script:
1. Loads raw SPSS files (2018, 2022 full datasets)
2. Filters to target countries
3. Replaces PISA missing codes
4. Harmonizes column names and encodings
5. Adds SES quintiles and sampling weight normalization
6. Saves per-country, per-cycle parquet files to data/processed/
7. Combines Albania CSVs for longitudinal dataset
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path when running as script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.extract import (
    load_legacy_csv,
    load_spss_full,
    replace_pisa_missing,
    save_processed,
)
from src.data.harmonize import combine_cycles, harmonize
from src.data.parse_fwf import read_pisa_fwf
from src.data.weights import compute_ses_quintiles, normalize_weights_within_country
from src.features.engineer import engineer_all
from src.utils.config import Config
from src.utils.logging import configure_logging
from src.utils.reproducibility import set_seeds

import structlog
logger = structlog.get_logger(__name__)


RAW_DIR = PROJECT_ROOT / "data/raw"

# Full-country SPSS (.SAV) files
RAW_FILES = {
    2022: RAW_DIR / "CY08MSP_STU_QQQ.SAV",
    2018: RAW_DIR / "CY07_MSU_STU_QQQ.sav",
    2015: RAW_DIR / "CY6_MS_CMB_STU_QQQ.sav",  # extracted from downloaded zip
}

# Fixed-width ASCII (.txt) files + their SPSS control files (2009, 2012)
FWF_FILES = {
    2012: {
        "data": RAW_DIR / "INT_STU12_DEC03.txt",
        "control": PROJECT_ROOT.parent
        / "DataScienceProject"
        / "SPSS syntax to read in student questionnaire data file.txt",
    },
    2009: {
        "data": RAW_DIR / "INT_STQ09_DEC11.txt",
        "control": RAW_DIR / "PISA2009_SPSS_student.txt",
    },
}

# Variables to extract from FWF cycles (PISA short names). BRR weights added below.
FWF_VARIABLES = [
    "ST04Q01",   # gender
    "ST07Q01",   # grade repetition
    "ST01Q01",   # grade (international)
    "ESCS", "HOMEPOS", "HISCED", "HISEI",
    "ANXMAT", "BELONG", "ATSCHL",  # ATSCHL is the 2009 belonging proxy
    "TEACHSUP", "IMMIG",
    "PV1MATH", "PV2MATH", "PV3MATH", "PV4MATH", "PV5MATH",
    "PV1READ", "PV2READ", "PV3READ", "PV4READ", "PV5READ",
    "PV1SCIE", "PV2SCIE", "PV3SCIE", "PV4SCIE", "PV5SCIE",
    "W_FSTUWT",
]

LEGACY_CSV_PATHS = {
    2009: PROJECT_ROOT.parent / "DataScienceProject" / "albania_pisa2009.csv",
    2012: PROJECT_ROOT.parent / "DataScienceProject" / "albania_pisa2012.csv",
    2015: PROJECT_ROOT.parent / "DataScienceProject" / "albania_pisa2015.csv",
    2018: PROJECT_ROOT.parent / "DataScienceProject" / "albania_pisa2018.csv",
    2022: PROJECT_ROOT.parent / "DataScienceProject" / "albania_pisa2022.csv",
}


def process_spss_cycle(
    cycle: int,
    countries: list[str],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Process one full-country SPSS cycle → save per-country parquet."""
    raw_path = RAW_FILES.get(cycle)
    if raw_path is None or not raw_path.exists():
        # Try alternate location (old project folder)
        alt = PROJECT_ROOT.parent / "DataScienceProject" / raw_path.name if raw_path else None
        if alt and alt.exists():
            raw_path = alt
        else:
            logger.warning("Raw SPSS file not found", cycle=cycle, path=str(raw_path))
            return {}

    logger.info("Processing SPSS cycle", cycle=cycle, countries=countries)
    df_raw = load_spss_full(str(raw_path), countries=countries)
    df_raw = replace_pisa_missing(df_raw)

    results: dict[str, pd.DataFrame] = {}
    cnt_col = "CNT" if "CNT" in df_raw.columns else "CNTRYID"

    for country in countries:
        df_country = df_raw[df_raw[cnt_col] == country].copy()
        if df_country.empty:
            logger.warning("Country not found in cycle", country=country, cycle=cycle)
            continue

        df_harm = harmonize(df_country, cycle=cycle, country=country)
        df_harm = compute_ses_quintiles(df_harm)
        df_harm = engineer_all(df_harm, country_col=cnt_col)

        out_path = output_dir / f"{country.lower()}_{cycle}.parquet"
        save_processed(df_harm, out_path)
        results[country] = df_harm

    return results


def process_fwf_cycle(
    cycle: int,
    countries: list[str],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Process one fixed-width ASCII cycle (2009/2012) → save per-country parquet."""
    entry = FWF_FILES.get(cycle)
    if entry is None:
        logger.warning("No FWF entry for cycle", cycle=cycle)
        return {}
    if not entry["data"].exists() or not entry["control"].exists():
        logger.warning("FWF data/control missing", cycle=cycle)
        return {}

    logger.info("Processing FWF cycle", cycle=cycle, countries=countries)
    df_raw = read_pisa_fwf(
        data_path=str(entry["data"]),
        control_path=str(entry["control"]),
        variables=FWF_VARIABLES,
        countries=countries,
    )

    results: dict[str, pd.DataFrame] = {}
    for country in countries:
        df_country = df_raw[df_raw["CNT"] == country].copy()
        if df_country.empty:
            logger.warning("Country absent from cycle (did not participate)", country=country, cycle=cycle)
            continue

        df_harm = harmonize(df_country, cycle=cycle, country=country)
        df_harm = compute_ses_quintiles(df_harm)
        df_harm = engineer_all(df_harm, country_col="CNT")

        out_path = output_dir / f"{country.lower()}_{cycle}.parquet"
        save_processed(df_harm, out_path)
        results[country] = df_harm

    return results


def process_cycle(cycle: int, countries: list[str], output_dir: Path) -> dict[str, pd.DataFrame]:
    """Dispatch to SPSS or FWF processor based on cycle file type."""
    if cycle in RAW_FILES and RAW_FILES[cycle].exists():
        return process_spss_cycle(cycle, countries, output_dir)
    if cycle in FWF_FILES:
        return process_fwf_cycle(cycle, countries, output_dir)
    logger.warning("No data source for cycle", cycle=cycle)
    return {}


def process_albania_longitudinal(output_dir: Path) -> pd.DataFrame:
    """
    Build the full Albanian longitudinal dataset from RAW sources (not legacy CSVs).

    Extracts Albania directly from each cycle's authoritative file:
    2009/2012 → fixed-width ASCII; 2015/2018/2022 → SPSS .SAV.
    This recovers the corrected 2012 ESCS and the real 2015 background indices,
    both of which were missing/buggy in the old project's CSVs.
    """
    cycle_dfs: dict[int, pd.DataFrame] = {}

    for cycle in [2009, 2012, 2015, 2018, 2022]:
        result = process_cycle(cycle, ["ALB"], output_dir)
        if "ALB" in result:
            df = engineer_all(result["ALB"], add_temporal=True)
            cycle_dfs[cycle] = df
            logger.info("Albania cycle processed", cycle=cycle, rows=len(df))
        else:
            logger.warning("Albania longitudinal: cycle unavailable", cycle=cycle)

    combined = combine_cycles(cycle_dfs)
    out_path = output_dir / "albania_longitudinal.parquet"
    save_processed(combined, out_path)
    logger.info("Albania longitudinal dataset saved", rows=len(combined))
    return combined


def build_comparison_dataset(
    cycles: list[int],
    countries: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Build cross-country comparison dataset for 2018 and 2022.
    Countries: ALB + comparison set (Balkans, top performers, GDP-matched).
    """
    all_dfs: list[pd.DataFrame] = []

    for cycle in cycles:
        country_dfs = process_cycle(cycle, countries, output_dir)
        for _country, df in country_dfs.items():
            all_dfs.append(df)

    if not all_dfs:
        logger.error("No data loaded for comparison dataset")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True, sort=False)
    combined = normalize_weights_within_country(combined)

    out_path = output_dir / "comparison_dataset.parquet"
    save_processed(combined, out_path)
    logger.info("Comparison dataset saved", rows=len(combined), countries=countries)
    return combined


def main() -> None:
    configure_logging(level="INFO")
    set_seeds(42)

    parser = argparse.ArgumentParser(description="PISA data pipeline")
    parser.add_argument("--cycles", nargs="+", type=int, default=[2018, 2022])
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["ALB", "MKD", "MNE", "SRB", "BGR", "EST", "FIN", "MEX", "COL"],
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--albania-longitudinal", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.albania_longitudinal:
        process_albania_longitudinal(output_dir)

    build_comparison_dataset(args.cycles, args.countries, output_dir)
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()
