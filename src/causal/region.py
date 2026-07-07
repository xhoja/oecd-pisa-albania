"""
Decode the PISA sampling STRATUM into geographic bands for Albania.

Albania's PISA public-use file has no subnational REGION/SUBNATIO breakdown
(both are single-valued for the country), but the *explicit sampling stratum*
labels every school by three axes:

    <Urbanicity> \\ <Region band> \\ <Sector>
    e.g. "ALB - stratum 03: Urban / Center / Public"

The region band (North / Center / South) is the geographic dimension we exploit
for the earthquake difference-in-differences. The three bands are stable across
the 2015, 2018 and 2022 cycles even though the numeric stratum *codes* differ
between cycles (2018 uses 4-digit ``ALB0203``, 2022 uses 2-digit ``ALB03``),
so we parse the *label text*, not the code, and build a per-cycle
code -> band lookup.

The raw SAV files carry the value labels; the processed parquet keeps only the
codes. ``build_stratum_lookup`` reads the SAV metadata once and writes a small
``stratum_region_lookup.csv`` that the analysis (and the notebook) consume, so
the causal notebook is reproducible without re-reading the multi-GB SAVs.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

REGION_BANDS = ("North", "Center", "South")
URBANICITIES = ("Urban", "Rural")
SECTORS = ("Public", "Private")

# The quake-affected band. The 26 Nov 2019 M6.4 earthquake was centred off
# Durres on the central coast; the declared state-of-emergency counties
# (Durres and Tirana) both fall in PISA's "Center" band.
TREATED_BAND = "Center"

# Regex tolerant of both "/" (2018/2022) and "\" (2015) separators and of the
# "Public and Private" merged sector that appears in 2022.
_LABEL_RE = re.compile(
    r"(Urban|Rural)\s*[\\/]\s*(North|Center|South)\s*[\\/]\s*"
    r"(Public(?:\s+and\s+Private)?|Private)",
    re.IGNORECASE,
)


def parse_stratum_label(label: str) -> dict[str, str] | None:
    """Extract (urbanicity, region, sector) from a STRATUM value label.

    Returns ``None`` for non-geographic strata (e.g. "Undisclosed STRATUM").
    """
    if not isinstance(label, str):
        return None
    m = _LABEL_RE.search(label)
    if not m:
        return None
    urban, region, sector = m.group(1), m.group(2), m.group(3)
    return {
        "urbanicity": urban.title(),
        "region": region.title(),
        "sector": "Public" if sector.lower().startswith("public") else "Private",
    }


def build_stratum_lookup(
    sav_paths: dict[int, str | Path],
    country_prefix: str = "ALB",
) -> pd.DataFrame:
    """Read STRATUM value labels from raw SAV files and decode region bands.

    Args:
        sav_paths: {cycle_year: path_to_sav} for SAV cycles (2015/2018/2022).
        country_prefix: keep only strata whose code starts with this prefix.

    Returns:
        DataFrame[CYCLE, STRATUM, label, urbanicity, region, sector] with one
        row per (cycle, stratum code). Undisclosed / unparseable strata are
        kept with NaN band fields so coverage can be audited.
    """
    import pyreadstat  # local import: heavy, only needed to (re)build the lookup

    rows: list[dict] = []
    for cycle, path in sav_paths.items():
        _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
        labels = meta.variable_value_labels.get("STRATUM", {})
        for code, label in labels.items():
            if not (isinstance(code, str) and code.startswith(country_prefix)):
                continue
            parsed = parse_stratum_label(label)
            rows.append(
                {
                    "CYCLE": cycle,
                    "STRATUM": code,
                    "label": label,
                    "urbanicity": parsed["urbanicity"] if parsed else None,
                    "region": parsed["region"] if parsed else None,
                    "sector": parsed["sector"] if parsed else None,
                }
            )
    lookup = pd.DataFrame(rows)
    n_bad = int(lookup["region"].isna().sum())
    logger.info(
        "Built stratum lookup",
        cycles=sorted(sav_paths),
        n_strata=len(lookup),
        n_undecoded=n_bad,
    )
    return lookup


def add_region_band(
    df: pd.DataFrame,
    lookup: pd.DataFrame,
    treated_band: str = TREATED_BAND,
) -> pd.DataFrame:
    """Merge region band onto microdata by (CYCLE, STRATUM) and add flags.

    Adds columns: ``region``, ``urbanicity``, ``sector`` and a binary
    ``TREATED`` (1 if region == ``treated_band``). Rows whose stratum cannot be
    decoded get NaN band fields and are excluded from ``TREATED`` (kept NaN) so
    the caller can drop them explicitly rather than silently miscoding them.
    """
    keys = ["CYCLE", "STRATUM"]
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"microdata missing join keys {missing}")
    band_cols = lookup[keys + ["region", "urbanicity", "sector"]].drop_duplicates(keys)
    out = df.merge(band_cols, on=keys, how="left")
    out["TREATED"] = pd.Series(
        pd.NA, index=out.index, dtype="Int64"
    ).mask(out["region"].notna(), (out["region"] == treated_band).astype("Int64"))
    return out
