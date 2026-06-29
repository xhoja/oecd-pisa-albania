"""
Parse PISA fixed-width ASCII data files (2009, 2012) using their SPSS control files.

PISA 2009 and 2012 ship as fixed-width ASCII text. The SPSS control (.txt) file
contains a DATA LIST block defining each variable's name, column span, and type:

    DATA LIST FILE "..." /
    CNT 1 - 3 (A)
    ST04Q01 47 - 47 (F,0)
    ...

This module:
1. Parses the DATA LIST block → list of (name, start, end, is_string)
2. Reads only the requested columns via pandas read_fwf (memory-efficient)
3. Filters to requested countries (column CNT)
4. Optionally applies VALUE LABELS / MISSING VALUES from the control file

References:
    OECD PISA 2012 Technical Report, Annex (file layout).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import structlog
logger = structlog.get_logger(__name__)


@dataclass
class ColSpec:
    name: str
    start: int   # 1-based inclusive (SPSS convention)
    end: int     # 1-based inclusive
    is_string: bool

    @property
    def colspec_0based(self) -> tuple[int, int]:
        """Return (start, end) as 0-based half-open for pandas read_fwf."""
        return (self.start - 1, self.end)


# Matches lines like:  ST04Q01 47 - 47 (F,0)   or   CNT 1 - 3 (A)
_DATALIST_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+(\d+)\s*-\s*(\d+)\s*\(([^)]*)\)\s*$"
)


def parse_control_file(control_path: str | Path) -> list[ColSpec]:
    """
    Parse the DATA LIST block of a PISA SPSS control file.

    Returns a list of ColSpec for every variable defined.
    """
    control_path = Path(control_path)
    specs: list[ColSpec] = []
    in_datalist = False

    with open(control_path, "r", encoding="latin-1") as f:
        for line in f:
            stripped = line.strip()
            upper = stripped.upper()

            if upper.startswith("DATA LIST"):
                in_datalist = True
                continue

            if in_datalist:
                # End of DATA LIST block: a period on its own or a new command
                if stripped == "." or upper.startswith(
                    ("VARIABLE LABELS", "VALUE LABELS", "MISSING VALUES",
                     "FORMATS", "EXECUTE", "SAVE", "*")
                ):
                    if not _DATALIST_RE.match(stripped):
                        in_datalist = False
                        continue

                m = _DATALIST_RE.match(stripped)
                if m:
                    name, start, end, fmt = m.groups()
                    is_string = fmt.strip().upper().startswith("A")
                    specs.append(ColSpec(
                        name=name,
                        start=int(start),
                        end=int(end),
                        is_string=is_string,
                    ))

    logger.info("Control file parsed", path=str(control_path), n_vars=len(specs))
    if not specs:
        raise ValueError(f"No variables parsed from {control_path}. Check DATA LIST format.")
    return specs


def read_pisa_fwf(
    data_path: str | Path,
    control_path: str | Path,
    variables: list[str] | None = None,
    countries: list[str] | None = None,
    country_col: str = "CNT",
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """
    Read a PISA fixed-width ASCII file, keeping only requested variables/countries.

    Args:
        data_path: path to the .txt fixed-width data file (e.g. INT_STU12_DEC03.txt)
        control_path: path to the SPSS control file with the DATA LIST block
        variables: variable names to keep (None = all). country_col always kept.
        countries: list of CNT codes to keep (None = all)
        country_col: name of the country column (default CNT)
        chunksize: rows per chunk (controls memory)

    Returns:
        DataFrame with requested variables, filtered to requested countries.
    """
    all_specs = parse_control_file(control_path)
    spec_by_name = {s.name: s for s in all_specs}

    # Always include the country column
    if variables is not None:
        wanted = list(dict.fromkeys([country_col] + variables))
        missing = [v for v in wanted if v not in spec_by_name]
        if missing:
            logger.warning("Variables not in control file", missing=missing)
        wanted = [v for v in wanted if v in spec_by_name]
    else:
        wanted = [s.name for s in all_specs]

    keep_specs = [spec_by_name[v] for v in wanted]
    colspecs = [s.colspec_0based for s in keep_specs]
    names = [s.name for s in keep_specs]
    string_cols = {s.name for s in keep_specs if s.is_string}

    logger.info(
        "Reading PISA FWF",
        data=str(data_path),
        n_vars=len(names),
        countries=countries,
    )

    chunks: list[pd.DataFrame] = []
    reader = pd.read_fwf(
        data_path,
        colspecs=colspecs,
        names=names,
        dtype={c: str for c in string_cols},
        chunksize=chunksize,
        encoding="latin-1",
    )

    total_read = 0
    for chunk in reader:
        total_read += len(chunk)
        if countries:
            chunk[country_col] = chunk[country_col].astype(str).str.strip()
            chunk = chunk[chunk[country_col].isin(countries)]
        if not chunk.empty:
            chunks.append(chunk)

    result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=names)
    logger.info("FWF read complete", total_scanned=total_read, rows_kept=len(result))
    return result


def list_available_variables(control_path: str | Path) -> list[str]:
    """Return all variable names defined in a control file."""
    return [s.name for s in parse_control_file(control_path)]
