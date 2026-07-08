"""
Behavioural *process* features from PISA computer-based-assessment log-derived
files (target #1: can a genuinely new data modality move the ~0.78 ceiling?).

Two public files carry per-item process data, both keyed by ``CNTSTUID`` and
joinable 1:1 to the student questionnaire:

* **STU_COG** (2022) - per cognitive item: total time ``...TT`` (ms), number of
  visits ``...V``, and a scored code ``...S`` (0/1 credit, plus 6=Not Reached,
  9=No Response).
* **STU_TTM** (2018) - per cognitive item: total time ``...TT`` and visits
  ``...V`` (no scored code in this file).
* **STU_TIM** (2022) - per *questionnaire* screen: total time ``..._TT`` (ms).

Leakage discipline
------------------
Item *correctness* (the 0/1 credit in ``...S``) mechanically determines the
plausible-value proficiency that defines the target, so it is **never** used as
a feature. The headline process features are purely *behavioural* - how long a
student spent and how they navigated, not whether they were right: timing,
visits, revisits and rapid-response fraction. ``not_reached`` / ``no_response``
counts are derived from the same test and are partly mechanical, so they are
computed separately and flagged, for use only as a labelled sensitivity.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Cognitive item columns: <C|P><M|R|S><digits>Q<digits><suffix>
# e.g. CM033Q01TT (time), CM033Q01V (visits), CM033Q01S (scored code).
COG_ITEM_RE = re.compile(r"^([CP][MRS]\d+Q\d+[A-Z]?)(TT|V|S)$")
# Questionnaire screen timing: <name>_TT  (e.g. ST001_TT, EFFORTT is special)
QTIM_RE = re.compile(r"^(.+)_TT$")

RAPID_MS = 5_000        # < 5 s on a cognitive item ~ rapid guess / disengaged
NOT_REACHED_CODE = 6
NO_RESPONSE_CODE = 9


def _country_col(cols: list[str]) -> str:
    return "CNT" if "CNT" in cols else "CNTRYID"


def load_cog_process(
    path: str | Path,
    country: str = "ALB",
    country_id: int = 8,
) -> pd.DataFrame:
    """Load the per-item time/visit/score columns for one country from a
    STU_COG or STU_TTM SAV file. Returns a student-by-item-metric wide frame
    with ``CNTSTUID`` plus every ``...TT`` / ``...V`` / ``...S`` column present.
    """
    import pyreadstat

    _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
    cols = meta.column_names
    item_cols = [c for c in cols if COG_ITEM_RE.match(c)]
    ccol = _country_col(cols)
    keep = [ccol, "CNTSTUID"] + item_cols
    df, _ = pyreadstat.read_sav(str(path), usecols=[c for c in keep if c in cols])
    mask = df[ccol] == (country if ccol == "CNT" else country_id)
    out = df.loc[mask].reset_index(drop=True)
    logger.info("Loaded cognitive process file", path=str(path),
                country=country, n=len(out), n_item_cols=len(item_cols))
    return out


def aggregate_cog_process(
    df: pd.DataFrame,
    rapid_ms: int = RAPID_MS,
    include_disengagement: bool = True,
) -> pd.DataFrame:
    """Aggregate per-item time/visit/score columns to one row per student.

    Behavioural features (leakage-safe, do not encode correctness):
        PROC_N_ITEMS         number of cognitive items the student saw (timed)
        PROC_TOTAL_TIME_S    total time on cognitive items (seconds)
        PROC_MEAN_TIME_S     mean time per item (seconds)
        PROC_MEDIAN_TIME_S   median time per item
        PROC_SD_TIME_S       SD of item time
        PROC_CV_TIME         coefficient of variation (SD/mean) - erratic pacing
        PROC_TOTAL_VISITS    total visit count across items
        PROC_MEAN_VISITS     mean visits per item
        PROC_REVISIT_RATE    fraction of items visited more than once
        PROC_RAPID_FRAC      fraction of items answered in < rapid_ms (guessing)

    Disengagement features (partly mechanical - flagged, sensitivity only):
        PROC_NOT_REACHED     count of items coded Not Reached (6)
        PROC_NO_RESPONSE     count of items coded No Response (9)
        PROC_FRAC_NOT_REACHED
    """
    tt_cols = [c for c in df.columns if COG_ITEM_RE.match(c) and c.endswith("TT")]
    v_cols = [c for c in df.columns if COG_ITEM_RE.match(c) and c.endswith("V")]
    s_cols = [c for c in df.columns if COG_ITEM_RE.match(c) and c.endswith("S")]
    if not tt_cols:
        raise ValueError("no ...TT timing columns found")

    tt = df[tt_cols].apply(pd.to_numeric, errors="coerce")
    seen = tt.notna()                                   # items the student saw
    n_seen = seen.sum(axis=1).replace(0, np.nan)        # guard 0/0 for non-tested
    tt_s = tt / 1000.0                                  # ms -> seconds

    out = pd.DataFrame({"CNTSTUID": df["CNTSTUID"].values})
    out["PROC_N_ITEMS"] = seen.sum(axis=1).values
    out["PROC_TOTAL_TIME_S"] = tt_s.sum(axis=1, min_count=1).values
    out["PROC_MEAN_TIME_S"] = tt_s.mean(axis=1).values
    out["PROC_MEDIAN_TIME_S"] = tt_s.median(axis=1).values
    out["PROC_SD_TIME_S"] = tt_s.std(axis=1).values
    out["PROC_CV_TIME"] = (out["PROC_SD_TIME_S"] / out["PROC_MEAN_TIME_S"]).replace(
        [np.inf, -np.inf], np.nan)
    out["PROC_RAPID_FRAC"] = (
        (tt < rapid_ms).where(seen).sum(axis=1) / n_seen
    ).values

    if v_cols:
        v = df[v_cols].apply(pd.to_numeric, errors="coerce")
        n_v = v.notna().sum(axis=1).replace(0, np.nan)
        out["PROC_TOTAL_VISITS"] = v.sum(axis=1, min_count=1).values
        out["PROC_MEAN_VISITS"] = v.mean(axis=1).values
        out["PROC_REVISIT_RATE"] = (
            (v > 1).where(v.notna()).sum(axis=1) / n_v
        ).values

    if include_disengagement and s_cols:
        s = df[s_cols].apply(pd.to_numeric, errors="coerce")
        answerable = s.notna()
        out["PROC_NOT_REACHED"] = (s == NOT_REACHED_CODE).sum(axis=1).values
        out["PROC_NO_RESPONSE"] = (s == NO_RESPONSE_CODE).sum(axis=1).values
        out["PROC_FRAC_NOT_REACHED"] = (
            (s == NOT_REACHED_CODE).sum(axis=1) / answerable.sum(axis=1).replace(0, np.nan)
        ).values

    feat_cols = [c for c in out.columns if c != "CNTSTUID"]
    out[feat_cols] = out[feat_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    logger.info("Aggregated cognitive process features",
                n_students=len(out), n_timed_items=len(tt_cols),
                n_features=out.shape[1] - 1)
    return out


# Behavioural columns safe for the headline ablation (exclude disengagement).
BEHAVIOURAL_COLS = [
    "PROC_N_ITEMS", "PROC_TOTAL_TIME_S", "PROC_MEAN_TIME_S", "PROC_MEDIAN_TIME_S",
    "PROC_SD_TIME_S", "PROC_CV_TIME", "PROC_RAPID_FRAC",
    "PROC_TOTAL_VISITS", "PROC_MEAN_VISITS", "PROC_REVISIT_RATE",
]
DISENGAGEMENT_COLS = ["PROC_NOT_REACHED", "PROC_NO_RESPONSE", "PROC_FRAC_NOT_REACHED"]


def load_questionnaire_timing(
    path: str | Path,
    country: str = "ALB",
    country_id: int = 8,
) -> pd.DataFrame:
    """Aggregate per-screen questionnaire response times (STU_TIM) to one row
    per student: total, mean, median, SD and rapid-response fraction. This is
    survey-taking behaviour (effort / careless responding), distinct from the
    cognitive-test behaviour above.
    """
    import pyreadstat

    _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
    cols = meta.column_names
    tt_cols = [c for c in cols if c.endswith("_TT") or c == "EFFORTT"]
    ccol = _country_col(cols)
    df, _ = pyreadstat.read_sav(str(path), usecols=[ccol, "CNTSTUID"] + tt_cols)
    df = df[df[ccol] == (country if ccol == "CNT" else country_id)].reset_index(drop=True)

    tt = df[tt_cols].apply(pd.to_numeric, errors="coerce") / 1000.0
    seen = tt.notna()
    n_seen = seen.sum(axis=1).replace(0, np.nan)        # guard 0/0
    out = pd.DataFrame({"CNTSTUID": df["CNTSTUID"].values})
    out["QTIM_TOTAL_S"] = tt.sum(axis=1, min_count=1).values
    out["QTIM_MEAN_S"] = tt.mean(axis=1).values
    out["QTIM_MEDIAN_S"] = tt.median(axis=1).values
    out["QTIM_SD_S"] = tt.std(axis=1).values
    out["QTIM_RAPID_FRAC"] = (
        (tt < 2.0).where(seen).sum(axis=1) / n_seen
    ).values
    feat_cols = [c for c in out.columns if c != "CNTSTUID"]
    out[feat_cols] = out[feat_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    logger.info("Aggregated questionnaire timing", n_students=len(out),
                n_screens=len(tt_cols))
    return out


QTIM_COLS = ["QTIM_TOTAL_S", "QTIM_MEAN_S", "QTIM_MEDIAN_S", "QTIM_SD_S", "QTIM_RAPID_FRAC"]


def link_process_features(
    student_df: pd.DataFrame,
    *process_frames: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join one or more per-student process frames onto the student data by
    ``CNTSTUID`` (unique within a cycle). Returns the student frame with the
    process columns appended.
    """
    out = student_df.copy()
    for pf in process_frames:
        out = out.merge(pf, on="CNTSTUID", how="left")
    return out
