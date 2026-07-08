"""
Build the processed 2022 school-questionnaire parquet (one row per CNTSCHID).

Source of truth for `data/processed/alb_2022_school.parquet`, consumed by the
questionnaire ablation (`run_school_questionnaire_experiment.py`) and notebook 06 (Part B).
Independent principal-reported school signal (resources, staff, leadership,
climate) - distinct from the student-composition `SCH_MEAN_*` aggregates.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.data.extract import load_school_questionnaire, save_processed
from src.utils.logging import configure_logging

FOCUS = "ALB"


def _school_vars(features_cfg: dict) -> list[str]:
    groups = features_cfg["school_questionnaire"]
    return [v for group in groups.values() for v in group]


def main() -> None:
    configure_logging(level="INFO")
    data_cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    feat_cfg = yaml.safe_load((ROOT / "configs/features.yaml").read_text())

    sch_path = (ROOT / data_cfg["paths"]["raw_2022_sch"]).resolve()
    keep = _school_vars(feat_cfg)

    sch = load_school_questionnaire(sch_path, countries=[FOCUS], keep_cols=keep)
    out = ROOT / "data/processed/alb_2022_school.parquet"
    save_processed(sch, out)

    miss = (sch.drop(columns=["CNTSCHID"]).isna().mean() * 100).round(1)
    print(f"\nSaved {len(sch)} schools x {sch.shape[1]-1} vars -> {out}")
    print("Missingness (%):")
    print(miss.to_string())
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
