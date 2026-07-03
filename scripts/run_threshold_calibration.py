"""
Model-improvement pass (pre-Phase-8): does post-hoc threshold tuning + isotonic
calibration lift the weak operating-point metrics (F1/MCC ~0.28 in the headline
table, scored at a fixed 0.5) without touching ROC-AUC?

Both fixes are leakage-safe (threshold + calibrator fit on out-of-fold TRAIN
predictions only) and survey-weighted, mirroring the rest of the pipeline. Writes
a before/after table and a bar figure. LightGBM is imported before scikit-learn
(macOS OpenMP import-order fix).
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import lightgbm  # noqa: F401  load libomp before sklearn (rc=-11 import-order fix)

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.models.experiment import threshold_calibration_cv
from src.models.prepare import build_model_data
from src.utils.logging import configure_logging
from src.visualization.style import apply_publication_style, save_figure
import pandas as pd

FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
         "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
# Strong, in-process-safe set (the boosters that matter + the interpretable baseline).
MODELS = ["logistic_regression", "gradient_boosting", "random_forest",
          "lightgbm", "catboost"]


def main() -> None:
    configure_logging(level="WARNING")
    apply_publication_style()
    res_dir = ROOT / "outputs/results"
    fig_dir = ROOT / "outputs/figures/models"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    data = build_model_data(df, FEATS, domain="math")
    print(f"Cohort n={len(data.y)} | at-risk rate={float(data.y.mean()):.3f}")
    print(f"Tuning threshold for MCC; models: {MODELS}\n")

    res = threshold_calibration_cv(data, MODELS, metric="mcc",
                                   outer_folds=5, outer_repeats=2, inner_folds=3)
    res = res.sort_values("mcc_cal_tuned_mean", ascending=False).reset_index(drop=True)

    cols = ["model", "roc_auc_mean", "mcc_at_0.5_mean", "mcc_tuned_mean",
            "mcc_cal_tuned_mean", "f1_at_0.5_mean", "f1_cal_tuned_mean",
            "brier_raw_mean", "brier_cal_mean", "ece_raw_mean", "ece_cal_mean",
            "threshold_mean"]
    out = res[cols]
    out.to_csv(res_dir / "threshold_calibration_2022.csv", index=False)
    print("\n" + out.to_string(index=False))

    # ---- Figure: MCC at 0.5 vs tuned vs calibrated+tuned ---------------------
    x = np.arange(len(res))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, res["mcc_at_0.5_mean"], width, label="MCC @ 0.5", color="#999999")
    ax.bar(x, res["mcc_tuned_mean"], width, label="MCC threshold-tuned", color="#0072B2")
    ax.bar(x + width, res["mcc_cal_tuned_mean"], width, label="MCC calibrated + tuned", color="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels(res["model"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Weighted MCC (mean over CV folds)")
    ax.set_title("Operating-point fix: threshold tuning + isotonic calibration\nAlbania 2022 (weighted, leakage-safe)")
    ax.legend(fontsize=8)
    save_figure(fig, str(fig_dir / "M1_threshold_calibration_mcc"))
    plt.close(fig)

    # Headline deltas for the console.
    best = res.iloc[0]
    print(f"\nBest (cal+tuned MCC): {best['model']} "
          f"MCC {best['mcc_at_0.5_mean']:.3f} -> {best['mcc_cal_tuned_mean']:.3f} "
          f"(+{best['mcc_cal_tuned_mean'] - best['mcc_at_0.5_mean']:.3f})")
    print(f"AUC unchanged by construction (ranking metric): {best['roc_auc_mean']:.3f}")
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
