"""
Regenerate the paper's five model/analysis figures from the saved result CSVs.

The figure titles once contained an em-dash; the paper must contain none. Rather
than re-run the heavy model/SHAP scripts, this reads the canonical result CSVs in
``outputs/results/`` (the source of truth those scripts already wrote) and re-renders
the five figures the manuscript uses, with clean (colon) titles and the project's
publication style. Deterministic, no model fitting.

Run:  python scripts/replot_paper_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.visualization.style import (
    apply_publication_style, save_figure, PALETTE, SEQUENTIAL_RANK_CMAP,
)
from src.visualization.model_plots import plot_cv_distribution
from src.visualization.eda import plot_ses_logistic_curve
from src.features.target import add_all_targets

RES = ROOT / "outputs" / "results"
FIG = ROOT / "outputs" / "figures"
PAPER_FIG = ROOT / "reports" / "paper" / "figures"


def main() -> None:
    apply_publication_style()

    # G3  descriptive SES gradient as a probability curve (2018 vs 2022).
    # Written straight into the paper's figure dir; sourced from the Albania
    # longitudinal parquet, no model fitting beyond the descriptive logistic.
    long = add_all_targets(
        pd.read_parquet(ROOT / "data" / "processed" / "albania_longitudinal.parquet")
    )
    fig = plot_ses_logistic_curve(long, cycles=(2018, 2022))
    save_figure(fig, str(PAPER_FIG / "G3_ses_logistic_curve"))
    plt.close(fig)

    # C4  model comparison (reuse the library plot; title fixed at source)
    df = pd.read_csv(RES / "albania_2022_model_comparison_school.csv")
    fig = plot_cv_distribution(df, "roc_auc")
    save_figure(fig, str(FIG / "models" / "C4_model_comparison_auc"))
    plt.close(fig)

    # D2  global SHAP importance bar
    imp = pd.read_csv(RES / "shap_global_importance_2022.csv").sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"], imp["mean_abs_shap"], color="#0072B2")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP): Albania 2022 (LightGBM)")
    fig.tight_layout()
    save_figure(fig, str(FIG / "shap" / "D2_shap_global_bar"))
    plt.close(fig)

    # F1  SHAP rank matrix by country
    rank = pd.read_csv(RES / "comparative_shap_rank_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(rank))))
    im = ax.imshow(rank.values, cmap=SEQUENTIAL_RANK_CMAP, aspect="auto", vmin=1, vmax=15)
    ax.set_xticks(range(len(rank.columns))); ax.set_xticklabels(rank.columns)
    ax.set_yticks(range(len(rank.index))); ax.set_yticklabels(rank.index, fontsize=8)
    for i in range(len(rank.index)):
        for j in range(len(rank.columns)):
            v = int(rank.values[i, j])
            ax.text(j, i, v, ha="center", va="center", fontsize=7,
                    color="white" if v <= 5 else "black")
    ax.set_title("SHAP importance rank by country: 2022 (1 = top driver)")
    fig.colorbar(im, ax=ax, label="rank (1 = most important, dark)")
    fig.tight_layout()
    save_figure(fig, str(FIG / "comparative" / "F1_shap_rank_matrix"))
    plt.close(fig)

    # E1  fairness gaps vs. threshold (gender)
    sw = pd.read_csv(RES / "fairness_threshold_sweep_gender.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sw["threshold"], sw["parity_gap"], "-o", label="Demographic-parity gap")
    ax.plot(sw["threshold"], sw["opportunity_gap"], "-s", label="Equal-opportunity (TPR) gap")
    ax.plot(sw["threshold"], sw["fpr_gap"], "-^", label="FPR gap")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Weighted group gap (max minus min)")
    ax.set_title("Fairness gaps vs. threshold: Albania 2022, GENDER (weighted)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, str(FIG / "fairness" / "E1_fairness_threshold_sweep"))
    plt.close(fig)

    # G2  calibration reliability (raw vs isotonic); ECE recomputed from bins
    raw = pd.read_csv(RES / "calibration_raw_2022.csv")
    cal = pd.read_csv(RES / "calibration_isotonic_2022.csv")

    def ece(d: pd.DataFrame) -> float:
        return float((d["weight_share"] * (d["observed_fraction"] - d["mean_predicted"]).abs()).sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1, label="Perfect calibration")
    ax.plot(raw["mean_predicted"], raw["observed_fraction"], "-o", color=PALETTE["blue"],
            lw=2, label=f"Raw (ECE {ece(raw):.3f})")
    ax.plot(cal["mean_predicted"], cal["observed_fraction"], "-s", color=PALETTE["green"],
            lw=2, label=f"Isotonic (ECE {ece(cal):.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed at-risk fraction (weighted)")
    ax.set_title("Calibration: Albania 2022 (weighted, out-of-fold)")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save_figure(fig, str(FIG / "models" / "G2_calibration"))
    plt.close(fig)

    print("Regenerated C4, D2, F1, E1, G2 with clean titles.")


if __name__ == "__main__":
    main()
