"""
Phase 6 explainability: local SHAP case studies + PDP/ICE (Albania 2022).

Complements ``run_shap_analysis.py`` (global importance/beeswarm) with the
*local* half of the story:

  1. One representative TP / TN / FP / FN instance, each with a SHAP waterfall —
     why the model was confidently right, and confidently wrong.
  2. PDP + centered-ICE curves for the top drivers, showing the shape (and
     instance-level heterogeneity) of each feature's effect on at-risk risk.

The explanatory model is a single LightGBM fit on all Albania-2022 data (fit on
all is fine for explanation, not evaluation). LightGBM is imported before
scikit-learn so its Homebrew libomp loads first (macOS import-order fix).
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
import pandas as pd

from src.explainability.partial_dependence import centered_ice, compute_pdp
from src.explainability.shap_analysis import (
    compute_local_shap,
    compute_shap_values,
    global_feature_importance,
    select_representative_cases,
)
from src.features.transformers import EngineeredFeatureBuilder
from src.models.prepare import build_model_data, impute_median
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.visualization.style import apply_publication_style, color_list, save_figure

FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
         "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]


def _waterfall(ax: plt.Axes, values: np.ndarray, base: float, pred: float,
               feature_names: list[str], data_row: np.ndarray, top_n: int = 10) -> None:
    """Minimal, dependency-light SHAP waterfall: the top_n features by |SHAP|,
    signed bars from the base value toward the prediction."""
    order = np.argsort(np.abs(values))[::-1][:top_n]
    contrib = values[order]
    labels = [f"{feature_names[i]} = {data_row[i]:.2f}" for i in order]
    colors = ["#D55E00" if c > 0 else "#0072B2" for c in contrib]  # red up-risk, blue down
    y = np.arange(len(contrib))[::-1]
    ax.barh(y, contrib, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("SHAP contribution (log-odds)")
    ax.set_title(f"base={base:.2f}  →  P(at-risk)={pred:.2f}", fontsize=9)


def main() -> None:
    configure_logging(level="WARNING")
    apply_publication_style()
    fig_dir = ROOT / "outputs/figures/shap"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir = ROOT / "outputs/results"
    res_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    data = build_model_data(df, FEATS, domain="math", add_school_context=True)
    X_eng = EngineeredFeatureBuilder().fit_transform(data.X)
    (X,) = impute_median(X_eng)
    y = data.y.values

    model = get_model("lightgbm")
    model.fit(X, y, sample_weight=data.weights.values)
    print(f"Model trained on {X.shape}. at-risk rate={y.mean():.3f}")

    # ---- 1. Local SHAP case studies (TP/TN/FP/FN) -------------------------
    cases = select_representative_cases(model, X, y)
    print("\nRepresentative cases:")
    rows = []
    for k, c in cases.items():
        print(f"  {k} ({c['label']}): P={c['prob']:.3f} y_true={c['y_true']} y_pred={c['y_pred']}")
        rows.append({"quadrant": k, **{kk: vv for kk, vv in c.items() if kk != "index"}})
    pd.DataFrame(rows).to_csv(res_dir / "shap_local_cases_2022.csv", index=False)

    idx = [c["index"] for c in cases.values()]
    X_bg = X.sample(min(300, len(X)), random_state=42)
    vals, base, names = compute_local_shap(model, X.iloc[idx], X_background=X_bg)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (k, c), v, b in zip(axes.ravel(), cases.items(), vals, base):
        _waterfall(ax, v, b, c["prob"], names, X.iloc[c["index"]].values)
        ax.set_ylabel(f"{k} — {c['label']}", fontsize=9)
    fig.suptitle("Local SHAP explanations — Albania 2022 (LightGBM)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, str(fig_dir / "D3_shap_local_cases"))
    plt.close(fig)

    # ---- 2. PDP + centered ICE for the top drivers ------------------------
    shap_vals, shap_names = compute_shap_values(model, X, X_background=X_bg, max_samples=2000)
    imp = global_feature_importance(shap_vals, shap_names)
    # top raw (non-interaction, non-missing) features so the axes are interpretable
    raw_top = [f for f in imp["feature"] if f in FEATS][:4]
    print(f"\nPDP/ICE for: {raw_top}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, feat in zip(axes.ravel(), raw_top):
        pdp = compute_pdp(model, X, feat, grid_resolution=30, ice=True)
        cice = centered_ice(pdp)
        # a subsample of centered-ICE lines (grey) + PDP mean (bold)
        rng = np.random.default_rng(42)
        sub = rng.choice(cice.shape[0], size=min(200, cice.shape[0]), replace=False)
        for row in cice[sub]:
            ax.plot(pdp.grid, row, color="0.7", lw=0.4, alpha=0.5)
        ax.plot(pdp.grid, pdp.average - pdp.average[0], color="#D55E00", lw=2.2, label="PDP (mean)")
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel(feat)
        ax.set_ylabel("Δ P(at-risk) vs. grid start")
        ax.legend(fontsize=8)
    fig.suptitle("Partial dependence + centered ICE — Albania 2022 (LightGBM)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, str(fig_dir / "D4_pdp_ice"))
    plt.close(fig)

    print("\nSAVED OK")


if __name__ == "__main__":
    main()
