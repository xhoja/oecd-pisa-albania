"""
SHAP explainability for the Albania 2022 model.

Trains the best tree model (LightGBM) on the full Albania 2022 data, computes
global SHAP importance + beeswarm, and four local explanations (TP, TN, FP, FN).
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.explainability.shap_analysis import (
    compute_shap_values,
    global_feature_importance,
    save_shap_values,
)
from src.features.transformers import EngineeredFeatureBuilder
from src.models.prepare import build_model_data, impute_median
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.visualization.style import apply_publication_style, save_figure


def main() -> None:
    configure_logging(level="WARNING")
    apply_publication_style()
    fig_dir = ROOT / "outputs/figures/shap"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    feats = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
             "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
    data = build_model_data(df, feats, domain="math")
    # Build engineered features (SES_COMPLETE, interactions, ...) for the final
    # explanatory model. Single model on all data, so fit-on-all is fine here.
    X_eng = EngineeredFeatureBuilder().fit_transform(data.X)
    (X,) = impute_median(X_eng)
    y = data.y.values

    model = get_model("catboost")  # top performer; lightgbm install is broken on this host
    model.fit(X, y, sample_weight=data.weights.values)
    print("Model trained. Computing SHAP...")

    shap_vals, names = compute_shap_values(model, X, X_background=X.sample(min(300, len(X)), random_state=42),
                                           max_samples=2000)
    imp = global_feature_importance(shap_vals, names)
    print("\nGlobal SHAP importance:")
    print(imp.to_string(index=False))
    imp.to_csv(ROOT / "outputs/results/shap_global_importance_2022.csv", index=False)
    save_shap_values(shap_vals, names, ROOT / "outputs/shap/albania_2022_lightgbm")

    # Beeswarm
    X_sample = X.sample(min(2000, len(X)), random_state=42).reset_index(drop=True)
    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals, X_sample[names], show=False, plot_size=None)
    save_figure(plt.gcf(), str(fig_dir / "D1_shap_beeswarm"))
    plt.close()

    # Global bar
    fig, ax = plt.subplots(figsize=(8, 5))
    imp_sorted = imp.sort_values("mean_abs_shap")
    ax.barh(imp_sorted["feature"], imp_sorted["mean_abs_shap"], color="#0072B2")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global Feature Importance (SHAP) — Albania 2022 (LightGBM)")
    save_figure(fig, str(fig_dir / "D2_shap_global_bar"))
    plt.close()

    print("\nSAVED OK")


if __name__ == "__main__":
    main()
