"""
Phase 8 — comparative modeling: Albania vs. peers (PISA 2022).

For each of the nine countries we (1) fit the school-context LightGBM and score its
weighted 5-fold CV AUC, (2) compute global SHAP importance and assemble a
feature × country **rank matrix**, and (3) measure the **SES gradient** (weighted
slope of math score on ESCS — steeper = more socioeconomic determinism). Together
these answer: are Albania's risk drivers universal or idiosyncratic, and is its
SES gradient unusually steep?

All metrics survey-weighted. LightGBM imported before scikit-learn (OpenMP fix).
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from src.explainability.shap_analysis import country_shap_comparison
from src.features.transformers import EngineeredFeatureBuilder
from src.models.evaluate import fit_with_sample_weight
from src.models.experiment import _wrap
from src.models.prepare import build_model_data, impute_median
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.visualization.style import apply_publication_style, save_figure

FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
         "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
GROUP = {"ALB": "Albania", "MKD": "Balkan", "MNE": "Balkan", "SRB": "Balkan",
         "BGR": "Balkan", "EST": "Top performer", "FIN": "Top performer",
         "MEX": "GDP-matched", "COL": "GDP-matched"}
ORDER = ["ALB", "MKD", "MNE", "SRB", "BGR", "COL", "MEX", "EST", "FIN"]


def cv_auc(data, seed: int = 42) -> tuple[float, float]:
    X, y, w = data.X, data.y, data.weights
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    with threadpool_limits(limits=1):
        for tr, te in skf.split(X, y):
            pipe = _wrap(get_model("lightgbm"))
            fit_with_sample_weight(pipe, X.iloc[tr], y.iloc[tr], w.iloc[tr].values)
            p = pipe.predict_proba(X.iloc[te])[:, 1]
            if len(np.unique(y.iloc[te])) < 2:
                continue
            aucs.append(roc_auc_score(y.iloc[te], p, sample_weight=w.iloc[te].values))
    return float(np.mean(aucs)), float(np.std(aucs))


def weighted_ses_slope(df: pd.DataFrame) -> float:
    """Weighted OLS slope of MATH_PV_MEAN on ESCS (score points per 1 SD of ESCS)."""
    m = df[["ESCS", "MATH_PV_MEAN", "W_FSTUWT"]].dropna()
    x, ys, w = m["ESCS"].values, m["MATH_PV_MEAN"].values, m["W_FSTUWT"].values
    xb = np.average(x, weights=w); yb = np.average(ys, weights=w)
    cov = np.average((x - xb) * (ys - yb), weights=w)
    var = np.average((x - xb) ** 2, weights=w)
    return float(cov / var) if var > 0 else float("nan")


def main() -> None:
    configure_logging(level="WARNING")
    apply_publication_style()
    res_dir = ROOT / "outputs/results"
    fig_dir = ROOT / "outputs/figures/comparative"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(ROOT / "data/processed/comparison_dataset.parquet")
    d22 = full[full.CYCLE == 2022]

    models_by_country: dict[str, object] = {}
    X_by_country: dict[str, pd.DataFrame] = {}
    rows = []
    for cnt in ORDER:
        sub = d22[d22.COUNTRY == cnt].copy()
        data = build_model_data(sub, FEATS, domain="math", add_school_context=True)
        auc_m, auc_s = cv_auc(data)
        slope = weighted_ses_slope(sub)
        rows.append({"country": cnt, "group": GROUP[cnt], "n": int(data.meta["n_samples"]),
                     "at_risk_rate": data.meta["at_risk_rate"],
                     "cv_auc": round(auc_m, 4), "cv_auc_std": round(auc_s, 4),
                     "ses_gradient": round(slope, 2)})
        # final explanatory fit on all data (fit-on-all fine for SHAP)
        X_eng = EngineeredFeatureBuilder().fit_transform(data.X)
        (X,) = impute_median(X_eng)
        model = get_model("lightgbm"); model.fit(X, data.y.values, sample_weight=data.weights.values)
        models_by_country[cnt] = model
        X_by_country[cnt] = X
        print(f"  {cnt} ({GROUP[cnt]}): n={data.meta['n_samples']} "
              f"at_risk={data.meta['at_risk_rate']:.3f} AUC={auc_m:.3f} SESgrad={slope:.1f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(res_dir / "comparative_country_2022.csv", index=False)
    print("\n" + table.to_string(index=False))

    # ---- SHAP rank matrix (feature x country) --------------------------------
    rank = country_shap_comparison(models_by_country, X_by_country, top_n=8)
    rank = rank[[c for c in ORDER if c in rank.columns]]
    rank.to_csv(res_dir / "comparative_shap_rank_matrix.csv")
    print("\nSHAP rank matrix (1 = most important per country):")
    print(rank.to_string())

    # heatmap
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(rank))))
    im = ax.imshow(rank.values, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=15)
    ax.set_xticks(range(len(rank.columns))); ax.set_xticklabels(rank.columns)
    ax.set_yticks(range(len(rank.index))); ax.set_yticklabels(rank.index, fontsize=8)
    for i in range(len(rank.index)):
        for j in range(len(rank.columns)):
            ax.text(j, i, int(rank.values[i, j]), ha="center", va="center", fontsize=7)
    ax.set_title("SHAP importance rank by country — 2022 (1 = top driver)")
    fig.colorbar(im, ax=ax, label="rank")
    save_figure(fig, str(fig_dir / "F1_shap_rank_matrix"))
    plt.close(fig)

    print("\nSAVED OK")


if __name__ == "__main__":
    main()
