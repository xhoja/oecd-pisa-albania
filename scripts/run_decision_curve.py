"""
Decision-curve analysis + calibration for the school-context screener (Albania 2022).

Reframes evaluation for a ~75%-prevalence triage tool, where ROC-AUC is capped and
uninformative: does acting on the model beat "screen everyone" / "screen no-one" at
a realistic risk threshold, and are the probabilities calibrated?

Protocol (leakage-safe, weighted):
- weighted 5-fold out-of-fold probabilities from the headline LightGBM school-context
  model (the same model SHAP/fairness use);
- isotonic recalibration fit **out-of-fold** on those OOF probabilities (a second CV
  loop), so the calibrated curve is honest, not fit-on-self;
- net benefit (Vickers & Elkin) for raw + calibrated vs the two reference policies,
  and weighted reliability curves + Brier/ECE.

LightGBM is imported before scikit-learn (macOS OpenMP import-order fix).
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
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from src.features.engineer import add_school_aggregates
from src.models.decision_curve import calibration_summary, net_benefit
from src.models.evaluate import fit_with_sample_weight
from src.models.experiment import _wrap
from src.models.prepare import build_model_data
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.visualization.style import PALETTE, apply_publication_style, save_figure

BASE_FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
              "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
SCHOOL_FEATS = ["SCH_MEAN_ESCS", "SCH_MEAN_HOMEPOS", "SCH_MEAN_ANXMAT",
                "SCH_MEAN_TEACHSUP", "SCH_N"]


def oof_raw(X, y, w, n_splits=5, seed=42):
    """Leakage-safe weighted out-of-fold probabilities (LightGBM school-context)."""
    prob = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with threadpool_limits(limits=1):
        for tr, te in skf.split(X, y):
            pipe = _wrap(get_model("lightgbm"))
            fit_with_sample_weight(pipe, X.iloc[tr], y.iloc[tr], w.iloc[tr].values)
            prob[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return prob


def oof_isotonic(prob, y, w, n_splits=5, seed=42):
    """Recalibrate the OOF probabilities with weighted isotonic, itself fit
    out-of-fold so the calibration is not evaluated on its own training data."""
    cal = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(prob.reshape(-1, 1), y):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(prob[tr], y.iloc[tr].values, sample_weight=w.iloc[tr].values)
        cal[te] = np.asarray(iso.transform(prob[te])).ravel()
    return cal


def main() -> None:
    configure_logging(level="WARNING")
    res_dir = ROOT / "outputs/results"
    fig_dir = ROOT / "outputs/figures/models"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    df = add_school_aggregates(df, cols=["ESCS", "HOMEPOS", "ANXMAT", "TEACHSUP"])
    data = build_model_data(df, BASE_FEATS + SCHOOL_FEATS, domain="math")
    X, y, w = data.X, data.y, data.weights
    print(f"cohort n = {len(y)} | at-risk rate = {float(y.mean()):.3f}")

    prob = oof_raw(X, y, w)
    prob_cal = oof_isotonic(prob, y, w)
    wv = w.values

    # ---- decision-curve analysis --------------------------------------------
    dca = net_benefit(y.values, prob, sample_weight=wv)
    dca_cal = net_benefit(y.values, prob_cal, sample_weight=wv)
    dca["net_benefit_model_calibrated"] = dca_cal["net_benefit_model"]
    dca.to_csv(res_dir / "decision_curve_2022.csv", index=False)

    # ---- calibration --------------------------------------------------------
    raw = calibration_summary(y.values, prob, sample_weight=wv)
    cal = calibration_summary(y.values, prob_cal, sample_weight=wv)
    raw["reliability"].to_csv(res_dir / "calibration_raw_2022.csv", index=False)
    cal["reliability"].to_csv(res_dir / "calibration_isotonic_2022.csv", index=False)
    print(f"\nCalibration (weighted):")
    print(f"  raw       Brier={raw['brier']:.4f}  ECE={raw['ece']:.4f}")
    print(f"  isotonic  Brier={cal['brier']:.4f}  ECE={cal['ece']:.4f}")

    # where does the model beat both reference policies?
    better = dca[(dca.net_benefit_model > dca.net_benefit_all) &
                 (dca.net_benefit_model > dca.net_benefit_none)]
    if len(better):
        print(f"\nModel beats treat-all AND treat-none over threshold "
              f"{better.threshold.min():.2f}–{better.threshold.max():.2f} "
              f"(prevalence {dca.attrs['prevalence']:.3f}).")
    else:
        print("\nModel never beats both reference policies — no decision value.")

    # ---- figures ------------------------------------------------------------
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dca.threshold, dca.net_benefit_model, color=PALETTE["blue"], lw=2, label="Model (raw)")
    ax.plot(dca.threshold, dca.net_benefit_model_calibrated, color=PALETTE["green"],
            lw=2, ls="-", label="Model (isotonic)")
    ax.plot(dca.threshold, dca.net_benefit_all, color=PALETTE["vermilion"], lw=1.5,
            ls="--", label="Screen everyone")
    ax.plot(dca.threshold, dca.net_benefit_none, color="0.4", lw=1.2, ls=":", label="Screen no-one")
    ax.set_ylim(bottom=min(-0.05, dca.net_benefit_model.min()))
    ax.set_xlabel("Threshold probability $p_t$ (risk tolerance)")
    ax.set_ylabel("Net benefit (weighted)")
    ax.set_title("Decision-curve analysis: Albania 2022 low-proficiency screener")
    ax.legend(fontsize=9)
    save_figure(fig, str(fig_dir / "G1_decision_curve"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1, label="Perfect calibration")
    rr, rc = raw["reliability"], cal["reliability"]
    ax.plot(rr.mean_predicted, rr.observed_fraction, "-o", color=PALETTE["blue"],
            lw=2, label=f"Raw (ECE {raw['ece']:.3f})")
    ax.plot(rc.mean_predicted, rc.observed_fraction, "-s", color=PALETTE["green"],
            lw=2, label=f"Isotonic (ECE {cal['ece']:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed at-risk fraction (weighted)")
    ax.set_title("Calibration: Albania 2022 (weighted, out-of-fold)")
    ax.legend(fontsize=9, loc="upper left")
    save_figure(fig, str(fig_dir / "G2_calibration"))
    plt.close(fig)

    print("\nSAVED OK")


if __name__ == "__main__":
    main()
