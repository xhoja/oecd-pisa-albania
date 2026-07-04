"""
Phase 7 fairness audit (Albania 2022, survey-weighted).

Produces out-of-fold predictions with a leakage-safe pipeline (engineered
features fit per fold), then audits demographic parity, equal opportunity (TPR),
FPR and calibration across protected attributes — GENDER, SES quintile (derived
from weighted ESCS), and immigrant status. Every metric is **survey-weighted**
(`W_FSTUWT`), consistent with the rest of the project; an unweighted audit would
misstate population-level gaps exactly like the descriptive stats it criticises.

LightGBM is imported before scikit-learn (macOS OpenMP import-order fix).
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import lightgbm  # noqa: F401  load libomp before sklearn (rc=-11 import-order fix)

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from src.data.weights import compute_ses_quintiles
from src.fairness.metrics import fairness_audit, threshold_sweep
from src.models.evaluate import fit_with_sample_weight
from src.models.experiment import _wrap
from src.models.prepare import build_model_data
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.visualization.style import apply_publication_style, save_figure

FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
         "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
MODEL = "lightgbm"


def oof_predictions(X, y, w, n_splits=5, seed=42):
    """Weighted, leakage-safe out-of-fold probabilities for the whole cohort."""
    prob = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with threadpool_limits(limits=1):
        for tr, te in skf.split(X, y):
            pipe = _wrap(get_model(MODEL))
            fit_with_sample_weight(pipe, X.iloc[tr], y.iloc[tr], w.iloc[tr].values)
            prob[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return prob


def main() -> None:
    configure_logging(level="WARNING")
    apply_publication_style()
    res_dir = ROOT / "outputs/results"
    fig_dir = ROOT / "outputs/figures/fairness"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    data = build_model_data(df, FEATS, domain="math", add_school_context=True)
    X, y, w = data.X, data.y, data.weights

    prob = oof_predictions(X, y, w)
    print(f"OOF predictions: {np.isfinite(prob).sum()}/{len(prob)} covered")

    # Assemble audit frame with protected attributes aligned to X's rows.
    audit = pd.DataFrame({
        "y_true": y.values.astype(float),
        "y_prob": prob,
        "y_pred": (prob >= 0.5).astype(float),
        "W_FSTUWT": w.values,
        "GENDER": X["GENDER"].values,
        "IMMIG": X["IMMIG"].values,
        "ESCS": X["ESCS"].values,
    })
    # SES quintile from weighted ESCS breaks (population quintiles, not sample).
    audit = compute_ses_quintiles(audit, ses_col="ESCS", weight_col="W_FSTUWT")
    audit["SES_QUINTILE"] = audit["SES_QUINTILE"].astype("float")  # groupable + NaN-skippable
    audit = audit.dropna(subset=["y_prob"])

    attrs = ["GENDER", "SES_QUINTILE", "IMMIG"]
    reports = fairness_audit(audit, "y_true", "y_pred", "y_prob", attrs, weight_col="W_FSTUWT")

    payload = {"model": MODEL, "n": int(len(audit)), "threshold": 0.5, "weighted": True,
               "attributes": {}}
    for attr, rep in reports.items():
        print("\n" + rep.summary())
        payload["attributes"][attr] = {
            "groups": rep.groups,
            "demographic_parity": _clean(rep.demographic_parity),
            "equal_opportunity_tpr": _clean(rep.equal_opportunity),
            "fpr_by_group": _clean(rep.fpr_by_group),
            "calibration_mean_prob": _clean(rep.calibration_by_group),
            "parity_gap": _nan(rep.parity_gap),
            "opportunity_gap": _nan(rep.opportunity_gap),
            "fpr_gap": _nan(rep.fpr_gap),
        }
    # allow_nan=False guards against silently writing non-standard NaN tokens if a
    # sparse group ever slips a NaN through the cleaners above.
    (res_dir / "fairness_audit_2022.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False)
    )

    # Threshold sweep on gender: does the operating point trade off parity vs TPR gap?
    sweep = threshold_sweep(audit, "y_true", "y_prob", "GENDER", weight_col="W_FSTUWT")
    sweep.to_csv(res_dir / "fairness_threshold_sweep_gender.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sweep["threshold"], sweep["parity_gap"], "-o", label="Demographic-parity gap")
    ax.plot(sweep["threshold"], sweep["opportunity_gap"], "-s", label="Equal-opportunity (TPR) gap")
    ax.plot(sweep["threshold"], sweep["fpr_gap"], "-^", label="FPR gap")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Weighted group gap (max − min)")
    ax.set_title("Fairness gaps vs. threshold: Albania 2022, GENDER (weighted)")
    ax.legend(fontsize=8)
    save_figure(fig, str(fig_dir / "E1_fairness_threshold_sweep"))
    plt.close(fig)

    print("\nSAVED OK")


def _nan(x: float):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def _clean(d: dict) -> dict:
    """Round group metrics and turn NaN (single-class groups) into JSON null."""
    return {k: _nan(v) for k, v in d.items()}


if __name__ == "__main__":
    main()
