"""
Target #4 - fairness *mitigation*, not just diagnosis (Albania 2022).

The audit (notebook 07) showed the screener over-flags the poorest students:
at a single 0.5 threshold the bottom SES quintile has a much higher false-
positive rate than the top. Here we *fix* that post-hoc - no retraining - with
group-specific decision thresholds that equalise the weighted FPR across SES
quintiles, and trace the fairness-utility trade-off against a single-threshold
baseline matched to the same overall selection rate.

Predictions are leakage-safe out-of-fold (StratifiedKFold, engineered features
per fold, survey-weighted), the school-context LightGBM - the same recipe as the
audit. All rates are weighted.

Outputs:
    outputs/results/fairness_mitigation_summary.json   baseline vs mitigated
    outputs/results/fairness_mitigation_frontier_ses.csv
    outputs/figures/fairness/E2_mitigation_tradeoff.png
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
from src.fairness.mitigation import (
    apply_group_thresholds,
    evaluate_assignment,
    fairness_utility_frontier,
    group_thresholds_equal_fpr,
)
from src.models.evaluate import fit_with_sample_weight
from src.models.experiment import _wrap
from src.models.prepare import build_model_data
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.visualization.style import PALETTE, apply_publication_style, save_figure

FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
         "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
GROUP = "SES_QUINTILE"


def oof_predictions(X, y, w, n_splits=5, seed=42):
    prob = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with threadpool_limits(limits=1):
        for tr, te in skf.split(X, y):
            pipe = _wrap(get_model("lightgbm"))
            fit_with_sample_weight(pipe, X.iloc[tr], y.iloc[tr], w.iloc[tr].values)
            prob[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return prob


def main() -> None:
    configure_logging(level="WARNING")
    apply_publication_style()
    res = ROOT / "outputs/results"; fig = ROOT / "outputs/figures/fairness"
    res.mkdir(parents=True, exist_ok=True); fig.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    data = build_model_data(df, FEATS, domain="math", add_school_context=True)
    X, y, w = data.X, data.y, data.weights
    prob = oof_predictions(X, y, w)
    print(f"OOF: {np.isfinite(prob).sum()}/{len(prob)} covered", flush=True)

    audit = pd.DataFrame({
        "y_true": y.values.astype(float), "y_prob": prob,
        "W_FSTUWT": w.values, "ESCS": X["ESCS"].values,
    })
    audit = compute_ses_quintiles(audit, ses_col="ESCS", weight_col="W_FSTUWT")
    audit["SES_QUINTILE"] = audit["SES_QUINTILE"].astype("float")
    audit = audit.dropna(subset=["y_prob", GROUP])

    # ---- baseline: single 0.5 threshold ----
    base_pred = (audit["y_prob"].values >= 0.5).astype(float)
    base = evaluate_assignment(audit, "y_true", base_pred, GROUP, "W_FSTUWT")

    # ---- mitigation: equalise FPR across quintiles at the overall baseline FPR ----
    target = round(float(base["overall_fpr"]), 3)
    thr = group_thresholds_equal_fpr(audit, "y_true", "y_prob", GROUP, "W_FSTUWT", target)
    mit_pred = apply_group_thresholds(audit, "y_prob", GROUP, thr)
    mit = evaluate_assignment(audit, "y_true", mit_pred, GROUP, "W_FSTUWT")

    summary = {
        "attribute": GROUP, "n": int(len(audit)),
        "target_fpr": target,
        "group_thresholds": {str(k): round(float(v), 3) for k, v in thr.items()},
        "baseline": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in base.items() if k != "by_group"},
        "mitigated": {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in mit.items() if k != "by_group"},
        "baseline_fpr_by_group": {k: round(v["fpr"], 4) for k, v in base["by_group"].items()},
        "mitigated_fpr_by_group": {k: round(v["fpr"], 4) for k, v in mit["by_group"].items()},
    }
    (res / "fairness_mitigation_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nBaseline  : FPR gap {base['fpr_gap']:.3f}  recall {base['overall_recall']:.3f}  "
          f"acc {base['overall_accuracy']:.3f}")
    print(f"Mitigated : FPR gap {mit['fpr_gap']:.3f}  recall {mit['overall_recall']:.3f}  "
          f"acc {mit['overall_accuracy']:.3f}  (target FPR {target})")
    print("FPR by quintile  baseline ->", summary["baseline_fpr_by_group"])
    print("                 mitigated->", summary["mitigated_fpr_by_group"])

    # ---- frontier ----
    frontier = fairness_utility_frontier(audit, "y_true", "y_prob", GROUP, "W_FSTUWT")
    frontier.to_csv(res / "fairness_mitigation_frontier_ses.csv", index=False)

    fig1, ax = plt.subplots(figsize=(7, 5))
    ax.plot(frontier.grp_recall, frontier.grp_fpr_gap, "-o", color=PALETTE["blue"],
            label="group-specific thresholds")
    ax.plot(frontier.single_recall, frontier.single_fpr_gap, "-s", color=PALETTE["vermilion"],
            label="single threshold (matched selection)")
    ax.set_xlabel("overall weighted recall (at-risk caught)")
    ax.set_ylabel("FPR gap across SES quintiles (max - min)")
    ax.set_title("Fairness-utility trade-off: mitigating SES over-flagging (2022)")
    ax.legend(fontsize=8)
    save_figure(fig1, str(fig / "E2_mitigation_tradeoff"))
    plt.close(fig1)
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
