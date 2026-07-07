"""
Target #3 - does the "0.78 is a data ceiling, not a feature ceiling" claim
generalise beyond Albania?

Albania's evidence (notebooks 11-12) was three-legged: (1) school socioeconomic
*composition* features give a large CV-AUC lift and then the accuracy plateaus,
(2) a big share of outcome variance is *between schools* (multilevel ICC), and
(3) the independent school questionnaire adds nothing beyond composition. Legs
(1) and (2) use only data every country has, so we run them on all nine
countries; leg (3) needs the linked school questionnaire (Albania-only) and
stays a single-country result.

For each country (PISA 2022, survey-weighted) we compute:
    * base CV AUC          - student background features only
    * school-context CV AUC - plus leave-one-out school-mean composition
    * lift + Nadeau-Bengio corrected-resampled-t-test p (is the lift real?)
    * null multilevel ICC  - between-school share of outcome variance
    * ceiling gap          - how far the school-context AUC sits below 1.0

The generalisation claim predicts, across countries: composition gives a
significant lift, the achievable AUC then sits in a similar moderate band, and
that band tracks the ICC (more between-school segregation -> more predictable).
A country that breaks the pattern (e.g. a low-ICC top performer with little lift)
is reported as a boundary condition, not hidden.

Outputs:
    outputs/results/ceiling_generalization_2022.csv
    outputs/figures/ceiling/F1_base_vs_ceiling.png
    outputs/figures/ceiling/F2_icc_vs_lift.png
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
from sklearn.model_selection import RepeatedStratifiedKFold
from threadpoolctl import threadpool_limits

from src.models.evaluate import corrected_resampled_ttest, fit_with_sample_weight
from src.models.experiment import _wrap
from src.models.multilevel import variance_partition_icc
from src.models.prepare import build_model_data
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.visualization.style import PALETTE, apply_publication_style, save_figure

FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
         "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
GROUP = {"ALB": "Albania", "MKD": "Balkan", "MNE": "Balkan", "SRB": "Balkan",
         "BGR": "Balkan", "EST": "Top performer", "FIN": "Top performer",
         "MEX": "GDP-matched", "COL": "GDP-matched"}
ORDER = ["ALB", "MKD", "MNE", "SRB", "BGR", "COL", "MEX", "EST", "FIN"]

N_SPLITS, N_REPEATS = 5, 2
TEST_TRAIN_RATIO = 1.0 / (N_SPLITS - 1)  # Nadeau-Bengio correction term


def _paired_cv(base, school, seed: int = 42):
    """Per-fold weighted AUC for base vs school-context on identical splits."""
    X_b, X_s, y, w = base.X, school.X, base.y, base.weights
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=seed)
    base_f, sch_f = [], []
    with threadpool_limits(limits=1):
        for tr, te in rskf.split(X_b, y):
            if len(np.unique(y.iloc[te])) < 2:
                continue
            wte = w.iloc[te].values
            for Xf, store in ((X_b, base_f), (X_s, sch_f)):
                pipe = _wrap(get_model("lightgbm"))
                fit_with_sample_weight(pipe, Xf.iloc[tr], y.iloc[tr], w.iloc[tr].values)
                p = pipe.predict_proba(Xf.iloc[te])[:, 1]
                store.append(roc_auc_score(y.iloc[te], p, sample_weight=wte))
    return np.array(base_f), np.array(sch_f)


def main() -> None:
    configure_logging(level="WARNING")
    apply_publication_style()
    res_dir = ROOT / "outputs/results"
    fig_dir = ROOT / "outputs/figures/ceiling"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(ROOT / "data/processed/comparison_dataset.parquet")
    d22 = full[full.CYCLE == 2022]

    rows = []
    for cnt in ORDER:
        sub = d22[d22.COUNTRY == cnt].copy()
        base = build_model_data(sub, FEATS, domain="math", add_school_context=False)
        school = build_model_data(sub, FEATS, domain="math", add_school_context=True)
        # both drop only missing-target rows -> identical index/y/weights
        assert base.y.index.equals(school.y.index)

        base_f, sch_f = _paired_cv(base, school)
        lift = float(sch_f.mean() - base_f.mean())
        _, _, p = corrected_resampled_ttest(sch_f - base_f, TEST_TRAIN_RATIO)

        # null (unconditional) between-school ICC on the same modelled sample
        groups = sub.loc[base.y.index, "CNTSCHID"]
        icc = variance_partition_icc(base.y, groups)["icc"]

        row = {
            "country": cnt, "group": GROUP[cnt],
            "n": int(base.meta["n_samples"]),
            "at_risk_rate": round(float(base.meta["at_risk_rate"]), 3),
            "base_auc": round(float(base_f.mean()), 4),
            "school_auc": round(float(sch_f.mean()), 4),
            "lift": round(lift, 4),
            "lift_p": round(float(p), 4),
            "icc": round(float(icc), 3),
            "ceiling_gap": round(1.0 - float(sch_f.mean()), 4),
        }
        rows.append(row)
        print(f"  {cnt} ({GROUP[cnt]}): base={row['base_auc']:.3f} "
              f"school={row['school_auc']:.3f} lift={lift:+.3f} (p={p:.3f}) "
              f"ICC={icc:.2f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(res_dir / "ceiling_generalization_2022.csv", index=False)
    print("\n" + table.to_string(index=False))

    # ---- figures -------------------------------------------------------------
    order_by_icc = table.sort_values("icc")
    x = np.arange(len(order_by_icc))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, order_by_icc.base_auc, 0.4, label="student features",
           color=PALETTE["blue"])
    ax.bar(x + 0.2, order_by_icc.school_auc, 0.4, label="+ school composition",
           color=PALETTE["vermilion"])
    ax.set_xticks(x); ax.set_xticklabels(order_by_icc.country)
    ax.set_ylim(0.5, 0.85); ax.axhline(0.78, ls=":", color="0.5", lw=1)
    ax.text(0, 0.785, "Albania ceiling 0.78", fontsize=8, color="0.4")
    ax.set_ylabel("weighted 5x2 CV AUC")
    ax.set_title("Predictive ceiling by country: student vs +school composition (2022)")
    ax.legend(frameon=False)
    save_figure(fig, str(fig_dir / "F1_base_vs_ceiling"))

    fig2, ax2 = plt.subplots(figsize=(6.5, 5))
    colors = {"Albania": PALETTE["vermilion"], "Balkan": PALETTE["blue"],
              "Top performer": PALETTE["green"], "GDP-matched": PALETTE["orange"]}
    for grp, s in table.groupby("group"):
        ax2.scatter(s.icc, s.school_auc, s=70, color=colors.get(grp, "gray"),
                    label=grp, zorder=3)
    for _, r in table.iterrows():
        ax2.annotate(r.country, (r.icc, r.school_auc), fontsize=8,
                     xytext=(4, 4), textcoords="offset points")
    ax2.set_xlabel("between-school ICC (null multilevel model)")
    ax2.set_ylabel("achievable CV AUC (+school composition)")
    ax2.set_title("More between-school segregation -> higher achievable AUC")
    ax2.legend(frameon=False, fontsize=8)
    save_figure(fig2, str(fig_dir / "F2_icc_vs_lift"))

    # correlation, honest boundary read
    corr = float(np.corrcoef(table.icc, table.school_auc)[0, 1])
    print(f"\nAcross 9 countries: corr(ICC, achievable AUC) = {corr:.2f}")
    print(f"Composition lift significant (p<0.05) in "
          f"{int((table.lift_p < 0.05).sum())}/{len(table)} countries.")


if __name__ == "__main__":
    main()
