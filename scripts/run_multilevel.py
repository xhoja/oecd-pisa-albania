"""
Multilevel (random-intercept) logistic model for Albania 2022 low-proficiency risk.

Two questions the flat models can't answer:
1. **How much risk is a school-level phenomenon?** The intraclass correlation (ICC)
   partitions variance into within- vs between-school. A large ICC is the principled
   justification for the project's school-context features.
2. **Does the statistically-correct model beat the hand-crafted one?** We compare, on
   identical weighted 5-fold splits, the random-intercept model (student features +
   a school random effect) against the headline school-mean LightGBM.

The random effect for a test student's school is estimated from that school's
*training* students (schools span folds), i.e. "predict a new student in a known
school" - the realistic deployment case. Unseen-school prediction (random effect = 0)
is the harder regime and is noted, not hidden.

Two mixed fits: the unweighted VB model (BinomialBayesMixedGLM takes no survey
weights) and a survey-weighted pseudo-likelihood PQL fit with within-cluster
scaled weights (the design-based refinement). AUCs are evaluated weighted. See
src/models/multilevel.py.

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from src.features.engineer import add_school_aggregates
from src.models.evaluate import fit_with_sample_weight
from src.models.experiment import _wrap
from src.models.multilevel import (
    _standardize,
    fit_random_intercept,
    fit_weighted_random_intercept,
    predict_random_intercept,
    variance_partition_icc,
)
from src.models.prepare import build_model_data
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.visualization.style import PALETTE, apply_publication_style, save_figure

BASE_FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
              "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
SCHOOL_FEATS = ["SCH_MEAN_ESCS", "SCH_MEAN_HOMEPOS", "SCH_MEAN_ANXMAT",
                "SCH_MEAN_TEACHSUP", "SCH_N"]
GROUP = "CNTSCHID"
N_SPLITS, SEED = 5, 42


def main() -> None:
    configure_logging(level="WARNING")
    res_dir = ROOT / "outputs/results"
    fig_dir = ROOT / "outputs/figures/models"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    df = add_school_aggregates(df, cols=["ESCS", "HOMEPOS", "ANXMAT", "TEACHSUP"])

    mdata = build_model_data(df, BASE_FEATS, domain="math")             # student-only (mixed)
    sdata = build_model_data(df, BASE_FEATS + SCHOOL_FEATS, domain="math")  # school-means (booster)
    feats = [f for f in BASE_FEATS if f in mdata.X.columns]
    X, y, w = mdata.X.reset_index(drop=True), mdata.y.reset_index(drop=True), mdata.weights.reset_index(drop=True)
    grp = df.loc[mdata.X.index, GROUP].reset_index(drop=True)
    Xs_full, ys, sw = sdata.X.reset_index(drop=True), sdata.y.reset_index(drop=True), sdata.weights.reset_index(drop=True)
    n_sch = grp.nunique()
    print(f"n = {len(y)} students in {n_sch} schools (avg {len(y)/n_sch:.1f}/school), "
          f"at-risk {float(y.mean()):.3f}")

    # ---- variance partition (full data) -------------------------------------
    null = variance_partition_icc(y, grp)
    fit = fit_random_intercept(X, y, grp, feats)
    print(f"\nICC (null, no predictors)      = {null['icc']:.3f}  "
          f"→ ~{null['icc']*100:.0f}% of risk variance is BETWEEN schools")
    print(f"ICC (conditional on students)  = {fit['icc']:.3f}  "
          f"(school effect σ = {fit['school_sd']:.2f})")

    # ---- survey-weighted pseudo-likelihood (PQL, scaled weights) ------------
    wfit = fit_weighted_random_intercept(X, y, grp, w, feats, scaling="effective")
    print(f"ICC (weighted PQL, scaled wts) = {wfit['icc']:.3f}  "
          f"(school effect σ = {wfit['school_sd']:.2f}; PQL mildly attenuates variance)")
    # side-by-side fixed-effect odds ratios: unweighted VB vs weighted PQL
    or_cmp = (fit['fixed_effects'][["term", "odds_ratio"]]
              .rename(columns={"odds_ratio": "OR_unweighted_vb"})
              .merge(wfit['fixed_effects'][["term", "odds_ratio", "coef", "sd"]]
                     .rename(columns={"odds_ratio": "OR_weighted_pql"}), on="term"))
    or_cmp.round(4).to_csv(res_dir / "multilevel_weighted_vs_unweighted_2022.csv", index=False)
    wfit['fixed_effects'].round(4).to_csv(res_dir / "multilevel_weighted_fixed_effects_2022.csv", index=False)

    # ---- weighted 5-fold CV: mixed vs school-mean booster -------------------
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        # multilevel (student features + school random intercept)
        f = fit_random_intercept(X.iloc[tr], y.iloc[tr], grp.iloc[tr], feats)
        Xs_te = _standardize(X[feats]).iloc[te]
        p_mix = predict_random_intercept(f['result'], Xs_te, grp.iloc[te].values, grp.iloc[tr].values)
        auc_mix = roc_auc_score(y.iloc[te], p_mix, sample_weight=w.iloc[te].values)

        # school-mean LightGBM on identical split
        with threadpool_limits(limits=1):
            pipe = _wrap(get_model("lightgbm"))
            fit_with_sample_weight(pipe, Xs_full.iloc[tr], ys.iloc[tr], sw.iloc[tr].values)
            p_gbm = pipe.predict_proba(Xs_full.iloc[te])[:, 1]
        auc_gbm = roc_auc_score(ys.iloc[te], p_gbm, sample_weight=sw.iloc[te].values)

        rows.append({"fold": k, "auc_multilevel": auc_mix, "auc_school_lgbm": auc_gbm})
        print(f"  fold {k}: multilevel AUC={auc_mix:.4f} | school-LGBM AUC={auc_gbm:.4f}")

    cv = pd.DataFrame(rows)
    summary = {
        "icc_null": round(null['icc'], 4),
        "icc_conditional": round(fit['icc'], 4),
        "school_sd": round(fit['school_sd'], 4),
        "icc_weighted_pql": round(wfit['icc'], 4),
        "school_sd_weighted_pql": round(wfit['school_sd'], 4),
        "weight_scaling": wfit['scaling'],
        "n_schools": int(n_sch),
        "auc_multilevel_mean": round(cv.auc_multilevel.mean(), 4),
        "auc_school_lgbm_mean": round(cv.auc_school_lgbm.mean(), 4),
    }
    cv.to_csv(res_dir / "multilevel_cv_2022.csv", index=False)
    fit['fixed_effects'].round(4).to_csv(res_dir / "multilevel_fixed_effects_2022.csv", index=False)
    pd.DataFrame([summary]).to_csv(res_dir / "multilevel_summary_2022.csv", index=False)
    print(f"\nCV mean AUC - multilevel {summary['auc_multilevel_mean']:.4f} vs "
          f"school-LGBM {summary['auc_school_lgbm_mean']:.4f}")

    # ---- odds-ratio forest plot (fixed effects, per-SD) ---------------------
    fe = fit['fixed_effects']
    fe = fe[fe.term != "Intercept"].copy()
    fe["lo"] = np.exp(fe.coef - 1.96 * fe.sd)
    fe["hi"] = np.exp(fe.coef + 1.96 * fe.sd)
    fe = fe.sort_values("odds_ratio")
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ypos = np.arange(len(fe))
    # colour risk-increasing (OR>1) vermilion, protective (OR<1) blue
    colors = [PALETTE["vermilion"] if o > 1 else PALETTE["blue"] for o in fe.odds_ratio]
    ax.errorbar(fe.odds_ratio, ypos,
                xerr=[fe.odds_ratio - fe.lo, fe.hi - fe.odds_ratio],
                fmt="none", ecolor="0.5", elinewidth=1, capsize=3, zorder=1)
    ax.scatter(fe.odds_ratio, ypos, color=colors, s=42, zorder=2)
    ax.axvline(1.0, color="0.4", ls="--", lw=1)
    ax.set_yticks(ypos); ax.set_yticklabels(fe.term)
    ax.set_xlabel("Odds ratio per 1 SD (within-school, 95% CI)")
    ax.set_title(f"Random-intercept logistic - Albania 2022 (school ICC {fit['icc']:.2f})")
    save_figure(fig, str(fig_dir / "H1_multilevel_odds_ratios"))
    plt.close(fig)

    print("\nSAVED OK")


if __name__ == "__main__":
    main()
