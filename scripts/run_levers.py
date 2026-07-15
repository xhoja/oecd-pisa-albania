"""
Malleable-lever ranking for Albania 2022: which *actionable* levers move risk.

SHAP tells us what the screener reads; this tells the Ministry what to *do*. We
split the fitted CatBoost school-context model's features into fixed endowments
(SES, parental education, home possessions, immigrant status, gender, grade) and
malleable levers (math anxiety, teacher support, belonging, grade-repetition,
ICT), then rank the levers by the population predicted at-risk reduction from a
standardized half-SD nudge in the beneficial direction (src/models/levers.py).

Two extra reads:
  * a fixed-vs-malleable "ceiling" bundle - how far actionable levers reach
    before the structural endowment floor, and
  * the lever ranking re-run inside the rural/North band vs urban, to see
    whether the best lever differs by geography (ties to notebook 02 Part C).

Caveat: a predictive what-if under the model's associations, not a causal
effect. See src/models/policy.py.

Outputs (outputs/results/):
    lever_ranking_2022.csv        national malleable-lever ranking
    lever_ceiling_2022.csv        malleable vs fixed bundle reach
    lever_ranking_by_geography_2022.csv   ranking within rural / urban
Figure (outputs/figures/models/):
    L1_lever_ranking.{png,pdf}
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
import pandas as pd
from threadpoolctl import threadpool_limits

from src.causal.region import add_region_band
from src.models.evaluate import fit_with_sample_weight, tune_threshold
from src.models.experiment import _wrap
from src.models.levers import bootstrap_lever_effects, lever_ceiling, rank_levers
from src.models.prepare import build_model_data
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.utils.reproducibility import set_seeds
from src.visualization.style import PALETTE, apply_publication_style

BASE_FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
              "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
SEED = 42
MAG = 0.5  # half-SD nudge per lever
N_BOOT = 400  # population-bootstrap resamples for lever-effect CIs
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures" / "models"


def _plot(rank: pd.DataFrame, base: float) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    r = rank.sort_values("delta_at_risk_rate", ascending=True).copy()
    r["reduction_pp"] = -r["delta_at_risk_rate"] * 100  # positive = fewer at-risk
    fig, ax = plt.subplots(figsize=(8, 4.2))
    xerr = None
    if "ci_low" in r.columns:
        # reductions flip the sign, so the CI bounds swap; build asymmetric xerr
        lo = -r["ci_high"] * 100  # lower reduction bound
        hi = -r["ci_low"] * 100   # upper reduction bound
        xerr = [(r["reduction_pp"] - lo).values, (hi - r["reduction_pp"]).values]
    ax.barh(r["description"], r["reduction_pp"], color=PALETTE["blue"],
            edgecolor="white", xerr=xerr,
            error_kw=dict(ecolor="0.35", capsize=3, lw=1.2))
    # place each label beyond the upper CI whisker so it never sits on a cap
    hi_red = (-r["ci_low"] * 100).values if "ci_low" in r.columns else r["reduction_pp"].values
    for y, (v, h) in enumerate(zip(r["reduction_pp"], hi_red)):
        ax.annotate(f"{v:+.1f} pp", xy=(max(v, h, 0.0), y), xytext=(7, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=9)
    ax.set_xlabel("Reduction in predicted at-risk rate (pp), 0.5-SD nudge (95% bootstrap CI)")
    ax.set_title(f"Actionable levers, ranked by reach\n(baseline predicted "
                 f"at-risk = {base*100:.0f}%; larger = more reduction)", fontsize=11)
    ax.axvline(0, color="0.6", lw=1)
    _lo = (-r["ci_high"] * 100).min() if "ci_high" in r.columns else r["reduction_pp"].min()
    ax.set_xlim(min(_lo - 0.3, -0.5), r["reduction_pp"].max() + 1.2)
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(FIGURES / f"L1_lever_ranking.{fmt}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_logging(level="WARNING")
    apply_publication_style()
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    lookup = pd.read_csv(RESULTS / "stratum_region_lookup.csv")
    df = add_region_band(df, lookup)

    data = build_model_data(df, BASE_FEATS, domain="math", add_school_context=True)
    X = data.X.copy()
    X = X.fillna(X.median(numeric_only=True))
    y = data.y.copy()
    w = data.weights.copy()
    geo = df.loc[X.index, ["region", "urbanicity"]]  # aligned by index

    set_seeds(SEED)
    with threadpool_limits(limits=1):
        pipe = _wrap(get_model("catboost"))
        fit_with_sample_weight(pipe, X, y, w.values)
        proba = pipe.predict_proba(X)[:, 1]
    thr = tune_threshold(y.values, proba, sample_weight=w.values, metric="f1_macro")

    # national ranking + population-bootstrap CIs on each lever effect
    rank = rank_levers(pipe, X, w.values, thr, magnitude=MAG)
    base = rank.attrs["baseline_at_risk_rate"]
    rank = bootstrap_lever_effects(pipe, X, w.values, thr, rank,
                                   magnitude=MAG, n_boot=N_BOOT, seed=SEED)
    rank.to_csv(RESULTS / "lever_ranking_2022.csv", index=False)

    ceil = lever_ceiling(pipe, X, w.values, thr, magnitude=MAG)
    ceil_row = pd.DataFrame([
        {"bundle": "all malleable levers", **ceil["malleable_bundle"]},
        {"bundle": "all fixed endowments (hypothetical)", **ceil["fixed_bundle"]},
    ])
    ceil_row.insert(0, "baseline_at_risk_rate", ceil["baseline_at_risk_rate"])
    ceil_row.to_csv(RESULTS / "lever_ceiling_2022.csv", index=False)

    # per-geography ranking: rural (any region) vs urban
    geo_rows = []
    for label, mask in [
        ("Rural", geo["urbanicity"] == "Rural"),
        ("Urban", geo["urbanicity"] == "Urban"),
        ("North", geo["region"] == "North"),
    ]:
        m = mask.reindex(X.index, fill_value=False).values
        if m.sum() < 100:
            continue
        rk = rank_levers(pipe, X[m], w[m].values, thr, magnitude=MAG)
        top = rk.iloc[0]
        for _, row in rk.iterrows():
            geo_rows.append({
                "group": label, "n": int(m.sum()),
                "lever": row["lever"], "description": row["description"],
                "delta_at_risk_rate": row["delta_at_risk_rate"],
                "is_top": row["lever"] == top["lever"],
            })
    geo_tab = pd.DataFrame(geo_rows)
    geo_tab.to_csv(RESULTS / "lever_ranking_by_geography_2022.csv", index=False)

    _plot(rank, base)

    # console summary
    print(f"threshold={thr:.3f}  baseline predicted at-risk={base:.3f}\n")
    print(f"=== NATIONAL MALLEABLE-LEVER RANKING (0.5-SD nudge, {N_BOOT}-boot 95% CI) ===")
    print(rank[["lever", "description", "direction", "delta_at_risk_rate",
                "ci_low", "ci_high"]].to_string(index=False))
    print("\n=== FIXED vs MALLEABLE BUNDLE (reach vs structural floor) ===")
    print(ceil_row.to_string(index=False))
    print("\n=== TOP LEVER BY GEOGRAPHY ===")
    if not geo_tab.empty:
        print(geo_tab[geo_tab.is_top][["group", "n", "lever", "delta_at_risk_rate"]]
              .to_string(index=False))
    print(f"\nFigure -> {FIGURES / 'L1_lever_ranking.png'}")


if __name__ == "__main__":
    main()
