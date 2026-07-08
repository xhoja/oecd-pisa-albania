"""
Target #5 - coverage robustness: Manski bounds on Albania's 2022 at-risk rate.

PISA 2022 covered about 79% of Albania's 15-year-olds (Coverage Index 3 ~ 0.79;
6 129 sampled students representing ~28 400 of ~36 000 - OECD PISA 2022 Results,
Country Note). The reported ~74% at-risk rate is a rate *among the covered*; the
~21% uncovered (disproportionately out-of-school, disadvantaged youth) are
unobserved. We report the range of population rates consistent with the data
(partial identification) instead of assuming the uncovered look like the sample.

Reports, all survey-weighted:
    * observed at-risk rate among the covered (design-based BRR+PV SE),
    * worst-case Manski bounds (no assumption on the uncovered),
    * monotone bounds (uncovered at least as at-risk as covered),
    * data-anchored scenarios that "reweight" the uncovered to look like the
      poorest covered SES quintile / the overall sample / the richest quintile,
    * sensitivity of the bounds to the assumed coverage share.

Outputs:
    outputs/results/coverage_bounds_2022.json
    outputs/results/coverage_sensitivity_2022.csv
    outputs/figures/coverage/F1_manski_bounds.png
"""
from __future__ import annotations

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

from src.coverage.manski import (
    coverage_sensitivity,
    monotone_bounds,
    scenario_rate,
    worst_case_bounds,
)
from src.data.weights import compute_ses_quintiles, pv_statistic_brr, weighted_proportion
from src.features.target import add_point_target
from src.utils.logging import configure_logging
from src.visualization.style import PALETTE, apply_publication_style, save_figure

# OECD PISA 2022 Results (Volume I), Albania Country Note: sampled students
# represent ~79% of the 15-year-old population.
COVERAGE_2022 = 0.79


def main() -> None:
    configure_logging(level="WARNING")
    apply_publication_style()
    res = ROOT / "outputs/results"; fig = ROOT / "outputs/figures/coverage"
    res.mkdir(parents=True, exist_ok=True); fig.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    stat = pv_statistic_brr(df, domain="math", statistic="at_risk")
    p_obs, se = float(stat["estimate"]), float(stat["se"])

    # SES-quintile at-risk rates as reweighting stand-ins for the uncovered
    d = compute_ses_quintiles(add_point_target(df.copy(), "math"),
                              ses_col="ESCS", weight_col="W_FSTUWT")
    def q_rate(q: float) -> float:
        s = d[d.SES_QUINTILE == q]
        return float(weighted_proportion(s["AT_RISK_MATH"].astype(float), s["W_FSTUWT"]))
    q1, q5 = q_rate(1.0), q_rate(5.0)

    wc = worst_case_bounds(p_obs, COVERAGE_2022)
    mono = monotone_bounds(p_obs, COVERAGE_2022, uncovered_worse=True)
    scenarios = {
        "uncovered_like_poorest_quintile": scenario_rate(p_obs, COVERAGE_2022, q1),
        "uncovered_like_overall_sample": scenario_rate(p_obs, COVERAGE_2022, p_obs),
        "uncovered_like_richest_quintile": scenario_rate(p_obs, COVERAGE_2022, q5),
    }

    payload = {
        "cycle": 2022, "coverage_index3": COVERAGE_2022,
        "observed_at_risk_covered": round(p_obs, 4),
        "observed_se_design_based": round(se, 4),
        "observed_ci95": [round(p_obs - 1.96 * se, 4), round(p_obs + 1.96 * se, 4)],
        "poorest_quintile_rate": round(q1, 4),
        "richest_quintile_rate": round(q5, 4),
        "worst_case_bounds": wc.as_dict(),
        "monotone_bounds_uncovered_worse": mono.as_dict(),
        "scenarios": {k: round(v, 4) for k, v in scenarios.items()},
    }
    (res / "coverage_bounds_2022.json").write_text(json.dumps(payload, indent=2))

    sens = pd.DataFrame(coverage_sensitivity(p_obs, np.round(np.arange(0.70, 0.96, 0.02), 2)))
    sens.to_csv(res / "coverage_sensitivity_2022.csv", index=False)

    print(f"Observed at-risk (covered): {p_obs:.3f} (design SE {se:.3f})")
    print(f"Coverage Index 3          : {COVERAGE_2022}")
    print(f"Worst-case bounds         : [{wc.lower:.3f}, {wc.upper:.3f}] (width {wc.width:.3f})")
    print(f"Monotone (uncovered worse): [{mono.lower:.3f}, {mono.upper:.3f}]")
    print(f"Scenario uncovered~=Q1    : {scenarios['uncovered_like_poorest_quintile']:.3f}")

    # ---- figure ----
    fig1, ax = plt.subplots(figsize=(8, 4.2))
    ax.axvspan(wc.lower, wc.upper, color=PALETTE["blue"], alpha=0.12, label="worst-case range")
    ax.axvspan(mono.lower, mono.upper, color=PALETTE["blue"], alpha=0.28,
               label="monotone range (uncovered >= covered)")
    ax.errorbar([p_obs], [1.0], xerr=[[1.96 * se], [1.96 * se]], fmt="o",
                color=PALETTE["black"], capsize=4, label="observed (covered) +/- 95% CI")
    marks = {"uncovered ~ poorest Q": scenarios["uncovered_like_poorest_quintile"],
             "uncovered ~ richest Q": scenarios["uncovered_like_richest_quintile"]}
    for i, (lbl, val) in enumerate(marks.items()):
        ax.scatter([val], [1.0], marker="D", s=55, zorder=5,
                   color=PALETTE["vermilion"] if "poorest" in lbl else PALETTE["green"])
        ax.annotate(lbl, (val, 1.0), xytext=(0, 12 + 14 * i), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_yticks([]); ax.set_ylim(0.85, 1.3)
    ax.set_xlim(0.55, 0.85)
    ax.set_xlabel("population at-risk rate (share below Level 2, math)")
    ax.set_title("Manski coverage bounds: Albania 2022 (79% coverage)")
    ax.legend(loc="lower left", fontsize=8)
    save_figure(fig1, str(fig / "F1_manski_bounds"))
    plt.close(fig1)
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
