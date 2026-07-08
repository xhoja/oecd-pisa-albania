"""
Earthquake natural-experiment analysis (target #2): a *rigorous null*.

The 26 Nov 2019 M6.4 Durres earthquake is used as a candidate identification
strategy. PISA's sampling STRATUM encodes a North / Center / South band; the
quake damage concentrated in the Center band (Durres + Tirana). We run a
difference-in-differences (Center vs North+South, 2018 -> 2022) and then subject
it to the tests a careful reviewer would demand:

    1. a **placebo DiD** on the pre-quake window 2015 -> 2018 (should be ~0),
    2. a **triple-difference** on urbanicity that nets out any capital-region
       secular trend, and
    3. a **school-clustered** regression cross-check.

Result: the naive DiD (~+4.7pp) does NOT survive. The placebo is large and
significant (parallel trends violated: the Center band was already on a steeper
improving trajectory), the urbanicity triple-difference is ~0 with the wrong
sign, and the effect is null once SEs cluster on schools. The 2018 -> 2022
at-risk explosion appears in *every* band and urbanicity cell (national COVID +
2022 sample-coverage shock), not localised to the quake zone. We therefore do
not claim a causal earthquake effect; the well-identified null reinforces the
paper's structural-national-crisis reading over a localised-shock story.

Outputs (outputs/results/):
    causal_event_study_2015_2022.csv   band x cycle at-risk with design-based CI
    causal_did_summary.json            DiD, placebo, DDD, DDD-placebo, regression
Figure: outputs/figures/causal/F1_event_study.png
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
import pandas as pd

from src.causal.did import (
    band_rate_by_cycle,
    ddd_brr_rubin,
    did_brr_rubin,
    did_regression,
)
from src.causal.region import add_region_band, build_stratum_lookup

RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures" / "causal"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

SAV_PATHS = {
    2015: RAW / "CY6_MS_CMB_STU_QQQ.sav",
    2018: RAW / "CY07_MSU_STU_QQQ.sav",
    2022: RAW / "CY08MSP_STU_QQQ.SAV",
}


def _load() -> pd.DataFrame:
    lookup = build_stratum_lookup(SAV_PATHS)
    # persist the small code->band lookup alongside the results (a committed,
    # SAV-free reference table the notebook renders) so notebook 14 is
    # reproducible without re-reading the multi-GB SAV files
    lookup.to_csv(RESULTS / "stratum_region_lookup.csv", index=False)
    df = pd.read_parquet(PROCESSED / "albania_longitudinal.parquet")
    df = df[df["CYCLE"].isin([2015, 2018, 2022])].copy()
    return add_region_band(df, lookup)


def _plot_event_study(es: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"Center": "#c1121f", "North": "#457b9d", "South": "#2a9d8f"}
    for band, sub in es.groupby("region"):
        sub = sub.sort_values("CYCLE")
        label = f"{band}" + (" (treated)" if band == "Center" else "")
        ax.errorbar(
            sub["CYCLE"], sub["at_risk"],
            yerr=1.96 * sub["se"], marker="o", capsize=3,
            color=colors.get(band, "gray"), label=label,
            linestyle="--" if band == "Center" else "-",
        )
    ax.axvspan(2019.9, 2022, color="gray", alpha=0.08)
    ax.axvline(2019.9, color="gray", ls=":", lw=1)
    ax.annotate("Nov 2019\nM6.4 quake", xy=(2019.9, ax.get_ylim()[0]),
                xytext=(2020.0, 0.45), fontsize=8, color="gray")
    ax.set_xticks([2015, 2018, 2022])
    ax.set_xlabel("PISA cycle")
    ax.set_ylabel("Share below Level 2 in mathematics")
    ax.set_title("Event study: at-risk rate by region band (design-based 95% CI)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "F1_event_study.png", dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = _load()

    es = band_rate_by_cycle(df)
    es.to_csv(RESULTS / "causal_event_study_2015_2022.csv", index=False)

    summary = {
        "did_main_2018_2022": did_brr_rubin(df, pre=2018, post=2022),
        "did_placebo_2015_2018": did_brr_rubin(df, pre=2015, post=2018),
        "ddd_main_2018_2022": ddd_brr_rubin(df, pre=2018, post=2022),
        "ddd_placebo_2015_2018": ddd_brr_rubin(df, pre=2015, post=2018),
        "did_regression_clustered_2018_2022": did_regression(df, pre=2018, post=2022),
    }
    with open(RESULTS / "causal_did_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot_event_study(es)

    # console summary
    print("\n=== EVENT STUDY ===")
    print(es.round(3).to_string(index=False))
    d, pl = summary["did_main_2018_2022"], summary["did_placebo_2015_2018"]
    ddd, dddpl = summary["ddd_main_2018_2022"], summary["ddd_placebo_2015_2018"]
    reg = summary["did_regression_clustered_2018_2022"]
    print("\n=== DiD (Center vs North+South) ===")
    print(f"  main    2018->2022  DiD={d['did']:+.4f}  se={d['se']:.4f}  "
          f"p={d['p_value']:.4g}  95%CI[{d['ci95_low']:.3f},{d['ci95_high']:.3f}]")
    print(f"  placebo 2015->2018  DiD={pl['did']:+.4f}  se={pl['se']:.4f}  "
          f"p={pl['p_value']:.4g}  <- parallel-trends test")
    print("=== Triple-difference (x urbanicity) ===")
    print(f"  main    2018->2022  DDD={ddd['ddd']:+.4f}  se={ddd['se']:.4f}  p={ddd['p_value']:.4g}")
    print(f"  placebo 2015->2018  DDD={dddpl['ddd']:+.4f}  se={dddpl['se']:.4f}  p={dddpl['p_value']:.4g}")
    print("=== Regression cross-check (WLS-LPM, school-clustered) ===")
    print(f"  DiD={reg['did']:+.4f}  se={reg['se']:.4f}  p={reg['p_value']:.4g}  "
          f"(n={reg['n']}, clusters={reg['n_clusters']})")
    print("\nConclusion: naive DiD does not survive placebo/DDD/clustered SE -> "
          "no credible causal quake effect; 2022 collapse is national.")


if __name__ == "__main__":
    main()
