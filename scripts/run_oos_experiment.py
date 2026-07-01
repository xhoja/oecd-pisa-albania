"""
Out-of-sample covariate-shift experiment.

Train on Albania 2009-2018, test on held-out 2022 cohort.
Hypothesis: tree models overfit historical patterns and collapse on 2022, while
regularized logistic regression generalizes (linear boundary robust to shift).
Generates the OOS comparison figure + JSON results.
"""
from __future__ import annotations

import os

# Must be set before numpy/sklearn/boosters import OpenMP (duplicate libomp on macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.models.experiment import out_of_sample, save_results
from src.utils.logging import configure_logging
from src.visualization.model_plots import plot_oos_comparison
from src.visualization.style import save_figure


def main() -> None:
    configure_logging(level="WARNING")
    long = pd.read_parquet(ROOT / "data/processed/albania_longitudinal.parquet")
    train = long[long.CYCLE.isin([2009, 2012, 2015, 2018])]
    test = long[long.CYCLE == 2022]
    print(f"Train (2009-2018): {len(train)} | Test (2022): {len(test)}")

    # Core features available across cycles (avoid 2015-only gaps biasing)
    feats = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG",
             "TEACHSUP", "ANXMAT", "GRADE", "HISCED", "HISEI"]
    models = ["logistic_regression", "random_forest", "extra_trees",
              "gradient_boosting", "xgboost", "lightgbm", "catboost"]

    res = out_of_sample(train, test, feats, models, domain="math")
    print(f"\nFeatures used ({len(res['features'])}): {res['features']}")
    print(f"n_train={res['n_train']} n_test={res['n_test']}\n")

    ok = {m: d for m, d in res["models"].items() if "roc_auc" in d}
    failed = [m for m, d in res["models"].items() if "error" in d]
    print(f"{'model':<22}{'test_AUC':>10}{'PR_AUC':>9}{'F1':>8}{'MCC':>8}")
    for m, d in sorted(ok.items(), key=lambda x: -x[1]["roc_auc"]):
        print(f"{m:<22}{d['roc_auc']:>10}{d['pr_auc']:>9}{d['f1_macro']:>8}{d['mcc']:>8}")
    if failed:
        print(f"\n[!] models that crashed at the C level (check local libomp install): {failed}")

    # DeLong pairwise AUC test on the shared 2022 test set.
    dl = res.get("delong_pairwise", [])
    if dl:
        print("\nDeLong pairwise AUC test (shared 2022 test set):")
        print(f"{'A':<20}{'B':<20}{'auc_a':>8}{'auc_b':>8}{'z':>8}{'p':>9}  sig")
        for r in dl:
            print(f"{r['model_a']:<20}{r['model_b']:<20}{r['auc_a']:>8}{r['auc_b']:>8}"
                  f"{r['z']:>8}{r['p_value']:>9}  {r['significant_0.05']}")

    save_results(res, ROOT / "outputs/results/oos_2022_experiment.json")
    fig = plot_oos_comparison({"models": ok}, metric="roc_auc")
    save_figure(fig, str(ROOT / "outputs/figures/models/oos_2022_comparison"))
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
