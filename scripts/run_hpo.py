"""
Phase 5 — Optuna hyper-parameter optimisation in nested CV for the top models.

Tunes each model on the Albania 2022 cohort with a leakage-safe, weighted,
OpenMP-isolated nested CV, and reports whether tuning beats the registry default
by a statistically significant margin (Nadeau-Bengio corrected resampled t-test).

Usage:
    python scripts/run_hpo.py                       # top 3, default budget
    python scripts/run_hpo.py --models catboost --trials 60 --repeats 3
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.models.experiment import save_results
from src.models.hpo import final_best_params, nested_cv_hpo
from src.models.prepare import build_model_data
from src.utils.logging import configure_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["catboost", "gradient_boosting", "logistic_regression"])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--outer-folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--inner-folds", type=int, default=4)
    args = ap.parse_args()

    configure_logging(level="WARNING")
    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    feats = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
             "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
    data = build_model_data(df, feats, domain="math")
    print(f"N={data.meta['n_samples']} at_risk={data.meta['at_risk_rate']} "
          f"feats={len(data.feature_names)} | budget: {args.trials} trials, "
          f"{args.outer_folds}x{args.repeats} outer / {args.inner_folds} inner\n")

    results = {}
    json_path = ROOT / "outputs/results/hpo_nested_cv.json"
    for model in args.models:
        print(f"=== HPO: {model} ===", flush=True)
        try:
            r = nested_cv_hpo(
                data, model, outer_folds=args.outer_folds, outer_repeats=args.repeats,
                inner_folds=args.inner_folds, n_trials=args.trials,
            )
        except Exception as e:  # don't let one model lose the others' results
            print(f"  [skip] {model}: {type(e).__name__}: {str(e).splitlines()[0]}", flush=True)
            results[model] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0]}"}
            save_results(results, json_path)  # incremental save
            continue
        r["final_best_params"] = final_best_params(r)
        results[model] = r
        sig = "significant" if r["hpo_significant_0.05"] else "NOT significant"
        print(f"  tuned={r['tuned_auc_mean']} {r['tuned_auc_95ci']} | "
              f"default={r['default_auc_mean']} {r['default_auc_95ci']} | "
              f"gain={r['hpo_gain']:+} (p={r['hpo_gain_p_value']}, {sig})\n", flush=True)
        save_results(results, json_path)  # incremental save after each model

    ok_results = {m: r for m, r in results.items() if "error" not in r}
    if not ok_results:
        print("No models completed HPO.")
        return

    summary = pd.DataFrame([{
        "model": m,
        "tuned_auc": r["tuned_auc_mean"],
        "default_auc": r["default_auc_mean"],
        "hpo_gain": r["hpo_gain"],
        "p_value": r["hpo_gain_p_value"],
        "significant": r["hpo_significant_0.05"],
    } for m, r in ok_results.items()]).sort_values("tuned_auc", ascending=False)
    summary.to_csv(ROOT / "outputs/results/hpo_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
