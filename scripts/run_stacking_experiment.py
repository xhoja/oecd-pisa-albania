"""
Phase 9 (Advanced) — stacking ensemble.

Question: with school context already lifting the ceiling to ~0.78, does *stacking*
the base learners beat the best single model (CatBoost, 0.783), or is the gain — as
with PV-stacking earlier — marginal and not worth the complexity?

Protocol (identical to the school-context ablation, so the comparison is fair):
- Albania 2022, 13 student features + 5 school-context features (leave-one-out
  survey-weighted school means), the headline regime.
- A StackingClassifier of bare tree/booster base learners with a plain
  class-balanced LogisticRegression meta-learner on their OOF probabilities
  (src.models.registry.build_stacking_ensemble; registered as "stacking").
- Run through compare_models_cv so stacking AND every single model share the
  SAME weighted RepeatedStratifiedKFold splits (leakage-safe in-fold engineer +
  median impute, Nadeau-Bengio corrected CIs), each in a fresh OpenMP-isolated
  interpreter.
- Paired Nadeau-Bengio corrected resampled t-test of stacking vs each single
  model on the identical per-fold AUCs.

Honest-reporting stance: report the delta and its p-value; a large-looking lift
with p > 0.05 is NOT a win. LightGBM is imported before scikit-learn (macOS
OpenMP import-order fix).
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

import numpy as np
import pandas as pd

from src.features.engineer import add_school_aggregates
from src.models.experiment import compare_models_cv, paired_nb_auc_test
from src.models.prepare import build_model_data
from src.utils.logging import configure_logging

BASE_FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
              "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
SCHOOL_FEATS = ["SCH_MEAN_ESCS", "SCH_MEAN_HOMEPOS", "SCH_MEAN_ANXMAT",
                "SCH_MEAN_TEACHSUP", "SCH_N"]
# The stacking base learners (catboost, lightgbm, gradient_boosting,
# random_forest) plus the meta-learner (logistic_regression), scored alongside
# "stacking" on identical folds so the ensemble is compared against its own parts.
SINGLES = ["catboost", "lightgbm", "gradient_boosting", "random_forest",
           "logistic_regression"]
MODELS = SINGLES + ["stacking"]
OUTER_FOLDS, OUTER_REPEATS = 5, 4


def main() -> None:
    configure_logging(level="WARNING")
    res_dir = ROOT / "outputs/results"
    res_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    df = add_school_aggregates(df, cols=["ESCS", "HOMEPOS", "ANXMAT", "TEACHSUP"])
    data = build_model_data(df, BASE_FEATS + SCHOOL_FEATS, domain="math")
    kept_school = [f for f in SCHOOL_FEATS if f in data.feature_names]
    print(f"Features: {len(data.feature_names)} (school feats kept: {kept_school})\n")

    print("== Stacking vs single models (school-context regime) ==")
    res = compare_models_cv(data, MODELS, OUTER_FOLDS, OUTER_REPEATS,
                            save_path=res_dir / "stacking_ensemble_2022.csv")
    print("\n=== Leaderboard (weighted CV AUC, Nadeau-Bengio CI) ===")
    print(res[["model", "roc_auc_mean", "roc_auc_95ci_low", "roc_auc_95ci_high",
               "pr_auc_mean", "f1_macro_mean", "mcc_mean"]].to_string(index=False))

    fold_scores = res.attrs.get("fold_scores", {})
    if "stacking" not in fold_scores:
        print("\n[!] stacking produced no valid folds — likely an OpenMP crash in "
              "the multi-booster subprocess. See stderr above.")
        print("\nSAVED OK")
        return

    pairwise = paired_nb_auc_test(fold_scores, OUTER_FOLDS)
    # Keep only the rows that involve stacking (the question is stack vs part).
    stk = pairwise[(pairwise.model_a == "stacking") | (pairwise.model_b == "stacking")].copy()
    stk.to_csv(res_dir / "stacking_pairwise_nb_2022.csv", index=False)
    print("\n=== Stacking vs each single model (paired Nadeau-Bengio t-test) ===")
    print(stk.to_string(index=False))

    stack_auc = float(np.mean(fold_scores["stacking"]))
    best_single = max((m for m in fold_scores if m != "stacking"),
                      key=lambda m: np.mean(fold_scores[m]))
    best_auc = float(np.mean(fold_scores[best_single]))
    delta = stack_auc - best_auc
    # p-value of stacking vs the best single model, whichever ordering the test used
    row = stk[((stk.model_a == "stacking") & (stk.model_b == best_single)) |
              ((stk.model_a == best_single) & (stk.model_b == "stacking"))]
    p = float(row["p_value"].iloc[0]) if len(row) else float("nan")
    print(f"\nStacking AUC {stack_auc:.4f} vs best single ({best_single}) "
          f"{best_auc:.4f}: delta {delta:+.4f}, p={p:.3f}")
    verdict = ("stacking significantly beats the best single model."
               if (delta > 0 and p < 0.05)
               else "no significant gain over the best single model — "
                    "single CatBoost remains the honest headline.")
    print("Verdict:", verdict)
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
