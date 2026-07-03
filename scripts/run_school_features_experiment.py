"""
Tier-1 model-improvement pass: does adding **school-level contextual SES** lift the
AUC ceiling the headline comparison hit (~0.73, HPO-confirmed)?

PISA's strongest single predictor of individual achievement is usually the
*school* socioeconomic composition, which the current 13-feature set omits. We add
leave-one-out survey-weighted school means (ESCS/HOMEPOS/ANXMAT/TEACHSUP) + cohort
size, then re-run the weighted repeated-CV comparison and test each model's
baseline-vs-augmented AUC with the Nadeau-Bengio corrected resampled t-test (folds
are identical across arms because the stratified split depends only on y).

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

import numpy as np
import pandas as pd

from src.features.engineer import add_school_aggregates
from src.models.evaluate import corrected_resampled_ttest
from src.models.experiment import compare_models_cv
from src.models.prepare import build_model_data
from src.utils.logging import configure_logging

BASE_FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
              "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
SCHOOL_FEATS = ["SCH_MEAN_ESCS", "SCH_MEAN_HOMEPOS", "SCH_MEAN_ANXMAT",
                "SCH_MEAN_TEACHSUP", "SCH_N"]
MODELS = ["logistic_regression", "gradient_boosting", "random_forest",
          "lightgbm", "catboost"]
OUTER_FOLDS, OUTER_REPEATS = 5, 4


def _fold_scores(res: pd.DataFrame) -> dict[str, list[float]]:
    return res.attrs.get("fold_scores", {})


def main() -> None:
    configure_logging(level="WARNING")
    res_dir = ROOT / "outputs/results"
    res_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    df = add_school_aggregates(df, cols=["ESCS", "HOMEPOS", "ANXMAT", "TEACHSUP"])

    data_base = build_model_data(df, BASE_FEATS, domain="math")
    data_aug = build_model_data(df, BASE_FEATS + SCHOOL_FEATS, domain="math")
    kept_school = [f for f in SCHOOL_FEATS if f in data_aug.feature_names]
    print(f"Baseline features: {len(data_base.feature_names)} | "
          f"Augmented: {len(data_aug.feature_names)} "
          f"(school feats kept: {kept_school})\n")

    print("== Baseline (13 student features) ==")
    base = compare_models_cv(data_base, MODELS, OUTER_FOLDS, OUTER_REPEATS)
    print("\n== Augmented (+ school context) ==")
    aug = compare_models_cv(data_aug, MODELS, OUTER_FOLDS, OUTER_REPEATS)

    fb, fa = _fold_scores(base), _fold_scores(aug)
    ratio = 1.0 / (OUTER_FOLDS - 1)
    rows = []
    for m in MODELS:
        if m not in fb or m not in fa:
            continue
        b = np.asarray(fb[m]); a = np.asarray(fa[m])
        n = min(len(b), len(a))
        diffs = a[:n] - b[:n]  # augmented - baseline, paired by fold
        _, _, p = corrected_resampled_ttest(diffs, ratio)
        rows.append({
            "model": m,
            "auc_base": round(float(b.mean()), 4),
            "auc_school": round(float(a.mean()), 4),
            "delta": round(float(a.mean() - b.mean()), 4),
            "p_value": round(float(p), 4),
            "significant_5pct": bool(p < 0.05),
        })

    out = pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)
    out.to_csv(res_dir / "school_features_ablation_2022.csv", index=False)
    print("\n=== School-context ablation (AUC, paired Nadeau-Bengio test) ===")
    print(out.to_string(index=False))

    best = out.iloc[0]
    print(f"\nLargest lift: {best['model']} {best['auc_base']:.4f} -> "
          f"{best['auc_school']:.4f} (delta {best['delta']:+.4f}, p={best['p_value']:.3f})")
    any_sig = out["significant_5pct"].any()
    print("Verdict:", "school context significantly helps at least one model."
          if any_sig else "no significant AUC lift — ceiling holds.")
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
