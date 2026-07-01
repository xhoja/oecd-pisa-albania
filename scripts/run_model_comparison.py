"""Run the Albania 2022 model comparison and save results + figures."""
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

from src.models.experiment import compare_models_cv, paired_nb_auc_test, save_results
from src.models.prepare import build_model_data
from src.utils.logging import configure_logging


def main() -> None:
    configure_logging(level="WARNING")
    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    feats = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
             "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
    data = build_model_data(df, feats, domain="math")
    print(f"N={data.meta['n_samples']} at_risk={data.meta['at_risk_rate']} feats={len(data.feature_names)}")
    print("Features:", data.feature_names)

    models = ["logistic_regression", "decision_tree", "random_forest", "extra_trees",
              "gradient_boosting", "xgboost", "lightgbm", "catboost", "knn", "naive_bayes"]
    csv_path = ROOT / "outputs/results/albania_2022_model_comparison.csv"
    res = compare_models_cv(data, models, outer_folds=5, outer_repeats=4, save_path=csv_path)
    print("\n" + res.to_string(index=False))
    save_results(res, ROOT / "outputs/results/albania_2022_model_comparison.csv")

    # Pairwise Nadeau-Bengio corrected resampled t-test: is the podium real?
    pw = paired_nb_auc_test(res.attrs["fold_scores"], res.attrs["outer_folds"])
    pw_path = ROOT / "outputs/results/albania_2022_pairwise_nb.csv"
    pw.to_csv(pw_path, index=False)
    print("\nPairwise corrected resampled t-test (AUC):")
    print(pw.to_string(index=False))
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
