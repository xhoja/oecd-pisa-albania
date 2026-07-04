"""
Actionable counterfactual recourse for at-risk Albanian students (2022).

For every student the CatBoost school-context screener flags at the operating
threshold, we find the smallest realistic change to *actionable* levers - lower
math anxiety, more teacher support / belonging, more ICT access - that would pull
predicted risk below the threshold. Immutable attributes (gender, immigrant
status, parental education, past repetition) are held fixed. This reframes the
model from a label into a menu of interventions.

Post-hoc explanation, so features are median-imputed up front for a well-defined
starting point; the CatBoost model has no OpenMP/libomp issue.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.explainability.counterfactuals import DEFAULT_ACTIONABLE, batch_counterfactuals
from src.models.evaluate import fit_with_sample_weight, tune_threshold
from src.models.experiment import _wrap
from src.models.prepare import build_model_data
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.utils.reproducibility import set_seeds

BASE_FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
              "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
SEED = 42
MAX_STUDENTS = 400   # cap the greedy search for runtime


def main() -> None:
    configure_logging(level="WARNING")
    res_dir = ROOT / "outputs/results"
    res_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    data = build_model_data(df, BASE_FEATS, domain="math", add_school_context=True)
    X = data.X.reset_index(drop=True)
    X = X.fillna(X.median(numeric_only=True))     # well-defined recourse start point
    y = data.y.reset_index(drop=True).values
    w = data.weights.reset_index(drop=True).values

    set_seeds(SEED)
    with threadpool_limits(limits=1):
        pipe = _wrap(get_model("catboost"))
        fit_with_sample_weight(pipe, X, pd.Series(y), w)
        proba = pipe.predict_proba(X)[:, 1]

    # operating threshold: weighted F1-macro tuned (matches the project's screener)
    thr = tune_threshold(y, proba, sample_weight=w, metric="f1_macro")
    at_risk = np.where(proba >= thr)[0]
    print(f"n={len(y)}  threshold={thr:.3f}  flagged at-risk={len(at_risk)}")

    stats = pd.DataFrame({
        "std": X.std(), "min": X.min(), "max": X.max(),
    })
    sel = at_risk[:MAX_STUDENTS]
    out = batch_counterfactuals(pipe, X.iloc[sel], stats, threshold=thr,
                                actionable=DEFAULT_ACTIONABLE, step_sd=0.25, max_sd=3.0)
    out.insert(0, "student_idx", sel)
    out.to_csv(res_dir / "counterfactuals_2022.csv", index=False)

    succ = out[out.success]
    print(f"\nRecourse found for {len(succ)}/{len(out)} flagged students "
          f"({100*len(succ)/max(len(out),1):.0f}%)")
    if len(succ):
        print(f"Median features changed: {succ.n_features_changed.median():.0f}  "
              f"| median total cost: {succ.cost_sd.median():.2f} SD")
        # which levers appear most often in recourse
        lever = Counter()
        for s in succ.recourse:
            for part in s.split(";"):
                tok = part.strip().split(" ")
                if tok and tok[0] in DEFAULT_ACTIONABLE:
                    lever[tok[0]] += 1
        print("Most-used levers:", dict(lever.most_common()))
        print("\nExample recourses:")
        print(succ[["student_idx", "p_original", "p_counterfactual", "recourse"]].head(6).to_string(index=False))
    # persist a small summary row
    pd.DataFrame([{
        "threshold": round(float(thr), 4),
        "n_flagged": int(len(out)),
        "recourse_rate": round(len(succ) / max(len(out), 1), 4),
        "median_features_changed": float(succ.n_features_changed.median()) if len(succ) else np.nan,
        "median_cost_sd": round(float(succ.cost_sd.median()), 3) if len(succ) else np.nan,
    }]).to_csv(res_dir / "counterfactuals_summary_2022.csv", index=False)
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
