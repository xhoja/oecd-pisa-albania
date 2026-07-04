"""
Confirm the fold-safe SchoolMeansTransformer matches full-cohort production means.

Production (`engineer.add_school_aggregates`) computes leave-one-out school means
once over the whole cohort; `SchoolMeansTransformer` recomputes them per training
fold. If the full-cohort means were leaking, the fold-safe pipeline would score
lower. We run both on the SAME 5x4 CV folds (per-fold AUCs paired) and report the
delta + a Nadeau-Bengio corrected resampled t-test. A delta ~0 (n.s.) confirms
the production means are empirically leakage-free - the conclusion the one-off
`run_school_features_foldsafe.py` reached, now via the reusable transformer.

LightGBM imported before scikit-learn (macOS OpenMP import-order fix).
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.features.engineer import add_school_aggregates
from src.features.transformers import EngineeredFeatureBuilder, SchoolMeansTransformer
from src.features.target import add_all_targets, THRESHOLDS
from src.models.evaluate import corrected_resampled_ttest, fit_with_sample_weight
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.utils.reproducibility import set_seeds

BASE_FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
              "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
AGG_COLS = ["ESCS", "HOMEPOS", "ANXMAT", "TEACHSUP"]
KEYS = ["CNTSCHID", "W_FSTUWT"]
MODELS = ["gradient_boosting", "lightgbm", "catboost"]
OUTER_FOLDS, OUTER_REPEATS = 5, 4


def _static_pipe(model):
    # production path: full-cohort SCH_MEAN_* already in X -> engineer -> impute
    return Pipeline([("engineer", EngineeredFeatureBuilder()),
                     ("imputer", SimpleImputer(strategy="median")),
                     ("model", model)])


def _foldsafe_pipe(model):
    # per-fold path: transformer recomputes SCH_MEAN_* on train fold only
    return Pipeline([("schoolmeans", SchoolMeansTransformer(cols=AGG_COLS)),
                     ("engineer", EngineeredFeatureBuilder()),
                     ("imputer", SimpleImputer(strategy="median")),
                     ("model", model)])


def _auc(pipe, Xtr, ytr, wtr, Xte, yte, wte) -> float:
    fit_with_sample_weight(pipe, Xtr, ytr, wtr)
    return float(roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1], sample_weight=wte))


def main() -> None:
    configure_logging(level="WARNING")
    res_dir = ROOT / "outputs/results"
    res_dir.mkdir(parents=True, exist_ok=True)

    raw = add_all_targets(pd.read_parquet(ROOT / "data/processed/alb_2022.parquet"))
    raw = raw[raw["MATH_PV_MEAN"].notna()].reset_index(drop=True)
    y = (raw["MATH_PV_MEAN"] < THRESHOLDS["math"]).astype(int).values
    w = raw["W_FSTUWT"].astype(float).values

    # static full-cohort composition (production) computed ONCE
    static_full = add_school_aggregates(raw, cols=AGG_COLS)
    X_static = static_full[BASE_FEATS + [f"SCH_MEAN_{c}" for c in AGG_COLS] + ["SCH_N"]].copy()
    # fold-safe input keeps the keys so the transformer can recompute per fold
    X_fs = raw[BASE_FEATS + KEYS].copy()

    cv = RepeatedStratifiedKFold(n_splits=OUTER_FOLDS, n_repeats=OUTER_REPEATS, random_state=42)
    fold = {m: {"static": [], "foldsafe": []} for m in MODELS}

    set_seeds(42)
    with threadpool_limits(limits=1):
        for k, (tr, te) in enumerate(cv.split(raw, y)):
            if len(np.unique(y[te])) < 2:
                continue
            for m in MODELS:
                fold[m]["static"].append(_auc(_static_pipe(get_model(m)),
                    X_static.iloc[tr], y[tr], w[tr], X_static.iloc[te], y[te], w[te]))
                fold[m]["foldsafe"].append(_auc(_foldsafe_pipe(get_model(m)),
                    X_fs.iloc[tr], y[tr], w[tr], X_fs.iloc[te], y[te], w[te]))
            print(f"  fold {k+1}/{OUTER_FOLDS*OUTER_REPEATS} done", flush=True)

    ratio = 1.0 / (OUTER_FOLDS - 1)
    rows = []
    for m in MODELS:
        s = np.asarray(fold[m]["static"]); f = np.asarray(fold[m]["foldsafe"])
        _, _, p = corrected_resampled_ttest(f - s, ratio)
        rows.append({"model": m, "auc_static_fullcohort": round(float(s.mean()), 4),
                     "auc_foldsafe_transformer": round(float(f.mean()), 4),
                     "delta": round(float(f.mean() - s.mean()), 4),
                     "p_value": round(float(p), 4)})
    out = pd.DataFrame(rows)
    out.to_csv(res_dir / "school_means_transformer_check_2022.csv", index=False)
    print("\n=== Fold-safe transformer vs full-cohort production means (AUC) ===")
    print(out.to_string(index=False))
    leak_free = (out["delta"].abs().max() < 0.01) and (out["p_value"].min() > 0.05)
    print("\nVerdict:", "full-cohort means are leakage-free (fold-safe delta ~0, n.s.)."
          if leak_free else "fold-safe recomputation moves the score - investigate.")
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
