"""
Does the independent school QUESTIONNAIRE break the ~0.78 AUC ceiling?

The compositional lever (`SCH_MEAN_*`, student ESCS/etc averaged to school) took
CatBoost 0.73 -> 0.78. That ceiling could still be a *feature-set* limit: ICC ~0.31
says ~a third of risk is school-level, and student rows cannot encode a principal's
report of staff shortage, class size, resources or leadership. This script adds the
PISA school-questionnaire variables (`SCHQ_*`, joined on CNTSCHID) on top of the
headline composition set and asks whether they add signal beyond it.

Three nested feature sets share one CV loop (per-fold AUCs perfectly paired for the
Nadeau-Bengio corrected resampled t-test):
    base            student-only (13 features)
    +composition    base + SCH_MEAN_* + SCH_N        (current headline)
    +questionnaire  +composition + SCHQ_*            (the new lever)

Questionnaire vars are exogenous school constants (a real, known attribute at
prediction time), so no fold-safe recomputation is needed - they are simply
left-joined once. Composition uses the same full-cohort means as the headline set;
the fold-safe script already showed that lift is not a leakage artefact.

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from threadpoolctl import threadpool_limits

from src.features.engineer import add_school_aggregates, add_school_questionnaire
from src.features.target import add_all_targets, THRESHOLDS
from src.models.evaluate import corrected_resampled_ttest, fit_with_sample_weight
from src.models.experiment import _wrap
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.utils.reproducibility import set_seeds

BASE_FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
              "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
AGG_COLS = ["ESCS", "HOMEPOS", "ANXMAT", "TEACHSUP"]
COMP_FEATS = [f"SCH_MEAN_{c}" for c in AGG_COLS] + ["SCH_N"]
MODELS = ["logistic_regression", "gradient_boosting", "random_forest",
          "lightgbm", "catboost"]
OUTER_FOLDS, OUTER_REPEATS = 5, 4


def _fit_auc(name, Xtr, ytr, wtr, Xte, yte, wte) -> float:
    pipe = _wrap(get_model(name))
    fit_with_sample_weight(pipe, Xtr, ytr, wtr)
    prob = pipe.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, prob, sample_weight=wte))


def main() -> None:
    configure_logging(level="WARNING")
    res_dir = ROOT / "outputs/results"
    res_dir.mkdir(parents=True, exist_ok=True)

    raw = add_all_targets(pd.read_parquet(ROOT / "data/processed/alb_2022.parquet"))
    thr = THRESHOLDS["math"]
    raw = raw[raw["MATH_PV_MEAN"].notna()].reset_index(drop=True)

    # Full-cohort composition (headline set) + exogenous questionnaire join, once.
    raw = add_school_aggregates(raw, cols=AGG_COLS)
    sch = pd.read_parquet(ROOT / "data/processed/alb_2022_school.parquet")
    q_cols = [c for c in sch.columns if c != "CNTSCHID"]
    raw = add_school_questionnaire(raw, sch, cols=q_cols)
    SCHQ_FEATS = [f"SCHQ_{c}" for c in q_cols]

    y_all = (raw["MATH_PV_MEAN"] < thr).astype(int).values
    w_all = raw["W_FSTUWT"].astype(float).values

    sets = {
        "base": BASE_FEATS,
        "composition": BASE_FEATS + COMP_FEATS,
        "questionnaire": BASE_FEATS + COMP_FEATS + SCHQ_FEATS,
    }

    cv = RepeatedStratifiedKFold(n_splits=OUTER_FOLDS, n_repeats=OUTER_REPEATS, random_state=42)
    fold_auc = {m: {s: [] for s in sets} for m in MODELS}

    set_seeds(42)
    with threadpool_limits(limits=1):
        for k, (tr, te) in enumerate(cv.split(raw, y_all)):
            train, test = raw.iloc[tr], raw.iloc[te]
            ytr, yte = y_all[tr], y_all[te]
            wtr, wte = w_all[tr], w_all[te]
            if len(np.unique(yte)) < 2:
                continue
            for s, feats in sets.items():
                Xtr, Xte = train[feats].copy(), test[feats].copy()
                for m in MODELS:
                    fold_auc[m][s].append(_fit_auc(m, Xtr, ytr, wtr, Xte, yte, wte))
            print(f"  fold {k+1}/{OUTER_FOLDS*OUTER_REPEATS} done", flush=True)

    ratio = 1.0 / (OUTER_FOLDS - 1)
    rows = []
    for m in MODELS:
        b = np.asarray(fold_auc[m]["base"])
        c = np.asarray(fold_auc[m]["composition"])
        q = np.asarray(fold_auc[m]["questionnaire"])
        # incremental questionnaire effect (over the headline composition set)
        _, _, p_q = corrected_resampled_ttest(q - c, ratio)
        _, _, p_c = corrected_resampled_ttest(c - b, ratio)
        rows.append({
            "model": m,
            "auc_base": round(float(b.mean()), 4),
            "auc_composition": round(float(c.mean()), 4),
            "auc_questionnaire": round(float(q.mean()), 4),
            "delta_comp_vs_base": round(float(c.mean() - b.mean()), 4),
            "p_comp": round(float(p_c), 4),
            "delta_q_vs_comp": round(float(q.mean() - c.mean()), 4),
            "p_q": round(float(p_q), 4),
            "q_significant_5pct": bool(p_q < 0.05),
        })
    out = pd.DataFrame(rows).sort_values("auc_questionnaire", ascending=False).reset_index(drop=True)
    out.to_csv(res_dir / "school_questionnaire_ablation_2022.csv", index=False)
    print("\n=== School-questionnaire ablation (weighted AUC, paired NB test) ===")
    print(out.to_string(index=False))

    q_helps = out["q_significant_5pct"].any() and out["delta_q_vs_comp"].max() > 0.005
    best_q = out["auc_questionnaire"].max()
    print(f"\nBest questionnaire-set AUC: {best_q:.4f} (composition ceiling ~0.78)")
    print("Verdict:", (
        "school questionnaire adds significant signal beyond composition - "
        "the 0.78 ceiling was a feature-set limit."
        if q_helps else
        "school questionnaire adds no significant signal beyond composition - "
        "the 0.78 ceiling is a genuine data limit for these students."
    ))
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
