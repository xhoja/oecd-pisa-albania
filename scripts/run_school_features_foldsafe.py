"""
Fold-safe confirmation of the school-context AUC lift.

The quick ablation (`run_school_features_experiment.py`) computed school means once
on the full cohort — mild *covariate* leakage (a test student's ESCS informs its
schoolmates' mean seen in training). Here school means are recomputed **from the
training fold only** each split, then mapped onto the test fold (unseen schools →
global train mean; train rows use a leave-one-out mean). Baseline and augmented
share the exact same folds (one CV loop), so per-fold AUCs are perfectly paired
for the Nadeau-Bengio corrected resampled t-test.

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

from src.features.target import add_all_targets, THRESHOLDS
from src.models.evaluate import corrected_resampled_ttest, fit_with_sample_weight
from src.models.experiment import _wrap
from src.models.prepare import build_model_data
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.utils.reproducibility import set_seeds

BASE_FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
              "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
AGG_COLS = ["ESCS", "HOMEPOS", "ANXMAT", "TEACHSUP"]
SCHOOL_FEATS = [f"SCH_MEAN_{c}" for c in AGG_COLS] + ["SCH_N"]
MODELS = ["logistic_regression", "gradient_boosting", "random_forest",
          "lightgbm", "catboost"]
OUTER_FOLDS, OUTER_REPEATS = 5, 4


def _train_school_means(train: pd.DataFrame, sch: np.ndarray, w: np.ndarray):
    """Weighted per-school means (per AGG col) and global means, from TRAIN only."""
    tbl, glob = {}, {}
    sidx = pd.Series(sch)
    for col in AGG_COLS:
        x = train[col].values.astype(float)
        nn = ~np.isnan(x)
        wv = np.where(nn, w, 0.0)
        wx = np.where(nn, w * x, 0.0)
        wv_sum = pd.Series(wv).groupby(sidx).transform("sum").values
        wx_sum = pd.Series(wx).groupby(sidx).transform("sum").values
        glob[col] = float(wx[nn].sum() / wv[nn].sum()) if wv[nn].sum() > 0 else float("nan")
        # per-school mean (full, for mapping onto test) and LOO mean (for train rows)
        mean_full = np.where(wv_sum > 0, wx_sum / np.maximum(wv_sum, 1e-9), np.nan)
        den_loo = wv_sum - wv
        mean_loo = np.where(den_loo > 0, (wx_sum - wx) / np.maximum(den_loo, 1e-9), np.nan)
        # school -> full-mean lookup for test mapping
        lut = pd.Series(mean_full, index=sch).groupby(level=0).first()
        tbl[col] = {"loo_train": mean_loo, "lut": lut}
    sch_n = pd.Series(1.0, index=range(len(sch))).groupby(sidx).transform("size").values
    return tbl, glob, sch_n


def _augment(train: pd.DataFrame, test: pd.DataFrame,
             sch_tr: np.ndarray, sch_te: np.ndarray, w_tr: np.ndarray):
    tbl, glob, sch_n_tr = _train_school_means(train, sch_tr, w_tr)
    Xtr = train[BASE_FEATS].copy()
    Xte = test[BASE_FEATS].copy()
    for col in AGG_COLS:
        Xtr[f"SCH_MEAN_{col}"] = np.where(np.isnan(tbl[col]["loo_train"]),
                                          glob[col], tbl[col]["loo_train"])
        mapped = pd.Series(sch_te).map(tbl[col]["lut"]).values
        Xte[f"SCH_MEAN_{col}"] = np.where(np.isnan(mapped), glob[col], mapped)
    # school size: train counts mapped onto test (unseen -> 1)
    Xtr["SCH_N"] = sch_n_tr
    size_lut = pd.Series(sch_n_tr, index=sch_tr).groupby(level=0).first()
    Xte["SCH_N"] = pd.Series(sch_te).map(size_lut).fillna(1.0).values
    return Xtr, Xte


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
    y_all = (raw["MATH_PV_MEAN"] < thr).astype(int).values
    w_all = raw["W_FSTUWT"].astype(float).values
    sch_all = raw["CNTSCHID"].values

    cv = RepeatedStratifiedKFold(n_splits=OUTER_FOLDS, n_repeats=OUTER_REPEATS, random_state=42)
    fold_auc = {m: {"base": [], "school": []} for m in MODELS}

    set_seeds(42)
    with threadpool_limits(limits=1):
        for k, (tr, te) in enumerate(cv.split(raw, y_all)):
            train, test = raw.iloc[tr], raw.iloc[te]
            ytr, yte = y_all[tr], y_all[te]
            wtr, wte = w_all[tr], w_all[te]
            if len(np.unique(yte)) < 2:
                continue
            Xtr_b, Xte_b = train[BASE_FEATS].copy(), test[BASE_FEATS].copy()
            Xtr_a, Xte_a = _augment(train, test, sch_all[tr], sch_all[te], wtr)
            for m in MODELS:
                fold_auc[m]["base"].append(_fit_auc(m, Xtr_b, ytr, wtr, Xte_b, yte, wte))
                fold_auc[m]["school"].append(_fit_auc(m, Xtr_a, ytr, wtr, Xte_a, yte, wte))
            print(f"  fold {k+1}/{OUTER_FOLDS*OUTER_REPEATS} done", flush=True)

    ratio = 1.0 / (OUTER_FOLDS - 1)
    rows = []
    for m in MODELS:
        b = np.asarray(fold_auc[m]["base"]); a = np.asarray(fold_auc[m]["school"])
        _, _, p = corrected_resampled_ttest(a - b, ratio)
        rows.append({"model": m, "auc_base": round(float(b.mean()), 4),
                     "auc_school_foldsafe": round(float(a.mean()), 4),
                     "delta": round(float(a.mean() - b.mean()), 4),
                     "p_value": round(float(p), 4),
                     "significant_5pct": bool(p < 0.05)})
    out = pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)
    out.to_csv(res_dir / "school_features_foldsafe_2022.csv", index=False)
    print("\n=== Fold-safe school-context ablation (AUC, paired NB test) ===")
    print(out.to_string(index=False))
    print("\nVerdict:", "gain survives fold-safe recomputation."
          if out["significant_5pct"].any() and out["delta"].max() > 0.01
          else "gain shrinks under fold-safe recomputation — was partly leakage.")
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
