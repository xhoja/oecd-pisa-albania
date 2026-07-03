"""
Tier-1 lever #2: does **plausible-value stacking** beat the PV-mean target?

The comparison trains on a single point target (at-risk = PV-*mean* < L2). PISA's
10 plausible values are 10 draws from each student's proficiency posterior;
collapsing them to the mean throws away that spread. PV-stacking instead trains on
all 10 — each student contributes 10 rows, row k labelled from PVk — so the model
sees the imputation uncertainty as label noise (a PV-aware, Rubin-flavoured fit).

Protocol (leakage-safe, weighted): StratifiedKFold over students; each fold's
pipeline (engineered features + booster) is fit on the training students only.
- baseline arm: 1 row/student, target = PV-mean < L2.
- stacked arm : 10 rows/student (train only), target = PVk < L2.
Both arms are evaluated on the SAME held-out students against the PV-mean target,
so AUCs are directly comparable and paired by fold for the Nadeau-Bengio test.
On top of the school-context feature set (the current headline).

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

from src.features.target import THRESHOLDS
from src.models.evaluate import corrected_resampled_ttest, fit_with_sample_weight
from src.models.experiment import _wrap
from src.models.prepare import build_model_data
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.utils.reproducibility import set_seeds

BASE_FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
              "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
MODELS = ["logistic_regression", "gradient_boosting", "lightgbm", "catboost"]
OUTER_FOLDS, OUTER_REPEATS = 5, 2
N_PV = 10


def main() -> None:
    configure_logging(level="WARNING")
    res_dir = ROOT / "outputs/results"
    res_dir.mkdir(parents=True, exist_ok=True)
    thr = THRESHOLDS["math"]

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    data = build_model_data(df, BASE_FEATS, domain="math", add_school_context=True)
    X = data.X.reset_index(drop=True)
    w = data.weights.reset_index(drop=True).values
    ymean = data.y.reset_index(drop=True).values  # PV-mean target (eval + baseline)
    # per-PV binary targets aligned to X's rows
    pv_cols = [f"PV{k}MATH" for k in range(1, N_PV + 1)]
    pv = df.loc[data.X.index, pv_cols].reset_index(drop=True)
    y_pv = (pv.values < thr).astype(int)  # (n, 10)
    print(f"n={len(X)} feats={X.shape[1]} at-risk(mean)={ymean.mean():.3f} "
          f"at-risk(PV avg)={y_pv.mean():.3f}\n")

    cv = RepeatedStratifiedKFold(n_splits=OUTER_FOLDS, n_repeats=OUTER_REPEATS, random_state=42)
    fold_auc = {m: {"base": [], "stack": []} for m in MODELS}

    set_seeds(42)
    with threadpool_limits(limits=1):
        for k, (tr, te) in enumerate(cv.split(X, ymean)):
            Xte, yte, wte = X.iloc[te], ymean[te], w[te]
            if len(np.unique(yte)) < 2:
                continue
            Xtr, wtr = X.iloc[tr], w[tr]
            # stacked training frame: 10 copies of train rows, target = PVk
            Xtr_stack = pd.concat([Xtr] * N_PV, ignore_index=True)
            ytr_stack = np.concatenate([y_pv[tr, j] for j in range(N_PV)])
            wtr_stack = np.tile(wtr, N_PV)
            for m in MODELS:
                pb = _wrap(get_model(m))
                fit_with_sample_weight(pb, Xtr, pd.Series(ymean[tr]), wtr)
                fold_auc[m]["base"].append(
                    roc_auc_score(yte, pb.predict_proba(Xte)[:, 1], sample_weight=wte))
                ps = _wrap(get_model(m))
                fit_with_sample_weight(ps, Xtr_stack, pd.Series(ytr_stack), wtr_stack)
                fold_auc[m]["stack"].append(
                    roc_auc_score(yte, ps.predict_proba(Xte)[:, 1], sample_weight=wte))
            print(f"  fold {k+1}/{OUTER_FOLDS*OUTER_REPEATS} done", flush=True)

    ratio = 1.0 / (OUTER_FOLDS - 1)
    rows = []
    for m in MODELS:
        b = np.asarray(fold_auc[m]["base"]); a = np.asarray(fold_auc[m]["stack"])
        _, _, p = corrected_resampled_ttest(a - b, ratio)
        rows.append({"model": m, "auc_pvmean": round(float(b.mean()), 4),
                     "auc_pvstack": round(float(a.mean()), 4),
                     "delta": round(float(a.mean() - b.mean()), 4),
                     "p_value": round(float(p), 4),
                     "significant_5pct": bool(p < 0.05)})
    out = pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)
    out.to_csv(res_dir / "pv_stacking_ablation_2022.csv", index=False)
    print("\n=== PV-stacking vs PV-mean (school-context model, AUC, paired NB) ===")
    print(out.to_string(index=False))
    sig = out[out["significant_5pct"] & (out["delta"] > 0)]
    print("\nVerdict:", f"PV-stacking significantly helps: {list(sig['model'])}."
          if len(sig) else "no significant gain from PV-stacking — PV-mean target is sufficient.")
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
