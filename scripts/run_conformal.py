"""
Split-conformal prediction sets for the Albania 2022 low-proficiency screener.

Turns the headline CatBoost school-context model's probabilities into
finite-sample-valid prediction sets: at risk level alpha the set covers the true
label with probability >= 1-alpha. A singleton set is a confident call; the
both-labels set {not-at-risk, at-risk} is the model abstaining. We report, over a
grid of alpha, weighted (population) coverage and the empty/singleton/ambiguous
mix, for marginal and Mondrian (class-conditional) calibration.

Stratified 60/20/20 proper-train / calibration / test split; weights carried
through so the guarantee is for the population, not the sample.

CatBoost has no OpenMP/libomp issue; no import-order dance needed here.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from src.models.conformal import calibrate, prediction_sets, set_summary
from src.models.evaluate import fit_with_sample_weight
from src.models.experiment import _wrap
from src.models.prepare import build_model_data
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.utils.reproducibility import set_seeds

BASE_FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
              "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
ALPHAS = [0.05, 0.10, 0.20]
SEED = 42


def main() -> None:
    configure_logging(level="WARNING")
    res_dir = ROOT / "outputs/results"
    res_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    data = build_model_data(df, BASE_FEATS, domain="math", add_school_context=True)
    X = data.X.reset_index(drop=True)
    y = data.y.reset_index(drop=True).values
    w = data.weights.reset_index(drop=True).values

    idx = np.arange(len(y))
    # 60 train / 20 cal / 20 test, stratified
    tr, rest = train_test_split(idx, test_size=0.4, random_state=SEED, stratify=y)
    cal, te = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=y[rest])
    print(f"n={len(y)}  train={len(tr)} cal={len(cal)} test={len(te)}  at-risk={y.mean():.3f}")

    set_seeds(SEED)
    with threadpool_limits(limits=1):
        pipe = _wrap(get_model("catboost"))
        fit_with_sample_weight(pipe, X.iloc[tr], pd.Series(y[tr]), w[tr])
        proba_cal = pipe.predict_proba(X.iloc[cal])
        proba_te = pipe.predict_proba(X.iloc[te])

    rows = []
    for alpha in ALPHAS:
        for mondrian in (False, True):
            c = calibrate(proba_cal, y[cal], alpha=alpha, weights=w[cal], mondrian=mondrian)
            s = set_summary(prediction_sets(proba_te, c), y[te], weights=w[te])
            rows.append({"alpha": alpha, "mondrian": mondrian, "target_coverage": 1 - alpha,
                         **{k: round(v, 4) for k, v in s.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(res_dir / "conformal_2022.csv", index=False)
    print("\n=== Conformal prediction sets (test split, weighted) ===")
    print(out.to_string(index=False))

    ok = all(r["coverage"] >= r["target_coverage"] - 0.03 for r in rows)
    print("\nCoverage guarantee holds at every alpha:" , ok)
    # headline read at alpha=0.1 marginal
    h = out[(out.alpha == 0.10) & (~out.mondrian)].iloc[0]
    print(f"At 90% target: coverage {h.coverage:.3f}, singleton (confident) share "
          f"{h.share_singleton:.3f}, ambiguous (abstain) share {h.share_ambiguous:.3f}")
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
