"""
Population policy simulation for Albania 2022 low-proficiency risk.

Asks the fitted CatBoost school-context screener a series of national what-ifs:
if an intervention shifted a lever across every student (lower math anxiety, more
teacher support / belonging, higher SES, lifted school composition), how does the
weighted (population) predicted at-risk rate move? Interactions propagate through
the engineered features automatically.

Caveat: a *predictive* what-if under the model's associations, not a causal
effect - shifting a proxy is not the intervention. See src/models/policy.py.
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

import pandas as pd
from threadpoolctl import threadpool_limits

from src.models.evaluate import fit_with_sample_weight, tune_threshold
from src.models.experiment import _wrap
from src.models.policy import scenario_table
from src.models.prepare import build_model_data
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.utils.reproducibility import set_seeds

BASE_FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
              "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
SEED = 42

SCENARIOS = {
    "anxiety -0.5 SD": {"ANXMAT": ("sd", -0.5)},
    "anxiety -1.0 SD": {"ANXMAT": ("sd", -1.0)},
    "teacher support +1 SD": {"TEACHSUP": ("sd", +1.0)},
    "belonging +1 SD": {"BELONG": ("sd", +1.0)},
    "psychosocial bundle": {"ANXMAT": ("sd", -0.5), "TEACHSUP": ("sd", +0.5), "BELONG": ("sd", +0.5)},
    "SES +0.5 SD": {"ESCS": ("sd", +0.5), "HOMEPOS": ("sd", +0.5)},
    "school composition +0.5 SD": {"SCH_MEAN_ESCS": ("sd", +0.5), "SCH_MEAN_HOMEPOS": ("sd", +0.5)},
    "everything (bundle + SES + composition)": {
        "ANXMAT": ("sd", -0.5), "TEACHSUP": ("sd", +0.5), "BELONG": ("sd", +0.5),
        "ESCS": ("sd", +0.5), "HOMEPOS": ("sd", +0.5),
        "SCH_MEAN_ESCS": ("sd", +0.5), "SCH_MEAN_HOMEPOS": ("sd", +0.5),
    },
}


def main() -> None:
    configure_logging(level="WARNING")
    res_dir = ROOT / "outputs/results"
    res_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(ROOT / "data/processed/alb_2022.parquet")
    data = build_model_data(df, BASE_FEATS, domain="math", add_school_context=True)
    X = data.X.reset_index(drop=True)
    X = X.fillna(X.median(numeric_only=True))
    y = data.y.reset_index(drop=True).values
    w = data.weights.reset_index(drop=True).values

    set_seeds(SEED)
    with threadpool_limits(limits=1):
        pipe = _wrap(get_model("catboost"))
        fit_with_sample_weight(pipe, X, pd.Series(y), w)
        proba = pipe.predict_proba(X)[:, 1]

    thr = tune_threshold(y, proba, sample_weight=w, metric="f1_macro")
    out = scenario_table(pipe, X, w, threshold=thr, scenarios=SCENARIOS)
    base = out.attrs["baseline_at_risk_rate"]
    out.to_csv(res_dir / "policy_simulation_2022.csv", index=False)

    print(f"threshold={thr:.3f}  baseline predicted at-risk rate={base:.3f}\n")
    print("=== Policy scenarios (weighted predicted at-risk rate) ===")
    print(out.to_string(index=False))
    best = out.iloc[0]
    print(f"\nLargest single reduction: '{best.scenario}' -> "
          f"{best.delta_at_risk_rate:+.3f} ({base:.3f} -> {best.at_risk_rate:.3f})")
    print("\nSAVED OK")


if __name__ == "__main__":
    main()
