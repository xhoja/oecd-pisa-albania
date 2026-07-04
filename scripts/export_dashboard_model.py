"""
Export the headline risk-screener model for the interactive dashboard.

Fits the project's headline configuration - a survey-weighted CatBoost with
school socioeconomic composition - on the *full* Albania-2022 cohort, then adds
the two things a deployable screener needs on top of the CV evidence:

  1. **Post-hoc isotonic calibration.** Raw CatBoost probabilities are sharp but
     mis-calibrated (weighted ECE ~0.15, see notebook 11); a screener reports a
     *probability of risk*, so we fit an out-of-fold isotonic map that pulls the
     weighted ECE to ~0.03. The map is learned on OUT-OF-FOLD predictions so it
     is not fit on the same rows it corrects.
  2. **A decision threshold** tuned (on the calibrated OOF probabilities) to
     maximise the survey-weighted MCC, plus the base rate and SES-quintile edges
     the dashboard needs to contextualise a score.

Everything is bundled into ``outputs/models/dashboard_bundle.joblib`` (the fitted
objects) and ``outputs/models/dashboard_meta.json`` (the JSON-serialisable slider
ranges / thresholds / metrics the Streamlit app reads).

This is an *associational* screener trained on one cross-section; the same
caveats as the paper apply (non-causal, over-flags the poorest quintile). The app
surfaces them.

Run:  python scripts/export_dashboard_model.py
"""
from __future__ import annotations

# Import the booster BEFORE sklearn so its Homebrew libomp loads ahead of
# sklearn's bundled copy (see memory: OpenMP macOS import-order segfault).
import catboost  # noqa: F401

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from src.models.prepare import build_model_data
from src.models.registry import get_model
from src.models.experiment import _wrap
from src.models.evaluate import fit_with_sample_weight, expected_calibration_error

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "alb_2022.parquet"
OUT_DIR = ROOT / "outputs" / "models"

# The headline student feature set (configs/features.yaml: core_features) +
# survey-weighted school-mean composition (add_school_context=True).
CORE_FEATURES = [
    "ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
    "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI",
]

# Features the dashboard exposes as controls, with human-readable labels and
# the widget kind. Everything else in the model row (SCH_N, _MISS_ indicators)
# is derived or held at a sensible default.
CONTINUOUS_LABELS = {
    "ESCS": "Economic, social & cultural status (ESCS)",
    "HOMEPOS": "Home possessions (HOMEPOS)",
    "HISCED": "Highest parental education (HISCED)",
    "HISEI": "Highest parental occupation status (HISEI)",
    "ANXMAT": "Mathematics anxiety (ANXMAT)",
    "BELONG": "Sense of school belonging (BELONG)",
    "TEACHSUP": "Teacher support in maths (TEACHSUP)",
    "GRADE": "Grade relative to modal (GRADE)",
    "ICTHOME": "ICT resources at home (ICTHOME)",
    "ICTSCH": "ICT resources at school (ICTSCH)",
    "SCH_MEAN_ESCS": "School-mean ESCS (composition)",
    "SCH_MEAN_HOMEPOS": "School-mean home possessions",
    "SCH_MEAN_ANXMAT": "School-mean maths anxiety",
    "SCH_MEAN_TEACHSUP": "School-mean teacher support",
    "SCH_N": "Students sampled in school (SCH_N)",
}
CATEGORICAL_LABELS = {
    "GENDER": ("Gender", {0.0: "Female", 1.0: "Male"}),
    "REPEAT": ("Ever repeated a grade", {0.0: "No", 1.0: "Yes"}),
    "IMMIG": ("Immigration status", {0.0: "Native", 1.0: "Second-gen", 2.0: "First-gen"}),
}


def weighted_mcc(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """Survey-weighted Matthews correlation coefficient."""
    y, pred, w = np.asarray(y), np.asarray(pred), np.asarray(w, dtype=float)
    tp = w[(y == 1) & (pred == 1)].sum()
    tn = w[(y == 0) & (pred == 0)].sum()
    fp = w[(y == 0) & (pred == 1)].sum()
    fn = w[(y == 1) & (pred == 0)].sum()
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def wquantiles(x: pd.Series, w: pd.Series, qs) -> list[float]:
    """Weighted quantiles, NaNs dropped."""
    m = x.notna()
    xv, wv = x[m].to_numpy(float), w[m].to_numpy(float)
    order = np.argsort(xv)
    xv, wv = xv[order], wv[order]
    cw = np.cumsum(wv) - 0.5 * wv
    cw /= wv.sum()
    return [float(np.interp(q, cw, xv)) for q in qs]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED)
    print(f"Loaded {PROCESSED.name}: {df.shape}")

    md = build_model_data(df, CORE_FEATURES, domain="math", add_school_context=True)
    X, y, w = md.X, md.y.to_numpy(), md.weights.to_numpy(float)
    print(f"Model data: n={md.meta['n_samples']} features={md.meta['n_features']} "
          f"at-risk={md.meta['at_risk_rate']}")

    with threadpool_limits(limits=1):
        # --- out-of-fold raw probabilities for a leak-free calibrator ---
        oof_raw = np.full(len(y), np.nan)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, va in skf.split(X, y):
            pipe = _wrap(get_model("catboost"))
            fit_with_sample_weight(pipe, X.iloc[tr], y[tr], w[tr])
            oof_raw[va] = pipe.predict_proba(X.iloc[va])[:, 1]

        # deployed calibrator: fit on ALL out-of-fold raw probs
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(oof_raw, y, sample_weight=w)
        oof_cal = iso.predict(oof_raw)

        # --- tune the decision threshold on calibrated OOF probs (weighted MCC) ---
        grid = np.linspace(0.15, 0.85, 71)
        mccs = [weighted_mcc(y, (oof_cal >= t).astype(int), w) for t in grid]
        best_t = float(grid[int(np.argmax(mccs))])

        # HONEST calibrated ECE: never evaluate isotonic on the rows it was fit on.
        # Nested split over the OOF pairs - fit iso on train part, score the held-out
        # part - so the reported ECE is a generalisation estimate (~project's 0.02-0.03),
        # not the in-sample ~0 that fitting-and-scoring the same rows would give.
        held_cal = np.full(len(y), np.nan)
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=7).split(oof_raw.reshape(-1, 1), y):
            iso_cv = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso_cv.fit(oof_raw[tr], y[tr], sample_weight=w[tr])
            held_cal[va] = iso_cv.predict(oof_raw[va])

        ece_raw = expected_calibration_error(y, oof_raw, sample_weight=w)
        ece_cal = expected_calibration_error(y, held_cal, sample_weight=w)

        # --- final deployed model: fit on the FULL cohort, weighted ---
        final = _wrap(get_model("catboost"))
        weighted = fit_with_sample_weight(final, X, y, w)
        print(f"Final fit weighted={weighted}  ECE raw={ece_raw:.3f} cal={ece_cal:.3f} "
              f"threshold={best_t:.3f}  MCC*={max(mccs):.3f}")

    # feature order AFTER the engineer step (what CatBoost + SHAP see)
    eng = final.named_steps["engineer"].transform(X.head())
    shap_feature_names = list(eng.columns)

    # slider metadata from the real cohort (weighted quantiles)
    w_series = md.weights
    sliders = {}
    for col, label in CONTINUOUS_LABELS.items():
        if col not in X.columns:
            continue
        p01, p25, p50, p75, p99 = wquantiles(X[col], w_series, [0.01, 0.25, 0.5, 0.75, 0.99])
        sliders[col] = {
            "label": label, "kind": "continuous",
            "min": round(p01, 3), "max": round(p99, 3),
            "default": round(p50, 3), "p25": round(p25, 3), "p75": round(p75, 3),
        }
    for col, (label, mapping) in CATEGORICAL_LABELS.items():
        if col not in X.columns:
            continue
        default = float(X[col].dropna().mode().iloc[0]) if X[col].notna().any() else 0.0
        sliders[col] = {
            "label": label, "kind": "categorical",
            "options": {str(k): v for k, v in mapping.items()},
            "default": default,
        }

    escs_edges = wquantiles(X["ESCS"], w_series, [0.2, 0.4, 0.6, 0.8]) if "ESCS" in X else []

    bundle = {
        "pipeline": final,
        "isotonic": iso,
        "feature_names": list(X.columns),        # full model-input row order
        "shap_feature_names": shap_feature_names,  # post-engineer, for SHAP labels
        "threshold": best_t,
    }
    joblib.dump(bundle, OUT_DIR / "dashboard_bundle.joblib")

    meta = {
        "model": "CatBoost + survey-weighted school composition (headline)",
        "trained_on": "Albania PISA 2022 (full cohort)",
        "n_samples": md.meta["n_samples"],
        "n_features": md.meta["n_features"],
        "base_rate_at_risk": md.meta["at_risk_rate"],
        "oecd_threshold_math": 420.07,
        "decision_threshold": round(best_t, 4),
        "ece_raw": round(ece_raw, 4),
        "ece_calibrated": round(ece_cal, 4),
        "weighted_mcc_at_threshold": round(max(mccs), 4),
        "escs_quintile_edges": [round(e, 3) for e in escs_edges],
        "feature_names": list(X.columns),
        "sliders": sliders,
    }
    (OUT_DIR / "dashboard_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_DIR/'dashboard_bundle.joblib'} and dashboard_meta.json")


if __name__ == "__main__":
    main()
