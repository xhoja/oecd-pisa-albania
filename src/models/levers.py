r"""
Malleable-lever ranking: turn the risk model into a *prescriptive* tool.

SHAP importance ranks features by how much the model *reads* them, but it mixes
two very different kinds of feature:

* **Fixed / endowment** — a student's socioeconomic status, parental education
  and occupation, home possessions, immigrant status, gender, grade. A screener
  reads these, but no education policy moves them in the short run; they describe
  *who* is at risk, not *what to do*.
* **Malleable / actionable** — math anxiety, teacher support, sense of
  belonging, grade-repetition policy, ICT access. These are the levers a school
  system can actually pull.

This module partitions the model's features into those two classes and then
ranks the malleable ones by the reduction in the population predicted at-risk
rate from a standardized "one nudge in the beneficial direction" intervention,
reusing :func:`src.models.policy.simulate_policy`. The ranking answers the
question the Ministry actually asks: *of the things we can change, which moves
the needle most, and how far can actionable levers take us before we hit the
structural (endowment) floor?*

Caveat (stated, not hidden): like all of :mod:`src.models.policy`, this is a
*predictive* what-if under the model's learned associations, not a causal effect.
Shifting a proxy is not the real-world intervention. Read alongside the
multilevel and causal caveats in the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.policy import predicted_at_risk_rate, simulate_policy

# --------------------------------------------------------------------------- #
# Feature classification. Keyed by the *base* feature name; the school-mean    #
# aggregates (SCH_MEAN_<F>) inherit their base feature's class, and the        #
# engineered interaction terms (A_x_B) are treated as derived, not levers.     #
# --------------------------------------------------------------------------- #
FIXED_FEATURES: dict[str, str] = {
    "ESCS": "family socioeconomic status (index)",
    "HOMEPOS": "home possessions",
    "HISCED": "parents' highest education",
    "HISEI": "parents' highest occupational status",
    "IMMIG": "immigrant status",
    "GENDER": "gender",
    "GRADE": "grade level (age-grade position)",
}
MALLEABLE_FEATURES: dict[str, str] = {
    "ANXMAT": "math anxiety (reducible through pedagogy)",
    "TEACHSUP": "teacher support",
    "BELONG": "sense of belonging at school",
    "REPEAT": "grade-repetition exposure (promotion policy)",
    "ICTHOME": "ICT access at home",
    "ICTSCH": "ICT access at school",
}

# The beneficial (risk-reducing) direction is discovered empirically per lever
# rather than hard-coded, but we record the expected sign for a sanity check.
EXPECTED_BENEFICIAL_SIGN: dict[str, int] = {
    "ANXMAT": -1, "TEACHSUP": +1, "BELONG": +1,
    "REPEAT": -1, "ICTHOME": +1, "ICTSCH": +1,
}

SCHOOL_MEAN_PREFIX = "SCH_MEAN_"


def _base_name(feat: str) -> str:
    """Strip the school-mean prefix to find a feature's base class."""
    if feat.startswith(SCHOOL_MEAN_PREFIX):
        return feat[len(SCHOOL_MEAN_PREFIX):]
    return feat


def classify_features(
    feature_names: list[str],
) -> dict[str, list[str]]:
    """Partition model features into ``fixed`` / ``malleable`` / ``other``.

    ``other`` collects engineered interaction terms and structural cohort
    features (e.g. ``SCH_N``, ``*_x_*``) that are neither a raw endowment nor a
    directly actionable lever. School-mean aggregates inherit their base class.
    """
    fixed, malleable, other = [], [], []
    for f in feature_names:
        base = _base_name(f)
        if "_x_" in f or f.endswith("_x_SES") or f == "SCH_N":
            other.append(f)
        elif base in FIXED_FEATURES:
            fixed.append(f)
        elif base in MALLEABLE_FEATURES:
            malleable.append(f)
        else:
            other.append(f)
    return {"fixed": fixed, "malleable": malleable, "other": other}


def rank_levers(
    model,
    X: pd.DataFrame,
    weights: np.ndarray,
    threshold: float,
    levers: list[str] | None = None,
    magnitude: float = 0.5,
    stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rank malleable levers by the population at-risk reduction they achieve.

    For each lever we apply a ``magnitude``-SD shift in *both* directions and
    keep the *beneficial* one (the shift that lowers the weighted predicted
    at-risk rate) — a "one standardized nudge in the helpful direction" — so the
    direction is discovered from the model rather than assumed. School-mean
    aggregates of the same base feature are shifted together with it, because a
    real intervention that lowers a student's math anxiety also lowers their
    school's mean.

    Returns a DataFrame sorted by the largest reduction, with columns
    ``[lever, base, description, direction, delta_at_risk_rate,
    delta_mean_prob, matches_expected_sign]``.
    """
    if stats is None:
        stats = pd.DataFrame({"std": X.std(), "min": X.min(), "max": X.max()})
    if levers is None:
        levers = classify_features(list(X.columns))["malleable"]

    # group each base lever with its school-mean twin so they move together
    base_to_cols: dict[str, list[str]] = {}
    for f in levers:
        base_to_cols.setdefault(_base_name(f), []).append(f)
    # also attach a present school-mean twin even if it was not in `levers`
    for base, cols in base_to_cols.items():
        twin = f"{SCHOOL_MEAN_PREFIX}{base}"
        if twin in X.columns and twin not in cols:
            cols.append(twin)

    base_rate = predicted_at_risk_rate(model, X, weights, threshold)["at_risk_rate"]
    rows = []
    for base, cols in base_to_cols.items():
        best = None
        for sign in (-1, +1):
            interv = {c: ("sd", sign * magnitude) for c in cols}
            r = simulate_policy(model, X, weights, threshold, interv, stats)
            # discover the beneficial direction on the *smooth* mean predicted
            # probability, not the discrete thresholded count, which is noisy
            # near the cutoff and can tie or invert for weak levers.
            if best is None or r["delta_mean_prob"] < best["delta_mean_prob"]:
                best = {**r, "sign": sign}
        rows.append({
            "lever": base,
            "cols": "+".join(cols),
            "description": MALLEABLE_FEATURES.get(base, base),
            "direction": "↓" if best["sign"] < 0 else "↑",
            "delta_at_risk_rate": round(best["delta_at_risk_rate"], 4),
            "delta_mean_prob": round(best["delta_mean_prob"], 4),
            "matches_expected_sign": (
                EXPECTED_BENEFICIAL_SIGN.get(base) == best["sign"]
                if base in EXPECTED_BENEFICIAL_SIGN else None
            ),
        })
    out = pd.DataFrame(rows).sort_values("delta_at_risk_rate").reset_index(drop=True)
    out.attrs["baseline_at_risk_rate"] = round(float(base_rate), 4)
    out.attrs["magnitude_sd"] = magnitude
    return out


def lever_ceiling(
    model,
    X: pd.DataFrame,
    weights: np.ndarray,
    threshold: float,
    magnitude: float = 0.5,
    stats: pd.DataFrame | None = None,
) -> dict:
    """How far can *actionable* levers take us vs. the *fixed* endowment floor?

    Contrasts two bundles at the same per-feature magnitude:

    * **all-malleable** — every lever nudged in its beneficial direction at once
      (the realistic best case for school policy), and
    * **all-fixed** — every endowment lifted in the risk-reducing direction (a
      *hypothetical* that no short-run policy can deliver, included only to size
      the structural share of predicted risk).

    Returns baseline, post-rate and delta for each bundle, so the paper can say
    "moving every actionable lever half an SD cuts the predicted at-risk rate
    from A to B, while the endowment gap that policy cannot touch is C".
    """
    if stats is None:
        stats = pd.DataFrame({"std": X.std(), "min": X.min(), "max": X.max()})
    cls = classify_features(list(X.columns))
    base_rate = predicted_at_risk_rate(model, X, weights, threshold)["at_risk_rate"]

    def _bundle(feats: list[str], expected: dict[str, int] | None) -> dict:
        interv = {}
        for f in feats:
            base = _base_name(f)
            if expected is not None and base in expected:
                sign = expected[base]
            else:
                # discover the beneficial direction on the smooth mean predicted
                # probability (see rank_levers), not the discrete at-risk count.
                down = simulate_policy(model, X, weights, threshold,
                                       {f: ("sd", -magnitude)}, stats)["delta_mean_prob"]
                up = simulate_policy(model, X, weights, threshold,
                                     {f: ("sd", +magnitude)}, stats)["delta_mean_prob"]
                sign = -1 if down <= up else +1
            interv[f] = ("sd", sign * magnitude)
        return simulate_policy(model, X, weights, threshold, interv, stats)

    mall = _bundle(cls["malleable"], EXPECTED_BENEFICIAL_SIGN)
    fixed = _bundle(cls["fixed"], None)
    return {
        "baseline_at_risk_rate": round(float(base_rate), 4),
        "malleable_bundle": {
            "n_features": len(cls["malleable"]),
            "post_at_risk_rate": round(mall["post_at_risk_rate"], 4),
            "delta_at_risk_rate": round(mall["delta_at_risk_rate"], 4),
        },
        "fixed_bundle": {
            "n_features": len(cls["fixed"]),
            "post_at_risk_rate": round(fixed["post_at_risk_rate"], 4),
            "delta_at_risk_rate": round(fixed["delta_at_risk_rate"], 4),
        },
        "magnitude_sd": magnitude,
    }
