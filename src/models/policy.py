r"""
Population-level policy simulation on the fitted risk model.

Counterfactual recourse (``explainability.counterfactuals``) acts one student at
a time; a *policy* acts on the whole population. Here we ask the fitted screener a
what-if: if an intervention shifted the distribution of some lever across all
students - e.g. "cut math anxiety by half a standard deviation nationally", "raise
teacher support", "lift the poorest schools' composition" - what happens to the
**weighted (population) share of students predicted at risk**?

An intervention is a set of feature shifts applied to every row:

- ``("sd", d)``  → add ``d`` training-SDs to the feature (signed);
- ``("abs", d)`` → add ``d`` in the feature's own units;
- ``("set", v)`` → clamp the feature to value ``v``.

Shifts are clipped to the observed feature range so scenarios stay in-support. The
model's engineered interactions propagate the change automatically (lowering
ANXMAT also moves ANXMAT-interaction terms), so second-order effects are included.

**Caveat (important, stated not hidden):** this is a *predictive* what-if under
the model's learned associations, **not** a causal effect estimate. It answers
"how would the model reclassify a population that looked like this", which is a
useful stress-test and screening-load projection, but shifting a proxy is not the
same as the real-world intervention. Read alongside the multilevel/causal caveats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _apply(X: pd.DataFrame, stats: pd.DataFrame, interventions: dict[str, tuple[str, float]]) -> pd.DataFrame:
    Xn = X.copy()
    for feat, (kind, val) in interventions.items():
        if feat not in Xn.columns:
            continue
        if kind == "sd":
            sd = float(stats.loc[feat, "std"]) if feat in stats.index else float(Xn[feat].std()) or 1.0
            Xn[feat] = Xn[feat] + val * sd
        elif kind == "abs":
            Xn[feat] = Xn[feat] + val
        elif kind == "set":
            Xn[feat] = val
        else:
            raise ValueError(f"Unknown intervention kind {kind!r} for {feat!r}")
        if feat in stats.index:
            Xn[feat] = Xn[feat].clip(stats.loc[feat, "min"], stats.loc[feat, "max"])
    return Xn


def predicted_at_risk_rate(model, X: pd.DataFrame, weights: np.ndarray, threshold: float) -> dict:
    """Weighted share predicted at risk and mean predicted probability for ``X``."""
    p = model.predict_proba(X)[:, 1]
    w = np.asarray(weights, dtype=float)
    wsum = w.sum()
    return {
        "at_risk_rate": float((w * (p >= threshold)).sum() / wsum),
        "mean_prob": float((w * p).sum() / wsum),
    }


def simulate_policy(
    model,
    X: pd.DataFrame,
    weights: np.ndarray,
    threshold: float,
    interventions: dict[str, tuple[str, float]],
    stats: pd.DataFrame | None = None,
) -> dict:
    """
    Apply one intervention set and return baseline vs post rates plus the change.

    Returns ``at_risk_rate`` / ``mean_prob`` for baseline and post, and their
    absolute deltas (post - baseline; negative = fewer students predicted at risk).
    """
    if stats is None:
        stats = pd.DataFrame({"std": X.std(), "min": X.min(), "max": X.max()})
    base = predicted_at_risk_rate(model, X, weights, threshold)
    Xn = _apply(X, stats, interventions)
    post = predicted_at_risk_rate(model, Xn, weights, threshold)
    return {
        "baseline_at_risk_rate": base["at_risk_rate"],
        "post_at_risk_rate": post["at_risk_rate"],
        "delta_at_risk_rate": post["at_risk_rate"] - base["at_risk_rate"],
        "baseline_mean_prob": base["mean_prob"],
        "post_mean_prob": post["mean_prob"],
        "delta_mean_prob": post["mean_prob"] - base["mean_prob"],
    }


def scenario_table(
    model,
    X: pd.DataFrame,
    weights: np.ndarray,
    threshold: float,
    scenarios: dict[str, dict[str, tuple[str, float]]],
    stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Run several named scenarios and return a tidy comparison table, sorted by the
    largest reduction in predicted at-risk rate.
    """
    if stats is None:
        stats = pd.DataFrame({"std": X.std(), "min": X.min(), "max": X.max()})
    rows = []
    for name, interv in scenarios.items():
        r = simulate_policy(model, X, weights, threshold, interv, stats)
        rows.append({
            "scenario": name,
            "at_risk_rate": round(r["post_at_risk_rate"], 4),
            "delta_at_risk_rate": round(r["delta_at_risk_rate"], 4),
            "delta_mean_prob": round(r["delta_mean_prob"], 4),
        })
    base = rows and simulate_policy(model, X, weights, threshold, {}, stats)["baseline_at_risk_rate"]
    out = pd.DataFrame(rows).sort_values("delta_at_risk_rate").reset_index(drop=True)
    out.attrs["baseline_at_risk_rate"] = round(float(base), 4) if rows else None
    return out
