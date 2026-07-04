r"""
Actionable counterfactual recourse for at-risk students.

A prediction says *who* is at risk; recourse says *what could change it*. For a
student the model flags (:math:`\hat p \ge \tau`), we search for the smallest,
**realistic** set of changes to *actionable* features that pulls the predicted
risk below the operating threshold :math:`\tau` - "if math anxiety fell by ~1 SD
and teacher support rose, this student would no longer be flagged".

Two constraints keep the recourse meaningful rather than a raw adversarial nudge:

- **Mutability.** Only actionable features may move. A school can plausibly act on
  math anxiety, sense of belonging, teacher support and ICT access; it cannot
  change a student's gender, immigrant status, parental education or past grade
  repetition. Those are held fixed.
- **Direction.** Each actionable feature may move only in its *beneficial*
  direction (lower anxiety, higher support/belonging), so recourse corresponds to
  an intervention you could actually deliver, and magnitudes are capped
  (default ±3 SD of the training distribution).

The model is a fitted probability classifier (here the CatBoost school-context
screener); the search is gradient-free (greedy coordinate descent in SD-sized
steps), so it works for any tree/black-box model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# beneficial direction per actionable feature: -1 = lowering reduces risk,
# +1 = raising reduces risk. (From the model / multilevel odds ratios: ANXMAT is
# risk-increasing; TEACHSUP, BELONG, ICT access are protective.)
DEFAULT_ACTIONABLE: dict[str, int] = {
    "ANXMAT": -1,     # lower math anxiety
    "TEACHSUP": +1,   # more teacher support
    "BELONG": +1,     # stronger school belonging
    "ICTHOME": +1,    # more ICT resources at home
    "ICTSCH": +1,     # more ICT resources at school
}


def _prob_at_risk(model, x_row: pd.DataFrame) -> float:
    return float(model.predict_proba(x_row)[:, 1][0])


def find_counterfactual(
    model,
    x: pd.Series,
    feature_stats: pd.DataFrame,
    threshold: float,
    actionable: dict[str, int] | None = None,
    max_sd: float = 3.0,
    step_sd: float = 0.25,
    max_iter: int = 200,
) -> dict:
    """
    Greedy minimal-change recourse for one student.

    Args:
        model: fitted classifier with ``predict_proba``; sees the full feature row.
        x: the student's feature row (a Series over the model's feature names).
        feature_stats: DataFrame indexed by feature with ``std``, ``min``, ``max``
            columns (training-set statistics) used for step size and clipping.
        threshold: operating decision threshold tau; recourse aims for p < tau.
        actionable: {feature: +1/-1 beneficial direction}; defaults to
            ``DEFAULT_ACTIONABLE`` intersected with available features.
        max_sd: cap on total movement per feature, in training SDs.
        step_sd: greedy step size, in training SDs.

    Returns a dict: ``success``, original/achieved probability, the per-feature
    ``changes`` (original → new, and SD magnitude), ``n_features_changed`` and
    total ``cost_sd`` (sum of |change| in SDs). If already below threshold,
    returns success with no changes.
    """
    feats = list(x.index)
    act = {f: d for f, d in (actionable or DEFAULT_ACTIONABLE).items()
           if f in feats and f in feature_stats.index}
    cols = feats

    cur = x.astype(float).copy()
    row = pd.DataFrame([cur], columns=cols)
    p0 = _prob_at_risk(model, row)
    if p0 < threshold:
        return {"success": True, "p_original": p0, "p_counterfactual": p0,
                "changes": {}, "n_features_changed": 0, "cost_sd": 0.0}

    moved = {f: 0.0 for f in act}          # SDs moved so far per feature
    for _ in range(max_iter):
        best_f, best_p, best_val = None, _prob_at_risk(model, pd.DataFrame([cur], columns=cols)), None
        for f, direction in act.items():
            sd = float(feature_stats.loc[f, "std"]) or 1.0
            if abs(moved[f]) + step_sd > max_sd:
                continue
            cand = cur.copy()
            newval = cand[f] + direction * step_sd * sd
            newval = float(np.clip(newval, feature_stats.loc[f, "min"], feature_stats.loc[f, "max"]))
            if newval == cand[f]:
                continue
            cand[f] = newval
            p = _prob_at_risk(model, pd.DataFrame([cand], columns=cols))
            if p < best_p:
                best_f, best_p, best_val = f, p, newval
        if best_f is None:
            break                          # no admissible improving move
        cur[best_f] = best_val
        moved[best_f] += step_sd
        if best_p < threshold:
            break

    p_final = _prob_at_risk(model, pd.DataFrame([cur], columns=cols))
    changes = {}
    for f in act:
        if cur[f] != x[f]:
            sd = float(feature_stats.loc[f, "std"]) or 1.0
            changes[f] = {"from": float(x[f]), "to": float(cur[f]),
                          "delta": float(cur[f] - x[f]), "delta_sd": float((cur[f] - x[f]) / sd)}
    return {
        "success": bool(p_final < threshold),
        "p_original": p0,
        "p_counterfactual": p_final,
        "changes": changes,
        "n_features_changed": len(changes),
        "cost_sd": float(sum(abs(c["delta_sd"]) for c in changes.values())),
    }


def batch_counterfactuals(
    model,
    X: pd.DataFrame,
    feature_stats: pd.DataFrame,
    threshold: float,
    actionable: dict[str, int] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Run recourse for every row of ``X`` (intended: the at-risk students). Returns
    a tidy per-student table (success, p_original, p_counterfactual,
    n_features_changed, cost_sd, and a human-readable ``recourse`` string).
    """
    recs = []
    for i in range(len(X)):
        r = find_counterfactual(model, X.iloc[i], feature_stats, threshold, actionable, **kwargs)
        desc = "; ".join(
            f"{f} {c['delta_sd']:+.2f} SD" for f, c in r["changes"].items()
        ) or "(already below threshold)"
        recs.append({
            "success": r["success"],
            "p_original": round(r["p_original"], 4),
            "p_counterfactual": round(r["p_counterfactual"], 4),
            "n_features_changed": r["n_features_changed"],
            "cost_sd": round(r["cost_sd"], 3),
            "recourse": desc,
        })
    return pd.DataFrame(recs, index=X.index)
