"""
Difference-in-differences for the 2019 Durres earthquake natural experiment.

Design
------
* **Treated** : schools in the Center region band (Durres + Tirana counties,
  the declared state-of-emergency area near the M6.4 epicentre).
* **Control** : schools in the North and South bands.
* **Pre**     : PISA 2018 (fielded before the Nov-2019 quake).
* **Post**    : PISA 2022 (fielded after; same 15-year-olds were ~11-13 at the
  time of the quake and lived the disruption through lower-secondary school).
* **Outcome** : weighted share below PISA Level-2 in mathematics (at-risk).

DiD estimate
------------
    DiD = (Y_treated,post - Y_treated,pre) - (Y_control,post - Y_control,pre)

interpreted as the causal effect of quake exposure on the at-risk rate under the
parallel-trends assumption. That assumption is probed two ways:
    * a placebo DiD on the pre-quake window 2015 -> 2018 (should be ~0), and
    * an event-study plot of band at-risk rates across 2015/2018/2022.

Standard errors are PISA design-based: Balanced Repeated Replication (80 Fay
replicate weights) for sampling variance, combined with Rubin's rules over the
10 plausible values for imputation variance -- the same machinery as
``src.data.weights.pv_statistic_brr``, applied to the DiD contrast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from src.data.weights import (
    FAYS_ADJUSTMENT,
    replicate_weight_cols,
    weighted_mean,
)
from src.features.target import THRESHOLDS, _pv_columns

logger = structlog.get_logger(__name__)


def _cell_means_did(
    ind: pd.Series,
    w: pd.Series,
    treated: pd.Series,
    post: pd.Series,
) -> dict[str, float]:
    """Weighted 2x2 cell means and the DiD contrast for a binary indicator.

    NaN indicator rows are dropped by ``weighted_mean`` (it excludes them),
    which is what we want for at-risk indicators built from missing PVs.
    """
    def cell(t: int, p: int) -> float:
        mask = (treated == t) & (post == p)
        return weighted_mean(ind[mask], w[mask])

    t_pre, t_post = cell(1, 0), cell(1, 1)
    c_pre, c_post = cell(0, 0), cell(0, 1)
    did = (t_post - t_pre) - (c_post - c_pre)
    return {
        "treated_pre": t_pre,
        "treated_post": t_post,
        "control_pre": c_pre,
        "control_post": c_post,
        "treated_change": t_post - t_pre,
        "control_change": c_post - c_pre,
        "did": did,
    }


def did_brr_rubin(
    df: pd.DataFrame,
    domain: str = "math",
    pre: int = 2018,
    post: int = 2022,
    treated_band: str = "Center",
    weight_col: str = "W_FSTUWT",
    threshold: float | None = None,
) -> dict[str, float]:
    """DiD on the at-risk rate with a PISA design-based standard error.

    Requires ``df`` to already carry a ``region`` band column (see
    ``src.causal.region.add_region_band``), ``CYCLE`` and the plausible-value
    and replicate-weight columns. ``df`` may span more than the two cycles; it
    is filtered to {pre, post} internally.

    Variance combines BRR sampling variance (per PV) with Rubin between-PV
    variance:  T = Ubar + (1 + 1/m) * B, evaluated on the DiD contrast itself.
    """
    thr = threshold if threshold is not None else THRESHOLDS[domain]

    sub = df[df["CYCLE"].isin([pre, post])].copy()
    sub = sub[sub["region"].notna()]
    treated = (sub["region"] == treated_band).astype(int)
    post_ind = (sub["CYCLE"] == post).astype(int)

    pv_cols = [p for p in _pv_columns(sub, domain) if sub[p].notna().any()]
    if not pv_cols:
        raise ValueError(f"no plausible-value columns for domain={domain}")
    rep_cols = replicate_weight_cols(sub)

    def indicator(pv: str) -> pd.Series:
        col = sub[pv]
        ind = (col < thr).astype(float)
        ind[col.isna()] = np.nan
        return ind

    per_pv_did: list[float] = []
    per_pv_var: list[float] = []
    per_pv_cells: list[dict[str, float]] = []
    base_w = sub[weight_col]
    for pv in pv_cols:
        ind = indicator(pv)
        cells = _cell_means_did(ind, base_w, treated, post_ind)
        per_pv_did.append(cells["did"])
        per_pv_cells.append(cells)
        if rep_cols:
            rep_did = [
                _cell_means_did(ind, sub[rc], treated, post_ind)["did"]
                for rc in rep_cols
            ]
            u = FAYS_ADJUSTMENT * float(
                np.nanmean([(r - cells["did"]) ** 2 for r in rep_did])
            )
        else:
            u = np.nan
        per_pv_var.append(u)

    m = len(per_pv_did)
    did = float(np.mean(per_pv_did))
    B = float(np.var(per_pv_did, ddof=1)) if m > 1 else 0.0
    if rep_cols:
        U_bar = float(np.mean(per_pv_var))
        T = U_bar + (1 + 1 / m) * B
        brr_available = True
    else:
        U_bar = np.nan
        T = (1 + 1 / m) * B
        brr_available = False
    se = float(np.sqrt(T)) if T > 0 else 0.0
    z = did / se if se > 0 else np.nan
    # two-sided normal p-value
    from math import erfc, sqrt
    p = float(erfc(abs(z) / sqrt(2))) if se > 0 else np.nan

    # average the cell means across PVs for reporting
    cell_keys = ["treated_pre", "treated_post", "control_pre", "control_post",
                 "treated_change", "control_change"]
    cells_avg = {k: float(np.mean([c[k] for c in per_pv_cells])) for k in cell_keys}

    n_treated = int(treated.sum())
    n_control = int((treated == 0).sum())
    result = {
        "did": did,
        "se": se,
        "ci95_low": did - 1.96 * se,
        "ci95_high": did + 1.96 * se,
        "z": z,
        "p_value": p,
        "sampling_var": U_bar,
        "imputation_var": B,
        "fmi": (((1 + 1 / m) * B / T) if (brr_available and T > 0) else np.nan),
        "n_pv": m,
        "n_replicates": len(rep_cols),
        "brr_available": brr_available,
        "pre": pre,
        "post": post,
        "treated_band": treated_band,
        "n_treated": n_treated,
        "n_control": n_control,
        **cells_avg,
    }
    logger.info(
        "DiD estimated",
        pre=pre, post=post, band=treated_band,
        did=round(did, 4), se=round(se, 4), p=round(p, 4) if se > 0 else None,
    )
    return result


def _cell_means_ddd(
    ind: pd.Series,
    w: pd.Series,
    treated: pd.Series,
    sub: pd.Series,
    post: pd.Series,
) -> dict[str, float]:
    """Weighted 8-cell triple-difference (band x subgroup x period).

    The subgroup ``sub`` is urbanicity (1=Urban). The DDD asks whether the
    *urban minus rural* gap moved differently in the treated (Center) band than
    in the control bands from pre to post -- differencing out any Center-wide
    (capital-region) trend that shifts urban and rural together.

        DDD = [ (C_U_post - C_U_pre) - (C_R_post - C_R_pre) ]
            - [ (K_U_post - K_U_pre) - (K_R_post - K_R_pre) ]
    """
    def cell(t: int, s: int, p: int) -> float:
        mask = (treated == t) & (sub == s) & (post == p)
        return weighted_mean(ind[mask], w[mask])

    center_gap_change = (cell(1, 1, 1) - cell(1, 1, 0)) - (cell(1, 0, 1) - cell(1, 0, 0))
    control_gap_change = (cell(0, 1, 1) - cell(0, 1, 0)) - (cell(0, 0, 1) - cell(0, 0, 0))
    return {
        "center_urban_rural_gap_change": center_gap_change,
        "control_urban_rural_gap_change": control_gap_change,
        "ddd": center_gap_change - control_gap_change,
    }


def ddd_brr_rubin(
    df: pd.DataFrame,
    domain: str = "math",
    pre: int = 2018,
    post: int = 2022,
    treated_band: str = "Center",
    weight_col: str = "W_FSTUWT",
    threshold: float | None = None,
) -> dict[str, float]:
    """Triple-difference (band x urbanicity x period) with design-based SE.

    This is the sharper identification test: the Nov-2019 quake damage was
    concentrated in the *urban* cores of the Center band (Durres and Tirana
    cities). A genuine quake effect should raise the urban-minus-rural at-risk
    gap in Center more than in the control bands. Differencing on urbanicity
    removes any capital-region-wide secular trend -- the confound that makes the
    plain DiD fail its placebo test.

    Requires ``region`` and ``urbanicity`` band columns (see
    ``src.causal.region.add_region_band``). SE is BRR (per PV) + Rubin over PVs,
    identical machinery to :func:`did_brr_rubin`.
    """
    thr = threshold if threshold is not None else THRESHOLDS[domain]

    sub_df = df[df["CYCLE"].isin([pre, post])].copy()
    sub_df = sub_df[sub_df["region"].notna() & sub_df["urbanicity"].notna()]
    treated = (sub_df["region"] == treated_band).astype(int)
    urban = (sub_df["urbanicity"] == "Urban").astype(int)
    post_ind = (sub_df["CYCLE"] == post).astype(int)

    pv_cols = [p for p in _pv_columns(sub_df, domain) if sub_df[p].notna().any()]
    if not pv_cols:
        raise ValueError(f"no plausible-value columns for domain={domain}")
    rep_cols = replicate_weight_cols(sub_df)

    def indicator(pv: str) -> pd.Series:
        col = sub_df[pv]
        ind = (col < thr).astype(float)
        ind[col.isna()] = np.nan
        return ind

    per_pv_ddd: list[float] = []
    per_pv_var: list[float] = []
    per_pv_parts: list[dict[str, float]] = []
    base_w = sub_df[weight_col]
    for pv in pv_cols:
        ind = indicator(pv)
        parts = _cell_means_ddd(ind, base_w, treated, urban, post_ind)
        per_pv_ddd.append(parts["ddd"])
        per_pv_parts.append(parts)
        if rep_cols:
            rep_ddd = [
                _cell_means_ddd(ind, sub_df[rc], treated, urban, post_ind)["ddd"]
                for rc in rep_cols
            ]
            u = FAYS_ADJUSTMENT * float(
                np.nanmean([(r - parts["ddd"]) ** 2 for r in rep_ddd])
            )
        else:
            u = np.nan
        per_pv_var.append(u)

    m = len(per_pv_ddd)
    ddd = float(np.mean(per_pv_ddd))
    B = float(np.var(per_pv_ddd, ddof=1)) if m > 1 else 0.0
    if rep_cols:
        U_bar = float(np.mean(per_pv_var))
        T = U_bar + (1 + 1 / m) * B
        brr_available = True
    else:
        U_bar = np.nan
        T = (1 + 1 / m) * B
        brr_available = False
    se = float(np.sqrt(T)) if T > 0 else 0.0
    z = ddd / se if se > 0 else np.nan
    from math import erfc, sqrt
    p = float(erfc(abs(z) / sqrt(2))) if se > 0 else np.nan

    keys = ["center_urban_rural_gap_change", "control_urban_rural_gap_change"]
    parts_avg = {k: float(np.mean([c[k] for c in per_pv_parts])) for k in keys}

    result = {
        "ddd": ddd,
        "se": se,
        "ci95_low": ddd - 1.96 * se,
        "ci95_high": ddd + 1.96 * se,
        "z": z,
        "p_value": p,
        "sampling_var": U_bar,
        "imputation_var": B,
        "fmi": (((1 + 1 / m) * B / T) if (brr_available and T > 0) else np.nan),
        "n_pv": m,
        "n_replicates": len(rep_cols),
        "brr_available": brr_available,
        "pre": pre,
        "post": post,
        "treated_band": treated_band,
        **parts_avg,
    }
    logger.info(
        "DDD estimated",
        pre=pre, post=post, band=treated_band,
        ddd=round(ddd, 4), se=round(se, 4), p=round(p, 4) if se > 0 else None,
    )
    return result


def band_rate_by_cycle(
    df: pd.DataFrame,
    domain: str = "math",
    cycles: tuple[int, ...] = (2015, 2018, 2022),
    weight_col: str = "W_FSTUWT",
) -> pd.DataFrame:
    """At-risk rate + design-based SE for every (cycle, region band).

    Feeds the event-study / parallel-trends plot. Reuses ``pv_statistic_brr``
    per slice so each estimate carries the full BRR+Rubin SE.
    """
    from src.data.weights import pv_statistic_brr

    rows = []
    for cyc in cycles:
        for band in ("North", "Center", "South"):
            slice_ = df[(df["CYCLE"] == cyc) & (df["region"] == band)]
            if slice_.empty:
                continue
            stat = pv_statistic_brr(slice_, domain=domain, statistic="at_risk",
                                    weight_col=weight_col)
            rows.append({
                "CYCLE": cyc,
                "region": band,
                "treated": band == "Center",
                "at_risk": stat["estimate"],
                "se": stat["se"],
                "ci95_low": stat["ci95_low"],
                "ci95_high": stat["ci95_high"],
                "n": len(slice_),
            })
    return pd.DataFrame(rows)


def did_regression(
    df: pd.DataFrame,
    domain: str = "math",
    pre: int = 2018,
    post: int = 2022,
    treated_band: str = "Center",
    weight_col: str = "W_FSTUWT",
    cluster_col: str = "CNTSCHID",
    threshold: float | None = None,
) -> dict[str, float]:
    """Cross-check: weighted linear-probability DiD with school-clustered SE.

    Uses the point at-risk indicator (mean of PVs) and statsmodels WLS with
    cluster-robust (by school) covariance. This is a robustness complement to
    the design-based ``did_brr_rubin`` -- it clusters at the sampling PSU rather
    than using replicate weights, and returns the interaction coefficient.
    """
    import statsmodels.formula.api as smf

    thr = threshold if threshold is not None else THRESHOLDS[domain]

    sub = df[df["CYCLE"].isin([pre, post])].copy()
    sub = sub[sub["region"].notna()]
    pv_cols = [p for p in _pv_columns(sub, domain) if sub[p].notna().any()]
    sub["_atrisk"] = (sub[pv_cols].mean(axis=1) < thr).astype(float)
    sub = sub.dropna(subset=["_atrisk", weight_col, cluster_col])
    sub["_treated"] = (sub["region"] == treated_band).astype(int)
    sub["_post"] = (sub["CYCLE"] == post).astype(int)

    model = smf.wls(
        "_atrisk ~ _treated * _post",
        data=sub,
        weights=sub[weight_col],
    ).fit(cov_type="cluster", cov_kwds={"groups": sub[cluster_col]})

    coef = "_treated:_post"
    return {
        "did": float(model.params[coef]),
        "se": float(model.bse[coef]),
        "p_value": float(model.pvalues[coef]),
        "ci95_low": float(model.conf_int().loc[coef, 0]),
        "ci95_high": float(model.conf_int().loc[coef, 1]),
        "n": int(model.nobs),
        "n_clusters": int(sub[cluster_col].nunique()),
        "method": "WLS-LPM cluster-robust(school)",
    }
