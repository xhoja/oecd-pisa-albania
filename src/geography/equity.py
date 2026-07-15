"""
Descriptive geographic-equity statistics for Albania across PISA cycles.

Every estimate is survey-weighted and carries a PISA design-based standard
error: BRR (80 Fay replicate weights) for sampling variance, combined with
Rubin's rules over the plausible values for imputation variance -- the same
machinery as :func:`src.data.weights.pv_statistic_brr`, reused per slice.

Three products:

* :func:`atrisk_by_groups` -- weighted at-risk rate + SE for every
  (cycle x group-cell), a generalisation of
  :func:`src.causal.did.band_rate_by_cycle` to any grouping columns
  (``region``, ``urbanicity``, ``sector`` or a cross of them).
* :func:`gap_brr_rubin` -- the at-risk *gap* between two levels of one grouping
  (e.g. Rural minus Urban) within a single cycle, with a design-based SE on the
  contrast itself, so the gap can be reported with a significance test rather
  than eyeballed from two overlapping CIs.
* :func:`region_urbanicity_matrix` -- a region x urbanicity pivot of at-risk
  rates for one cycle, the input to the equity heatmap.

The grouping columns are added upstream by
:func:`src.causal.region.add_region_band` (region / urbanicity / sector), which
decodes them from the sampling STRATUM label. Rows whose stratum cannot be
decoded carry NaN group fields and are dropped here rather than miscoded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from src.data.weights import (
    FAYS_ADJUSTMENT,
    pv_statistic_brr,
    replicate_weight_cols,
    weighted_mean,
)
from src.features.target import THRESHOLDS, _pv_columns

logger = structlog.get_logger(__name__)


def atrisk_by_groups(
    df: pd.DataFrame,
    group_cols: list[str],
    domain: str = "math",
    cycles: tuple[int, ...] = (2015, 2018, 2022),
    weight_col: str = "W_FSTUWT",
    min_n: int = 30,
) -> pd.DataFrame:
    """Weighted at-risk rate + design-based SE for every (cycle x group cell).

    Args:
        df: microdata carrying ``CYCLE``, the plausible-value and replicate-weight
            columns, and every column named in ``group_cols`` (added by
            :func:`src.causal.region.add_region_band`).
        group_cols: one or more grouping columns, e.g. ``["region"]`` or
            ``["region", "urbanicity"]`` for a cross.
        cycles: cycles to report, in order.
        min_n: cells with fewer than this many students are still returned but
            flagged ``small=True`` (a thin-cell warning, not a drop) so the
            caller can suppress or annotate them.

    Returns:
        One row per (cycle x populated group cell) with columns
        ``[CYCLE, *group_cols, at_risk, se, ci95_low, ci95_high, n, small]``,
        each estimate carrying the full BRR + Rubin standard error.
    """
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise KeyError(f"microdata missing grouping columns {missing}")

    rows: list[dict] = []
    for cyc in cycles:
        cyc_df = df[df["CYCLE"] == cyc]
        # drop rows with an undecoded cell on any grouping column
        cyc_df = cyc_df.dropna(subset=group_cols)
        if cyc_df.empty:
            continue
        for cell_vals, slice_ in cyc_df.groupby(group_cols, dropna=True):
            if slice_.empty:
                continue
            stat = pv_statistic_brr(
                slice_, domain=domain, statistic="at_risk", weight_col=weight_col
            )
            cell_vals = cell_vals if isinstance(cell_vals, tuple) else (cell_vals,)
            rows.append(
                {
                    "CYCLE": cyc,
                    **dict(zip(group_cols, cell_vals)),
                    "at_risk": stat["estimate"],
                    "se": stat["se"],
                    "ci95_low": stat["ci95_low"],
                    "ci95_high": stat["ci95_high"],
                    "n": len(slice_),
                    "small": len(slice_) < min_n,
                    "brr_available": stat["brr_available"],
                }
            )
    out = pd.DataFrame(rows)
    logger.info(
        "at-risk by groups",
        group_cols=group_cols,
        cycles=list(cycles),
        n_cells=len(out),
    )
    return out


def _weighted_gap(
    ind: pd.Series,
    w: pd.Series,
    is_hi: pd.Series,
    is_lo: pd.Series,
) -> float:
    """Weighted at-risk(hi) - at-risk(lo) for one binary indicator / weight."""
    return weighted_mean(ind[is_hi], w[is_hi]) - weighted_mean(ind[is_lo], w[is_lo])


def gap_brr_rubin(
    df: pd.DataFrame,
    group_col: str,
    level_hi: str,
    level_lo: str,
    domain: str = "math",
    cycle: int | None = None,
    weight_col: str = "W_FSTUWT",
    threshold: float | None = None,
) -> dict[str, float]:
    """Design-based at-risk gap ``rate(level_hi) - rate(level_lo)`` within a cycle.

    The gap is estimated on the contrast itself so its standard error is exact
    under the design rather than an approximation from two marginal CIs. Variance
    combines BRR sampling variance (per plausible value, recomputing the contrast
    with each replicate weight) with Rubin between-PV variance:
    ``T = Ubar + (1 + 1/m) * B``. Mirrors :func:`src.causal.did.did_brr_rubin`
    but for a single-cycle two-level contrast (e.g. Rural vs Urban, North vs
    South).

    Args:
        group_col: the grouping column (e.g. ``"urbanicity"`` or ``"region"``).
        level_hi, level_lo: the two levels to contrast; the gap is hi minus lo.
        cycle: restrict to this cycle; ``None`` pools all cycles present.

    Returns:
        dict with ``gap, se, ci95_low, ci95_high, z, p_value``, the two marginal
        rates ``rate_hi``/``rate_lo``, cell sizes, and design bookkeeping.
    """
    thr = threshold if threshold is not None else THRESHOLDS[domain]

    sub = df if cycle is None else df[df["CYCLE"] == cycle]
    sub = sub[sub[group_col].isin([level_hi, level_lo])].copy()
    is_hi = (sub[group_col] == level_hi).to_numpy()
    is_lo = (sub[group_col] == level_lo).to_numpy()
    if is_hi.sum() == 0 or is_lo.sum() == 0:
        raise ValueError(
            f"empty level for {group_col}: "
            f"n({level_hi})={int(is_hi.sum())}, n({level_lo})={int(is_lo.sum())}"
        )

    pv_cols = [p for p in _pv_columns(sub, domain) if sub[p].notna().any()]
    if not pv_cols:
        raise ValueError(f"no plausible-value columns for domain={domain}")
    rep_cols = replicate_weight_cols(sub)
    base_w = sub[weight_col]
    is_hi_s = pd.Series(is_hi, index=sub.index)
    is_lo_s = pd.Series(is_lo, index=sub.index)

    def indicator(pv: str) -> pd.Series:
        col = sub[pv]
        ind = (col < thr).astype(float)
        ind[col.isna()] = np.nan
        return ind

    per_pv_gap: list[float] = []
    per_pv_var: list[float] = []
    per_pv_hi: list[float] = []
    per_pv_lo: list[float] = []
    for pv in pv_cols:
        ind = indicator(pv)
        gap = _weighted_gap(ind, base_w, is_hi_s, is_lo_s)
        per_pv_gap.append(gap)
        per_pv_hi.append(weighted_mean(ind[is_hi_s], base_w[is_hi_s]))
        per_pv_lo.append(weighted_mean(ind[is_lo_s], base_w[is_lo_s]))
        if rep_cols:
            rep_gap = [
                _weighted_gap(ind, sub[rc], is_hi_s, is_lo_s) for rc in rep_cols
            ]
            u = FAYS_ADJUSTMENT * float(
                np.nanmean([(r - gap) ** 2 for r in rep_gap])
            )
        else:
            u = np.nan
        per_pv_var.append(u)

    m = len(per_pv_gap)
    gap = float(np.mean(per_pv_gap))
    B = float(np.var(per_pv_gap, ddof=1)) if m > 1 else 0.0
    if rep_cols:
        U_bar = float(np.mean(per_pv_var))
        T = U_bar + (1 + 1 / m) * B
        brr_available = True
    else:
        U_bar = np.nan
        T = (1 + 1 / m) * B
        brr_available = False
    se = float(np.sqrt(T)) if T > 0 else 0.0
    z = gap / se if se > 0 else np.nan
    from math import erfc, sqrt

    p = float(erfc(abs(z) / sqrt(2))) if se > 0 else np.nan

    result = {
        "group_col": group_col,
        "level_hi": level_hi,
        "level_lo": level_lo,
        "cycle": cycle,
        "gap": gap,
        "se": se,
        "ci95_low": gap - 1.96 * se,
        "ci95_high": gap + 1.96 * se,
        "z": z,
        "p_value": p,
        "rate_hi": float(np.mean(per_pv_hi)),
        "rate_lo": float(np.mean(per_pv_lo)),
        "sampling_var": U_bar,
        "imputation_var": B,
        "n_pv": m,
        "n_replicates": len(rep_cols),
        "brr_available": brr_available,
        "n_hi": int(is_hi.sum()),
        "n_lo": int(is_lo.sum()),
    }
    logger.info(
        "geographic gap",
        group_col=group_col, hi=level_hi, lo=level_lo, cycle=cycle,
        gap=round(gap, 4), se=round(se, 4),
        p=round(p, 4) if se > 0 else None,
    )
    return result


def region_urbanicity_matrix(
    df: pd.DataFrame,
    cycle: int,
    domain: str = "math",
    weight_col: str = "W_FSTUWT",
    regions: tuple[str, ...] = ("North", "Center", "South"),
    urbanicities: tuple[str, ...] = ("Urban", "Rural"),
) -> pd.DataFrame:
    """Region x urbanicity at-risk-rate pivot for one cycle (heatmap input).

    Returns a DataFrame indexed by ``urbanicity`` with one column per region,
    each entry the survey-weighted at-risk rate. Empty cells are NaN. The long
    form with SEs and cell sizes is available from :func:`atrisk_by_groups`; this
    is the compact matrix the heatmap renders.
    """
    long = atrisk_by_groups(
        df, ["urbanicity", "region"], domain=domain, cycles=(cycle,),
        weight_col=weight_col,
    )
    mat = (
        long.pivot(index="urbanicity", columns="region", values="at_risk")
        .reindex(index=list(urbanicities), columns=list(regions))
    )
    return mat
