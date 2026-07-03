"""
Publication-quality EDA figures for PISA analysis.

All figures use weighted statistics (PISA sampling weights) and the shared
publication style. Each function returns a matplotlib Figure for saving.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import structlog

from src.data.weights import weighted_mean, weighted_proportion, weighted_std
from src.visualization.style import (
    AT_RISK_COLORS,
    COUNTRY_COLORS,
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    apply_publication_style,
    color_list,
    save_figure,
)

logger = structlog.get_logger(__name__)

WEIGHT_COL = "W_FSTUWT"


def _w(df: pd.DataFrame) -> pd.Series:
    """Return weight series, falling back to unit weights."""
    if WEIGHT_COL in df.columns:
        return df[WEIGHT_COL]
    return pd.Series(np.ones(len(df)), index=df.index)


# ---------------------------------------------------------------------------
# A1: At-risk trajectory with 95% CI
# ---------------------------------------------------------------------------

def plot_atrisk_trajectory(
    df: pd.DataFrame,
    target_col: str = "AT_RISK_MATH",
    cycle_col: str = "CYCLE",
    score_col: str = "MATH_PV_MEAN",
) -> plt.Figure:
    """At-risk rate and mean score over cycles with 95% CI (weighted)."""
    apply_publication_style()
    cycles = sorted(df[cycle_col].dropna().unique())

    rates, rate_ci, scores, score_ci = [], [], [], []
    for c in cycles:
        s = df[df[cycle_col] == c]
        w = _w(s)
        p = weighted_proportion(s[target_col], w)
        n = len(s)
        se = np.sqrt(p * (1 - p) / n)
        rates.append(p * 100)
        rate_ci.append(1.96 * se * 100)

        ms = weighted_mean(s[score_col], w)
        sd = weighted_std(s[score_col], w)
        scores.append(ms)
        score_ci.append(1.96 * sd / np.sqrt(n))

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    ax1.errorbar(cycles, rates, yerr=rate_ci, marker="o", capsize=4,
                 color=AT_RISK_COLORS["at_risk"], label="At-risk rate (%)", linewidth=2.2)
    ax2.errorbar(cycles, scores, yerr=score_ci, marker="s", capsize=4,
                 color=AT_RISK_COLORS["proficient"], label="Mean math score",
                 linewidth=2.2, linestyle="--")

    ax1.set_xlabel("PISA Cycle")
    ax1.set_ylabel("Low-proficiency rate (%)", color=AT_RISK_COLORS["at_risk"])
    ax2.set_ylabel("Mean math score (PV)", color=AT_RISK_COLORS["proficient"])
    ax1.set_xticks(cycles)
    ax1.tick_params(axis="y", labelcolor=AT_RISK_COLORS["at_risk"])
    ax2.tick_params(axis="y", labelcolor=AT_RISK_COLORS["proficient"])
    ax1.set_title("Low-Proficiency Rate and Mean Math Score Over Time (95% CI)")
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# B1: Score distributions by cycle (ridgeline-style)
# ---------------------------------------------------------------------------

def plot_score_distributions(
    df: pd.DataFrame,
    score_col: str = "MATH_PV_MEAN",
    cycle_col: str = "CYCLE",
    threshold: float = 420.0,
) -> plt.Figure:
    """Overlaid KDE of score distributions per cycle."""
    apply_publication_style()
    cycles = sorted(df[cycle_col].dropna().unique())
    colors = color_list(len(cycles))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for c, col in zip(cycles, colors):
        s = df[df[cycle_col] == c][score_col].dropna()
        sns.kdeplot(s, ax=ax, label=str(int(c)), color=col, fill=True, alpha=0.15, linewidth=2)

    ax.axvline(threshold, color="black", linestyle=":", linewidth=1.5,
               label=f"L2 threshold ({threshold:.0f})")
    ax.set_xlabel("Math score (plausible value mean)")
    ax.set_ylabel("Density")
    ax.set_title("Math Score Distribution by PISA Cycle")
    ax.legend(title="Cycle")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# B2: Missingness heatmap (feature × cycle)
# ---------------------------------------------------------------------------

def plot_missingness_heatmap(
    df: pd.DataFrame,
    features: list[str],
    cycle_col: str = "CYCLE",
) -> plt.Figure:
    """Heatmap of % missing per feature per cycle."""
    apply_publication_style()
    cycles = sorted(df[cycle_col].dropna().unique())
    avail = [f for f in features if f in df.columns]

    matrix = pd.DataFrame(index=avail, columns=[int(c) for c in cycles], dtype=float)
    for c in cycles:
        s = df[df[cycle_col] == c]
        for f in avail:
            matrix.loc[f, int(c)] = s[f].isna().mean() * 100

    fig, ax = plt.subplots(figsize=(8, max(4, len(avail) * 0.4)))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".0f", cmap=SEQUENTIAL_CMAP,
                vmin=0, vmax=100, cbar_kws={"label": "% missing"}, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title("Feature Missingness by Cycle (%)")
    ax.set_xlabel("PISA Cycle")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# B3: SES quintile × at-risk heatmap
# ---------------------------------------------------------------------------

def plot_ses_quintile_heatmap(
    df: pd.DataFrame,
    target_col: str = "AT_RISK_MATH",
    quintile_col: str = "SES_QUINTILE",
    cycle_col: str = "CYCLE",
) -> plt.Figure:
    """Weighted at-risk rate by SES quintile across cycles."""
    apply_publication_style()
    cycles = sorted(df[cycle_col].dropna().unique())
    quintiles = [1, 2, 3, 4, 5]

    matrix = pd.DataFrame(index=quintiles, columns=[int(c) for c in cycles], dtype=float)
    for c in cycles:
        for q in quintiles:
            s = df[(df[cycle_col] == c) & (df[quintile_col] == q)]
            if len(s) > 0:
                matrix.loc[q, int(c)] = weighted_proportion(s[target_col], _w(s)) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".0f", cmap=SEQUENTIAL_CMAP,
                cbar_kws={"label": "At-risk rate (%)"}, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title("Low-Proficiency Rate by SES Quintile and Cycle (%)")
    ax.set_xlabel("PISA Cycle")
    ax.set_ylabel("SES Quintile (1=lowest)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# B4: Correlation matrix
# ---------------------------------------------------------------------------

def plot_correlation_matrix(
    df: pd.DataFrame,
    features: list[str],
) -> plt.Figure:
    """Clustered correlation heatmap of features."""
    apply_publication_style()
    avail = [f for f in features if f in df.columns]
    corr = df[avail].corr()

    fig, ax = plt.subplots(figsize=(max(7, len(avail) * 0.6), max(6, len(avail) * 0.55)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=DIVERGING_CMAP,
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"label": "Pearson r", "shrink": 0.7}, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# B5: 2022 crisis feature drift (standardized mean difference)
# ---------------------------------------------------------------------------

def plot_feature_drift(
    df: pd.DataFrame,
    features: list[str],
    cycle_col: str = "CYCLE",
    baseline_cycle: int = 2018,
    crisis_cycle: int = 2022,
) -> plt.Figure:
    """Standardized mean difference of features between baseline and crisis cycle."""
    apply_publication_style()
    base = df[df[cycle_col] == baseline_cycle]
    crisis = df[df[cycle_col] == crisis_cycle]

    smd = {}
    for f in features:
        if f not in df.columns:
            continue
        v1 = base[f].dropna()
        v2 = crisis[f].dropna()
        if len(v1) < 10 or len(v2) < 10:
            continue
        pooled = np.sqrt((v1.var() + v2.var()) / 2)
        if pooled > 0:
            smd[f] = (v2.mean() - v1.mean()) / pooled

    smd_s = pd.Series(smd).sort_values()
    colors = [AT_RISK_COLORS["at_risk"] if v < 0 else AT_RISK_COLORS["proficient"]
              for v in smd_s.values]

    fig, ax = plt.subplots(figsize=(8, max(4, len(smd_s) * 0.4)))
    ax.barh(smd_s.index, smd_s.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Standardized mean difference ({crisis_cycle} − {baseline_cycle})")
    ax.set_title(f"Feature Drift: {crisis_cycle} vs {baseline_cycle} (Cohen's d)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# G: Country comparison — at-risk ranking
# ---------------------------------------------------------------------------

def plot_country_atrisk_ranking(
    df: pd.DataFrame,
    target_col: str = "AT_RISK_MATH",
    country_col: str = "COUNTRY",
    cycle: int | None = None,
    cycle_col: str = "CYCLE",
) -> plt.Figure:
    """Sorted bar of weighted at-risk rate by country (single cycle)."""
    apply_publication_style()
    data = df[df[cycle_col] == cycle] if cycle is not None else df

    rows = []
    for cnt in data[country_col].dropna().unique():
        s = data[data[country_col] == cnt]
        rows.append({"country": cnt, "at_risk": weighted_proportion(s[target_col], _w(s)) * 100})
    res = pd.DataFrame(rows).sort_values("at_risk")

    colors = [COUNTRY_COLORS.get(c, "#888888") for c in res["country"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(res["country"], res["at_risk"], color=colors)
    for i, v in enumerate(res["at_risk"]):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=9)
    title_cycle = f" ({cycle})" if cycle else ""
    ax.set_xlabel("Low-proficiency rate (%)")
    ax.set_title(f"Low-Proficiency Rate by Country{title_cycle}")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SES gradient comparison across countries
# ---------------------------------------------------------------------------

def plot_ses_gradient_by_country(
    df: pd.DataFrame,
    score_col: str = "MATH_PV_MEAN",
    ses_col: str = "ESCS",
    country_col: str = "COUNTRY",
    cycle: int | None = None,
    cycle_col: str = "CYCLE",
) -> plt.Figure:
    """Regression slope of score on ESCS per country (SES gradient steepness)."""
    apply_publication_style()
    data = df[df[cycle_col] == cycle] if cycle is not None else df

    fig, ax = plt.subplots(figsize=(9, 6))
    for cnt in sorted(data[country_col].dropna().unique()):
        s = data[data[country_col] == cnt][[ses_col, score_col]].dropna()
        if len(s) < 50:
            continue
        coef = np.polyfit(s[ses_col], s[score_col], 1)
        xs = np.linspace(s[ses_col].quantile(0.02), s[ses_col].quantile(0.98), 50)
        ys = np.polyval(coef, xs)
        ax.plot(xs, ys, label=f"{cnt} (slope={coef[0]:.0f})",
                color=COUNTRY_COLORS.get(cnt, "#888888"), linewidth=2)

    ax.set_xlabel("ESCS (socioeconomic status)")
    ax.set_ylabel("Math score")
    title_cycle = f" ({cycle})" if cycle else ""
    ax.set_title(f"SES Gradient by Country{title_cycle}")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def generate_all_eda(
    df_long: pd.DataFrame,
    df_comparison: pd.DataFrame,
    features: list[str],
    output_dir: str | Path,
) -> dict[str, plt.Figure]:
    """Generate and save the full EDA figure suite."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figs: dict[str, plt.Figure] = {}

    figs["A1_atrisk_trajectory"] = plot_atrisk_trajectory(df_long)
    figs["B1_score_distributions"] = plot_score_distributions(df_long)
    figs["B2_missingness"] = plot_missingness_heatmap(df_long, features)
    figs["B3_ses_quintile"] = plot_ses_quintile_heatmap(df_long)
    figs["B4_correlation"] = plot_correlation_matrix(df_long, features)
    figs["B5_feature_drift"] = plot_feature_drift(df_long, features)
    figs["G1_country_ranking_2022"] = plot_country_atrisk_ranking(df_comparison, cycle=2022)
    figs["G2_ses_gradient_2022"] = plot_ses_gradient_by_country(df_comparison, cycle=2022)

    for name, fig in figs.items():
        save_figure(fig, str(output_dir / name))
        logger.info("EDA figure saved", name=name)
        plt.close(fig)

    return figs
