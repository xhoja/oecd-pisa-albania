"""Smoke tests for EDA figures that don't have coverage elsewhere."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from conftest import make_pisa_frame  # noqa: E402
from src.visualization.eda import plot_ses_logistic_curve  # noqa: E402


def _two_cycle_frame() -> pd.DataFrame:
    a = make_pisa_frame(n=500, cycle=2018, seed=1)
    b = make_pisa_frame(n=500, cycle=2022, seed=2)
    df = pd.concat([a, b], ignore_index=True)
    df["AT_RISK_MATH"] = (df["ESCS"] < df["ESCS"].median()).astype(int)
    return df


def test_plot_ses_logistic_curve_returns_figure_with_both_cycles():
    df = _two_cycle_frame()
    fig = plot_ses_logistic_curve(df, cycles=(2018, 2022))
    ax = fig.axes[0]
    # one logistic line per cycle
    assert len(ax.lines) == 2
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("2018" in l for l in labels) and any("2022" in l for l in labels)
    plt.close(fig)


def test_plot_ses_logistic_curve_skips_cycle_without_data():
    df = _two_cycle_frame()
    fig = plot_ses_logistic_curve(df, cycles=(2018, 1999))  # 1999 absent
    assert len(fig.axes[0].lines) == 1
    plt.close(fig)
