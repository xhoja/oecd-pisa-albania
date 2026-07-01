"""
Publication-quality matplotlib style configuration.
All figures use this style for consistency across the paper.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# Color palette — colorblind-safe (Wong 2011)
PALETTE = {
    "black":    "#000000",
    "orange":   "#E69F00",
    "sky_blue": "#56B4E9",
    "green":    "#009E73",
    "yellow":   "#F0E442",
    "blue":     "#0072B2",
    "vermilion":"#D55E00",
    "purple":   "#CC79A7",
}

COUNTRY_COLORS = {
    "ALB": "#D55E00",    # Albania — vermilion, stands out
    "MKD": "#56B4E9",    # North Macedonia — sky blue
    "MNE": "#009E73",    # Montenegro — green
    "SRB": "#CC79A7",    # Serbia — purple
    "BGR": "#E69F00",    # Bulgaria — orange
    "EST": "#0072B2",    # Estonia — deep blue
    "FIN": "#F0E442",    # Finland — yellow
    "MEX": "#000000",    # Mexico — black
    "COL": "#999999",    # Colombia — grey
    "OECD": "#AAAAAA",  # OECD average — light grey
}

AT_RISK_COLORS = {
    "at_risk": "#D55E00",
    "proficient": "#0072B2",
}


def apply_publication_style() -> None:
    """Apply publication-quality matplotlib rcParams."""
    mpl.rcParams.update({
        # Figure
        "figure.dpi": 150,
        "figure.figsize": (8, 5),
        "figure.facecolor": "white",
        # Font
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        # Saving
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        # PDF metadata
        "pdf.fonttype": 42,  # TrueType (editable in Illustrator/Inkscape)
        "ps.fonttype": 42,
    })
    sns.set_style("white")


def save_figure(
    fig: plt.Figure,
    path: str,
    formats: list[str] | None = None,
) -> None:
    """Save figure in multiple formats."""
    if formats is None:
        formats = ["pdf", "png"]
    base = path.rsplit(".", 1)[0] if "." in path else path
    for fmt in formats:
        fig.savefig(f"{base}.{fmt}", format=fmt, bbox_inches="tight")


def color_list(n: int) -> list[str]:
    """Return n colorblind-safe colors."""
    colors = list(PALETTE.values())
    if n <= len(colors):
        return colors[:n]
    # Cycle if more than 8 needed
    return [colors[i % len(colors)] for i in range(n)]
