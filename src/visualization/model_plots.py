"""
Model-results figures: ROC, PR, calibration, confusion matrix, CV distribution,
and the out-of-sample (covariate-shift) comparison.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

import structlog

from src.models.evaluate import get_calibration_data, get_pr_data, get_roc_data
from src.visualization.style import (
    SEQUENTIAL_CMAP,
    apply_publication_style,
    color_list,
    save_figure,
)

logger = structlog.get_logger(__name__)


def plot_roc_curves(curves: dict[str, dict], title: str = "ROC Curves") -> plt.Figure:
    """curves: {model_name: {'y_true':..., 'y_prob':..., 'auc':...}}"""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = color_list(len(curves))
    for (name, d), c in zip(curves.items(), colors):
        roc = get_roc_data(d["y_true"], d["y_prob"])
        ax.plot(roc["fpr"], roc["tpr"], color=c, linewidth=2,
                label=f"{name} (AUC={d.get('auc', np.nan):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_pr_curves(curves: dict[str, dict], title: str = "Precision-Recall Curves") -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = color_list(len(curves))
    for (name, d), c in zip(curves.items(), colors):
        pr = get_pr_data(d["y_true"], d["y_prob"])
        ax.plot(pr["recall"], pr["precision"], color=c, linewidth=2,
                label=f"{name} (AP={d.get('pr_auc', np.nan):.3f})")
    base = d["y_true"].mean()
    ax.axhline(base, color="k", linestyle="--", linewidth=1, alpha=0.5, label=f"Baseline ({base:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_calibration(curves: dict[str, dict], title: str = "Calibration") -> plt.Figure:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = color_list(len(curves))
    for (name, d), c in zip(curves.items(), colors):
        cal = get_calibration_data(d["y_true"], d["y_prob"], n_bins=10)
        ax.plot(cal["mean_predicted_value"], cal["fraction_of_positives"],
                marker="o", color=c, linewidth=2, label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix") -> plt.Figure:
    apply_publication_style()
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap=SEQUENTIAL_CMAP, cbar=False,
                xticklabels=["Proficient", "At-risk"], yticklabels=["Proficient", "At-risk"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_cv_distribution(comparison_df: pd.DataFrame, metric: str = "roc_auc") -> plt.Figure:
    """Bar of mean metric with 95% CI error bars across models."""
    apply_publication_style()
    df = comparison_df.sort_values(f"{metric}_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    yerr = [
        df[f"{metric}_mean"] - df[f"{metric}_95ci_low"],
        df[f"{metric}_95ci_high"] - df[f"{metric}_mean"],
    ]
    ax.barh(df["model"], df[f"{metric}_mean"], xerr=yerr, capsize=4,
            color=color_list(len(df)))
    ax.set_xlabel(f"{metric.upper()} (mean ± 95% CI)")
    ax.set_title(f"Model Comparison: {metric.upper()}")
    ax.set_xlim(left=max(0.4, df[f"{metric}_95ci_low"].min() - 0.02))
    fig.tight_layout()
    return fig


def plot_oos_comparison(oos_results: dict, metric: str = "roc_auc") -> plt.Figure:
    """
    Bar chart of out-of-sample test metric per model.
    Highlights the covariate-shift story: tree models collapse, linear generalizes.
    """
    apply_publication_style()
    models = list(oos_results["models"].keys())
    vals = [oos_results["models"][m][metric] for m in models]
    order = np.argsort(vals)
    models = [models[i] for i in order]
    vals = [vals[i] for i in order]
    colors = ["#D55E00" if v < 0.55 else ("#E69F00" if v < 0.65 else "#009E73") for v in vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(models, vals, color=colors)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (0.5)")
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel(f"Out-of-sample test {metric.upper()}")
    ax.set_title("Out-of-Sample Performance (train 2009–2018 → test 2022)")
    ax.legend()
    fig.tight_layout()
    return fig
