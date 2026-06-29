"""
SHAP-based explainability: global and local explanations.
Supports TreeSHAP (tree models), LinearSHAP (LR), and KernelSHAP (others).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

import structlog
logger = structlog.get_logger(__name__)


def get_explainer(model: Any, X_background: pd.DataFrame) -> shap.Explainer:
    """
    Select appropriate SHAP explainer based on model type.
    TreeSHAP for tree models (exact), LinearSHAP for LR, KernelSHAP for others.
    """
    model_type = type(model).__name__

    # Handle sklearn Pipeline: unwrap to the actual estimator
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", list(model.named_steps.values())[-1])
        background = model.named_steps["scaler"].transform(X_background) if "scaler" in model.named_steps else X_background.values
        background_df = pd.DataFrame(background, columns=X_background.columns)
    else:
        clf = model
        background_df = X_background

    tree_models = {
        "RandomForestClassifier", "ExtraTreesClassifier",
        "GradientBoostingClassifier", "HistGradientBoostingClassifier",
        "XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
        "DecisionTreeClassifier",
    }
    linear_models = {"LogisticRegression", "LinearSVC", "Ridge", "Lasso"}

    clf_type = type(clf).__name__
    if clf_type in tree_models:
        logger.info("Using TreeSHAP", model_type=clf_type)
        return shap.TreeExplainer(clf)
    elif clf_type in linear_models:
        logger.info("Using LinearSHAP", model_type=clf_type)
        return shap.LinearExplainer(clf, background_df)
    else:
        logger.info("Using KernelSHAP", model_type=clf_type)
        background_summary = shap.kmeans(background_df, 50)
        return shap.KernelExplainer(model.predict_proba, background_summary)


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    X_background: pd.DataFrame | None = None,
    max_samples: int = 2000,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute SHAP values for X.

    Returns:
        (shap_values, feature_names) where shap_values shape is (n_samples, n_features)
        for binary classification, returns values for positive class.
    """
    if X_background is None:
        X_background = X.sample(min(200, len(X)), random_state=42)

    # Limit samples for performance
    if len(X) > max_samples:
        X_explain = X.sample(max_samples, random_state=42)
        logger.info("Subsampling for SHAP", original=len(X), sampled=max_samples)
    else:
        X_explain = X

    explainer = get_explainer(model, X_background)

    # Handle Pipeline: transform X before explanation
    if hasattr(model, "named_steps") and "scaler" in model.named_steps:
        X_transformed = model.named_steps["scaler"].transform(X_explain)
        X_for_shap = pd.DataFrame(X_transformed, columns=X_explain.columns)
    else:
        X_for_shap = X_explain

    raw = explainer(X_for_shap)

    # For binary classification: extract positive class SHAP values
    if isinstance(raw.values, list):
        shap_vals = raw.values[1]  # positive class
    elif raw.values.ndim == 3:
        shap_vals = raw.values[:, :, 1]  # (n_samples, n_features, n_classes)
    else:
        shap_vals = raw.values

    feature_names = list(X_explain.columns)
    logger.info("SHAP values computed", shape=shap_vals.shape)
    return shap_vals, feature_names


def global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Mean absolute SHAP values as global feature importance."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    result = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


def country_shap_comparison(
    models_by_country: dict[str, Any],
    X_by_country: dict[str, pd.DataFrame],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Compute SHAP feature importance for multiple countries, return rank matrix.
    Rows = features, Columns = countries, Values = rank (1=most important).
    """
    importance_by_country: dict[str, pd.DataFrame] = {}
    for country, model in models_by_country.items():
        X = X_by_country[country]
        shap_vals, feat_names = compute_shap_values(model, X)
        imp = global_feature_importance(shap_vals, feat_names)
        importance_by_country[country] = imp.set_index("feature")["mean_abs_shap"]

    # Build rank matrix
    all_features = sorted(
        set(f for imp in importance_by_country.values() for f in imp.index)
    )
    rank_matrix = pd.DataFrame(index=all_features)
    for country, imp in importance_by_country.items():
        full = imp.reindex(all_features).fillna(0)
        rank_matrix[country] = full.rank(ascending=False).astype(int)

    return rank_matrix.loc[rank_matrix.min(axis=1) <= top_n].sort_values(
        list(rank_matrix.columns)[0]
    )


def save_shap_values(
    shap_values: np.ndarray,
    feature_names: list[str],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path.with_suffix(".npy")), shap_values)
    pd.Series(feature_names).to_csv(path.with_suffix(".features.csv"), index=False, header=False)
    logger.info("SHAP values saved", path=str(path))
