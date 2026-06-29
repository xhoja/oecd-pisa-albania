"""
Build model-ready (X, y, weights) matrices from processed PISA data.

Handles:
- feature selection (drop all-missing / zero-variance / collinear)
- median imputation INSIDE a column transformer (no leakage when wrapped in CV)
- alignment of target and sample weights

For rigorous PISA inference the target should be built per plausible value and
combined via Rubin's rules (see src.features.target.per_pv_targets and
src.data.impute.fit_per_pv). For model *comparison* we use the PV-mean point
target, which is standard practice and far cheaper; the per-PV combination is
applied to the final selected model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import structlog

from src.features.engineer import engineer_all
from src.features.select import select_features
from src.features.target import add_point_target

logger = structlog.get_logger(__name__)

WEIGHT_COL = "W_FSTUWT"


@dataclass
class ModelData:
    X: pd.DataFrame
    y: pd.Series
    weights: pd.Series
    feature_names: list[str]
    meta: dict


def build_model_data(
    df: pd.DataFrame,
    candidate_features: list[str],
    domain: str = "math",
    threshold: float | None = None,
    add_engineered: bool = True,
    drop_missing_target: bool = True,
) -> ModelData:
    """
    Construct (X, y, weights) for modeling from a processed dataframe.

    Args:
        df: processed dataframe (one or more cycles)
        candidate_features: features to consider (subset that exists is used)
        domain: 'math' | 'reading' | 'science'
        threshold: low-proficiency threshold (defaults to PISA L2)
        add_engineered: include engineered features if present
        drop_missing_target: drop rows where target cannot be computed
    """
    df = add_point_target(df, domain=domain, threshold=threshold)
    target_col = {"math": "AT_RISK_MATH", "reading": "AT_RISK_READ", "science": "AT_RISK_SCIE"}[domain]

    engineered = [
        "SES_COMPLETE", "MATERIAL_DEFICIT", "DIGITAL_READINESS", "DIGITAL_GAP",
        "SES_x_ANXIETY", "BELONG_x_TEACHSUP", "GRADE_x_SES", "GENDER_x_ANXMAT",
        "HOME_EDU_SCORE",
    ]
    feats = list(candidate_features)
    if add_engineered:
        feats += [f for f in engineered if f in df.columns]

    feats = [f for f in feats if f in df.columns]

    work = df.copy()
    if drop_missing_target:
        work = work[work[target_col].notna()]

    # Feature selection: drop all-missing-in-this-slice, zero-variance, collinear
    feats = [f for f in feats if work[f].notna().any()]
    feats = select_features(work, feats, max_missingness=0.95, max_correlation=0.97)

    X = work[feats].copy()
    y = work[target_col].astype(int)
    weights = work[WEIGHT_COL] if WEIGHT_COL in work.columns else pd.Series(np.ones(len(work)), index=work.index)

    meta = {
        "domain": domain,
        "n_samples": len(X),
        "n_features": len(feats),
        "at_risk_rate": round(float(y.mean()), 4),
        "target_col": target_col,
    }
    logger.info("Model data built", **meta)
    return ModelData(X=X, y=y, weights=weights, feature_names=feats, meta=meta)


def impute_median(train_X: pd.DataFrame, *other_X: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """
    Median-impute using TRAIN medians only (no leakage), apply to all frames.
    Returns imputed (train, *others) in the same order.
    """
    medians = train_X.median()
    out = [train_X.fillna(medians)]
    for X in other_X:
        out.append(X.fillna(medians))
    return tuple(out)
