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

from src.data.impute import add_missingness_indicators
from src.data.weights import normalize_weights_within_country
from src.features.engineer import add_school_aggregates
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
    add_indicators: bool = True,
    use_normalized_weights: bool | None = None,
    drop_missing_target: bool = True,
    add_school_context: bool = False,
    school_agg_cols: list[str] | None = None,
) -> ModelData:
    """
    Construct (X, y, weights) for modeling from a processed dataframe.

    Engineered features (SES_COMPLETE, interactions, ...) are intentionally NOT
    added here: they are built inside the CV pipeline by
    ``src.features.transformers.EngineeredFeatureBuilder`` so their
    standardization/rank statistics are fit on the train fold only (no leakage).
    Pass the raw component columns (ESCS, HOMEPOS, ANXMAT, ICTHOME, ...) in
    ``candidate_features`` and the builder derives the rest per fold.

    Args:
        df: processed dataframe (one or more cycles)
        candidate_features: raw features to consider (subset that exists is used)
        domain: 'math' | 'reading' | 'science'
        threshold: low-proficiency threshold (defaults to PISA L2)
        add_indicators: add binary _MISS_<feat> columns for partially-missing
            features so tree models can use missingness as signal
        use_normalized_weights: divide W_FSTUWT so it sums to n within each
            country (stops a large country dominating the loss in cross-country
            models). Default: auto-on when >1 country is present.
        drop_missing_target: drop rows where target cannot be computed
        add_school_context: add survey-weighted, leave-one-out school-mean features
            (``SCH_MEAN_<col>`` + ``SCH_N``) from ``CNTSCHID``. School socioeconomic
            composition is PISA's strongest contextual predictor; an ablation
            (``scripts/run_school_features_experiment.py`` and its fold-safe
            confirmation) shows it lifts weighted CV AUC ~+0.05, significant, with
            an identical delta under fold-safe recomputation (so the full-cohort
            means used here are not a leakage artefact).
        school_agg_cols: which student indices to aggregate to school level
            (default ESCS/HOMEPOS/ANXMAT/TEACHSUP).
    """
    df = add_point_target(df, domain=domain, threshold=threshold)
    target_col = {"math": "AT_RISK_MATH", "reading": "AT_RISK_READ", "science": "AT_RISK_SCIE"}[domain]

    candidate_features = list(candidate_features)
    if add_school_context:
        agg_cols = school_agg_cols or ["ESCS", "HOMEPOS", "ANXMAT", "TEACHSUP"]
        df = add_school_aggregates(df, cols=agg_cols)
        candidate_features += [f"SCH_MEAN_{c}" for c in agg_cols] + ["SCH_N"]

    feats = [f for f in candidate_features if f in df.columns]

    work = df.copy()
    if drop_missing_target:
        work = work[work[target_col].notna()]

    # Feature selection: drop all-missing-in-this-slice, zero-variance, collinear
    feats = [f for f in feats if work[f].notna().any()]
    feats = select_features(work, feats, max_missingness=0.95, max_correlation=0.97)

    if add_indicators:
        before = set(work.columns)
        work = add_missingness_indicators(work, feats, threshold=0.95)
        feats += [c for c in work.columns if c.startswith("_MISS_") and c not in before]

    X = work[feats].copy()
    y = work[target_col].astype(int)

    # Weight selection. Normalize within country for cross-country slices.
    country_col = "COUNTRY" if "COUNTRY" in work.columns else ("CNT" if "CNT" in work.columns else None)
    n_countries = work[country_col].nunique() if country_col else 1
    if use_normalized_weights is None:
        use_normalized_weights = n_countries > 1
    if WEIGHT_COL in work.columns:
        if use_normalized_weights:
            work = normalize_weights_within_country(work)
            weights = work[f"{WEIGHT_COL}_NORM"]
        else:
            weights = work[WEIGHT_COL]
    else:
        weights = pd.Series(np.ones(len(work)), index=work.index)

    meta = {
        "domain": domain,
        "n_samples": len(X),
        "n_features": len(feats),
        "n_countries": int(n_countries),
        "weights_normalized": bool(use_normalized_weights and WEIGHT_COL in work.columns),
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
