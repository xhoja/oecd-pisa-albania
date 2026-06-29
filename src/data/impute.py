"""
Missing value handling for PISA data.

Two imputation strategies:
1. Multiple Imputation by Chained Equations (MICE) for MAR features
   (student-level missing within a cycle)
2. Cycle-level missingness: features completely absent in a cycle get
   a sentinel NaN + a binary indicator column.

Plausible value handling:
    PVs are NOT imputed — they are multiple imputations already (by OECD).
    Models are fit 10× (once per PV draw), then combined via Rubin's rules.

References:
    van Buuren & Groothuis-Oudshoorn (2011). mice: Multivariate Imputation
        by Chained Equations in R.
    Rubin (1987). Multiple Imputation for Nonresponse in Surveys. Ch. 3.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline

import structlog
logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Little's MCAR test (simplified chi-square approximation)
# ---------------------------------------------------------------------------

def little_mcar_test(df: pd.DataFrame) -> dict[str, float]:
    """
    Approximate Little's MCAR test.
    H0: data is MCAR. p < 0.05 → MAR or MNAR (imputation model needed).
    Returns dict with statistic and p_value.
    """
    complete = df.dropna()
    n_complete = len(complete)
    n_total = len(df)

    # Pattern matrix: which rows have same missing pattern
    missing_pattern = df.isnull().astype(int)
    patterns = missing_pattern.drop_duplicates()

    test_stat = 0.0
    df_dof = 0
    grand_means = complete.mean()
    grand_cov = complete.cov()

    for _, pattern_row in patterns.iterrows():
        is_missing = pattern_row == 1
        obs_cols = list(pattern_row[~is_missing].index)
        if not obs_cols:
            continue

        # Rows matching this pattern
        mask = (missing_pattern == pattern_row).all(axis=1)
        group = df.loc[mask, obs_cols].dropna()
        n_k = len(group)
        if n_k < 2:
            continue

        group_mean = group.mean()
        diff = (group_mean - grand_means[obs_cols]).values

        try:
            sub_cov = grand_cov.loc[obs_cols, obs_cols]
            inv_cov = np.linalg.pinv(sub_cov.values)
            test_stat += n_k * float(diff @ inv_cov @ diff)
            df_dof += len(obs_cols)
        except np.linalg.LinAlgError:
            continue

    p_value = 1 - stats.chi2.cdf(test_stat, df=max(df_dof - len(df.columns), 1))
    result = {"statistic": round(test_stat, 4), "df": df_dof, "p_value": round(p_value, 6)}
    logger.info("Little's MCAR test", **result)
    return result


# ---------------------------------------------------------------------------
# MICE implementation
# ---------------------------------------------------------------------------

@dataclass
class MICEConfig:
    n_imputations: int = 5
    max_iter: int = 10
    random_state: int = 42
    estimator: object = field(default_factory=BayesianRidge)


def mice_impute(
    df: pd.DataFrame,
    features: list[str],
    config: MICEConfig | None = None,
) -> list[pd.DataFrame]:
    """
    Multiple Imputation by Chained Equations.

    Returns a list of n_imputations dataframes, each with imputed values.
    Only imputes columns listed in `features` that are numeric.
    Cycle-level missing (entire column NaN) is NOT imputed via MICE.
    """
    if config is None:
        config = MICEConfig()

    numeric_features = [
        f for f in features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]

    # Separate cycle-level missing (>95% NaN) from student-level missing
    cycle_missing = [f for f in numeric_features if df[f].isna().mean() > 0.95]
    imputable = [f for f in numeric_features if f not in cycle_missing]

    if cycle_missing:
        logger.info("Cycle-level missing (skipping MICE)", features=cycle_missing)

    if not imputable:
        logger.warning("No features to impute via MICE")
        return [df.copy() for _ in range(config.n_imputations)]

    imputed_datasets: list[pd.DataFrame] = []
    for m in range(config.n_imputations):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=config.max_iter,
            random_state=config.random_state + m,
            sample_posterior=True,  # introduces stochastic variation across imputations
        )
        df_copy = df.copy()
        df_copy[imputable] = imputer.fit_transform(df_copy[imputable])
        imputed_datasets.append(df_copy)
        logger.debug("MICE imputation complete", m=m + 1)

    logger.info("MICE complete", n_datasets=config.n_imputations, features_imputed=imputable)
    return imputed_datasets


# ---------------------------------------------------------------------------
# Rubin's combining rules
# ---------------------------------------------------------------------------

def rubins_combine(
    estimates: list[float],
    variances: list[float],
) -> tuple[float, float, float]:
    """
    Rubin's rules for combining multiply-imputed estimates.

    Args:
        estimates: point estimates from each imputed dataset
        variances: within-imputation variances (SE²) from each dataset

    Returns:
        (combined_estimate, combined_se, fraction_missing_info)
    """
    m = len(estimates)
    Q_bar = float(np.mean(estimates))

    # Within-imputation variance
    U_bar = float(np.mean(variances))

    # Between-imputation variance
    B = float(np.var(estimates, ddof=1))

    # Total variance
    T = U_bar + (1 + 1 / m) * B

    # Fraction of missing information
    fmi = (1 + 1 / m) * B / T if T > 0 else 0.0

    return Q_bar, float(np.sqrt(T)), fmi


# ---------------------------------------------------------------------------
# Per-PV model pipeline (the correct PISA methodology)
# ---------------------------------------------------------------------------

def fit_per_pv(
    df: pd.DataFrame,
    pv_cols: list[str],
    threshold: float,
    fit_fn: callable,
    **fit_kwargs,
) -> dict[str, object]:
    """
    Fit a model once per plausible value, combine results via Rubin's rules.

    fit_fn signature: fit_fn(X, y, **fit_kwargs) -> (estimator, metrics_dict)
    where metrics_dict maps metric_name -> (estimate, variance).

    Returns:
        dict of {metric_name: (combined_estimate, combined_se, fmi)}
    """
    per_pv_metrics: dict[str, list[tuple[float, float]]] = {}
    estimators = []

    for pv_col in pv_cols:
        if pv_col not in df.columns:
            continue
        y = (df[pv_col] < threshold).astype(int)
        mask = y.notna()
        estimator, metrics = fit_fn(df[mask], y[mask], **fit_kwargs)
        estimators.append(estimator)
        for metric_name, (est, var) in metrics.items():
            per_pv_metrics.setdefault(metric_name, []).append((est, var))

    combined = {}
    for metric_name, ev_pairs in per_pv_metrics.items():
        ests = [e for e, _ in ev_pairs]
        vars_ = [v for _, v in ev_pairs]
        combined[metric_name] = rubins_combine(ests, vars_)

    return {"combined_metrics": combined, "estimators": estimators}


# ---------------------------------------------------------------------------
# Cycle-level missingness indicators
# ---------------------------------------------------------------------------

def add_missingness_indicators(
    df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.95,
) -> pd.DataFrame:
    """
    For features with >threshold fraction missing, add binary indicator column
    _MISS_<feature> = 1 if missing. Helps tree models learn missingness patterns.
    """
    df = df.copy()
    for feat in features:
        if feat not in df.columns:
            continue
        miss_rate = df[feat].isna().mean()
        if miss_rate > 0.01:  # any non-trivial missingness
            df[f"_MISS_{feat}"] = df[feat].isna().astype(int)
    return df
