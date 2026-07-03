r"""
Multilevel (hierarchical) logistic model for low-proficiency risk.

PISA has a two-stage design: schools are sampled, then students *within* schools.
Students in the same school share unobserved context (peers, teaching, resources),
so their outcomes are correlated — an assumption a flat logistic regression
violates. The project's headline model handles this with *hand-crafted* survey-
weighted school-mean features; the statistically principled alternative is a
**random-intercept logistic model** that lets each school have its own baseline
log-odds:

.. math::
    \operatorname{logit} P(y_{ij}=1) = \beta_0 + \beta^\top x_{ij} + u_j,
    \qquad u_j \sim \mathcal{N}(0,\ \sigma_u^2),

for student *i* in school *j*. The school random effect :math:`u_j` absorbs
between-school variation the student features miss. Two things fall out:

- the **intraclass correlation** on the latent (logit) scale,
  :math:`\mathrm{ICC} = \sigma_u^2 / (\sigma_u^2 + \pi^2/3)`, where
  :math:`\pi^2/3 \approx 3.29` is the logistic residual variance — the share of
  variance that is *between schools*; a large ICC is the quantitative
  justification for modelling school context at all;
- fixed-effect odds ratios :math:`e^{\beta}` that are now *within-school* (adjusted
  for the school baseline), a cleaner read of each student factor.

Fit via variational Bayes (`statsmodels` `BinomialBayesMixedGLM`). **Caveat:**
`BinomialBayesMixedGLM` does not take survey weights; fits here are unweighted, so
the ICC/structure conclusions are robust but the fixed effects are sample- (not
population-) estimates. Proper multilevel pseudo-likelihood with scaled weights
(Rabe-Hesketh & Skrondal) is the further refinement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOGISTIC_RESIDUAL_VAR = np.pi ** 2 / 3.0  # ≈ 3.29


def icc_logistic(school_var: float) -> float:
    r"""Latent-scale intraclass correlation
    :math:`\sigma_u^2/(\sigma_u^2+\pi^2/3)`."""
    return float(school_var / (school_var + LOGISTIC_RESIDUAL_VAR))


def _standardize(X: pd.DataFrame) -> pd.DataFrame:
    """Median-impute then z-score (helps the VB optimizer converge)."""
    Xi = X.fillna(X.median(numeric_only=True))
    sd = Xi.std(ddof=0).replace(0, 1.0)
    return (Xi - Xi.mean()) / sd


def fit_random_intercept(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    feature_names: list[str] | None = None,
) -> dict:
    """
    Fit a random-intercept logistic model (school = grouping) and return a summary
    dict: school variance & SD, ICC, and a tidy fixed-effects table with odds
    ratios. Features are standardized so ORs are per-SD.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    feats = feature_names or list(X.columns)
    Xs = _standardize(X[feats]).reset_index(drop=True)
    d = Xs.copy()
    d["y"] = np.asarray(y, dtype=float)
    d["grp"] = np.asarray(groups)

    formula = "y ~ " + " + ".join(feats)
    model = BinomialBayesMixedGLM.from_formula(formula, {"grp": "0 + C(grp)"}, d)
    res = model.fit_vb()

    # vcp_mean is the posterior mean of the log-SD of the variance component.
    school_sd = float(np.exp(res.vcp_mean[0]))
    school_var = school_sd ** 2
    icc = icc_logistic(school_var)

    # fixed effects: res.fe_mean / res.fe_sd, ordered like res.model.exog_names
    fe_names = list(res.model.exog_names)
    fe = pd.DataFrame({
        "term": fe_names,
        "coef": np.asarray(res.fe_mean, dtype=float),
        "sd": np.asarray(res.fe_sd, dtype=float),
    })
    fe["odds_ratio"] = np.exp(fe["coef"])
    return {
        "school_sd": school_sd,
        "school_var": school_var,
        "icc": icc,
        "fixed_effects": fe,
        "n_groups": int(pd.Series(groups).nunique()),
        "n_obs": int(len(y)),
        "result": res,
    }


def variance_partition_icc(y: pd.Series, groups: pd.Series) -> dict:
    """
    Null random-intercept model (intercept only) — the *unconditioned* variance
    partition: how much of the outcome variance is between schools before any
    student predictor is added. This is the honest baseline ICC.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = pd.DataFrame({"y": np.asarray(y, dtype=float), "grp": np.asarray(groups)})
    res = BinomialBayesMixedGLM.from_formula("y ~ 1", {"grp": "0 + C(grp)"}, d).fit_vb()
    school_var = float(np.exp(res.vcp_mean[0])) ** 2
    return {"school_var": school_var, "icc": icc_logistic(school_var), "result": res}


def predict_random_intercept(res, X_std: pd.DataFrame, groups: np.ndarray,
                             train_groups: np.ndarray) -> np.ndarray:
    r"""
    Predicted probabilities :math:`\sigma(\beta_0+\beta^\top x + \hat u_j)` for new
    rows. The school random effect is added when the school was seen in training
    (its :math:`\hat u_j` is estimable) and set to 0 (the population mean) for
    unseen schools — the honest out-of-fold behaviour.
    """
    fe_names = list(res.model.exog_names)
    beta = np.asarray(res.fe_mean, dtype=float)
    # design with intercept in the same order as fe_names
    D = pd.DataFrame({"Intercept": 1.0}, index=X_std.index)
    for name in fe_names:
        if name == "Intercept":
            continue
        D[name] = X_std[name].values
    lp = D[fe_names].values @ beta

    # random effects: res.vc_mean aligns with the sorted unique training groups
    uniq = np.array(sorted(pd.unique(train_groups)))
    re_map = {g: float(u) for g, u in zip(uniq, np.asarray(res.vc_mean, dtype=float))}
    lp = lp + np.array([re_map.get(g, 0.0) for g in groups])
    return 1.0 / (1.0 + np.exp(-lp))
