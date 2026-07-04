r"""
Multilevel (hierarchical) logistic model for low-proficiency risk.

PISA has a two-stage design: schools are sampled, then students *within* schools.
Students in the same school share unobserved context (peers, teaching, resources),
so their outcomes are correlated - an assumption a flat logistic regression
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
  :math:`\pi^2/3 \approx 3.29` is the logistic residual variance - the share of
  variance that is *between schools*; a large ICC is the quantitative
  justification for modelling school context at all;
- fixed-effect odds ratios :math:`e^{\beta}` that are now *within-school* (adjusted
  for the school baseline), a cleaner read of each student factor.

Two estimators are provided:

- ``fit_random_intercept`` - variational Bayes (`statsmodels`
  `BinomialBayesMixedGLM`), **unweighted** (the VB GLM takes no survey weights);
  ICC/structure conclusions are robust but fixed effects are sample- (not
  population-) estimates.
- ``fit_weighted_random_intercept`` - **survey-weighted pseudo-likelihood** via
  penalized quasi-likelihood (Schall 1991) with level-1 weights scaled within
  cluster (Rabe-Hesketh & Skrondal 2006). This is the design-based refinement:
  population-consistent fixed effects and a weight-aware variance component. PQL
  slightly attenuates the variance component for binary outcomes, so its ICC is a
  mild lower bound - read the two ICCs together.
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
    Null random-intercept model (intercept only) - the *unconditioned* variance
    partition: how much of the outcome variance is between schools before any
    student predictor is added. This is the honest baseline ICC.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = pd.DataFrame({"y": np.asarray(y, dtype=float), "grp": np.asarray(groups)})
    res = BinomialBayesMixedGLM.from_formula("y ~ 1", {"grp": "0 + C(grp)"}, d).fit_vb()
    school_var = float(np.exp(res.vcp_mean[0])) ** 2
    return {"school_var": school_var, "icc": icc_logistic(school_var), "result": res}


def scale_survey_weights(
    weights: np.ndarray, groups: np.ndarray, method: str = "effective"
) -> np.ndarray:
    r"""
    Scale level-1 (student) survey weights **within each cluster** for multilevel
    pseudo-likelihood (Rabe-Hesketh & Skrondal 2006; Pfeffermann et al. 1998).
    Raw weights make the cluster contribution depend on its total weight, which
    biases the variance component; scaling fixes the effective cluster size.

    - ``"effective"`` (method 2, recommended for informative weights): scale so
      the scaled weights sum to the *effective* cluster size
      :math:`(\sum_i w_{ij})^2 / \sum_i w_{ij}^2`.
    - ``"cluster"`` (method 1): scale so scaled weights sum to the raw cluster
      size :math:`n_j` (i.e. :math:`w^*_{ij} = w_{ij} n_j / \sum_i w_{ij}`).
    """
    w = np.asarray(weights, dtype=float)
    g = np.asarray(groups)
    out = np.ones_like(w)
    for gj in np.unique(g):
        m = g == gj
        wj = w[m]
        s1 = wj.sum()
        if s1 <= 0:
            out[m] = 1.0
            continue
        if method == "cluster":
            target = m.sum()  # n_j
        else:  # effective
            s2 = (wj ** 2).sum()
            target = (s1 ** 2) / s2 if s2 > 0 else m.sum()
        out[m] = wj * target / s1
    return out


def fit_weighted_random_intercept(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    weights: pd.Series,
    feature_names: list[str] | None = None,
    scaling: str = "effective",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    r"""
    Survey-weighted random-intercept logistic model via **penalized quasi-
    likelihood** (Schall 1991) with **scaled** level-1 weights - the "proper
    multilevel pseudo-likelihood with scaled survey weights" refinement over the
    unweighted VB ``fit_random_intercept``.

    Each PQL iteration linearizes the logistic model to a working response
    :math:`z_{ij}=\eta_{ij}+(y_{ij}-\mu_{ij})/v_{ij}` with IRLS weight
    :math:`v_{ij}=\mu_{ij}(1-\mu_{ij})`, then solves the weighted linear mixed
    model (combined weight :math:`W_{ij}=w^*_{ij}v_{ij}`). For a random intercept
    the mixed-model step has closed forms - BLUP shrinkage for :math:`u_j` and
    weighted GLS for :math:`\beta` - alternated to convergence, and
    :math:`\sigma_u^2` is updated by Schall's trace correction. Features are
    standardized so odds ratios are per-SD, matching ``fit_random_intercept``.

    Returns the same summary keys as ``fit_random_intercept`` plus ``scaling``.
    """
    feats = feature_names or list(X.columns)
    Xs = _standardize(X[feats]).reset_index(drop=True)
    Xmat = np.column_stack([np.ones(len(Xs)), Xs.values])  # intercept + feats
    yv = np.asarray(y, dtype=float)
    g = np.asarray(groups)
    w_scaled = scale_survey_weights(np.asarray(weights, dtype=float), g, method=scaling)

    uniq = np.array(sorted(pd.unique(g)))
    gi = np.searchsorted(uniq, g)  # 0..J-1 cluster index per row
    J = len(uniq)
    p = Xmat.shape[1]

    beta = np.zeros(p)
    u = np.zeros(J)
    sigma_u2 = 1.0

    for _ in range(max_iter):
        eta = Xmat @ beta + u[gi]
        eta = np.clip(eta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        v = np.clip(mu * (1.0 - mu), 1e-6, None)
        z = eta + (yv - mu) / v                 # working response
        W = w_scaled * v                        # combined IRLS x survey weight

        # alternate beta (weighted GLS) and u (BLUP shrinkage) given sigma_u2
        for _inner in range(50):
            # u_j = sum_i W_ij (z_ij - x_ij beta) / (sum_i W_ij + 1/sigma_u2)
            r = z - Xmat @ beta
            Tj = np.bincount(gi, weights=W, minlength=J)
            Sj = np.bincount(gi, weights=W * r, minlength=J)
            u_new = Sj / (Tj + 1.0 / sigma_u2)
            # beta via weighted LS on (z - u)
            resid = z - u_new[gi]
            XtW = Xmat.T * W
            beta_new = np.linalg.solve(XtW @ Xmat + 1e-8 * np.eye(p), XtW @ resid)
            if np.max(np.abs(beta_new - beta)) < tol and np.max(np.abs(u_new - u)) < tol:
                beta, u = beta_new, u_new
                break
            beta, u = beta_new, u_new

        # Schall update of the random-intercept variance
        Tj = np.bincount(gi, weights=W, minlength=J)
        eff_df = np.sum(Tj / (Tj + 1.0 / sigma_u2))   # effective # of REs
        sigma_u2_new = float(np.sum(u ** 2) / max(eff_df, 1e-8))
        if abs(sigma_u2_new - sigma_u2) < tol:
            sigma_u2 = sigma_u2_new
            break
        sigma_u2 = sigma_u2_new

    # fixed-effect SEs from the final weighted GLS information matrix
    XtW = Xmat.T * W
    cov = np.linalg.inv(XtW @ Xmat + 1e-8 * np.eye(p))
    se = np.sqrt(np.diag(cov))
    fe_names = ["Intercept"] + feats
    fe = pd.DataFrame({"term": fe_names, "coef": beta, "sd": se})
    fe["odds_ratio"] = np.exp(fe["coef"])

    school_var = float(sigma_u2)
    return {
        "school_sd": float(np.sqrt(school_var)),
        "school_var": school_var,
        "icc": icc_logistic(school_var),
        "fixed_effects": fe,
        "random_effects": pd.Series(u, index=uniq),
        "n_groups": J,
        "n_obs": int(len(yv)),
        "scaling": scaling,
    }


def predict_random_intercept(res, X_std: pd.DataFrame, groups: np.ndarray,
                             train_groups: np.ndarray) -> np.ndarray:
    r"""
    Predicted probabilities :math:`\sigma(\beta_0+\beta^\top x + \hat u_j)` for new
    rows. The school random effect is added when the school was seen in training
    (its :math:`\hat u_j` is estimable) and set to 0 (the population mean) for
    unseen schools - the honest out-of-fold behaviour.
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
