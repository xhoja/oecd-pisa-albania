r"""
Per-notebook "Methods & formulas" reference cells.

Single source of truth for the mathematical exposition inserted just after each
notebook's intro. Imported by `_build_notebooks.py` (so a rebuild includes them)
and by `_inject_formulas.py` (so the already-executed notebooks get them without
losing their outputs). Notebook 10 already carries its formulas inline, so it has
no entry here.

Each value is Markdown with LaTeX ($...$ / $$...$$). Keep the leading marker line
`## Methods & formulas (reference)` — the injector uses it to stay idempotent.
"""
from __future__ import annotations

MARKER = "## Methods & formulas (reference)"

FORMULAS: dict[str, str] = {
    "01": r"""## Methods & formulas (reference)

Every statistic below is **survey-weighted**, and every inferential one carries a
**design-based** standard error. The machinery used throughout the project:

**Weighted mean / proportion.** For values $x_i$ with final student weights
$w_i=$ `W_FSTUWT`,
$$\bar x_w=\frac{\sum_i w_i x_i}{\sum_i w_i},\qquad
\hat p_w=\frac{\sum_i w_i\,\mathbb{1}[x_i=1]}{\sum_i w_i}.$$

**Plausible values (PVs).** Achievement is not observed but *drawn*: PISA ships
$L=10$ plausible values per student (imputations from the item-response
posterior). A statistic is computed on each PV and combined (Rubin, below);
collapsing the PVs to their mean understates uncertainty.

**BRR (Balanced Repeated Replication).** The *sampling* variance uses $R=80$ Fay
replicate weights $w_i^{(r)}$:
$$\operatorname{Var}_{\text{smp}}(\hat\theta)=\frac{1}{R\,(1-k)^2}
\sum_{r=1}^{R}\big(\hat\theta^{(r)}-\hat\theta\big)^2,\qquad k=0.5\ (\text{Fay}).$$

**Rubin's rules (combine PVs).** With per-PV estimate $\hat\theta_\ell$ and its
sampling variance $U_\ell$,
$$\bar\theta=\tfrac1L\!\sum_\ell\hat\theta_\ell,\quad
\bar U=\tfrac1L\!\sum_\ell U_\ell,\quad
B=\tfrac{1}{L-1}\!\sum_\ell(\hat\theta_\ell-\bar\theta)^2,$$
$$\operatorname{Var}(\bar\theta)=\bar U+\Big(1+\tfrac1L\Big)B,\qquad
\text{FMI}\approx\frac{(1+1/L)\,B}{\operatorname{Var}(\bar\theta)}.$$
$B$ is the between-PV (imputation) variance; **FMI** is the share of total
variance due to the PV draw.

**Effect sizes.** Weighted **Cohen's $d$** (at-risk vs proficient on a feature):
$d=\dfrac{\bar x_{w,1}-\bar x_{w,0}}{s_{w,\text{pool}}}$.
**Cramér's $V$** for an $r\times c$ table: $V=\sqrt{\dfrac{\chi^2/n}{\min(r-1,c-1)}}$,
with $\chi^2$ on **normalized** weights (rescaled to sample $n$) so
population-scale weights don't inflate it.""",

    "02": r"""## Methods & formulas (reference)

The cross-country comparison reuses the weighted estimators and the **BRR + Rubin**
design-based SE from notebook 01 (each 2022 cohort has 80 Fay replicate weights, so
every SE here is fully design-based). Two additions:

**SES gradient.** The socioeconomic gradient is the *weighted* OLS slope of math
score on the standardized SES index (ESCS):
$$\widehat{\text{score}}_i=\alpha+\beta\,\text{ESCS}_i,\qquad
\hat\beta=\frac{\sum_i w_i\,(\text{ESCS}_i-\bar{\text{ESCS}}_w)(y_i-\bar y_w)}
{\sum_i w_i\,(\text{ESCS}_i-\bar{\text{ESCS}}_w)^2}.$$
$\hat\beta$ = score points per 1 SD of ESCS. A **steep** slope = strong
socioeconomic determinism; a **flat** slope sitting at a low mean is a **floor
effect** (everyone low regardless of SES), not equity.

**Significance of a country gap.** Two countries differ significantly when their
design-based 95% intervals $\hat\theta\pm1.96\,\text{SE}$ do not overlap — a
conservative visual test.""",

    "03": r"""## Methods & formulas (reference)

**Covariate shift** = the feature distribution changes between cohorts,
$P_{2018}(x)\neq P_{2022}(x)$, even if the labelling rule $P(y\mid x)$ is stable.
Three quantifications:

**Domain classifier (classifier two-sample test).** Label 2018 rows $0$ and 2022
rows $1$ and train a classifier to separate them from features alone. Held-out
AUC $\gg 0.5$ ⇒ the distributions are distinguishable; AUC $\approx0.5$ ⇒ no
detectable shift.

**Standardized mean difference (per feature).**
$\text{SMD}_f=\dfrac{\bar x_{f,2022}-\bar x_{f,2018}}{s_{f,\text{pool}}}$ — which
features moved, and how far, in pooled-SD units.

**Maximum Mean Discrepancy (MMD).** A kernel two-sample statistic; with Gaussian
kernel $k(a,b)=\exp(-\lVert a-b\rVert^2/2\sigma^2)$,
$$\text{MMD}^2=\mathbb{E}[k(x,x')]+\mathbb{E}[k(z,z')]-2\,\mathbb{E}[k(x,z)],
\qquad x\sim P_{2018},\ z\sim P_{2022}.$$
It is $0$ iff the distributions match; larger = more divergence.""",

    "04": r"""## Methods & formulas (reference)

**Task.** Binary classification of *low proficiency* (math PV $<420$, below
Level 2). Each model outputs $\hat p(x)=P(\text{at-risk}\mid x)$.

**The model zoo — what each optimizes.**
- **Logistic regression** — linear log-odds
  $\log\frac{\hat p}{1-\hat p}=\beta_0+\beta^\top x$, fit by weighted maximum
  likelihood with an $\ell_2$ penalty; the interpretable baseline.
- **Decision tree** — recursive splits minimizing Gini impurity
  $G=\sum_c \hat\pi_c(1-\hat\pi_c)$.
- **Random forest / Extra-Trees** — average of $B$ decorrelated bagged trees
  (variance reduction).
- **Gradient boosting / XGBoost / LightGBM / CatBoost** — additive stagewise
  trees $F_m=F_{m-1}+\nu\,h_m$, each $h_m$ fitting the negative gradient of
  log-loss; $\nu$ = learning rate.

**Class imbalance.** `class_weight='balanced'` scales class $c$ by $n/(2n_c)$ so
the minority (proficient) isn't ignored, multiplied by each student's PISA weight.

**Evaluation — repeated stratified $k$-fold CV** ($5\times4$, preserving the
at-risk ratio); all preprocessing fit **inside** each train fold (no leakage).
Metrics:
- **ROC-AUC** $=P(\hat p_{+}>\hat p_{-})$ — threshold-free ranking; **PR-AUC**
  (average precision) for the positive class.
- **Macro-F1** $=\tfrac12(F1_0+F1_1)$, $F1=\frac{2\,\text{prec}\cdot\text{rec}}
  {\text{prec}+\text{rec}}$.
- **MCC** $=\dfrac{TP\cdot TN-FP\cdot FN}
  {\sqrt{(TP{+}FP)(TP{+}FN)(TN{+}FP)(TN{+}FN)}}$ — balanced under skew.

**Corrected inference.** CV folds share training data, so fold scores are
correlated and a naïve $t$-CI is too narrow; the **Nadeau-Bengio** correction
inflates the variance by $\big(\tfrac1J+\tfrac{n_\text{test}}{n_\text{train}}\big)$
(notebook 10). For a single held-out set we use the **DeLong** test for correlated
ROC-AUCs. The headline AUC is combined across the 10 PVs with **Rubin's rules**
(notebook 01).

**Out-of-sample protocol.** Train 2009–2018, test 2022: if the covariate shift
(notebook 03) matters, transfer AUC falls.""",

    "05": r"""## Methods & formulas (reference)

**SHAP (SHapley Additive exPlanations).** Attributes a prediction to features via
the game-theoretic **Shapley value** — the fair payout of feature $i$ averaged
over all orderings:
$$\phi_i=\!\!\sum_{S\subseteq F\setminus\{i\}}\!\!
\frac{|S|!\,(|F|-|S|-1)!}{|F|!}\,\big[f(S\cup\{i\})-f(S)\big].$$
They obey **local accuracy** (additivity): $f(x)=\phi_0+\sum_i\phi_i$, where the
base value $\phi_0=\mathbb{E}[f]$ — contributions plus base reconstruct the
prediction (here in log-odds). **TreeSHAP** computes them *exactly* for tree
ensembles in polynomial time.

**Global importance.** Mean absolute SHAP over the sample,
$\text{Imp}_i=\tfrac1n\sum_k|\phi_i^{(k)}|$ — the average magnitude of feature
$i$'s push on the prediction.

**Beeswarm / dependence.** Each dot is one student's $\phi_i$; a dependence plot
shows $\phi_i$ vs $x_i$ (coloured by an interacting feature) to reveal effect
direction and moderation. SHAP explains the **model** (associational), not
causation.""",

    "06": r"""## Methods & formulas (reference)

**Local SHAP (waterfall).** For one student the signed $\phi_i$ (log-odds) stack
from the base value $\phi_0=\mathbb{E}[f]$ to the prediction
$f(x)=\phi_0+\sum_i\phi_i$ (local accuracy). Red $\phi_i>0$ push toward at-risk,
blue $\phi_i<0$ toward proficient.

**Partial Dependence (PDP).** Marginal effect of feature $S$, averaging over the
other features' empirical distribution:
$$\text{PDP}_S(v)=\frac1n\sum_{k=1}^{n} f\big(v,\,x^{(k)}_{\setminus S}\big).$$

**Individual Conditional Expectation (ICE).** One curve per instance,
$\text{ICE}^{(k)}_S(v)=f(v,x^{(k)}_{\setminus S})$; the PDP is their average.
**Centered ICE** anchors each curve at the grid start,
$\text{ICE}^{(k)}(v)-\text{ICE}^{(k)}(v_0)$, so heterogeneous slopes — the
fingerprint of interactions — show up where the average curve hides them.""",

    "07": r"""## Methods & formulas (reference)

**Group fairness metrics** for a protected attribute $A$, prediction $\hat y$ and
truth $y$, all **survey-weighted**. Per group $a$:
- **Demographic parity** (selection rate) $\ \text{SR}_a=P(\hat y=1\mid A=a)$.
- **Equal opportunity** (TPR) $\ \text{TPR}_a=P(\hat y=1\mid y=1,A=a)$.
- **FPR** $\ \text{FPR}_a=P(\hat y=1\mid y=0,A=a)$.
- **Calibration** $\ P(y=1\mid \hat p,A=a)\approx\hat p$ within probability bins.

Each **gap** $=\max_a(\cdot)-\min_a(\cdot)$; $0$ = parity. These criteria are
mutually incompatible in general (they cannot all hold at once unless base rates
are equal or prediction is perfect), so we report the *set*, not one number.

**Leakage-safe audit.** Predictions are **out-of-fold** (each fold's pipeline fit
on train only); **SES quintiles** use *weighted* ESCS breakpoints (population, not
sample, quintiles). The **threshold sweep** recomputes every gap over decision
thresholds $t$ (predict $\hat y=\mathbb{1}[\hat p\ge t]$) to test whether the
operating point can buy fairness.""",

    "08": r"""## Methods & formulas (reference)

**Per-country models.** One school-context LightGBM per country, scored by weighted
5-fold CV AUC (notebook 04 metrics). Per-country AUCs are **not** directly
comparable as "model quality": separability is hardest near a saturated base rate
(when $\approx\!3/4$ are at-risk there is little contrast), so AUC partly reflects
prevalence.

**SHAP rank matrix.** Within each country, features are ranked by mean $|\phi_i|$
(notebook 05); rank 1 = top driver. The feature $\times$ country matrix exposes
**universal** drivers (similar rank everywhere) vs **country-specific** ones. The
colormap follows the project schema — **dark = more important** (colorblind-safe),
rank 1 darkest.

**SES gradient** = weighted slope of score on ESCS (notebook 02); Albania's is the
*flattest* — a floor effect, not equity.""",

    "09": r"""## Methods & formulas (reference)

**Why scenarios, not a point forecast.** Five cycles and a 2022 structural break
make any single extrapolation indefensible; we propagate the **design-based SE** of
each cycle (BRR + PV, notebook 01) through a Monte-Carlo and report **scenarios**.

**Monte-Carlo.** Draw each historical rate
$\tilde p_c\sim\mathcal{N}(\hat p_c,\ \text{SE}_c^2)$ over $N$ simulations, form each
scenario's next-cycle value per draw, and read off the median and the 5th/95th
percentiles (a 90% predictive interval).

**Scenarios.**
- **Persistence:** $p_{\text{next}}=p_{2022}+\varepsilon$ with a small drift SD
  (the crisis level holds).
- **Recovery:** the pre-COVID trend (weighted OLS line through 2009–2018) resumes,
  extrapolated forward (the 2022 spike fully reverses).
- **Partial:** the mean of persistence and recovery (the shock half-reverses).
- **Naïve linear** *(discarded):* a weighted line through **all five** cycles —
  shown only to demonstrate that ignoring the break manufactures false confidence.

The honest signal is the **width** of the plausible band, not a single number.""",
}
