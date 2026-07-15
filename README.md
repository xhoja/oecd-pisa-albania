# Predicting Albanian Student Low-Proficiency Risk Using OECD PISA Data (2009–2022)
### A Comparative Machine-Learning Framework

A data-science project predicting which Albanian 15-year-olds are
at risk of low mathematics proficiency (below PISA Level 2, score < 420), benchmarked
against Balkan peers, top performers, and GDP-matched economies, with rigorous PISA
methodology (sampling weights, plausible values), explainability, and a covariate-shift
analysis of Albania's dramatic 2022 regression.

---

## Technology Stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-latest-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-latest-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-latest-8CAAE6?logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-F7931E?logo=scikitlearn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-latest-3B5998)
![CatBoost](https://img.shields.io/badge/CatBoost-headline-FFCC00?logoColor=black)
![LightGBM](https://img.shields.io/badge/LightGBM-boosters-9ACD32)
![XGBoost](https://img.shields.io/badge/XGBoost-boosters-337AB7)
![Optuna](https://img.shields.io/badge/Optuna-nested--CV%20HPO-8A2BE2)
![SHAP](https://img.shields.io/badge/SHAP-explainability-FF6F61)
![Altair](https://img.shields.io/badge/Altair-charts-1F77B4)
![matplotlib](https://img.shields.io/badge/matplotlib-figures-11557C)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-article%20report-008080?logo=latex&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-188%20passing-0A9EDC?logo=pytest&logoColor=white)

| Area | Tools |
|---|---|
| **Language & core** | Python 3, pandas, NumPy, SciPy, scikit-learn, statsmodels |
| **Models** | CatBoost, LightGBM, XGBoost, scikit-learn (GBM / RF / ExtraTrees / LogReg / SVM / KNN / MLP / NaiveBayes / DecisionTree), Optuna (nested-CV HPO) |
| **PISA methodology** | pyreadstat (SPSS `.sav`) + custom fixed-width parser; survey weights, plausible values with Rubin's rules, BRR (80 Fay replicates) |
| **Explainability & fairness** | SHAP (TreeSHAP), partial dependence / ICE, weighted fairness metrics, conformal prediction, counterfactual recourse |
| **Visualisation** | matplotlib, seaborn, Altair; colorblind-safe palette |
| **Testing & quality** | pytest (188 tests on synthetic fixtures), structlog, ruff, mypy |
| **Paper** | LaTeX `article` (single-column report: title page, table of contents, running headers), BibTeX |
| **Interactive dashboard (UI)** | Streamlit (app), Altair (charts), fpdf2 (PDF report), joblib (model bundle) |

Logic lives in tested `src/` modules; notebooks are narrative layers over them, so every
statistic in this README is backed by a unit-tested function.

---

## Research Question

> Can machine learning identify Albanian students at risk of low academic proficiency
> from background variables, and do the predictive factors differ from OECD peers, and
> can those differences explain Albania's catastrophic 2018→2022 reversal?

**Headline findings so far:**
1. A domain classifier distinguishes 2018 from 2022 Albanian students with **AUC 0.98** on
   background features alone, the 2022 collapse is a *structural* distributional shift, not
   merely lower scores.
2. **School socioeconomic composition is the missing lever.** Adding survey-weighted
   school-mean features lifts every model's weighted CV AUC ~**+0.05** (CatBoost 0.73→**0.78**,
   significant, fold-safe), and once school context is in, the **gradient boosters pull
   significantly ahead of logistic regression** (GBM vs LR *p* = 0.001). The earlier "LR ties
   the boosters / data-limited ceiling" result was an artefact of an impoverished,
   student-only feature set, the ceiling was the *features*, not the data or the model.
3. **...but the ~0.78 ceiling above school composition is a genuine data limit.** Linking the
   PISA **school questionnaire** (principal-reported resources, staff, class size, leadership)
   directly onto students adds **no significant signal beyond school composition** (best +0.004
   AUC, all *p* > 0.24): school-mean SES already proxies school inputs. Three independent routes:
   composition booster, multilevel ICC ≈ 0.31, and the direct questionnaire linkage, converge on
   the same ceiling (notebook 06).

---

## Key Results (current)

### Albania 2022, model comparison (5×4 repeated stratified CV, weighted metrics)

**With school context** (13 student features + survey-weighted school-mean
ESCS/HOMEPOS/ANXMAT/TEACHSUP + cohort size). This is the current headline set; the
`Δ AUC` column is the lift over the student-only baseline (see ablation below).

| Model | ROC-AUC | Δ AUC vs. student-only | PR-AUC | F1-macro | MCC |
|---|---|---|---|---|---|
| **CatBoost** | **0.783** | +0.052 | 0.907 | 0.693 | 0.393 |
| Gradient Boosting | 0.777 | +0.050 | 0.905 | 0.665 | 0.365 |
| LightGBM | 0.775 | +0.060 | 0.902 | 0.688 | 0.381 |
| Random Forest | 0.769 | +0.060 | 0.895 | 0.644 | 0.345 |
| Extra Trees | 0.758 | +0.051 | 0.884 | 0.641 | 0.332 |
| XGBoost | 0.756 | +0.058 | 0.892 | 0.657 | 0.333 |
| Logistic Regression | 0.754 | +0.027 | 0.897 | 0.649 | 0.332 |
| SVM (RBF) | 0.749 | +0.037 | 0.894 | 0.609 | 0.282 |
| MLP (128-64) | 0.743 | +0.034 | 0.884 | 0.620 | 0.295 |
| Naive Bayes | 0.701 | +0.019 | 0.856 | 0.610 | 0.263 |
| KNN | 0.700 | +0.036 | 0.861 | 0.613 | 0.257 |
| Decision Tree | 0.609 | +0.041 | 0.797 | 0.608 | 0.217 |

The four boosters (CatBoost/GBM/LightGBM/RF) form the top band and all beat Logistic
Regression significantly (e.g. GBM vs LR corrected-resampled *p* = 0.001); the interpretable
baseline gains least from school context (+0.027) because the boosters exploit the
student × school-composition interaction a linear model cannot. **SVM and MLP (added in the
Phase 5b completeness pass) land mid-table, below LR and well below the boosters**, confirming
the podium is a booster story, not an artefact of omitting kernel/neural learners; neither
non-linear margin nor a small dense net beats gradient boosting on this tabular, ~6k-row,
weighted problem. MLP is fit unweighted (`MLPClassifier` takes no `sample_weight`). See
`outputs/results/albania_2022_model_comparison_school.csv` and `..._pairwise_nb_school.csv`.

<details><summary><b>Student-only baseline</b> (original 13-feature set, click to expand)</summary>

| Model | ROC-AUC (Nadeau-Bengio 95% CI) | PR-AUC | F1-macro | MCC |
|---|---|---|---|---|
| **CatBoost** | **0.731** (0.714–0.749) | 0.880 | 0.647 | 0.305 |
| Logistic Regression | 0.727 (0.704–0.751) | 0.882 | 0.621 | 0.281 |
| Gradient Boosting | 0.727 (0.711–0.743) | 0.878 | 0.609 | 0.283 |
| LightGBM | 0.715 (0.700–0.730) | 0.872 | 0.638 | 0.282 |
| SVM (RBF) | 0.712 (0.685–0.739) | 0.873 | 0.571 | 0.242 |
| Random Forest | 0.710 (0.692–0.727) | 0.868 | 0.599 | 0.261 |
| MLP (128-64) | 0.709 (0.684–0.735) | 0.867 | 0.580 | 0.251 |
| Extra Trees | 0.707 (0.688–0.726) | 0.866 | 0.608 | 0.260 |
| XGBoost | 0.698 (0.681–0.715) | 0.863 | 0.617 | 0.262 |
| Naive Bayes | 0.682 (0.659–0.705) | 0.846 | 0.599 | 0.246 |
| KNN | 0.664 (0.638–0.689) | 0.842 | 0.588 | 0.215 |
| Decision Tree | 0.568 (0.548–0.589) | 0.780 | 0.567 | 0.135 |

In the student-only set the top three (CatBoost ≈ LR ≈ GBM) were a statistical tie, the
finding that motivated adding school context.

</details>

All metrics are **weighted** with the survey weights (a correctness fix: scaler-wrapped
models such as Logistic Regression were previously fit unweighted because the weight kwarg
was routed to the wrong pipeline step, now fixed, which lifted LR from 0.726 to 0.727 and
into second place). KNN is the one exception: `KNeighborsClassifier` has no `sample_weight`,
so its fit is inherently unweighted (its metrics are still weighted at evaluation).

The 95% intervals use the **Nadeau-Bengio corrected-resampled variance**, not a naïve
`S/√k`: repeated-CV folds share training data, so their scores are positively correlated
and the naïve interval is too narrow. The correction replaces the `1/k` term with
`1/k + n_test/n_train`, giving an honest (wider) interval.

> LightGBM used to segfault (`rc=-11`) at the C level on this host. The cause was OpenMP
> **import order**, not a `libomp` install: LightGBM links Homebrew's libomp, and if
> scikit-learn's bundled libomp loaded first the two runtimes collided and aborted the fit.
> The isolated-subprocess workers now import the boosters before scikit-learn, so LightGBM
> runs cleanly and is included above. Fits are still isolated in subprocesses, so any
> genuine C-level crasher is skipped gracefully rather than taking down the run.

### Are the differences real? (pairwise significance)

A ranking table alone invites over-reading a 0.005 AUC gap. We test it directly.

- **In CV**, pairwise **Nadeau-Bengio corrected resampled t-test** on the paired per-fold
  AUCs. In the **student-only** set (`albania_2022_pairwise_nb.csv`) the top three were a
  statistical tie: CatBoost ≈ LR (*p* = 0.50), CatBoost ≈ GBM (*p* = 0.37), GBM ≈ LR
  (*p* = 0.91), a transparent logistic regression matched the best boosters. **Adding school
  context breaks that tie** (`albania_2022_pairwise_nb_school.csv`): the boosters
  (CatBoost/GBM/LightGBM/RF) become a top band that beats LR significantly (GBM vs LR
  *p* = 0.001), because they exploit the student × school-composition interaction LR cannot.
- **Out-of-sample**, pairwise **DeLong test** on the shared 2022 test set
  (`oos_2022_experiment.json → delong_pairwise`): the OOS leaders Random Forest, Gradient
  Boosting and LightGBM are mutually indistinguishable (RF vs GBM *p* = 0.64, RF vs LightGBM
  *p* = 0.66, RF vs CatBoost *p* = 0.05), while they all beat XGBoost and LR significantly.

### Model-improvement pass (pre-Phase-8)

Two levers were tested against the student-only ceiling; both are leakage-safe and weighted.

**1. School context, breaks the AUC ceiling.** HPO gave no significant gain, suggesting a
data ceiling; the real cause was a missing feature family. Adding survey-weighted,
leave-one-out **school-mean** ESCS/HOMEPOS/ANXMAT/TEACHSUP + cohort size lifts weighted CV
AUC ~**+0.05** for every model (table above), all *p* < 0.001 by the paired corrected
resampled t-test. A **fold-safe** rerun (school means recomputed from the training fold only)
reproduces an identical delta (`school_features_foldsafe_2022.csv`), so the lift is a genuine
compositional effect, not leakage. `scripts/run_school_features_experiment.py` (+ `_foldsafe`).

**2. PV-stacking, marginal.** Training on all 10 plausible values as stacked rows (target
= PVk < L2, evaluated on the PV-mean target, StratifiedKFold by student) beats the PV-mean
target only slightly and only for two models, LightGBM +0.011 (*p* = 0.05), GBM +0.009
(*p* = 0.007); CatBoost/LR flat. It is the more PISA-correct target treatment but the practical
AUC gain is small (≪ school context). Kept as an ablation, not folded into the headline.
`scripts/run_pv_stacking_experiment.py` → `pv_stacking_ablation_2022.csv`.

**3. Operating point, threshold tuning + isotonic calibration.** The comparison scores
F1/MCC at a fixed 0.5, but `class_weight='balanced'` recenters the scores. Tuning the
threshold on out-of-fold train predictions lifts MCC ~+0.02–0.03 and F1-macro ~+0.03–0.05;
weighted **isotonic calibration** cuts ECE ~5–7× (CatBoost 0.147→0.027, LR 0.209→0.029) and
improves Brier, directly strengthening the fairness/calibration story. AUC is unchanged (a
ranking metric). `scripts/run_threshold_calibration.py` → `threshold_calibration_2022.csv`,
figure `M1`.

### Out-of-sample (train 2009–2018 → test 2022, weighted)

GBM & RF lead at **0.674** AUC, with LightGBM level at 0.673; CatBoost 0.665, Extra Trees
0.663, XGBoost 0.639, LR 0.622.
The shift is **detectable** (AUC 0.98) yet models still transfer moderately, a key nuance
(covariate shift *without* total collapse). Threshold tuned on train only; ECE reported.

### Hyper-parameter optimisation (Phase 5, nested CV)

Optuna tuning of the top three models in *proper nested CV* (inner tuning, outer evaluation,
leakage-safe & weighted) is compared against the registry defaults with the corrected
resampled t-test (`outputs/results/hpo_summary.csv`):

| Model | Default AUC | Tuned AUC | Gain | *p* | Significant? |
|---|---|---|---|---|---|
| CatBoost | 0.733 | 0.735 | +0.002 | 0.81 | No |
| Gradient Boosting | 0.727 | 0.731 | +0.004 | 0.45 | No |
| Logistic Regression | 0.726 | 0.726 | −0.001 | 0.43 | No |

**Tuning yields no statistically significant gain** for any model, the registry defaults are
already at the ceiling *for the student-only feature set*. This looked like a data ceiling, but
it was a **feature** ceiling: adding school context (below) lifts AUC ~+0.05 where tuning could
not. The lesson holds in reverse, capacity/tuning were maxed; the missing signal was a feature.

### SHAP global importance (Albania 2022, LightGBM, school-context model)

`SCH_MEAN_HOMEPOS` > `ANXMAT` > `SCH_MEAN_TEACHSUP` > `SCH_MEAN_ESCS` > `HISCED` > `HOMEPOS` >
`SCH_N` > `SCH_MEAN_ANXMAT` … **Four of the top seven drivers are school-level**, the model
reads a student's risk more from the socioeconomic *composition* of their school than from their
own resources (the compositional effect made explicit). Math anxiety (`ANXMAT`) is the strongest
individual factor; immigration status remains negligible. (On the student-only model the ranking
was `ANXMAT` > `HISCED` > `HOMEPOS` …; adding school context reorders the whole story.)

**Local explanations (Phase 6).** `scripts/run_explainability_cases.py` adds the local view:
SHAP waterfalls for one confidently-correct TP (P = 0.997) and TN (P = 0.012) and one
confidently-*wrong* FP (P = 0.88) and FN (P = 0.07), the cases worth inspecting, plus PDP +
centered-ICE curves for the top raw drivers (`ANXMAT`, `HISCED`, `HOMEPOS`, `BELONG`). Figures:
`outputs/figures/shap/D3_shap_local_cases`, `D4_pdp_ice`. Notebook 04.

### Fairness audit (Albania 2022, survey-weighted, out-of-fold @ 0.5, school-context model)

Out-of-fold LightGBM predictions audited across protected groups; every rate is
survey-weighted (`outputs/results/fairness_audit_2022.json`). Deltas are vs. the student-only
model:

| Attribute | Parity gap | Equal-opp (TPR) gap | FPR gap |
|---|---|---|---|
| SES quintile | **0.48** | 0.33 | **0.63** |
| Immigrant status | 0.22 (↑ from 0.13) | 0.18 | **0.59** (↑ from 0.38) |
| GENDER | 0.11 (↓ from 0.18) | 0.09 | 0.05 (↓ from 0.14) |

**SES is still where the model is least fair, and school context does not fix it**, school-mean
SES *is* SES, so making it a top feature doubly encodes the gradient. Bottom-quintile students
are flagged at 91% vs. 43% for the top, with a **0.63 FPR gap** (equalized-odds concern). Adding
school context has a **split effect**: the **gender gap shrank** (parity 0.18→0.11, FPR 0.14→0.05
- school context absorbed part of the gender signal), while the **immigrant FPR gap grew**
(0.38→0.59; small groups, one FPR=1.0 cell, a flag, not a precise estimate). The threshold sweep
(`fairness_threshold_sweep_gender.csv`, figure `E1`) shows the gaps shift rather than vanish with
the operating point, mitigation needs constraints, not a knob. Notebook 07.

### Cross-country (PISA 2022, weighted low-proficiency rate)

Albania **75%** (highest), above Balkan peers (N. Macedonia 67%, Montenegro 60%) and
GDP-matched economies (Colombia 73%, Mexico 67%); Estonia lowest at 14%.

### Comparative modeling (Phase 8), Albania vs. peers

Per-country school-context LightGBM (nine countries, 2022), weighted 5-fold CV
(`scripts/run_comparative_modeling.py`, notebook 05):

- **Albania: high prevalence, low separability, flat gradient.** Highest at-risk rate (0.75),
  one of the *lowest* model AUCs (**0.77**, with 3-in-4 at-risk there's little contrast to
  separate), and the **flattest SES gradient (~17 pts/SD)** of all nine countries. The 2022
  crisis is **broad-based**, depressed across the whole SES distribution, not concentrated in
  the poor. *(This corrects an earlier narrative that misread the gradient plot as "steep";
  Albania's slope is in fact the flattest, a floor effect, even advantaged students score ~369.)*
- **Steep-but-high contrast.** Estonia/Finland have *steeper* gradients (~38–40) yet far higher
  levels; SES predicts more there only because there is more score range to predict.
- **Drivers are shared, not unique.** A SHAP feature × country **rank matrix**
  (`comparative_shap_rank_matrix.csv`, figure `F1`) shows **math anxiety universal** (top-3 almost
  everywhere, incl. Estonia) and **school socioeconomic composition** dominant in most systems.
  Albania mirrors its Balkan peers, what differs is the *level/breadth* of low proficiency, not
  the mechanism. Country-specific: grade repetition is #2 in Colombia; individual ESCS is #1 in
  Finland.
- **Policy read:** flat gradient + school-composition dominance means SES-targeted transfers
  alone miss most at-risk Albanian students, system-wide levers (teacher support, math-anxiety
  reduction, raising the floor across all schools) are needed.

### Forecast, next cycle (2026)

PISA moved to a **4-year** cadence after COVID delayed 2021 → 2022, so the next assessment is
**2026**. With only five cycles and a 2022 structural break, a point forecast is indefensible, so
we report **scenarios** and propagate each cycle's design-based (BRR + PV) SE through a
Monte-Carlo (`src/forecast/scenarios.py`, notebook 10):

| Scenario | 2026 median | 90% PI |
|---|---|---|
| Persistence (crisis holds) | **0.74** | 0.66–0.82 |
| Partial reversion | 0.47 | 0.43–0.52 |
| Pre-COVID recovery (trend resumes) | 0.21 | 0.18–0.24 |
| *naive all-cycles line (discarded)* | *0.73* | *0.72–0.74* |

The honest message is the **width (~21–74%)**, not a single number. Because the 2022 spike
reflects durable disruptions (the notebook-03 covariate shift), the defensible zone is
**partial-reversion (~47%) to persistence (~74%)**; the ~21% recovery is an optimistic floor
that assumes the pre-COVID trajectory resumes untouched. The naive line only *looks* confident
because it ignores the break. Bottom line: absent a strong targeted recovery, plan for 2026 to
sit well above Albania's 2018 low of 42%.

### Causal probe — 2019 Durrës earthquake (a well-identified null)

The whole framework is associational, so we test the one natural experiment in reach: the
26 Nov 2019 M6.4 Durrës earthquake. PISA's sampling `STRATUM` encodes a North/Center/South band;
quake damage concentrated in the **Center** band (Durrës + Tirana). A difference-in-differences
(Center = treated vs North+South, **2018 → 2022**) with design-based BRR + PV standard errors
(`src/causal/`, `scripts/run_causal_did.py`, notebook 09):

| Estimator | Effect on at-risk rate | p |
|---|---|---|
| Naïve DiD, 2018→2022 | **+4.7 pp** | <0.001 |
| Placebo DiD, 2015→2018 (parallel-trends test) | **−7.0 pp** | <0.001 |
| Triple-difference (× urbanicity), 2018→2022 | −3.3 pp | 0.37 |
| DiD, school-clustered SE | +5.2 pp | 0.22 |

The naïve DiD **does not survive**: the pre-quake placebo rejects parallel trends (Center was
already on a steeper improving trajectory), the urbanicity triple-difference is ~0 with the wrong
sign, and the effect is null once inference clusters on schools. At-risk jumps ~30–40 pp in *every*
band, so the 2022 collapse is **national** (COVID + sample-coverage shock), not a localised Durrës
signal. We report the null for transparency — it forecloses the "earthquake caused the crisis"
over-claim and reinforces the structural-crisis reading. The public file cannot separate Durrës
from Tirana and only 2015/2018 carry a decodable band, so no estimator can rescue identification.

### Does the data-ceiling claim generalise? (9 countries)

Albania's thesis — its ~0.78 ceiling is a **data property, not a feature shortfall** — is tested
on all nine countries by running its two data-only legs (`src/models/multilevel.py`,
`scripts/run_ceiling_generalization.py`, notebook 05): the **school-composition lift** (weighted 5×2
CV AUC, student features vs + leave-one-out school-mean composition, Nadeau-Bengio corrected test)
and the **between-school ICC** (null multilevel model).

| System | ICC | base → +composition AUC | lift | sig |
|---|---|---|---|---|
| Balkans (MKD/MNE/SRB/BGR) | .34–.56 | up to .79 → .86 | +.06 to +.08 | ✓ |
| GDP-matched (COL/MEX) | .31–.45 | .75–.83 → .79–.86 | +.03 to +.04 | ✓ |
| Albania | .31 | .71 → .77 | +.059 | ✓ |
| Estonia *(boundary)* | .21 | .73 → .75 | +.022 | n.s. (p=.09) |
| Finland *(boundary)* | .10 | .78 → .79 | +.008 | weak |

The composition lift is **significant in 8/9 countries** and the achievable AUC tracks the ICC at
**r = 0.80** — the ceiling is set by how much risk lives *between* schools, not by the feature set.
The boundary is the **equitable top performers**: in Finland (ICC 0.10) schools are socially mixed,
composition adds almost nothing, and the mechanism weakens. So the claim generalises to segregated
systems and self-limits in equitable ones — reported as a boundary, not buried. (The third Albanian
leg, the school-questionnaire null, needs the linked questionnaire only Albania has, so it stays
single-country.)

### Can a new modality move the ceiling? (CBA process data)

Every feature so far is one questionnaire modality. We link a genuinely new source — PISA's
**computer-based-assessment process data** (`src/features/process.py`,
`scripts/run_process_modality.py`, notebook 06), joined 1:1 by `CNTSTUID`: per-item response
**time + visits** (2022 `STU_COG`, 2018 `STU_TTM`) and questionnaire **response latency** (2022
`STU_TIM`). Only *behavioural* features are used — item correctness mechanically defines the target,
so it is never a feature.

| 2022, at the student+school ceiling | +process AUC | lift | p |
|---|---|---|---|
| cognitive + questionnaire | **0.858** | +0.085 | <.001 |
| cognitive-only | 0.851 | +0.078 | <.001 |
| questionnaire-only *(deployable)* | 0.784 | **+0.011** | .019 |

A new modality **does** move the number — 0.773 → **0.86**, the first thing to break 0.78. **But
it is not deployable:** ~90% of the lift is cognitive-test process data, which is endogenous to the
proficiency it predicts and only exists *after* a student sits the CBA. The only pre-test modality
(questionnaire latency) adds **+1.1 pp**. So the **screening** ceiling holds at ~0.78 — the ceiling
is about *what is knowable before the test*. Reported honestly, including the large cognitive-process
lift we cannot use for early warning. (Albania fielded neither the parent nor teacher questionnaire —
both ruled out at source.)

### Fairness mitigation, not just diagnosis

The audit found the screener over-flags the poorest students. We *fix* it post-hoc — no retraining —
with **group-specific decision thresholds** on frozen out-of-fold predictions
(`src/fairness/mitigation.py`, `scripts/run_fairness_mitigation.py`, notebook 07):

| SES-quintile FPR | single 0.5 threshold | equal-FPR group thresholds |
|---|---|---|
| poorest Q1 | 0.83 | 0.42 |
| richest Q5 | 0.20 | 0.42 |
| **FPR gap** | **0.63** | **0.003** |

Equalising the false-positive rate collapses the equalized-odds gap from **0.63 to ~0.00** — the
disparity was an artefact of a blanket operating point, removable without touching the model. It
costs ~6 pp of overall recall (0.81 → 0.75), and on the fairness-utility frontier group-specific
thresholds strictly dominate a single threshold. We present it as a quantified policy option (with
the disparate-treatment caveat), not a default.

### Coverage robustness — Manski bounds

PISA 2022 covered only ~**79%** of Albania's 15-year-olds (Coverage Index 3 ≈ 0.79; 6,129 students
representing ~28,400 of ~36,000 — OECD Country Note). The reported ~74% at-risk rate is a rate
*among the covered*; the ~21% uncovered (out-of-school, disadvantaged) are unobserved. We report
partial-identification bounds instead of assuming they resemble the sample (`src/coverage/manski.py`,
`scripts/run_coverage_bounds.py`, notebook 09):

| | population at-risk rate |
|---|---|
| observed (covered), design-based ±95% CI | 0.739 ± 0.008 |
| worst-case Manski bounds | **[0.58, 0.79]** |
| monotone (uncovered ≥ covered) | **[0.74, 0.79]** |
| scenario: uncovered ~ poorest quintile | ~0.76 |

The sampling SE (0.004) badly understates total uncertainty — the coverage gap does. But the credible
(monotone) bound is one-sided: out-of-school youth are not *less* at risk, so coverage most likely
means the headline **understates** the crisis. Crucially, even the worst-case lower bound (0.58) sits
~16 pp above Albania's 2018 rate (0.42) across every plausible coverage share — **the 2018→2022
deterioration is robust to any coverage assumption**; only the level is uncertain, not the conclusion.

### The geography of the crisis — where *inside* Albania it concentrates

The national ~74% at-risk rate hides *where* the crisis lives. Albania's public file has no
subnational `REGION`, but the sampling `STRATUM` decodes into **region band (North / Center / South)
× urbanicity (Urban / Rural) × sector (Public / Private)** (`src/geography/equity.py`,
`scripts/run_geographic_equity.py`, notebook 02 Part C). Every rate is survey-weighted with a PISA
design-based SE (BRR + Rubin), and each *gap* is estimated on the contrast itself so it carries a
significance test rather than being eyeballed from two marginal CIs.

| 2022, share below Level 2 (math) | Urban | Rural |
|---|---|---|
| **North** | 0.79 | **0.83** |
| **Center** (Tirana) | 0.69 | 0.82 |
| **South** | 0.71 | 0.79 |

- **The North is highest in every cycle** and the rural bands carry the heaviest load; rural North
  (0.83) and rural Center (0.82) are the worst cells, while even urban Center — which contains the
  capital — is the *lowest* cell at 0.69, still a majority at risk.
- **The gaps are real under design-based SEs (2022):** Rural − Urban **+9.9 pp** (*p* ≈ 1e-12),
  North − South **+7.0 pp** (*p* ≈ 3e-10), and **Public − Private +25.5 pp** (*p* < 0.001) — the
  public/private divide is the widest of the three and has *grown* (0.15 in 2018 → 0.26 in 2022),
  though private schools are a small, self-selected ~13% of students (a composition signal, not a
  like-for-like school effect).
- **The 2018 → 2022 reversal is broad-based** (every cell jumped ~28–37 pp) but stacked *on top of*
  pre-existing rural disadvantage, so the rural bands ended highest.
- **Policy read:** because the SES gradient is flat (comparative section above), income-targeted
  transfers alone miss most at-risk students — **geography is a sharper targeting axis than family
  SES.** The North, rural and public-school bands are where a fixed remediation budget buys the most
  reach. Figures `GEO1`–`GEO3` (trajectory, region × urbanicity heatmap, 2018→2022 dumbbell).

### From prediction to prescription — malleable-lever ranking

SHAP ranks features by how much the model *reads* them, mixing **fixed endowments** (SES, parental
education, home possessions, immigrant status, gender, grade — a screener reads them but no policy
moves them short-run) with **malleable levers** (math anxiety, teacher support, belonging,
grade-repetition, ICT). We split the two and rank only the *actionable* levers by the population
predicted at-risk reduction from a standardized half-SD nudge in the beneficial direction — the
direction discovered on the smooth mean predicted probability, not the noisy thresholded count
(`src/models/levers.py`, `scripts/run_levers.py`, notebook 08 Part D).

| Actionable lever (0.5-SD beneficial nudge) | Δ predicted at-risk |
|---|---|
| **Math anxiety** ↓ | **−5.1 pp** |
| Teacher support ↑ | −1.0 pp |
| Grade-repetition / ICT / belonging | ≈ 0 |

- **Math anxiety is the one soft lever with real reach**, matching the counterfactual-recourse and
  multilevel evidence; everything else is essentially flat.
- **The structural floor, made explicit:** moving *every* actionable lever half an SD at once takes
  the predicted at-risk rate ~0.72 → ~0.66 (−6.9 pp), and a *hypothetical* that lifts every fixed
  endowment by the same amount (which no short-run policy can deliver) buys only ~5 pp more. Both
  bundles leave a large majority at risk — the crisis is **structural, not a tuning problem**, the
  same conclusion the policy what-ifs, the covariate shift and the flat SES gradient reach from
  other directions.
- **Levers and geography are complements:** the anxiety lever is *weaker where the crisis is
  deepest* (its reduction is far smaller in the rural band than the urban one), so anxiety
  programmes help most in urban schools while the rural / North floor needs system-level
  investment. Figure `L1`. Model-associational, not causal (stated, not hidden).

---

## Data

| Cycle | Source | Coverage |
|---|---|---|
| 2009 | `INT_STQ09_DEC11.txt` (fixed-width ASCII) + SPSS control | All countries |
| 2012 | `INT_STU12_DEC03.txt` (fixed-width ASCII) + SPSS control | All countries |
| 2015 | `CY6_MS_CMB_STU_QQQ.sav` (downloaded, 1.4 GB) | All countries |
| 2018 | `CY07_MSU_STU_QQQ.sav` | All countries |
| 2022 | `CY08MSP_STU_QQQ.SAV` | All countries |

- **Albania longitudinal:** 27,042 students, 5 cycles
- **Comparison set:** 323,121 students, ALB, MKD, MNE, SRB, BGR, EST, FIN, MEX, COL
- Raw files are **not** committed (multi-GB). Place them in `data/raw/`; processed
  parquet files are regenerated by the pipeline.

> **Data notes (verified):** Albania genuinely lacks ESCS in 2012 & 2015 (an OECD gap,
> not a bug); peer countries have full 2015 background data. North Macedonia joined PISA
> in 2015; Serbia skipped 2015.

**Design-based estimates.** PISA ships 80 Fay balanced-repeated-replication (BRR) weights.
The pipeline preserves them, and `src/data/weights.pv_statistic_brr` reports point estimates
with the *correct* PISA standard error, BRR sampling variance (per plausible value) combined
with the between-PV imputation variance via Rubin's rules. Albania's weighted at-risk math
trajectory with design-based SEs:

| Cycle | At-risk (math < L2) | SE | 95% CI | BRR |
|---|---|---|---|---|
| 2009 | 0.677 | 0.0076 | 0.663–0.692 | - (FWF) |
| 2012 | 0.607 | 0.0051 | 0.597–0.617 | - (FWF) |
| 2015 | 0.533 | 0.0108 | 0.512–0.554 | ✓ |
| 2018 | 0.424 | 0.0088 | 0.407–0.441 | ✓ |
| **2022** | **0.739** | **0.0041** | **0.731–0.747** | ✓ |

> **Replicate-weight caveat:** the 80 replicate weights ship only on the SAV cycles
> (2015/2018/2022). The FWF cycles (2009/2012) fall back to an imputation-only SE, flagged
> `BRR=False`, an honest limit for any cross-cycle significance claim.

---

## Repository Structure

```
OECD_PISA_Project/
├── configs/              # data / features / models / experiment YAML
├── src/
│   ├── data/             # extract, parse_fwf, harmonize, weights (BRR+Rubin,
│   │                     #   weighted_describe), validate (data contracts), impute, pipeline
│   ├── features/         # engineer, select, target, transformers (fold-safe), process (CBA)
│   ├── models/           # registry, hpo (Optuna nested CV), optimize (search space),
│   │                     #   evaluate (metrics, Nadeau-Bengio, DeLong), prepare,
│   │                     #   experiment (Rubin's rules), train, _isolated_worker,
│   │                     #   conformal, decision_curve, multilevel, policy, levers
│   ├── explainability/   # SHAP analysis, partial dependence (PDP/ICE), counterfactuals
│   ├── causal/           # earthquake DiD / triple-difference (design-based SE)
│   ├── geography/        # within-Albania equity: region x urbanicity x sector at-risk (BRR+Rubin)
│   ├── coverage/         # Manski partial-identification bounds (coverage robustness)
│   ├── fairness/         # group-fairness metrics
│   ├── forecast/         # scenario forecast (WLS trend + Monte-Carlo, BRR-aware)
│   ├── statistics/       # tests (chi², MMD, covariate shift, effect sizes,
│   │                     #   feature_effect_sizes)
│   ├── visualization/    # style, eda (incl. plot_ses_logistic_curve), model_plots
│   └── utils/            # config, logging, reproducibility
├── notebooks/            # 01_eda_albania, 02_crisis_in_context, 03_modeling,
│                         #   04_explainability, 05_cross_country, 06_the_ceiling,
│                         #   07_fairness, 08_decision_support, 09_crisis_robustness,
│                         #   10_forecast_2026  (all executed; each a multi-Part notebook
│                         #   consolidating the earlier analyses)
│                         #   built by _build_notebooks.py (+ _formulas.py)
├── scripts/              # run_model_comparison (--school), run_oos_experiment, run_hpo,
│                         #   run_shap_analysis, run_explainability_cases, run_fairness_audit,
│                         #   run_school_features_experiment (+_foldsafe), run_threshold_calibration,
│                         #   run_pv_stacking_experiment, run_stacking_experiment,
│                         #   run_comparative_modeling, run_multilevel, run_decision_curve,
│                         #   run_conformal, run_counterfactuals, run_policy,
│                         #   build_school_questionnaire, run_school_questionnaire_experiment,
│                         #   run_school_means_transformer_check, replot_paper_figures,
│                         #   run_geographic_equity (within-Albania map), run_levers (lever ranking),
│                         #   export_dashboard_model
├── reports/              # paper/ (LaTeX article report → main.pdf; sections/, figures/),
│                         #   dashboard/ (Streamlit risk-screener app.py), presentation/
├── tests/                # 188 pytest unit tests (weights, impute, target, transformers,
│                         #   validate, evaluate, engineer/school, forecast, statistics,
│                         #   eda_viz, dashboard), no real data needed
├── outputs/              # figures/{eda,models,shap,fairness,comparative} + results/ (csv, json)
│                         #   + models/ (dashboard bundle) + shap/
└── data/                 # raw/ (git-ignored) + processed/ (parquet)
```

---

## Reproduce

```bash
# 1. Install
pip install -e ".[dev]"     # runtime + pytest/ruff/mypy
brew install libomp         # macOS: required by XGBoost/LightGBM/CatBoost

# 2. Run the test suite (no data required, 188 tests on synthetic fixtures)
pytest                      # or: pytest -q -o addopts=""  to skip coverage

# 3. Place raw PISA files in data/raw/ (see Data table above)

# 4. Build processed datasets (all cycles, all countries)
python -m src.data.pipeline --albania-longitudinal \
    --cycles 2009 2012 2015 2018 2022 \
    --countries ALB MKD MNE SRB BGR EST FIN MEX COL

# 5. EDA notebooks (weighted stats + design-based BRR standard errors)
python notebooks/_build_notebooks.py
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

# 6. Modeling + explainability
python scripts/run_model_comparison.py   # CV table + pairwise Nadeau-Bengio test
python scripts/run_oos_experiment.py      # OOS metrics + pairwise DeLong test
python scripts/run_hpo.py                  # Optuna nested-CV HPO (tuned vs default)
python scripts/run_shap_analysis.py         # SHAP global importance + beeswarm
python scripts/run_explainability_cases.py  # local SHAP waterfalls (TP/TN/FP/FN) + PDP/ICE
python scripts/run_fairness_audit.py        # weighted fairness audit + threshold sweep

# Model-improvement pass (Tier-1/2 levers)
python scripts/run_model_comparison.py --school    # headline table WITH school context
python scripts/run_school_features_experiment.py   # school-context ablation (+ _foldsafe.py)
python scripts/run_pv_stacking_experiment.py       # PV-stacking vs PV-mean target
python scripts/run_threshold_calibration.py        # threshold tuning + isotonic calibration

# Robustness extensions (feed notebooks 05/06/07/09)
python scripts/run_ceiling_generalization.py       # data-ceiling claim across 9 countries
python scripts/run_process_modality.py             # CBA process-data modality ablation
python scripts/run_fairness_mitigation.py          # group-specific thresholds (fairness fix)
python scripts/run_causal_did.py                   # earthquake difference-in-differences
python scripts/run_coverage_bounds.py              # Manski coverage bounds
```

Notebook 04 (explainability) and notebook 07 (fairness) narrate the audit scripts;
`_build_notebooks.py` refuses to overwrite an executed notebook unless `--force`, so
re-running it only (re)builds missing notebooks and leaves existing outputs intact.

Validate any processed slice before modelling:

```python
from src.data.validate import validate_processed, assert_valid
violations = validate_processed(df, feature_cols=["ESCS", "HOMEPOS", "ANXMAT"])
assert_valid(df)   # raises on ERROR-level contract breaches; WARN is allowed
```

### Environment note (macOS OpenMP)
The boosters (LLVM `libomp`) and scikit-learn (`libgomp`) can abort the process if their
OpenMP runtimes collide in one run. The code wraps CV fitting in `threadpool_limits(1)`,
isolates each model in a fresh subprocess (`src/models/_isolated_worker.py`), and skips any
C-level crasher gracefully. Always set `KMP_DUPLICATE_LIB_OK=TRUE`.

**LightGBM's `rc=-11` segfault was an OpenMP import-order bug, not a `libomp` install
problem.** LightGBM's wheel links Homebrew's libomp; if scikit-learn's bundled libomp loaded
first (which happened because the booster was imported lazily, after `experiment.py` had
already pulled in scikit-learn), the two runtimes collided and the fit aborted at the C-level
`Dataset` construction. The isolated workers now import the boosters *before* scikit-learn, so
LightGBM's libomp goes in first and it runs cleanly, it is back in the comparison, OOS, and
SHAP.

---

## Statistical Rigor

This project deliberately corrects the shortcuts common in student PISA analyses. Each
choice below is implemented and unit-tested.

- **Sampling weights** (`W_FSTUWT`) on *every* descriptive statistic and model metric,
  never an unweighted mean.
- **Design-based standard errors.** `pv_statistic_brr` combines Balanced Repeated
  Replication (80 Fay replicate weights) with the between-plausible-value variance via
  Rubin's rules, the OECD PISA Data Analysis Manual method, not a bootstrap that ignores
  the design. (`src/data/weights.py`)
- **Plausible values** treated as the 10 multiple imputations they are: per-PV targets and
  Rubin's combining rules, reporting the fraction of missing information (FMI). Collapsing
  the PVs to a single mean understates uncertainty. (`src/data/impute.py`,
  `src/features/target.py`, `src/models/experiment.per_pv_cv_evaluate`)
- **No leakage.** Feature standardisation, ranking and interaction terms are recomputed on
  the training fold only, inside the CV pipeline. (`src/features/transformers.py`)
- **Honest CV intervals.** Repeated-CV fold scores are correlated (shared training data), so
  metric CIs use the **Nadeau-Bengio** corrected-resampled variance instead of `S/√k`.
  (`src/models/evaluate.nadeau_bengio_interval`)
- **Significance testing, not eyeballing.** Model-vs-model differences are tested with the
  **corrected resampled t-test** in CV and the **DeLong** test on the shared OOS test set,
  so a small AUC gap is reported as a tie when it is one.
  (`src/models/evaluate.corrected_resampled_ttest`, `delong_roc_test`)
- **Exact OECD Level-2 thresholds** (math 420.07 / reading 407.47 / science 409.54); the
  decision threshold is tuned on train data only, with ECE and Brier calibration reported.
- **Nested-CV Optuna HPO**, leakage-safe, weighted, OpenMP-isolated tuning that reports
  whether it beats the registry default significantly (`src/models/hpo.py`).

## Testing & Validation

- **188 unit tests** (`tests/`, `pytest`) run on synthetic fixtures, no PISA data required.
  They pin down the weighted statistics, Rubin's rules, the leakage-safe transformer, the
  BRR+PV variance, the Nadeau-Bengio / DeLong maths (e.g. identical predictors → DeLong
  *p* = 1; replicates ≡ base weight → BRR SE = 0), the weight-routing fix, the HPO plumbing,
  the leave-one-out **school aggregates**, the **forecast** WLS/Monte-Carlo scenarios, the
  **stacking** ensemble builder, the **decision-curve / calibration** algebra (net-benefit
  references, weighted Brier/ECE), the **multilevel** ICC + random-intercept fit, and the
  **causal DiD / triple-difference** (planted-effect recovery, stratum-label parsing).
- **Data contracts** (`src/data/validate.py`) turn silent pipeline regressions into explicit
  `ERROR`/`WARN` violations: weight presence & positivity, plausible-value counts, target
  range, valid cycles, replicate-weight availability, all-missing features. Wire
  `assert_valid` into the pipeline to fail fast.

---

## Deliverables (Phase 10)

### Written paper, `reports/paper/` (done)
A single-column **LaTeX `article` report** (standalone title page, table of contents,
running headers, colored inline statistics), authored by **Xhoi Ikonomi, Metropolitan
University of Tirana**. The Data section carries a full exploratory pass, a data
dictionary, survey-weighted descriptives, an effect-size ranking, a cross-country
comparability table, and the descriptive SES-gradient probability curve. It follows the
Harvard hourglass structure (hook, explicit thesis + roadmap, topic-sentence body,
hook-returning conclusion) and frames the work around the *science*: the structural-shift
thesis (AUC 0.98), the feature-vs-data ceiling (~0.78, three convergent routes), the
broad-based flat-gradient crisis, and an honest fairness account. It integrates the five
robustness and extension analyses in full, the ceiling generalising across nine systems
(ICC-tracked, r~0.80), the CBA process-modality test (0.86 but endogenous, so the
deployable ceiling holds), the group-threshold fairness mitigation (FPR gap 0.63 to
0.003), the Manski coverage bounds ([0.74, 0.79]), and the earthquake
difference-in-differences (a well-identified null: the collapse is national). A dedicated
**economic-context section** ties the collapse to Albania's 2019 Durrës earthquake and
COVID-19 school closures (the OECD country note makes the same attribution; the DiD then
shows the loss was national, not localised to the quake), with web-verified macro facts
(remittances, GDP, education spending, the 2023 census) and real citations. No em-dashes
anywhere (prose and figure titles). Build:

```bash
cd reports/paper && make          # pdflatex + bibtex -> main.pdf (24 pp)
```

### Interactive risk-screener dashboard, `reports/dashboard/` (done)
A **bilingual (English / Albanian) Streamlit** web app over the headline model
(survey-weighted CatBoost + school composition, isotonically calibrated). A student answers
**13 plain-language questions** (books at home, parents' education and work, feelings about
maths, school resources), no PISA jargon, and each answer is mapped onto the model's OECD
indices using the real Albania-2022 weighted quantiles. The UI returns a **calibrated risk
probability** with a **cohort-percentile marker**, a **stable Altair diverging chart** of the
factor groups pushing risk up (red) or down (blue) with a per-factor drill-down, a
**SES-risk gradient** card (the student's own fifth highlighted), a leak-free **calibration
reliability curve**, one-click **preset profiles**, prominent caveats plus a
provenance/license footer, and a **downloadable PDF report card** (`fpdf2`) in the chosen
language. Header KPI tiles surface the cohort at-risk rate, the ~0.78 ceiling, and the
sample size. The model bundle and three precomputed leak-free context series are exported
by `scripts/export_dashboard_model.py`.

- **UI framework:** Streamlit (`reports/dashboard/app.py`); card layout, language toggle;
  theme follows the viewer's system light/dark (via `st.context.theme`, matched in the charts).
- **Charts:** Altair (rounded diverging bars, fixed order + colours so they never reshuffle).
- **Report export:** fpdf2 (native single-page PDF, EN/SQ).
- **Run:** `pip install -e ".[dashboard]"` then `streamlit run reports/dashboard/app.py`.

### Slides & poster, **not done yet (planned next)**
The **presentation (slides)** and the **poster** are the only remaining Phase 10
items. Everything else (analysis, paper, dashboard) is complete.

---

## Roadmap

### Done
- **Phase 1, Infrastructure & data:** repo, configs, FWF parser, SPSS loader,
  harmonization, weights/PV handling, full pipeline. 5 cycles × 9 countries processed.
- **Phase 2, EDA & statistics:** 8 publication figures, executed notebooks (01–02),
  covariate-shift analysis (AUC 0.98), weighted stats, effect sizes.
- **Phase 3, Feature engineering:** SES composites, digital indices, interaction terms,
  country-normalized features, selection (VIF / correlation / missingness).
- **Phase 4, Modeling:** 9-model weighted comparison (repeated stratified CV) with
  Nadeau-Bengio CIs and pairwise significance testing (corrected resampled t-test in CV,
  DeLong out-of-sample); weighted OOS 2022 experiment with Rubin's-rules per-PV evaluation
  and train-only threshold tuning; SHAP global + importance. Notebooks 03–04.
- **Rigor hardening (Phases 1–4):** test suite, dependency-free data contracts, BRR+Rubin
  design-based standard errors wired into the EDA notebooks, Nadeau-Bengio CIs and
  DeLong/corrected-resampled significance tests. Fixed a weighting bug (scaler-wrapped models
  were fit unweighted).
- **Phase 5, HPO:** Optuna in proper nested CV (inner tuning, outer evaluation),
  leakage-safe and weighted, OpenMP-isolated per fold, for CatBoost / GBM / LR
  (`src/models/hpo.py`, `scripts/run_hpo.py`). Each tuned model tested against its registry
  default with the corrected resampled t-test. Result: **no significant gain from tuning**,
  defaults are already at the accuracy ceiling.
- **Model-improvement pass:** two levers against the student-only ceiling. **School context**
  (survey-weighted leave-one-out school-mean features) lifts every model ~+0.05 AUC
  (CatBoost 0.73→**0.78**), fold-safe and significant, breaking the ceiling HPO could not, and
  flipping the podium so boosters beat LR. **Threshold tuning + isotonic calibration** lift
  MCC/F1 and cut ECE ~5–7×. `build_model_data(..., add_school_context=True)`; scripts
  `run_school_features_experiment[_foldsafe]`, `run_threshold_calibration`. SHAP (nb04) and
  the fairness audit (nb07) were then re-run on the school-augmented model.
- **Phase 6, Explainability (local + PDP/ICE):** SHAP local case studies for one
  representative TP/TN/FP/FN each (`scripts/run_explainability_cases.py` → waterfalls +
  `shap_local_cases_2022.csv`) and PDP + centered-ICE for the top drivers, narrated in
  notebook 04.
- **Phase 7, Fairness audit (survey-weighted):** demographic parity, equal opportunity (TPR),
  FPR and calibration across gender, SES quintile and immigrant status, plus a threshold sweep,
  all **weighted** (`scripts/run_fairness_audit.py` → `fairness_audit_2022.json`,
  `fairness_threshold_sweep_gender.csv`, `E1` figure), narrated in notebook 07. School context
  *shrinks* the gender gap but leaves the SES gap intact and enlarges the immigrant FPR gap.
- **Phase 8, Comparative modeling:** per-country school-context LightGBM (nine countries),
  weighted CV AUC, SHAP feature × country rank matrix, and SES-gradient comparison
  (`scripts/run_comparative_modeling.py`, notebook 05). Headline: Albania is high-prevalence /
  low-separability / flat-gradient (broad-based crisis); drivers (math anxiety, school
  composition) are shared with peers, not unique.
- **Forecast, next cycle (2026):** scenario Monte-Carlo (`src/forecast/`, notebook 10)
  propagating design-based SEs: plausible 2026 low-proficiency range ~21–74%, defensible zone
  partial-reversion (~47%) to persistence (~74%). Honest about the five-cycle / structural-break
  limits.
- **Phase 9, Stacking ensemble (negligible, worse where it counts):** a StackingClassifier of
  the four bare tree/booster base learners (CatBoost + LightGBM + GBM + RF) with a
  class-balanced Logistic-Regression meta-learner on their out-of-fold probabilities, scored in
  the school-context regime through the identical weighted / leakage-safe / OpenMP-isolated CV
  path (`scripts/run_stacking_experiment.py`, registry `"stacking"`, notebook 03). Stacking AUC
  **0.786** vs single CatBoost **0.783**, a paired Nadeau-Bengio gain of **+0.003 (p=0.030)**:
  significant but practically nil, and it *regresses* the decision-point metrics (MCC 0.386 vs
  0.393, F1 0.683 vs 0.693) at ~19× the compute. The ~0.78 ceiling holds; **single CatBoost
  remains the deployable headline**. Kept as an ablation, alongside the earlier **PV-stacking**
  result (all 10 PVs as training rows; ~+0.01 AUC for LightGBM/GBM only), both confirm
  diminishing returns (`stacking_ensemble_2022.csv`, `stacking_pairwise_nb_2022.csv`).
- **Documentation & rigor pass:** per-notebook *Methods & formulas* reference cells
  (`notebooks/_formulas.py`), a unified colorblind-safe colormap schema across all figures
  (`visualization/style`: SEQUENTIAL / SEQUENTIAL_RANK / DIVERGING; dark = more-important),
  and an internal performance / quality review.
- **Screener evaluation + multilevel model (robustness strengthening, notebook 06):**
  **Decision-curve analysis** (`src/models/decision_curve.py`, `scripts/run_decision_curve.py`,
  figure `G1`) reframes the ~75%-prevalence problem around *net benefit*, the model beats
  screen-everyone/screen-no-one over the selective threshold band **0.51–0.95**, and weighted
  isotonic **calibration** cuts ECE **~0.10 → ~0.01** (`G2`). A **random-intercept logistic**
  model (`src/models/multilevel.py`, `scripts/run_multilevel.py`, figure `H1`), the correct
  spec for PISA's nested design, **matches** the hand-crafted school-mean booster on identical
  folds (CV AUC ~0.78 either way, confirming the ceiling is real) and shows a school **ICC ≈ 0.31**:
  ~a third of risk variance is *between schools* and barely touched by student features. A
  **survey-weighted pseudo-likelihood** refit (PQL with within-cluster scaled weights, Schall /
  Rabe-Hesketh & Skrondal, `multilevel.fit_weighted_random_intercept`) gives design-consistent
  fixed effects: odds ratios shift by ≤0.05 and the ICC stays in the **0.22–0.31** band, so the
  variance-partition story is robust to how sampling weights are handled.
- **School-questionnaire linkage, ceiling is data, not features (notebook 06):** the ICC ≈ 0.31
  motivated linking the PISA 2022 **school questionnaire** (`CY08MSP_SCH_QQQ`: resources, staff
  shortage, class size, leadership, climate), joined onto students on `CNTSCHID`
  (`src/features/engineer.py:add_school_questionnaire`, `scripts/run_school_questionnaire_experiment.py`).
  A paired ablation on shared folds finds these *independent* school inputs add **no significant
  signal beyond school composition** (best incremental **+0.004** AUC, all *p* > 0.24). School-mean
  ESCS already proxies school resources/staffing, so the **~0.78 ceiling is a genuine data limit,
  not a feature-set limit**, a third independent route (composition booster, multilevel ICC,
  direct questionnaire linkage) converging on the same conclusion.

- **Decision support, uncertainty, recourse, policy (notebook 08):** three tools that turn the
  screener into something actionable. **Conformal prediction** (`src/models/conformal.py`,
  `scripts/run_conformal.py`) gives distribution-free prediction sets, the coverage guarantee
  holds under survey weighting, ~60% confident singletons / ~40% honest abstentions at 90%
  coverage. **Counterfactual recourse** (`src/explainability/counterfactuals.py`,
  `scripts/run_counterfactuals.py`) finds the minimal actionable change to un-flag a student;
  it succeeds for only a minority and is dominated by **math anxiety**, the individual echo of
  the structural thesis. **Policy simulation** (`src/models/policy.py`, `scripts/run_policy.py`)
  shows even an everything-at-once population intervention only cuts predicted at-risk ~72%→60%
  (anxiety the top single lever, ~-7 pp), no soft lever rescues a 75%-prevalence crisis. All
  three are model-associational, not causal (stated, not hidden).

- **Fold-safe transformer + Phase 5b completeness:** fold-safe per-fold school-means transformer
  (`features.transformers.SchoolMeansTransformer`; `run_school_means_transformer_check.py`
  confirms full-cohort means are leakage-free, delta <0.003, n.s.); MLP & SVM added to the
  comparison (both mid-table, boosters still lead) with the sklearn 1.8 `penalty` deprecation
  fixed (`l1_ratio` replaces the l1/l2/elasticnet categorical); proper multilevel
  pseudo-likelihood with scaled survey weights (`fit_weighted_random_intercept`, see above).

- **Robustness extensions (five):** hardening the analysis against the strongest methodological counter-arguments.
  **(1)** A *causal* test of the 2019 Durrës earthquake (difference-in-differences + placebo +
  triple-difference, `src/causal/`) — a well-identified **null**: the 2022 collapse is national,
  not a localised shock. **(2)** The data-ceiling claim **generalises** across nine countries
  (composition lift significant in 8/9, achievable AUC tracks between-school ICC at *r* = 0.80),
  with an equitable-system boundary (Finland). **(3)** A genuinely new **modality** (CBA process
  data, `src/features/process.py`) moves the AUC to 0.86, but ~90% of that **gain** is endogenous
  cognitive-test process — the *screening* ceiling holds ~0.78. **(4)** Fairness **mitigation**
  (`src/fairness/mitigation.py`): group-specific thresholds collapse the SES false-positive gap
  0.63 → 0.003. **(5)** **Manski coverage bounds** (`src/coverage/`): at Albania's ~79% PISA
  coverage, the crisis conclusion is robust to *any* assumption about the excluded ~21%. See
  the Key Results sections above.

- **Notebook consolidation:** the 18 topic-installment notebooks were merged into **10 coherent
  multi-Part notebooks** (01–10), built by `_build_notebooks.py`.

- **Within-Albania geographic equity & malleable-lever ranking (Albania-native contributions):**
  two additions that turn the national story into something a Ministry can target. **(1)** A
  design-based (BRR + Rubin) map of the below-Level-2 rate by **region band × urbanicity × sector**
  across 2015/2018/2022 (`src/geography/equity.py`, notebook 02 Part C): the North and rural bands
  carry the heaviest load in every cycle, and the Rural−Urban (+9.9 pp), North−South (+7.0 pp) and
  Public−Private (+25.5 pp) gaps are all significant on the contrast itself. **(2)** A
  **malleable-lever ranking** (`src/models/levers.py`, notebook 08 Part D) that splits fixed
  endowments from actionable levers and ranks the latter by predicted at-risk reduction: math
  anxiety is the one lever with real reach (−5.1 pp), and even the all-lever bundle leaves a large
  majority at risk — the structural floor, made explicit. See the two Key Results sections above.

### Remaining
- **Report & deliverables:** the **written paper** (`reports/paper/`, compiles to `main.pdf`)
  now **incorporates all five robustness extensions above** (methods + results), and the
  **dashboard** (`reports/dashboard/`) is built. The **slides** and **poster** remain outstanding.

---

*Data: OECD PISA 2009–2022 Public Use Files, https://www.oecd.org/pisa/*
