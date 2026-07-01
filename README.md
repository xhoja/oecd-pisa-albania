# Predicting Albanian Student Low-Proficiency Risk Using OECD PISA Data (2009–2022)
### A Comparative Machine-Learning Framework

A data-science project predicting which Albanian 15-year-olds are
at risk of low mathematics proficiency (below PISA Level 2, score < 420), benchmarked
against Balkan peers, top performers, and GDP-matched economies — with rigorous PISA
methodology (sampling weights, plausible values), explainability, and a covariate-shift
analysis of Albania's dramatic 2022 regression.

---

## Research Question

> Can machine learning identify Albanian students at risk of low academic proficiency
> from background variables, and do the predictive factors differ from OECD peers — and
> can those differences explain Albania's catastrophic 2018→2022 reversal?

**Headline findings so far:**
1. A domain classifier distinguishes 2018 from 2022 Albanian students with **AUC 0.98** on
   background features alone — the 2022 collapse is a *structural* distributional shift, not
   merely lower scores.
2. With proper significance testing, a **transparent logistic regression is statistically
   tied with the best gradient boosters** (CatBoost/GBM) for predicting risk — the accuracy
   ceiling here is set by the data, not model complexity.

---

## Key Results (current)

### Albania 2022 — model comparison (5×4 repeated stratified CV, weighted metrics)

| Model | ROC-AUC (Nadeau-Bengio 95% CI) | PR-AUC | F1-macro | MCC |
|---|---|---|---|---|
| **CatBoost** | **0.731** (0.714–0.749) | 0.880 | 0.647 | 0.305 |
| Logistic Regression | 0.727 (0.704–0.751) | 0.882 | 0.621 | 0.281 |
| Gradient Boosting | 0.727 (0.711–0.743) | 0.878 | 0.609 | 0.283 |
| Random Forest | 0.710 (0.692–0.727) | 0.868 | 0.599 | 0.261 |
| Extra Trees | 0.707 (0.688–0.726) | 0.866 | 0.608 | 0.260 |
| XGBoost | 0.698 (0.681–0.715) | 0.863 | 0.617 | 0.262 |
| Naive Bayes | 0.682 (0.659–0.705) | 0.846 | 0.599 | 0.246 |
| KNN | 0.664 (0.638–0.689) | 0.842 | 0.588 | 0.215 |
| Decision Tree | 0.568 (0.548–0.589) | 0.780 | 0.567 | 0.135 |

All metrics are **weighted** with the survey weights (a correctness fix: scaler-wrapped
models such as Logistic Regression were previously fit unweighted because the weight kwarg
was routed to the wrong pipeline step — now fixed, which lifted LR from 0.726 to 0.727 and
into second place). KNN is the one exception: `KNeighborsClassifier` has no `sample_weight`,
so its fit is inherently unweighted (its metrics are still weighted at evaluation).

The 95% intervals use the **Nadeau-Bengio corrected-resampled variance**, not a naïve
`S/√k`: repeated-CV folds share training data, so their scores are positively correlated
and the naïve interval is too narrow. The correction replaces the `1/k` term with
`1/k + n_test/n_train`, giving an honest (wider) interval.

> LightGBM is excluded: it segfaults (`rc=-11`) at the C level on this host — a `libomp`
> install issue, not a code bug. Model fits are isolated in subprocesses so a C-level
> crasher is skipped gracefully rather than taking down the run.

### Are the differences real? (pairwise significance)

A ranking table alone invites over-reading a 0.005 AUC gap. We test it directly.

- **In CV** — pairwise **Nadeau-Bengio corrected resampled t-test** on the paired per-fold
  AUCs (`outputs/results/albania_2022_pairwise_nb.csv`). The **top three are a statistical
  tie**: CatBoost ≈ Logistic Regression (*p* = 0.50), CatBoost ≈ Gradient Boosting
  (*p* = 0.37), GBM ≈ LR (*p* = 0.91). All three beat Random Forest and everything below it
  (*p* < 0.01). **Takeaway: a transparent logistic regression matches the best boosters** on
  this task — an interpretability win, not a cost.
- **Out-of-sample** — pairwise **DeLong test** on the shared 2022 test set
  (`oos_2022_experiment.json → delong_pairwise`): the OOS leaders Random Forest, Gradient
  Boosting and CatBoost are mutually indistinguishable (RF vs GBM *p* = 0.64, RF vs CatBoost
  *p* = 0.05), while all three beat XGBoost and LR significantly.

### Out-of-sample (train 2009–2018 → test 2022, weighted)

GBM & RF lead at **0.674** AUC; CatBoost 0.665, Extra Trees 0.663, XGBoost 0.639, LR 0.622.
The shift is **detectable** (AUC 0.98) yet models still transfer moderately — a key nuance
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

**Tuning yields no statistically significant gain** for any model — the sensible registry
defaults are already at the task's accuracy ceiling. This reinforces the headline: on these
features the limit is the data, not model capacity or tuning.

### SHAP global importance (Albania 2022, CatBoost)

`ANXMAT` > `MATERIAL_DEFICIT` > `HISCED` > `HOMEPOS` > `HISEI` > `BELONG` > `ESCS` …
Math anxiety and material home resources dominate; immigration status is negligible.
(SHAP switched from LightGBM to CatBoost because LightGBM crashes on this host.)

### Cross-country (PISA 2022, weighted low-proficiency rate)

Albania **75%** (highest) — above Balkan peers (N. Macedonia 67%, Montenegro 60%) and
GDP-matched economies (Colombia 73%, Mexico 67%); Estonia lowest at 14%.

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
- **Comparison set:** 323,121 students — ALB, MKD, MNE, SRB, BGR, EST, FIN, MEX, COL
- Raw files are **not** committed (multi-GB). Place them in `data/raw/`; processed
  parquet files are regenerated by the pipeline.

> **Data notes (verified):** Albania genuinely lacks ESCS in 2012 & 2015 (an OECD gap,
> not a bug); peer countries have full 2015 background data. North Macedonia joined PISA
> in 2015; Serbia skipped 2015.

**Design-based estimates.** PISA ships 80 Fay balanced-repeated-replication (BRR) weights.
The pipeline preserves them, and `src/data/weights.pv_statistic_brr` reports point estimates
with the *correct* PISA standard error — BRR sampling variance (per plausible value) combined
with the between-PV imputation variance via Rubin's rules. Albania's weighted at-risk math
trajectory with design-based SEs:

| Cycle | At-risk (math < L2) | SE | 95% CI | BRR |
|---|---|---|---|---|
| 2009 | 0.677 | 0.0076 | 0.663–0.692 | — (FWF) |
| 2012 | 0.607 | 0.0051 | 0.597–0.617 | — (FWF) |
| 2015 | 0.533 | 0.0108 | 0.512–0.554 | ✓ |
| 2018 | 0.424 | 0.0088 | 0.407–0.441 | ✓ |
| **2022** | **0.739** | **0.0041** | **0.731–0.747** | ✓ |

> **Replicate-weight caveat:** the 80 replicate weights ship only on the SAV cycles
> (2015/2018/2022). The FWF cycles (2009/2012) fall back to an imputation-only SE, flagged
> `BRR=False` — an honest limit for any cross-cycle significance claim.

---

## Repository Structure

```
OECD_PISA_Project/
├── configs/              # data / features / models / experiment YAML
├── src/
│   ├── data/             # extract, parse_fwf, harmonize, weights (BRR+Rubin),
│   │                     #   validate (data contracts), impute, pipeline
│   ├── features/         # engineer, select, target, transformers (fold-safe)
│   ├── models/           # registry, hpo (Optuna nested CV), optimize (search space),
│   │                     #   evaluate (metrics, Nadeau-Bengio, DeLong), prepare,
│   │                     #   experiment (Rubin's rules), train, _isolated_worker
│   ├── explainability/   # SHAP analysis
│   ├── fairness/         # group-fairness metrics
│   ├── statistics/       # tests (chi², MMD, covariate shift, effect sizes)
│   └── visualization/    # style, eda, model_plots
├── notebooks/            # 01_eda_albania, 02_eda_comparative, 03_covariate_shift,
│                         #   04_modeling, 05_explainability  (all executed)
├── scripts/              # run_model_comparison, run_oos_experiment, run_hpo, run_shap_analysis
├── tests/                # 56 pytest unit tests (weights, impute, target,
│                         #   transformers, validate, evaluate) — no real data needed
├── outputs/              # figures/{eda,models,shap} + results/ (csv, json)
└── data/                 # raw/ (git-ignored) + processed/ (parquet)
```

---

## Reproduce

```bash
# 1. Install
pip install -e ".[dev]"     # runtime + pytest/ruff/mypy
brew install libomp         # macOS: required by XGBoost/LightGBM/CatBoost

# 2. Run the test suite (no data required — 56 tests on synthetic fixtures)
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
python scripts/run_shap_analysis.py
```

Validate any processed slice before modelling:

```python
from src.data.validate import validate_processed, assert_valid
violations = validate_processed(df, feature_cols=["ESCS", "HOMEPOS", "ANXMAT"])
assert_valid(df)   # raises on ERROR-level contract breaches; WARN is allowed
```

### Environment note (macOS OpenMP)
XGBoost (LLVM `libomp`) and scikit-learn (`libgomp`) can abort the process if both load
in one run. The code wraps CV fitting in `threadpool_limits(1)`, isolates each model in a
fresh subprocess (`src/models/_isolated_worker.py`), and skips any C-level crasher
gracefully. Always set `KMP_DUPLICATE_LIB_OK=TRUE`.

**LightGBM is currently broken on this host** — it segfaults (`rc=-11`) even in a pristine
single-thread interpreter, a `libomp` install issue rather than a code bug. It is skipped
in comparisons and OOS, and SHAP uses CatBoost instead. Reinstalling `libomp` should
restore it.

---

## Statistical Rigor

This project deliberately corrects the shortcuts common in student PISA analyses. Each
choice below is implemented and unit-tested.

- **Sampling weights** (`W_FSTUWT`) on *every* descriptive statistic and model metric —
  never an unweighted mean.
- **Design-based standard errors.** `pv_statistic_brr` combines Balanced Repeated
  Replication (80 Fay replicate weights) with the between-plausible-value variance via
  Rubin's rules — the OECD PISA Data Analysis Manual method, not a bootstrap that ignores
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
  **corrected resampled t-test** in CV and the **DeLong** test on the shared OOS test set —
  so a small AUC gap is reported as a tie when it is one.
  (`src/models/evaluate.corrected_resampled_ttest`, `delong_roc_test`)
- **Exact OECD Level-2 thresholds** (math 420.07 / reading 407.47 / science 409.54); the
  decision threshold is tuned on train data only, with ECE and Brier calibration reported.
- **Nested-CV Optuna HPO** — leakage-safe, weighted, OpenMP-isolated tuning that reports
  whether it beats the registry default significantly (`src/models/hpo.py`).

## Testing & Validation

- **63 unit tests** (`tests/`, `pytest`) run on synthetic fixtures — no PISA data required.
  They pin down the weighted statistics, Rubin's rules, the leakage-safe transformer, the
  BRR+PV variance, the Nadeau-Bengio / DeLong maths (e.g. identical predictors → DeLong
  *p* = 1; replicates ≡ base weight → BRR SE = 0), the weight-routing fix, and the HPO
  plumbing.
- **Data contracts** (`src/data/validate.py`) turn silent pipeline regressions into explicit
  `ERROR`/`WARN` violations: weight presence & positivity, plausible-value counts, target
  range, valid cycles, replicate-weight availability, all-missing features. Wire
  `assert_valid` into the pipeline to fail fast.

---

## Roadmap

###  Done
- **Phase 1 — Infrastructure & data:** repo, configs, FWF parser, SPSS loader,
  harmonization, weights/PV handling, full pipeline. 5 cycles × 9 countries processed.
- **Phase 2 — EDA & statistics:** 8 publication figures, 3 executed notebooks,
  covariate-shift analysis (AUC 0.98), weighted stats, effect sizes.
- **Phase 3 — Feature engineering:** SES composites, digital indices, interaction terms,
  country-normalized features, selection (VIF / correlation / missingness).
- **Phase 4 — Modeling:** 9-model weighted comparison (repeated stratified CV) with
  Nadeau-Bengio CIs and pairwise significance testing (corrected resampled t-test in CV,
  DeLong out-of-sample); weighted OOS 2022 experiment with Rubin's-rules per-PV evaluation
  and train-only threshold tuning; SHAP global + importance. Notebooks 04–05.
- **Rigor hardening (Phases 1–4):** 63-test suite, dependency-free data contracts, BRR+Rubin
  design-based standard errors wired into the EDA notebooks, Nadeau-Bengio CIs and
  DeLong/corrected-resampled significance tests. Verified: the top three CV models are a
  statistical tie (Logistic Regression matches the best boosters). Fixed a weighting bug
  (scaler-wrapped models were fit unweighted).
- **Phase 5 — HPO:** Optuna in proper nested CV (inner tuning, outer evaluation),
  leakage-safe and weighted, OpenMP-isolated per fold, for CatBoost / GBM / LR
  (`src/models/hpo.py`, `scripts/run_hpo.py`). Each tuned model is tested against its
  registry default with the corrected resampled t-test. Result: **no significant gain from
  tuning** — defaults are already at the accuracy ceiling.

### Next
- **Phase 5b (deferred):** add MLP & SVM to the comparison; restore LightGBM once `libomp`
  is fixed (fix `_suggest_params` for the sklearn ≥1.8 `penalty` deprecation first).
- **Phase 6 — Explainability:** SHAP local case studies (TP/TN/FP/FN), dependence &
  interaction plots, PDP/ICE, per-country SHAP comparison.
- **Phase 7 — Fairness:** demographic parity / equal opportunity / equalized odds across
  gender, SES quintile, immigration; threshold sweep; calibration by group.
- **Phase 8 — Comparative modeling:** train per country group; SHAP rank matrix;
  SES-gradient comparison; Albania-vs-peers narrative.
- **Phase 9 — Advanced:** stacking ensemble, probability calibration, conformal
  prediction (uncertainty), counterfactual explanations, policy simulation.
- **Phase 10 — Deliverables:** written report, slides, poster,
  interactive risk-score dashboard.

---

*Data: OECD PISA 2009–2022 Public Use Files — https://www.oecd.org/pisa/*
