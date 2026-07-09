# Albania PISA Risk Screener, interactive dashboard

A **bilingual (English / Albanian)** Streamlit front-end over the project's
**headline model** (survey-weighted CatBoost with school socioeconomic composition,
isotonically calibrated). A student answers **plain-language questions** (books at
home, parents' education and work, how they feel about maths) with no PISA jargon;
the app maps each answer onto the model's OECD indices using the real Albania-2022
weighted quantiles, then returns a **calibrated probability** of scoring below
mathematics Level 2 (< 420), the grouped drivers behind it, and a **downloadable
PDF report card** (native PDF via fpdf2, in the chosen language). Language toggles
with the switch near the top.

## Run

```bash
pip install -e ".[dashboard]"    # once: streamlit + altair + fpdf2 (catboost/sklearn already in the env)
python scripts/export_dashboard_model.py   # 1. train + persist the model bundle (needs data/processed/alb_2022.parquet)
streamlit run reports/dashboard/app.py     # 2. launch the app
```

Step 1 writes two files to `outputs/models/`:

- `dashboard_bundle.joblib`: the fitted pipeline, the out-of-fold isotonic
  calibrator, the tuned decision threshold, and the feature/SHAP column order.
- `dashboard_meta.json`: slider ranges (weighted quantiles of the real cohort),
  thresholds, calibration metrics, SES-quintile edges, and three precomputed,
  leak-free context series the app reads (see below).

## What it shows

- **Header context**: a provenance strip plus three KPI tiles (cohort at-risk 75%,
  the ~0.78 AUC ceiling, N = 6,129 students).
- **Calibrated risk %** and the screening decision at a threshold tuned to maximise
  survey-weighted MCC (~0.58), with the cohort base rate (75%) marked for context.
- **Cohort percentile marker**: where this estimate sits in the 2022 cohort's
  calibrated-risk distribution ("higher risk than X% of the cohort").
- **Grouped SHAP drivers** in a *fixed order with fixed colours* (blue lowers risk,
  red raises it) so the chart never reshuffles as answers change; you can watch a
  single answer move its bar. An expander breaks the groups into the top individual
  factors.
- **SES-risk gradient**: survey-weighted mean risk across ESCS quintiles (poorest to
  richest fifth), the student's own fifth highlighted. Albania's gentle slope is the
  headline finding.
- **Calibration reliability curve** (in the technical panel): leak-free
  predicted-vs-observed on held-out calibrated probabilities, dot size by cohort
  share, diagonal = ideal.
- **Preset profiles**: one-click illustrative composites (cohort average /
  under-resourced / advantaged), clearly labelled synthetic, not real students.
- **Caveats, prominently**: the model is *associational* (single 2022 cross-section,
  an answer is not an intervention), it *over-flags the poorest areas* (fairness
  audit), and it sits at a genuine **~0.78 AUC data ceiling**. A provenance/license
  footer restates the source and that no individual student data is stored or shown.

## Theming

`base` is intentionally left unset in `.streamlit/config.toml`, so the app follows
the viewer's operating-system / browser light-dark preference (Streamlit 1.58 has no
in-app theme switch). The app reads `st.context.theme.type` and matches its own
chart and card palette to whatever Streamlit resolves, so native widgets and the
custom charts stay in sync. The two data palettes (a blue/red diverging pair per
surface) are each validated against their own light or dark background.

## Answer to index mapping

Students don't know their ESCS or ANXMAT. Each concrete answer is mapped to a
data-grounded anchor (min / p25 / median / p75 / max) of that index's **weighted
Albania-2022 distribution**, read from `dashboard_meta.json`. So "0-10 books" lands
at the low end of the real home-possessions distribution, "University degree" at the
high end of ESCS, and so on. The landing state is a *typical* student near the cohort
average, not a worst case.

## Honesty notes

This is a **triage and communication aid, not a lever simulator**. Moving a slider
shows how the model *reads* a profile, not what would happen if that factor changed
for a real student; math anxiety, for instance, is partly a *consequence* of low
achievement, not only a cause. Use scores to widen support, never to ration it.

The precomputed context series (`risk_percentile_edges`, `ses_risk_gradient`,
`reliability_curve`) are all derived from leak-free out-of-fold predictions, and no
individual student rows are shipped to the app.

Provenance: the same pipeline scored ~0.78 weighted CV AUC in
`notebooks/03_modeling.ipynb`; the export script reuses the project's leakage-safe,
survey-weighted training path (`src/models/`).
