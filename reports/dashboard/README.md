# Albania PISA Risk Screener — interactive dashboard

A **bilingual (English / Albanian)** Streamlit front-end over the project's
**headline model** (survey-weighted CatBoost with school socioeconomic composition,
isotonically calibrated). A student answers **plain-language questions** — books at
home, parents' education and work, how they feel about maths — with no PISA jargon;
the app maps each answer onto the model's OECD indices using the real Albania-2022
weighted quantiles, then returns a **calibrated probability** of scoring below
mathematics Level 2 (< 420), the grouped drivers behind it, and a **downloadable
PDF report card** (native PDF via fpdf2, in the chosen language). Toggle language
with the switch at the top-right.

## Run

```bash
pip install -e ".[dashboard]"    # once: streamlit + altair + fpdf2 (catboost/sklearn already in the env)
python scripts/export_dashboard_model.py   # 1. train + persist the model bundle (needs data/processed/alb_2022.parquet)
streamlit run reports/dashboard/app.py     # 2. launch the app
```

Step 1 writes two files to `outputs/models/`:

- `dashboard_bundle.joblib` — the fitted pipeline, the out-of-fold isotonic
  calibrator, the tuned decision threshold, and the feature/SHAP column order.
- `dashboard_meta.json` — slider ranges (weighted quantiles of the real cohort),
  thresholds, calibration metrics, and SES-quintile edges the app reads.

## What it shows

- **Calibrated risk %** and the screening decision at a threshold tuned to maximise
  survey-weighted MCC (~0.58), with the cohort base rate (75%) marked for context.
- **Grouped SHAP drivers** in a *fixed order with fixed colours* (blue lowers risk,
  red raises it) so the chart never reshuffles as answers change — you can watch a
  single answer move its bar.
- **Caveats, prominently**: the model is *associational* (single 2022 cross-section —
  an answer is not an intervention), it *over-flags the poorest areas* (fairness
  audit), and it sits at a genuine **~0.78 AUC data ceiling**.

## Answer → index mapping

Students don't know their ESCS or ANXMAT. Each concrete answer is mapped to a
data-grounded anchor (min / p25 / median / p75 / max) of that index's **weighted
Albania-2022 distribution**, read from `dashboard_meta.json`. So "0–10 books" lands
at the low end of the real home-possessions distribution, "University degree" at the
high end of ESCS, and so on. The landing state is a *typical* student near the cohort
average, not a worst case.

## Honesty notes

This is a **triage and communication aid, not a lever simulator**. Moving a slider
shows how the model *reads* a profile, not what would happen if that factor changed
for a real student — math anxiety, for instance, is partly a *consequence* of low
achievement, not only a cause. Use scores to widen support, never to ration it.

Provenance: the same pipeline scored ~0.78 weighted CV AUC in
`notebooks/03_modeling.ipynb`; the export script reuses the project's leakage-safe,
survey-weighted training path (`src/models/`).
