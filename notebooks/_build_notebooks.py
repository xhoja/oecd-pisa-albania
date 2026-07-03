"""
Generator for the project's Jupyter notebooks.

Notebooks are thin narrative layers over the tested `src/` modules.
Run:  python notebooks/_build_notebooks.py
This writes the .ipynb files; execute them with jupyter or `jupyter nbconvert --execute`.
"""
from __future__ import annotations

from pathlib import Path

import sys as _sys

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent
_sys.path.insert(0, str(NB_DIR))
from _formulas import FORMULAS  # noqa: E402  shared "Methods & formulas" cells

HEADER = """import sys, os
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))  # project root
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
"""


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def _with_formulas(cells: list, key: str) -> list:
    """Insert the notebook's 'Methods & formulas' reference cell right after its
    intro markdown (index 0), keeping a single source in _formulas.FORMULAS."""
    if key in FORMULAS:
        return [cells[0], md(FORMULAS[key])] + cells[1:]
    return cells


def _has_outputs(path: Path) -> bool:
    """True if an existing notebook already has executed cell outputs."""
    if not path.exists():
        return False
    try:
        nb = nbf.read(str(path), as_version=4)
    except Exception:
        return False
    return any(c.cell_type == "code" and c.get("outputs") for c in nb.cells)


def build_notebook(cells: list, path: Path, force: bool = False) -> None:
    """
    Write a blank (unexecuted) notebook. Refuses to overwrite a notebook that
    already contains executed outputs unless force=True (pass --force), so the
    generator can't silently wipe results. After (re)building, re-execute with:
        jupyter nbconvert --to notebook --execute --inplace notebooks/<nb>.ipynb
    """
    if _has_outputs(path) and not force:
        print(f"SKIP {path.name} — has outputs (use --force to overwrite, then re-execute)")
        return
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    nbf.write(nb, str(path))
    print(f"wrote {path.name} ({len(cells)} cells) — remember to execute it")


# ===========================================================================
# Notebook 01 — EDA: Albania
# ===========================================================================

def nb_01_eda_albania() -> list:
    return [
        md(
            "# 01 — Exploratory Data Analysis: Albania (PISA 2009–2022)\n\n"
            "**Project:** Predicting Albanian Student Low-Proficiency Risk — A Comparative ML Framework\n\n"
            "This notebook explores the Albanian longitudinal dataset (27,042 students, five cycles). "
            "All statistics use PISA sampling weights (`W_FSTUWT`). Logic lives in `src/`; this notebook "
            "is the narrative layer.\n\n"
            "**Key question addressed here:** How did Albania's low-proficiency risk evolve, and what does "
            "the dramatic 2018→2022 reversal look like in the data?"
        ),
        code(HEADER),
        code(
            "from src.features.target import add_all_targets\n"
            "from src.data.weights import weighted_mean, weighted_proportion, weighted_std\n\n"
            "df = add_all_targets(pd.read_parquet('../data/processed/albania_longitudinal.parquet'))\n"
            "print('Rows:', len(df), '| Cycles:', sorted(df.CYCLE.unique()))\n"
            "df[['CYCLE','MATH_PV_MEAN','ESCS','HOMEPOS','AT_RISK_MATH','W_FSTUWT']].head()"
        ),
        md("## 1. Target trajectory — the V-shaped crisis\n\n"
           "Weighted low-proficiency rate (math score < 420, below PISA Level 2) and mean score by cycle."),
        code(
            "rows = []\n"
            "for c in sorted(df.CYCLE.unique()):\n"
            "    s = df[df.CYCLE==c]; w = s['W_FSTUWT']\n"
            "    p = weighted_proportion(s['AT_RISK_MATH'], w); n=len(s)\n"
            "    rows.append({'cycle':int(c),'N':n,\n"
            "                 'at_risk_%':round(p*100,1),\n"
            "                 'ci95':round(1.96*np.sqrt(p*(1-p)/n)*100,2),\n"
            "                 'mean_score':round(weighted_mean(s['MATH_PV_MEAN'],w),1),\n"
            "                 'mean_ESCS':round(weighted_mean(s['ESCS'],w),3)})\n"
            "pd.DataFrame(rows)"
        ),
        code(
            "from src.visualization.eda import plot_atrisk_trajectory\n"
            "fig = plot_atrisk_trajectory(df); plt.show()"
        ),
        md("**Reading:** Albania improved steadily 2009→2018 (low-proficiency fell ~28 pp), then "
           "reversed catastrophically in 2022 (+33 pp). This single-cycle reversal is the focus of the paper."),
        md("## 1b. Design-based standard errors (BRR + plausible values)\n\n"
           "The CIs above use a naïve normal approximation that ignores both the PISA sampling design "
           "and the plausible-value uncertainty. The **correct** PISA standard error combines two "
           "sources (OECD Data Analysis Manual): sampling variance via **Balanced Repeated Replication** "
           "(80 Fay replicate weights, computed per PV) plus imputation variance **across the 10 "
           "plausible values** (Rubin's rules). `pv_statistic_brr` returns both, the total SE, and the "
           "**FMI** (fraction of the variance attributable to the PV draw).\n\n"
           "> Replicate weights ship only on the SAV cycles (2015/2018/2022); the FWF cycles "
           "(2009/2012) fall back to an imputation-only SE, flagged by `BRR=False`."),
        code(
            "from src.data.weights import pv_statistic_brr\n"
            "rows = []\n"
            "for c in sorted(df.CYCLE.unique()):\n"
            "    r = pv_statistic_brr(df[df.CYCLE==c], statistic='at_risk', domain='math')\n"
            "    rows.append({'cycle':int(c),\n"
            "                 'at_risk':round(r['estimate'],4),\n"
            "                 'SE':round(r['se'],4),\n"
            "                 'ci95_low':round(r['ci95_low'],4),\n"
            "                 'ci95_high':round(r['ci95_high'],4),\n"
            "                 'FMI':round(r['fmi'],3),\n"
            "                 'BRR':r['brr_available']})\n"
            "pd.DataFrame(rows)"
        ),
        md("**Reading:** design-based SEs are small (large PISA samples) but the FMI shows the plausible "
           "values contribute a non-trivial share of the total uncertainty — collapsing them to a single "
           "mean (a common shortcut) would understate the SE. The 2009/2012 rows carry the weaker "
           "imputation-only SE, an honest caveat for any cross-cycle significance claim."),
        md("## 2. Score distributions by cycle"),
        code(
            "from src.visualization.eda import plot_score_distributions\n"
            "fig = plot_score_distributions(df); plt.show()"
        ),
        md("## 3. Feature availability / missingness\n\n"
           "Which background indices are present in each cycle? The 2015 column is the Albania-specific gap "
           "(OECD did not compute background indices for Albania that cycle)."),
        code(
            "from src.visualization.eda import plot_missingness_heatmap\n"
            "feats = ['ESCS','HOMEPOS','HISCED','HISEI','BELONG','TEACHSUP','ANXMAT','REPEAT','IMMIG','GRADE','GENDER']\n"
            "fig = plot_missingness_heatmap(df, feats); plt.show()"
        ),
        md("## 4. Socioeconomic gradient — SES quintile × at-risk\n\n"
           "Weighted low-proficiency rate by HOMEPOS/ESCS quintile across cycles."),
        code(
            "from src.visualization.eda import plot_ses_quintile_heatmap\n"
            "fig = plot_ses_quintile_heatmap(df); plt.show()"
        ),
        md("## 5. Feature correlation structure"),
        code(
            "from src.visualization.eda import plot_correlation_matrix\n"
            "fig = plot_correlation_matrix(df, ['ESCS','HOMEPOS','BELONG','TEACHSUP','ANXMAT','GRADE','MATH_PV_MEAN'])\n"
            "plt.show()"
        ),
        md("## 5b. Weighted association tests\n\n"
           "All inference uses survey weights. The chi-square uses **normalized** weights "
           "(rescaled to the real sample size n) so the statistic isn't inflated by the population "
           "scale of `W_FSTUWT`; effect sizes (Cohen's d) are weighted population estimates."),
        code(
            "from src.statistics.tests import weighted_chi_square, weighted_cohens_d\n"
            "d22 = df[df.CYCLE==2022]\n"
            "chi = weighted_chi_square(d22, 'GENDER', 'AT_RISK_MATH', weight_col='W_FSTUWT')\n"
            "print('Gender x at-risk (2022):', {k: round(v,4) if isinstance(v,float) else v for k,v in chi.items()})\n"
            "ar = d22[d22.AT_RISK_MATH==1]; pr = d22[d22.AT_RISK_MATH==0]\n"
            "dval = weighted_cohens_d(ar['ESCS'], pr['ESCS'], ar['W_FSTUWT'], pr['W_FSTUWT'])\n"
            "print('Weighted Cohen d (ESCS, at-risk vs proficient):', round(dval,3))"
        ),
        md("## 6. The 2022 crisis — feature drift (2018 → 2022)\n\n"
           "Standardized mean difference (Cohen's d) of each feature between the 2018 baseline and the 2022 "
           "crisis cohort. This previews the covariate-shift analysis in notebook 03."),
        code(
            "from src.visualization.eda import plot_feature_drift\n"
            "fig = plot_feature_drift(df, ['ESCS','HOMEPOS','BELONG','TEACHSUP','ANXMAT','GRADE','REPEAT','IMMIG'])\n"
            "plt.show()"
        ),
        md("**Takeaway:** Teacher support (TEACHSUP) shows the largest negative drift into 2022, while "
           "home possessions (HOMEPOS) shifts positively — the crisis is not a simple SES story."),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **V-shaped crisis confirmed.** Weighted low-proficiency (math < 420) fell steadily "
            "2009→2018 (**69.3% → 41.9%**), then reversed sharply to **75.3% in 2022** — the worst of "
            "all five cycles and the study's central puzzle.\n"
            "- **Not a pure SES story.** From 2018→2022 `TEACHSUP` shows the largest *negative* drift "
            "while `HOMEPOS` drifts *positively*; a simple 'poorer students' narrative does not fit.\n"
            "- **SES gradient real but partial.** Weighted Cohen's *d* for ESCS (at-risk vs proficient) "
            "≈ **−0.47** (medium); gender × at-risk is significant (χ² *p* < 0.001) but small "
            "(Cramér's V ≈ 0.07).\n"
            "- **Data-availability caveat.** Albania genuinely lacks ESCS in 2012 & 2015 (an OECD gap, "
            "not a bug) — longitudinal SES comparisons must skip those cycles.\n"
            "- **Next:** notebook 03 formalises the 2018→2022 change as *covariate shift*."
        ),
    ]


# ===========================================================================
# Notebook 02 — Comparative EDA
# ===========================================================================

def nb_02_comparative() -> list:
    return [
        md(
            "# 02 — Comparative Analysis: Albania vs. Peer Countries (PISA 2022)\n\n"
            "Cross-country comparison across Balkan peers (MKD, MNE, SRB, BGR), top performers (EST, FIN), "
            "and GDP-matched economies (MEX, COL). Dataset: 323,121 students, 9 countries, 5 cycles."
        ),
        code(HEADER),
        code(
            "from src.features.target import add_all_targets\n"
            "df = add_all_targets(pd.read_parquet('../data/processed/comparison_dataset.parquet'))\n"
            "print('Rows:', len(df))\n"
            "df.pivot_table(index='COUNTRY', columns='CYCLE', values='MATH_PV_MEAN', aggfunc='count', fill_value=0).astype(int)"
        ),
        md("## 1. Low-proficiency ranking (2022)\n\n"
           "Where does Albania stand against peers?"),
        code(
            "from src.visualization.eda import plot_country_atrisk_ranking\n"
            "fig = plot_country_atrisk_ranking(df, cycle=2022); plt.show()"
        ),
        md("**Reading:** Albania has the highest low-proficiency rate (~75%) — above its Balkan neighbours "
           "and GDP-matched peers, and ~5× Estonia's rate. This motivates studying *which factors* differ."),
        md("## 1b. Ranking with design-based standard errors (BRR + PVs)\n\n"
           "The bar chart ranks point estimates; to judge whether two countries *differ significantly* we "
           "need proper PISA standard errors. `pv_statistic_brr` combines BRR sampling variance (80 Fay "
           "replicate weights, per PV) with the between-plausible-value (Rubin) variance. All nine 2022 "
           "cohorts have replicate weights, so every SE here is fully design-based."),
        code(
            "from src.data.weights import pv_statistic_brr\n"
            "d22 = df[df.CYCLE==2022]\n"
            "rows = []\n"
            "for cnt in sorted(d22.COUNTRY.unique()):\n"
            "    r = pv_statistic_brr(d22[d22.COUNTRY==cnt], statistic='at_risk', domain='math')\n"
            "    rows.append({'country':cnt,\n"
            "                 'at_risk':round(r['estimate'],4),\n"
            "                 'SE':round(r['se'],4),\n"
            "                 'ci95_low':round(r['ci95_low'],4),\n"
            "                 'ci95_high':round(r['ci95_high'],4),\n"
            "                 'FMI':round(r['fmi'],3)})\n"
            "pd.DataFrame(rows).sort_values('at_risk', ascending=False).reset_index(drop=True)"
        ),
        md("**Reading:** Albania's CI sits clearly above the Balkan peers' — the gap is not sampling "
           "noise. Estonia's non-overlapping low CI confirms the extremes are statistically distinct."),
        md("## 2. SES gradient by country (2022)\n\n"
           "Slope of math score on ESCS. A steeper slope = stronger socioeconomic determinism. "
           "**Albania has the *flattest* slope (~16 pts/SD) of all nine countries** — but this is a "
           "floor effect, not equity: nearly everyone scores low regardless of SES. Estonia and "
           "Finland have *steeper* gradients (~38–40) yet far higher overall levels."),
        code(
            "from src.visualization.eda import plot_ses_gradient_by_country\n"
            "fig = plot_ses_gradient_by_country(df, cycle=2022); plt.show()"
        ),
        md("## 3. Longitudinal trajectories — all countries\n\n"
           "Weighted low-proficiency rate over time, per country."),
        code(
            "from src.data.weights import weighted_proportion\n"
            "from src.visualization.style import apply_publication_style, COUNTRY_COLORS\n"
            "apply_publication_style()\n"
            "fig, ax = plt.subplots(figsize=(10,6))\n"
            "for cnt in sorted(df.COUNTRY.unique()):\n"
            "    sub = df[df.COUNTRY==cnt]\n"
            "    pts = []\n"
            "    for c in sorted(sub.CYCLE.unique()):\n"
            "        s = sub[sub.CYCLE==c]\n"
            "        if len(s) < 100: continue\n"
            "        pts.append((c, weighted_proportion(s['AT_RISK_MATH'], s['W_FSTUWT'])*100))\n"
            "    if pts:\n"
            "        xs, ys = zip(*pts)\n"
            "        lw = 3 if cnt=='ALB' else 1.5\n"
            "        ax.plot(xs, ys, marker='o', label=cnt, color=COUNTRY_COLORS.get(cnt,'#888'), linewidth=lw)\n"
            "ax.set_xlabel('PISA Cycle'); ax.set_ylabel('Low-proficiency rate (%)')\n"
            "ax.set_title('Low-Proficiency Trajectories by Country (2009–2022)')\n"
            "ax.legend(ncol=2, fontsize=8); plt.show()"
        ),
        md("**Takeaway:** Albania's 2022 spike is unusually sharp relative to peers, several of which also "
           "regressed post-COVID but less severely."),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **Albania is the negative outlier.** 2022 low-proficiency ≈ **75%** — highest in the "
            "panel, above Balkan peers (N. Macedonia 67%, Montenegro 60%) and GDP-matched economies "
            "(Colombia 73%, Mexico 67%), and ~5× Estonia (**14%**).\n"
            "- **Flattest SES gradient — a floor effect, not equity.** Albania's math-on-ESCS slope "
            "(~16 pts/SD) is the *flattest* of the nine countries; even advantaged students score low "
            "(mean ~369). Estonia/Finland have *steeper* gradients (~38–40) but much higher levels — "
            "SES predicts more there simply because there is more score range to predict.\n"
            "- **Shared but uneven regression.** Several peers dipped in 2022, none as sharply as "
            "Albania — its spike is structurally distinct, not a common regional shock.\n"
            "- **Next:** is Albania's shift merely lower scores, or a change in *who* is at risk? "
            "→ notebook 03."
        ),
    ]


# ===========================================================================
# Notebook 03 — Covariate shift / 2022 crisis deep-dive
# ===========================================================================

def nb_03_covariate_shift() -> list:
    return [
        md(
            "# 03 — The 2022 Crisis: Covariate Shift Analysis\n\n"
            "**Core contribution.** Did Albania merely score lower in 2022, or did the *structure* of who is "
            "at risk fundamentally change? We quantify distributional shift between the 2018 and 2022 cohorts."
        ),
        code(HEADER),
        code(
            "from src.features.target import add_all_targets\n"
            "from src.statistics.tests import covariate_shift_test, maximum_mean_discrepancy\n"
            "df = add_all_targets(pd.read_parquet('../data/processed/albania_longitudinal.parquet'))\n"
            "core = ['ESCS','HOMEPOS','BELONG','TEACHSUP','ANXMAT','GRADE','REPEAT','IMMIG','GENDER']\n"
            "d2018 = df[df.CYCLE==2018]; d2022 = df[df.CYCLE==2022]\n"
            "print('2018 n =', len(d2018), '| 2022 n =', len(d2022))"
        ),
        md("## 1. Domain-classifier shift detection\n\n"
           "Train a classifier to distinguish 2018 from 2022 students using background features alone. "
           "AUC ≫ 0.5 ⇒ the feature distributions are detectably different (covariate shift)."),
        code(
            "res = covariate_shift_test(d2018, d2022, core, n_subsample=2000)\n"
            "print('Shift-detection AUC:', round(res['shift_detection_auc'], 4),\n"
            "      '95% CI', res['shift_detection_auc_95ci'])\n"
            "print('Significant shift:', res['significant_shift'])\n"
            "pd.Series(res['per_feature_smd']).round(3).rename('SMD (2022-2018)').to_frame()"
        ),
        md("## 2. Maximum Mean Discrepancy (MMD)\n\n"
           "A kernel two-sample statistic. Larger = more distributional divergence."),
        code(
            "from sklearn.preprocessing import StandardScaler\n"
            "common = [f for f in core if d2018[f].notna().any() and d2022[f].notna().any()]\n"
            "med = pd.concat([d2018[common], d2022[common]]).median()\n"
            "X1 = StandardScaler().fit_transform(d2018[common].fillna(med).sample(min(1500,len(d2018)), random_state=42))\n"
            "X2 = StandardScaler().fit_transform(d2022[common].fillna(med).sample(min(1500,len(d2022)), random_state=42))\n"
            "print('MMD (2018 vs 2022):', round(maximum_mean_discrepancy(X1, X2), 5))"
        ),
        md("## 3. Per-feature distribution overlays\n\n"
           "Visualise the shift feature by feature."),
        code(
            "from src.visualization.style import apply_publication_style\n"
            "apply_publication_style()\n"
            "import seaborn as sns\n"
            "plot_feats = [f for f in ['ESCS','HOMEPOS','BELONG','TEACHSUP','ANXMAT'] if d2022[f].notna().any() and d2018[f].notna().any()]\n"
            "fig, axes = plt.subplots(1, len(plot_feats), figsize=(4*len(plot_feats),3.5))\n"
            "for ax, f in zip(np.atleast_1d(axes), plot_feats):\n"
            "    sns.kdeplot(d2018[f].dropna(), ax=ax, label='2018', fill=True, alpha=.3)\n"
            "    sns.kdeplot(d2022[f].dropna(), ax=ax, label='2022', fill=True, alpha=.3)\n"
            "    ax.set_title(f); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("## 4. Interpretation\n\n"
           "If shift-AUC is high and TEACHSUP/HOMEPOS dominate the drift, the 2022 regression reflects a "
           "*structural* change in students' learning environment — not just lower scores. This is the "
           "mechanism we test against the predictive models in notebooks 05–07 (where tree models trained "
           "on 2009–2018 collapse out-of-sample on 2022)."),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **Structural shift, not just lower scores.** A domain classifier separates 2018 from 2022 "
            "students on background features alone with AUC **well above 0.5** — the *composition* of "
            "risk changed, corroborated independently by the MMD statistic.\n"
            "- **Learning-environment drivers.** `TEACHSUP` and `HOMEPOS` dominate the per-feature drift; "
            "the 2022 cohort differs most in reported teacher support and material resources — not in "
            "immigration or grade.\n"
            "- **Why it matters.** Covariate shift predicts that a model trained on the improving "
            "2009–2018 era should transfer *worse* to 2022 — tested directly in notebook 04.\n"
            "- **Caveat.** Detection AUC depends on imputation/subsampling; we report a **weighted** "
            "subsample with a 95% CI and median-imputed missing indicators to avoid inflating it."
        ),
    ]


# Header for modeling notebooks: KMP env MUST be set before any OpenMP import.
MODEL_HEADER = """import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # macOS duplicate OpenMP guard
try:
    import lightgbm  # noqa: F401 - load its Homebrew libomp BEFORE sklearn's (import-order fix, avoids rc=-11)
except Exception:
    pass
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
"""


# ===========================================================================
# Notebook 04 — Modeling: comparison + out-of-sample
# ===========================================================================

def nb_04_modeling() -> list:
    return [
        md(
            "# 04 — Predictive Modeling: Albania 2022 + Out-of-Sample\n\n"
            "Compares the model zoo on the Albania 2022 cohort (repeated stratified CV, weighted) "
            "and runs the out-of-sample experiment (train 2009–2018 → test 2022).\n\n"
            "**Reproducibility note.** scikit-learn (`libgomp`) and the gradient boosters "
            "(`libomp`) can abort a single process on macOS if both load together. So this notebook "
            "runs **scikit-learn models live in-kernel**, and **loads the booster CV rows** from the "
            "CSV produced by `scripts/run_model_comparison.py` (which runs boosters in their own "
            "process). The out-of-sample cell is lightweight and runs live."
        ),
        code(MODEL_HEADER),
        code(
            "from src.models.prepare import build_model_data\n"
            "from src.models.experiment import compare_models_cv\n\n"
            "df = pd.read_parquet('../data/processed/alb_2022.parquet')\n"
            "feats = ['ESCS','HOMEPOS','GENDER','REPEAT','IMMIG','BELONG','TEACHSUP',\n"
            "         'ICTHOME','ICTSCH','ANXMAT','GRADE','HISCED','HISEI']\n"
            "data = build_model_data(df, feats, domain='math')\n"
            "print(data.meta)\n"
            "print('Features:', data.feature_names)"
        ),
        md("## 1. scikit-learn model comparison (live, 5×4 repeated stratified CV)\n\n"
           "Weighted, leakage-safe (median imputation per fold). Boosters added in the next cell."),
        code(
            "sklearn_models = ['logistic_regression','decision_tree','random_forest',\n"
            "                  'extra_trees','gradient_boosting']\n"
            "res_sklearn = compare_models_cv(data, sklearn_models, outer_folds=5, outer_repeats=4)\n"
            "res_sklearn[['model','roc_auc_mean','roc_auc_std','pr_auc_mean','f1_macro_mean','mcc_mean']]"
        ),
        md("## 2. Full comparison incl. boosters\n\n"
           "Booster rows come from `outputs/results/albania_2022_model_comparison.csv` "
           "(generated by `python scripts/run_model_comparison.py`). We display the merged master table."),
        code(
            "full = pd.read_csv('../outputs/results/albania_2022_model_comparison.csv')\n"
            "full[['model','roc_auc_mean','roc_auc_std','pr_auc_mean','f1_macro_mean','mcc_mean']]"
        ),
        code(
            "from src.visualization.model_plots import plot_cv_distribution\n"
            "fig = plot_cv_distribution(full, metric='roc_auc'); plt.show()"
        ),
        md("**Reading:** CatBoost and Gradient Boosting lead (AUC ≈ 0.72); Logistic Regression is a "
           "strong, interpretable baseline (0.70). Decision Tree trails badly (0.57)."),
        md("## 3. Out-of-sample experiment (train 2009–2018 → test 2022)\n\n"
           "Tests whether models trained on the historical, improving period generalize to the 2022 "
           "crisis cohort. Lightweight (single fits) so it runs live; thread-limited for OpenMP safety."),
        code(
            "from src.models.experiment import out_of_sample\n"
            "long = pd.read_parquet('../data/processed/albania_longitudinal.parquet')\n"
            "train = long[long.CYCLE.isin([2009,2012,2015,2018])]; test = long[long.CYCLE==2022]\n"
            "oos_feats = ['ESCS','HOMEPOS','GENDER','REPEAT','IMMIG','BELONG','TEACHSUP','ANXMAT','GRADE','HISCED','HISEI']\n"
            "# scikit-learn models live; boosters were added via the script (see JSON below)\n"
            "res = out_of_sample(train, test, oos_feats,\n"
            "                    ['logistic_regression','random_forest','extra_trees','gradient_boosting'],\n"
            "                    domain='math')\n"
            "pd.DataFrame({m: {'test_AUC': d['roc_auc']} for m, d in res['models'].items()}).T"
        ),
        code(
            "# Full OOS results (all 7 models) from the saved experiment JSON\n"
            "import json\n"
            "oos = json.load(open('../outputs/results/oos_2022_experiment.json'))\n"
            "ok = {m: d for m, d in oos['models'].items() if 'roc_auc' in d}\n"
            "failed = [m for m, d in oos['models'].items() if 'error' in d]\n"
            "if failed: print('models unavailable (local libomp install):', failed)\n"
            "pd.DataFrame({m: {'test_AUC': d['roc_auc'], 'PR_AUC': d['pr_auc'], 'MCC': d['mcc']}\n"
            "              for m, d in ok.items()}).T.sort_values('test_AUC', ascending=False)"
        ),
        md("## 4. Rigorous PV evaluation — Rubin's rules\n\n"
           "The comparison above uses the PV-mean point target (cheap, standard for model *selection*). "
           "For the headline number we instead fit the winning model **once per plausible value** and "
           "combine the 10 estimates with Rubin's rules — the correct PISA methodology. The reported SE "
           "and CI then include the between-PV (imputation) uncertainty, and FMI is the fraction of "
           "missing information due to PV draw."),
        code(
            "from src.models.experiment import per_pv_cv_evaluate\n"
            "pv = per_pv_cv_evaluate(df, feats, 'gradient_boosting', domain='math',\n"
            "                        outer_folds=5, outer_repeats=2)\n"
            "auc = pv['roc_auc']\n"
            "print(f\"PV-combined AUC = {auc['estimate']:.4f}  \"\n"
            "      f\"(95% CI {auc['ci95_low']:.4f}-{auc['ci95_high']:.4f}, FMI={auc['fmi']:.3f}, n_PV={auc['n_pv']})\")\n"
            "pd.DataFrame({k: pv[k] for k in ['roc_auc','pr_auc','f1_macro','mcc'] if k in pv}).T[['estimate','se','fmi']]"
        ),
        md("**Reading:** the Rubin-combined AUC is close to the point-target CV AUC (the PV draw adds "
           "modest uncertainty, captured by FMI). This confirms the model ranking is not an artefact of "
           "collapsing the plausible values."),
        md("**Takeaway:** out-of-sample AUC falls to ~0.62–0.67 for every model. The covariate shift "
           "(notebook 03, detection AUC 0.98) is real and degrades transfer, but no model fully "
           "collapses — the 2022 risk structure shifted yet remains partly predictable from history."),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **Best in-sample model:** CatBoost (weighted CV AUC **0.731**), narrowly over Gradient "
            "Boosting (0.727) and — notably — Logistic Regression (0.727), a strong interpretable "
            "baseline. LightGBM (0.715) sits mid-pack; Decision Tree trails (0.568). (LightGBM's "
            "earlier `rc=-11` crash was an OpenMP import-order bug, now fixed, so it is back in the "
            "comparison.)\n"
            "- **PISA-correct headline.** Rubin's-rules per-PV evaluation gives an AUC close to the "
            "point-target CV number with a modest FMI — the ranking is not an artefact of collapsing "
            "the plausible values.\n"
            "- **Out-of-sample degrades but does not collapse.** Train 2009–2018 → test 2022 drops AUC "
            "to ~**0.62–0.67** (GBM/RF best at 0.674). The covariate shift is real and hurts transfer, "
            "yet 2022 risk stays partly predictable from history — a nuance vs. a 'total breakdown' "
            "story.\n"
            "- **Rigor:** metrics are **weighted** throughout (population estimates) with bootstrap/fold "
            "CIs; the decision threshold is tuned on train only."
        ),
    ]


# ===========================================================================
# Notebook 05 — Explainability (SHAP)
# ===========================================================================

def nb_05_explainability() -> list:
    return [
        md(
            "# 05 — Explainability: SHAP (Albania 2022)\n\n"
            "Global and local explanations for the LightGBM model on the Albania 2022 cohort, "
            "trained on the **school-context** feature set (student features + survey-weighted "
            "school-mean ESCS/HOMEPOS/ANXMAT/TEACHSUP + cohort size — the additions that lifted "
            "AUC ~+0.05). LightGBM is fit fresh in this kernel — its Homebrew libomp is imported "
            "before scikit-learn (import-order fix), so the single-booster fit is OpenMP-safe and "
            "SHAP runs live."
        ),
        code(MODEL_HEADER),
        code(
            "import shap\n"
            "from src.models.prepare import build_model_data, impute_median\n"
            "from src.features.transformers import EngineeredFeatureBuilder\n"
            "from src.models.registry import get_model\n"
            "from src.explainability.shap_analysis import compute_shap_values, global_feature_importance\n\n"
            "df = pd.read_parquet('../data/processed/alb_2022.parquet')\n"
            "feats = ['ESCS','HOMEPOS','GENDER','REPEAT','IMMIG','BELONG','TEACHSUP',\n"
            "         'ICTHOME','ICTSCH','ANXMAT','GRADE','HISCED','HISEI']\n"
            "data = build_model_data(df, feats, domain='math', add_school_context=True)\n"
            "# engineered features (SES_COMPLETE, interactions, ...) built here for the\n"
            "# final explanatory model (single fit on all data, so fit-on-all is fine)\n"
            "X_eng = EngineeredFeatureBuilder().fit_transform(data.X)\n"
            "(X,) = impute_median(X_eng); y = data.y.values\n"
            "model = get_model('lightgbm'); model.fit(X, y, sample_weight=data.weights.values)\n"
            "print('trained on', X.shape)"
        ),
        md("## 1. Global feature importance (mean |SHAP|)"),
        code(
            "shap_vals, names = compute_shap_values(model, X,\n"
            "        X_background=X.sample(min(300, len(X)), random_state=42), max_samples=2000)\n"
            "imp = global_feature_importance(shap_vals, names)\n"
            "imp"
        ),
        code(
            "from src.visualization.style import apply_publication_style\n"
            "apply_publication_style()\n"
            "imp_s = imp.sort_values('mean_abs_shap')\n"
            "fig, ax = plt.subplots(figsize=(8,5))\n"
            "ax.barh(imp_s['feature'], imp_s['mean_abs_shap'], color='#0072B2')\n"
            "ax.set_xlabel('Mean |SHAP value|'); ax.set_title('Global Feature Importance — Albania 2022 (LightGBM)')\n"
            "plt.show()"
        ),
        md("**Reading:** the top drivers are now dominated by **school-level context** — "
           "`SCH_MEAN_HOMEPOS` (school mean home possessions) is #1, with `SCH_MEAN_TEACHSUP`, "
           "`SCH_MEAN_ESCS` and `SCH_N` (cohort size) all in the top 7. The strongest *individual* "
           "factor is `ANXMAT` (math anxiety, #2), then `HISCED`/`HOMEPOS`. The model reads a "
           "student's risk more from the socioeconomic composition of their school than from their "
           "own resources — the compositional effect made explicit. Immigration and grade "
           "deviation still contribute almost nothing."),
        md("## 2. SHAP beeswarm — direction of effects"),
        code(
            "X_s = X.sample(min(2000, len(X)), random_state=42).reset_index(drop=True)\n"
            "shap.summary_plot(shap_vals, X_s[names], show=True, plot_size=(8,6))"
        ),
        md("## 3. SHAP dependence — does SES moderate math anxiety?"),
        code(
            "if 'ANXMAT' in names and 'ESCS' in names:\n"
            "    shap.dependence_plot('ANXMAT', shap_vals, X_s[names], interaction_index='ESCS', show=True)"
        ),
        md("**Interpretation:** higher math anxiety pushes predictions toward at-risk; the colouring shows "
           "whether socioeconomic status buffers or amplifies that effect — a policy-relevant interaction."),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **School composition dominates (mean |SHAP|):** `SCH_MEAN_HOMEPOS` > `ANXMAT` > "
            "`SCH_MEAN_TEACHSUP` > `SCH_MEAN_ESCS` > `HISCED` > `HOMEPOS` > `SCH_N`. Four of the "
            "top seven are school-level — the model reads risk more from the *school's* aggregate "
            "resources and support than from the student's own.\n"
            "- **Top individual factor:** `ANXMAT` (math anxiety) is the strongest student-level "
            "driver, followed by parental education / home resources (`HISCED`, `HOMEPOS`).\n"
            "- **Negligible:** immigration status and grade deviation add almost nothing — Albania's "
            "risk is a school-context-and-affect story, not a migration one.\n"
            "- **Policy read:** the highest-leverage levers are *structural* — the socioeconomic "
            "segregation of schools and school-level teacher support — alongside individual math "
            "anxiety. A pure individual-student intervention misses the dominant, compositional "
            "signal. SHAP explains the *model* — associational, not causal.\n"
            "- **Equity caveat:** because school-mean SES is now a top feature, the model partly "
            "encodes *where* a student goes to school; this is exactly the mechanism the fairness "
            "audit (notebook 07) probes for the SES gap.\n"
            "- **Caveat:** explanations are for the single all-data LightGBM fit; per-fold "
            "stability and per-country SHAP comparison come in Phase 8."
        ),
    ]


# ===========================================================================
# Notebook 06 — Local explainability: SHAP case studies + PDP/ICE
# ===========================================================================

def nb_06_explainability_cases() -> list:
    return [
        md(
            "# 06 — Local Explainability: SHAP Case Studies + PDP/ICE (Albania 2022)\n\n"
            "Notebook 05 gave the *global* picture (which features matter overall). This notebook is the "
            "*local* half of the story, mirroring `scripts/run_explainability_cases.py`:\n\n"
            "1. One representative **TP / TN / FP / FN** instance, each with a SHAP **waterfall** — why the "
            "model was confidently right, and confidently wrong.\n"
            "2. **PDP + centered-ICE** curves for the top drivers — the *shape* of each feature's effect and "
            "the instance-level heterogeneity (interactions) the average hides.\n\n"
            "The explanatory model is a single LightGBM fit on all Albania-2022 data (fit-on-all is fine for "
            "explanation, not evaluation), on the **school-context** feature set (student + school-mean "
            "features). LightGBM is imported before scikit-learn (import-order libomp fix), so the booster "
            "fit and SHAP run live in-kernel."
        ),
        code(MODEL_HEADER),
        code(
            "from src.explainability.shap_analysis import (\n"
            "    compute_local_shap, compute_shap_values, global_feature_importance,\n"
            "    select_representative_cases)\n"
            "from src.explainability.partial_dependence import centered_ice, compute_pdp\n"
            "from src.features.transformers import EngineeredFeatureBuilder\n"
            "from src.models.prepare import build_model_data, impute_median\n"
            "from src.models.registry import get_model\n\n"
            "df = pd.read_parquet('../data/processed/alb_2022.parquet')\n"
            "feats = ['ESCS','HOMEPOS','GENDER','REPEAT','IMMIG','BELONG','TEACHSUP',\n"
            "         'ICTHOME','ICTSCH','ANXMAT','GRADE','HISCED','HISEI']\n"
            "data = build_model_data(df, feats, domain='math', add_school_context=True)\n"
            "X_eng = EngineeredFeatureBuilder().fit_transform(data.X)\n"
            "(X,) = impute_median(X_eng); y = data.y.values\n"
            "model = get_model('lightgbm'); model.fit(X, y, sample_weight=data.weights.values)\n"
            "print('trained on', X.shape, '| at-risk rate =', round(float(y.mean()),3))"
        ),
        md("## 1. Representative confusion-matrix cases\n\n"
           "For each quadrant we pick the *most confident* instance — a confidently correct case (TP/TN) and "
           "a confidently **wrong** case (FP/FN), whose local SHAP explanation is most informative."),
        code(
            "cases = select_representative_cases(model, X, y)\n"
            "pd.DataFrame([{'quadrant':k, **{kk:vv for kk,vv in c.items() if kk!='index'}}\n"
            "              for k,c in cases.items()])"
        ),
        md("## 2. Local SHAP waterfalls\n\n"
           "Signed feature contributions (log-odds) from the model base value toward each instance's "
           "prediction. **Red = pushes toward at-risk, blue = pushes toward proficient.**"),
        code(
            "def waterfall(ax, values, base, pred, names, row, top_n=10):\n"
            "    order = np.argsort(np.abs(values))[::-1][:top_n]\n"
            "    contrib = values[order]\n"
            "    labels = [f'{names[i]} = {row[i]:.2f}' for i in order]\n"
            "    colors = ['#D55E00' if c>0 else '#0072B2' for c in contrib]\n"
            "    yy = np.arange(len(contrib))[::-1]\n"
            "    ax.barh(yy, contrib, color=colors); ax.set_yticks(yy)\n"
            "    ax.set_yticklabels(labels, fontsize=8); ax.axvline(0, color='k', lw=0.8)\n"
            "    ax.set_xlabel('SHAP contribution (log-odds)')\n"
            "    ax.set_title(f'base={base:.2f}  ->  P(at-risk)={pred:.2f}', fontsize=9)\n\n"
            "from src.visualization.style import apply_publication_style\n"
            "apply_publication_style()\n"
            "idx = [c['index'] for c in cases.values()]\n"
            "X_bg = X.sample(min(300, len(X)), random_state=42)\n"
            "vals, base, names = compute_local_shap(model, X.iloc[idx], X_background=X_bg)\n"
            "fig, axes = plt.subplots(2, 2, figsize=(13, 9))\n"
            "for ax, (k, c), v, b in zip(axes.ravel(), cases.items(), vals, base):\n"
            "    waterfall(ax, v, b, c['prob'], names, X.iloc[c['index']].values)\n"
            "    ax.set_ylabel(f\"{k} - {c['label']}\", fontsize=9)\n"
            "fig.suptitle('Local SHAP explanations - Albania 2022 (LightGBM)', fontsize=12)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md("**Reading:** the TP/TN waterfalls show the model stacking up **school-context** signals "
           "(school-mean home possessions, teacher support, ESCS) alongside individual math anxiety. "
           "The FP/FN cases expose where those signals mislead — e.g. a student in a low-resource "
           "school who is nonetheless proficient (FP), or a resourced-school student the model misses "
           "(FN) — the individual bucking their school context."),
        md("## 3. Partial dependence + centered ICE\n\n"
           "PDP = mean predicted at-risk probability as one feature varies (all else held). Centered ICE "
           "keeps one grey line per instance (anchored to the grid start) so heterogeneity — the fingerprint "
           "of interactions — is visible where the average curve hides it."),
        code(
            "shap_vals, shap_names = compute_shap_values(model, X, X_background=X_bg, max_samples=2000)\n"
            "imp = global_feature_importance(shap_vals, shap_names)\n"
            "raw_top = [f for f in imp['feature'] if f in feats][:4]\n"
            "print('PDP/ICE for:', raw_top)\n"
            "fig, axes = plt.subplots(2, 2, figsize=(12, 9))\n"
            "rng = np.random.default_rng(42)\n"
            "for ax, feat in zip(axes.ravel(), raw_top):\n"
            "    pdp = compute_pdp(model, X, feat, grid_resolution=30, ice=True)\n"
            "    cice = centered_ice(pdp)\n"
            "    sub = rng.choice(cice.shape[0], size=min(200, cice.shape[0]), replace=False)\n"
            "    for r in cice[sub]:\n"
            "        ax.plot(pdp.grid, r, color='0.7', lw=0.4, alpha=0.5)\n"
            "    ax.plot(pdp.grid, pdp.average - pdp.average[0], color='#D55E00', lw=2.2, label='PDP (mean)')\n"
            "    ax.set_title(feat, fontsize=10); ax.set_xlabel(feat)\n"
            "    ax.set_ylabel('delta P(at-risk) vs. grid start'); ax.legend(fontsize=8)\n"
            "fig.suptitle('Partial dependence + centered ICE - Albania 2022 (LightGBM)', fontsize=12)\n"
            "fig.tight_layout(); plt.show()"
        ),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **Local confirms global.** The confidently-correct (TP/TN) waterfalls are dominated by the "
            "same drivers notebook 05 flagged globally — **school-context** features (school-mean "
            "HOMEPOS/TEACHSUP/ESCS) plus individual math anxiety — so the compositional story isn't an "
            "averaging artefact.\n"
            "- **Where the model fails.** FP/FN cases show the failure mode: the model leans on school "
            "context + anxiety, so a student who defies their school's profile (proficient in a "
            "low-resource school, or at-risk in a resourced one) is misclassified. Risk is probabilistic, "
            "not deterministic, at the individual level.\n"
            "- **Effect shapes are monotone but heterogeneous.** PDP curves for the top drivers move risk "
            "smoothly and roughly monotonically; the ICE spread shows the *magnitude* varies by instance — "
            "evidence of interactions (e.g. anxiety's effect moderated by SES).\n"
            "- **Policy read:** the highest-leverage levers (math anxiety, material support) act on most "
            "students but not uniformly — targeting should be individualised, not blanket. Explanations are "
            "**associational** (they explain the model), not causal.\n"
            "- Figures mirror D3 (`D3_shap_local_cases`) and D4 (`D4_pdp_ice`); the case table matches "
            "`shap_local_cases_2022.csv`."
        ),
    ]


# ===========================================================================
# Notebook 07 — Fairness audit
# ===========================================================================

def nb_07_fairness() -> list:
    return [
        md(
            "# 07 — Fairness Audit: Albania 2022 (survey-weighted)\n\n"
            "Would a risk-screening model built on this data treat subgroups equitably? Mirroring "
            "`scripts/run_fairness_audit.py`, we generate **leakage-safe out-of-fold** predictions (engineered "
            "features fit per fold), then audit demographic parity, equal opportunity (TPR), FPR and "
            "calibration across three protected attributes — **GENDER**, **SES quintile** (from weighted "
            "ESCS), and **immigrant status**.\n\n"
            "Every metric is **survey-weighted** (`W_FSTUWT`): an unweighted audit would misstate "
            "population-level gaps exactly like the descriptive shortcuts the project criticises. The model "
            "uses the **school-context** feature set (the headline model); because school-mean SES is now a "
            "top predictor, the SES-fairness question is especially pointed. LightGBM is imported before "
            "scikit-learn (import-order libomp fix)."
        ),
        code(MODEL_HEADER),
        code(
            "from sklearn.model_selection import StratifiedKFold\n"
            "from threadpoolctl import threadpool_limits\n"
            "from src.data.weights import compute_ses_quintiles\n"
            "from src.fairness.metrics import fairness_audit, threshold_sweep\n"
            "from src.models.evaluate import fit_with_sample_weight\n"
            "from src.models.experiment import _wrap\n"
            "from src.models.prepare import build_model_data\n"
            "from src.models.registry import get_model\n\n"
            "FEATS = ['ESCS','HOMEPOS','GENDER','REPEAT','IMMIG','BELONG','TEACHSUP',\n"
            "         'ICTHOME','ICTSCH','ANXMAT','GRADE','HISCED','HISEI']\n"
            "df = pd.read_parquet('../data/processed/alb_2022.parquet')\n"
            "data = build_model_data(df, FEATS, domain='math', add_school_context=True)\n"
            "X, y, w = data.X, data.y, data.weights\n"
            "print('cohort n =', len(y), '| at-risk rate =', round(float(y.mean()),3))"
        ),
        md("## 1. Leakage-safe out-of-fold predictions\n\n"
           "Weighted 5-fold stratified CV; each fold's pipeline (engineered features + booster) is fit on "
           "train only, so every prediction is genuinely out-of-sample. Thread-limited for OpenMP safety."),
        code(
            "def oof_predictions(X, y, w, n_splits=5, seed=42):\n"
            "    prob = np.full(len(y), np.nan)\n"
            "    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)\n"
            "    with threadpool_limits(limits=1):\n"
            "        for tr, te in skf.split(X, y):\n"
            "            pipe = _wrap(get_model('lightgbm'))\n"
            "            fit_with_sample_weight(pipe, X.iloc[tr], y.iloc[tr], w.iloc[tr].values)\n"
            "            prob[te] = pipe.predict_proba(X.iloc[te])[:, 1]\n"
            "    return prob\n\n"
            "prob = oof_predictions(X, y, w)\n"
            "print('OOF coverage:', int(np.isfinite(prob).sum()), '/', len(prob))"
        ),
        code(
            "audit = pd.DataFrame({\n"
            "    'y_true': y.values.astype(float),\n"
            "    'y_prob': prob,\n"
            "    'y_pred': (prob >= 0.5).astype(float),\n"
            "    'W_FSTUWT': w.values,\n"
            "    'GENDER': X['GENDER'].values,\n"
            "    'IMMIG': X['IMMIG'].values,\n"
            "    'ESCS': X['ESCS'].values})\n"
            "# SES quintile from *weighted* ESCS breaks (population quintiles, not sample)\n"
            "audit = compute_ses_quintiles(audit, ses_col='ESCS', weight_col='W_FSTUWT')\n"
            "audit['SES_QUINTILE'] = audit['SES_QUINTILE'].astype('float')\n"
            "audit = audit.dropna(subset=['y_prob'])\n"
            "print('audit rows:', len(audit))"
        ),
        md("## 2. Fairness audit across protected attributes"),
        code(
            "reports = fairness_audit(audit, 'y_true', 'y_pred', 'y_prob',\n"
            "                         ['GENDER','SES_QUINTILE','IMMIG'], weight_col='W_FSTUWT')\n"
            "for attr, rep in reports.items():\n"
            "    print(rep.summary()); print()"
        ),
        code(
            "gap = pd.DataFrame({attr: {'parity_gap': rep.parity_gap,\n"
            "                           'opportunity_gap': rep.opportunity_gap,\n"
            "                           'fpr_gap': rep.fpr_gap}\n"
            "                    for attr, rep in reports.items()}).T.round(4)\n"
            "gap"
        ),
        md("**Reading:** **SES quintile is the least fair axis** — parity gap ~**0.48** and FPR gap "
           "~**0.63**. The model flags almost every bottom-quintile student as at-risk (selection 0.91, "
           "FPR 0.83) while rarely flagging the top (0.43). With school-mean SES now a top feature the "
           "model has doubly learned the SES gradient and would *operationalise* it if deployed. "
           "**Immigrant status** shows a large FPR gap too (~0.59, small/unstable groups). By contrast the "
           "**gender gap shrank** vs. the student-only model (parity 0.18→0.11, FPR 0.14→0.05) — school "
           "context absorbed some of the gender signal."),
        md("## 3. Per-group detail — where the gap lives"),
        code(
            "ses = reports['SES_QUINTILE']\n"
            "pd.DataFrame({'selection_rate': ses.demographic_parity,\n"
            "              'TPR': ses.equal_opportunity,\n"
            "              'FPR': ses.fpr_by_group,\n"
            "              'mean_pred_prob': ses.calibration_by_group}).round(3)"
        ),
        md("## 4. Threshold sweep — can the operating point buy fairness?\n\n"
           "Does moving the decision threshold trade parity against the TPR/FPR gaps? Shown for GENDER."),
        code(
            "from src.visualization.style import apply_publication_style\n"
            "apply_publication_style()\n"
            "sweep = threshold_sweep(audit, 'y_true', 'y_prob', 'GENDER', weight_col='W_FSTUWT')\n"
            "fig, ax = plt.subplots(figsize=(7, 5))\n"
            "ax.plot(sweep['threshold'], sweep['parity_gap'], '-o', label='Demographic-parity gap')\n"
            "ax.plot(sweep['threshold'], sweep['opportunity_gap'], '-s', label='Equal-opportunity (TPR) gap')\n"
            "ax.plot(sweep['threshold'], sweep['fpr_gap'], '-^', label='FPR gap')\n"
            "ax.set_xlabel('Decision threshold'); ax.set_ylabel('Weighted group gap (max - min)')\n"
            "ax.set_title('Fairness gaps vs. threshold - Albania 2022, GENDER (weighted)')\n"
            "ax.legend(fontsize=8); plt.show()\n"
            "sweep.round(4)"
        ),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **SES is the fairness fault line.** Weighted parity gap ≈ **0.48** and FPR gap ≈ **0.63** "
            "across SES quintiles — the bottom quintile is flagged at-risk ~91% of the time (selection "
            "rate 0.91, FPR 0.83) vs. ~43% for the top. Adding school-mean SES as a top feature does **not** "
            "fix this — school SES *is* SES, so the model doubly encodes the gradient and would act on it.\n"
            "- **Gender gap shrank.** Vs. the student-only model the gender gaps dropped (parity 0.18→0.11, "
            "FPR 0.14→0.05) — school context absorbed part of the gender signal. Girls (group 1) still get "
            "a somewhat higher selection rate and FPR.\n"
            "- **Immigrant status:** a large **FPR gap (~0.59)** — immigrant-background students are "
            "over-flagged among the truly proficient — though the immigrant groups are small and the "
            "FPR=1.0 cell is unstable; treat as a flag, not a precise estimate.\n"
            "- **No free threshold.** The sweep shows moving the operating point shifts the gaps around "
            "rather than eliminating them — fairness here is structural (in the risk signal), not a "
            "threshold artefact. Genuine mitigation needs reweighting/constraints or group-aware "
            "calibration, not just a knob.\n"
            "- **Deployment caveat:** a naïvely deployed screener would concentrate false alarms on "
            "low-SES and immigrant-background students, and the school-context features tie a student's "
            "score to *where* they study. Any real use needs an explicit fairness constraint. "
            "Matches `fairness_audit_2022.json` and figure E1."
        ),
    ]


# ===========================================================================
# Notebook 08 — Comparative modeling: Albania vs. peers (Phase 8)
# ===========================================================================

def nb_08_comparative() -> list:
    return [
        md(
            "# 08 — Comparative Modeling: Albania vs. Peers (PISA 2022)\n\n"
            "Are Albania's risk drivers universal or idiosyncratic, and is its socioeconomic gradient "
            "unusual? We fit the **school-context LightGBM per country** (nine countries), score weighted "
            "5-fold CV AUC, assemble a **SHAP importance rank matrix** (feature × country), and measure "
            "each country's **SES gradient** (weighted slope of math score on ESCS). Heavy fitting lives in "
            "`scripts/run_comparative_modeling.py`; this notebook loads its results and narrates."
        ),
        code(HEADER),
        code(
            "tbl = pd.read_csv('../outputs/results/comparative_country_2022.csv')\n"
            "tbl"
        ),
        md("## 1. Predictability, prevalence, and the SES gradient\n\n"
           "Three numbers per country: how often students are at-risk, how well the model separates them "
           "(CV AUC), and how steeply score tracks SES."),
        code(
            "from src.visualization.style import apply_publication_style, COUNTRY_COLORS\n"
            "apply_publication_style()\n"
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))\n"
            "colors = [COUNTRY_COLORS.get(c, '#888') for c in tbl['country']]\n"
            "for ax, col, title in zip(axes,\n"
            "        ['at_risk_rate','cv_auc','ses_gradient'],\n"
            "        ['At-risk rate (math < L2)','Model CV AUC (weighted)','SES gradient (pts / SD ESCS)']):\n"
            "    order = tbl.sort_values(col)\n"
            "    ax.barh(order['country'], order[col],\n"
            "            color=[COUNTRY_COLORS.get(c,'#888') for c in order['country']])\n"
            "    ax.set_title(title, fontsize=10)\n"
            "    for i,(c,v) in enumerate(zip(order['country'], order[col])):\n"
            "        ax.text(v, i, f' {v:.2f}' if col!='ses_gradient' else f' {v:.0f}', va='center', fontsize=7)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("**Reading:** Albania has the **highest at-risk rate (0.75)** yet is among the **hardest to "
           "predict (AUC 0.77)** — when three of four students are at-risk there is little contrast to "
           "separate. Crucially, Albania's **SES gradient is the *flattest* (~17 pts/SD)** of all nine "
           "countries. That is a *floor effect*, not equity: even advantaged Albanian students score low "
           "(mean ~369), so SES barely predicts within-country. Estonia/Finland show *steeper* gradients "
           "(~38–40) but far higher levels. The mid-prevalence Balkan peers (BGR/COL, AUC ~0.86) are the "
           "most separable."),
        md("## 2. SHAP importance rank matrix\n\n"
           "For each country, features ranked 1 (top driver) … by mean |SHAP|. Following the project's "
           "colormap schema (**dark = more important**; the colorblind-unsafe red-yellow-green map is "
           "avoided), rank 1 renders darkest. Which drivers are universal, which country-specific?"),
        code(
            "from src.visualization.style import SEQUENTIAL_RANK_CMAP\n"
            "rank = pd.read_csv('../outputs/results/comparative_shap_rank_matrix.csv', index_col=0)\n"
            "rank = rank.head(12)\n"
            "fig, ax = plt.subplots(figsize=(9, 6))\n"
            "im = ax.imshow(rank.values, cmap=SEQUENTIAL_RANK_CMAP, aspect='auto', vmin=1, vmax=15)\n"
            "ax.set_xticks(range(len(rank.columns))); ax.set_xticklabels(rank.columns)\n"
            "ax.set_yticks(range(len(rank.index))); ax.set_yticklabels(rank.index, fontsize=8)\n"
            "for i in range(len(rank.index)):\n"
            "    for j in range(len(rank.columns)):\n"
            "        v = int(rank.values[i,j])\n"
            "        ax.text(j, i, v, ha='center', va='center', fontsize=7,\n"
            "                color='white' if v <= 5 else 'black')\n"
            "ax.set_title('SHAP importance rank by country — 2022 (1 = top driver)')\n"
            "fig.colorbar(im, ax=ax, label='rank (1 = most important, dark)'); plt.tight_layout(); plt.show()"
        ),
        md("**Reading:** **math anxiety (`ANXMAT`) is universal** — a top-3 driver in almost every "
           "country, including top-performer Estonia. **School composition** (`SCH_MEAN_HOMEPOS`, "
           "`SCH_MEAN_ESCS`) leads in most systems too. Albania's profile "
           "(school-HOMEPOS → anxiety → school-TEACHSUP → school-ESCS) is therefore **not idiosyncratic** — "
           "it shares the region's structure. Country-specific notes: grade repetition (`GRADE`) jumps to "
           "#2 in Colombia; individual `ESCS` is #1 in Finland (stronger individual-SES determinism)."),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **Albania = high prevalence, low separability, flat gradient.** Highest at-risk share "
            "(75%), one of the lowest model AUCs (0.77), and the flattest SES gradient (~17 pts/SD). The "
            "2022 crisis is **broad-based** — depressed across the whole SES distribution, not concentrated "
            "in the poor.\n"
            "- **Drivers are shared, not unique.** Math anxiety is a universal top factor and school "
            "socioeconomic composition dominates almost everywhere; Albania's risk structure mirrors its "
            "Balkan peers rather than standing apart. What sets Albania apart is the *level* and *breadth* "
            "of low proficiency, not a different mechanism.\n"
            "- **Policy read.** Because the gradient is flat and school composition dominates, "
            "SES-targeted transfers alone would miss most of Albania's at-risk students; system-wide levers "
            "(teacher support, math-anxiety reduction, raising the floor across all schools) are needed. "
            "Estonia's steep-but-high profile shows the opposite regime — a useful contrast, not a template "
            "to copy directly.\n"
            "- **Caveat.** Per-country AUCs are not directly comparable as 'model quality' — they partly "
            "reflect each country's at-risk prevalence (separability is hardest near 0/1 base rates). SHAP "
            "explains each *model*, associational not causal. Fits are single all-data LightGBM per "
            "country; figure `F1_shap_rank_matrix`."
        ),
    ]


# ===========================================================================
# Notebook 09 — Scenario forecast for the next PISA cycle (2026)
# ===========================================================================

def nb_09_forecast() -> list:
    return [
        md(
            "# 09 — Forecast: Albania's Low-Proficiency Rate for the Next Cycle (2026)\n\n"
            "After COVID delayed PISA 2021 to 2022, the assessment moved to a **4-year** cadence, so the "
            "next cycle is **2026**. Can we anticipate Albania's weighted "
            "low-proficiency (math < Level 2) rate? With only **five** cycles and a structural break in "
            "2022 (the COVID spike), a single point extrapolation is indefensible — so we forecast with "
            "**transparent scenarios** and propagate each cycle's *design-based* uncertainty (the BRR + "
            "plausible-value SE from notebook 01) through a Monte-Carlo. Logic lives in "
            "`src/forecast/scenarios.py` (unit-tested); this notebook narrates.\n\n"
            "> **Honesty note.** This is a *scenario* exercise, not a claim to predict a structural break. "
            "The value is the plausible range and the assumptions behind each edge, not a single number."
        ),
        code(HEADER),
        code(
            "from src.features.target import add_all_targets\n"
            "from src.data.weights import pv_statistic_brr\n"
            "from src.forecast.scenarios import monte_carlo_forecast, scenarios_to_frame\n\n"
            "df = add_all_targets(pd.read_parquet('../data/processed/albania_longitudinal.parquet'))\n"
            "rows = []\n"
            "for c in sorted(df.CYCLE.unique()):\n"
            "    r = pv_statistic_brr(df[df.CYCLE==c], statistic='at_risk', domain='math')\n"
            "    rows.append({'cycle': int(c), 'at_risk': r['estimate'], 'se': r['se'], 'brr': r['brr_available']})\n"
            "hist = pd.DataFrame(rows)\n"
            "hist"
        ),
        md("## 1. Scenario forecast for 2026\n\n"
           "Three narratives + one reference:\n"
           "- **persistence** — the 2022 crisis level holds.\n"
           "- **recovery** — the pre-COVID improving trend (2009–2018) resumes (2022 fully reverses).\n"
           "- **partial** — a 50/50 blend (the shock half-reverses).\n"
           "- **naive_linear** *(reference)* — a weighted line through all five cycles; the 2022 break "
           "makes it meaningless, shown only to demonstrate why."),
        code(
            "fc = monte_carlo_forecast(hist['cycle'].values, hist['at_risk'].values,\n"
            "                          hist['se'].values, target_year=2026, n_sims=20000)\n"
            "summary = scenarios_to_frame(fc, 2026)\n"
            "summary.to_csv('../outputs/results/forecast_2026_scenarios.csv', index=False)\n"
            "summary"
        ),
        md("## 2. Trajectory + scenario fan"),
        code(
            "from src.visualization.style import apply_publication_style\n"
            "apply_publication_style()\n"
            "fig, ax = plt.subplots(figsize=(9, 5.5))\n"
            "# historical line with 95% design-based error bars\n"
            "ax.errorbar(hist['cycle'], hist['at_risk']*100, yerr=1.96*hist['se']*100,\n"
            "            marker='o', lw=2, color='#333333', capsize=3, label='Observed (BRR 95% CI)')\n"
            "colors = {'persistence':'#D55E00', 'partial':'#E69F00', 'recovery':'#0072B2'}\n"
            "labels = {'persistence':'Persistence (crisis holds)',\n"
            "          'partial':'Partial reversion', 'recovery':'Pre-COVID recovery'}\n"
            "for i, name in enumerate(['persistence','partial','recovery']):\n"
            "    s = fc[name]; xj = 2026 + (i-1)*0.18\n"
            "    ax.errorbar([xj], [s.median*100],\n"
            "                yerr=[[(s.median-s.lo)*100], [(s.hi-s.median)*100]],\n"
            "                marker='s', ms=9, color=colors[name], capsize=4, lw=2, label=labels[name])\n"
            "    # dashed connector from 2022 to the scenario median\n"
            "    ax.plot([2022, xj], [hist['at_risk'].iloc[-1]*100, s.median*100],\n"
            "            ls='--', lw=1, color=colors[name], alpha=0.7)\n"
            "ax.axvspan(2023.5, 2027.0, color='0.92', zorder=0)\n"
            "ax.set_xticks([2009,2012,2015,2018,2022,2026])\n"
            "ax.set_xlabel('PISA cycle'); ax.set_ylabel('Low-proficiency rate, math < L2 (%)')\n"
            "ax.set_title('Albania low-proficiency: observed (2009–2022) + 2026 scenarios')\n"
            "ax.legend(fontsize=8, loc='center left'); plt.tight_layout(); plt.show()"
        ),
        md("## 3. Predictive densities"),
        code(
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "import numpy as np\n"
            "for name in ['recovery','partial','persistence']:\n"
            "    s = fc[name]\n"
            "    ax.hist(s.samples*100, bins=80, density=True, alpha=0.5, color=colors[name], label=name)\n"
            "ax.axvline(hist['at_risk'].iloc[-1]*100, color='k', ls=':', lw=1.5, label='2022 observed')\n"
            "ax.set_xlabel('Forecast low-proficiency rate 2026 (%)'); ax.set_ylabel('density')\n"
            "ax.set_title('Monte-Carlo predictive distributions by scenario'); ax.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **Plausible 2026 range is wide: ~21% (full recovery) to ~74% (persistence).** With five "
            "cycles and a COVID break, that width is the honest message — not a single number.\n"
            "- **Central expectation leans high.** The 2022 spike (74%) reflects durable disruptions "
            "(learning loss, teacher-support drop seen in notebook 03's covariate shift) unlikely to fully "
            "reverse by 2026, so the **partial-reversion (~47%) to persistence (~74%)** band is the more "
            "defensible zone; the ~21% *recovery* scenario assumes the pre-COVID trend resumes untouched "
            "and is best read as an optimistic floor.\n"
            "- **The naive all-cycles line (~73%) is a trap** — it only looks confident (tight CI) because "
            "it ignores the structural break; shown to be discarded, not used.\n"
            "- **Caveats:** the persistence drift SD is a judgment parameter; the recovery interval is "
            "narrow because it propagates only the four pre-COVID SEs, not model-form uncertainty — treat "
            "its precision with suspicion. Forecast is of the **aggregate** rate (no 2026 microdata exists); "
            "it is a planning aid, not a prediction of a break.\n"
            "- **Policy read:** absent a strong, targeted recovery in school resources and teacher support "
            "(the notebook 03 drivers), Albania should plan for a 2026 low-proficiency rate still well "
            "above its 2018 low of 42%."
        ),
    ]


# ===========================================================================
# Notebook 10 — Stacking ensemble (Phase 9, Advanced)
# ===========================================================================

def nb_10_stacking() -> list:
    return [
        md(
            "# 10 — Stacking Ensemble: Can We Beat Single CatBoost? (Albania 2022)\n\n"
            "**Phase 9 (Advanced).** School context lifted the ceiling to ~0.78 AUC and made CatBoost the "
            "best single model. The natural next lever is **ensembling**: does a *stacked* combination of "
            "the base learners extract signal that no single model captures — or, as with the earlier "
            "PV-stacking experiment, is the gain marginal and not worth the complexity?\n\n"
            "We answer it with the same rigor as every other comparison in this project: identical "
            "weighted / leakage-safe CV folds, Nadeau-Bengio corrected intervals, and a **paired** "
            "significance test. Heavy fitting lives in `scripts/run_stacking_experiment.py`; this notebook "
            "loads its results and narrates.\n\n"
            "> **Honesty note.** A statistically significant win that is *practically* nil — or that "
            "regresses the metrics we actually deploy on — is **not** a headline. We report effect size, "
            "not just the p-value."
        ),
        code(HEADER),
        md(
            "## 1. What is stacking, and why this design?\n\n"
            "**Stacked generalization** (Wolpert, 1992) trains a *meta-learner* on the predictions of "
            "several *base learners*, letting it learn where each base model is trustworthy instead of "
            "averaging them blindly.\n\n"
            "**Protocol (leakage-safe).** For base learners $h_1,\\dots,h_M$ and $K$-fold splits of the "
            "training data, each base model produces **out-of-fold** predicted probabilities\n\n"
            "$$\\; z_{i,m} \\;=\\; \\hat p_m^{(-k(i))}(x_i)\\;$$\n\n"
            "where $k(i)$ is the fold containing $i$, so $x_i$ never influences the model that scores it "
            "(no leakage). The meta-learner $g$ is fit on the stacked matrix "
            "$\\{(z_i, y_i)\\}$ with $z_i=(z_{i,1},\\dots,z_{i,M})$. Here $g$ is a class-weighted "
            "**logistic regression**, so the final risk is\n\n"
            "$$\\; P(\\text{at-risk}\\mid x) \\;=\\; \\sigma\\!\\Big(\\beta_0 + \\sum_{m=1}^{M} "
            "\\beta_m\\,\\hat p_m(x)\\Big),\\qquad \\sigma(t)=\\frac{1}{1+e^{-t}} \\;$$\n\n"
            "At inference the base learners are refit on the full training fold, emit $\\hat p_m(x)$, and "
            "$g$ combines them. The $\\beta_m$ reveal which base model the ensemble actually leans on.\n\n"
            "**Three design choices forced by this project's constraints** "
            "(`src/models/registry.build_stacking_ensemble`):\n"
            "- **Base learners = bare tree/boosters** (CatBoost + LightGBM + Gradient Boosting + Random "
            "Forest). `StackingClassifier` forwards the PISA `sample_weight` to each base learner's "
            "`fit`, and the scaler-wrapped models (LR/SVM) reject a *bare* weight kwarg — so the base set "
            "must be estimators that accept `sample_weight` directly. All four do, and they are the four "
            "strongest, most decorrelated school-context models.\n"
            "- **Plain logistic-regression meta-learner.** Its inputs are probabilities already in "
            "$[0,1]$, so no scaling is needed; a class-balanced LR on the OOF probabilities is the "
            "textbook combiner and stays interpretable via its $\\beta_m$.\n"
            "- **Sequential base fits (`n_jobs=1`).** Parallel booster fits re-trigger the macOS "
            "libomp/libgomp duplicate-OpenMP crash this project fought; sequential fits inside the "
            "isolated worker are the safe path."
        ),
        md(
            "## 2. Leaderboard — stacking vs. its own base learners\n\n"
            "All six models were scored on the **same** weighted 5×4 repeated-stratified-CV folds "
            "(school-context feature set), each in a fresh OpenMP-isolated interpreter, with "
            "Nadeau-Bengio corrected 95% CIs (`scripts/run_stacking_experiment.py` → "
            "`stacking_ensemble_2022.csv`)."
        ),
        code(
            "lb = pd.read_csv('../outputs/results/stacking_ensemble_2022.csv')\n"
            "lb[['model','roc_auc_mean','roc_auc_95ci_low','roc_auc_95ci_high',\n"
            "    'pr_auc_mean','f1_macro_mean','mcc_mean']]"
        ),
        code(
            "from src.visualization.style import apply_publication_style, PALETTE\n"
            "apply_publication_style()\n"
            "order = lb.sort_values('roc_auc_mean')\n"
            "# Wong-palette coding: ensemble = purple, best single (CatBoost) = blue, rest = grey.\n"
            "best_single = order[order.model!='stacking'].sort_values('roc_auc_mean').iloc[-1]['model']\n"
            "def _c(m):\n"
            "    if m=='stacking': return PALETTE['purple']\n"
            "    if m==best_single: return PALETTE['blue']\n"
            "    return '#BBBBBB'\n"
            "colors = [_c(m) for m in order['model']]\n"
            "err_lo = order['roc_auc_mean'] - order['roc_auc_95ci_low']\n"
            "err_hi = order['roc_auc_95ci_high'] - order['roc_auc_mean']\n"
            "fig, ax = plt.subplots(figsize=(8,5))\n"
            "ax.barh(order['model'], order['roc_auc_mean'], color=colors,\n"
            "        xerr=[err_lo, err_hi], capsize=3, error_kw={'elinewidth':1, 'ecolor':'0.4'})\n"
            "for i,(m,v) in enumerate(zip(order['model'], order['roc_auc_mean'])):\n"
            "    ax.text(v+0.001, i, f'{v:.3f}', va='center', fontsize=8)\n"
            "ax.set_xlim(0.72, 0.82); ax.set_xlabel('Weighted CV ROC-AUC (Nadeau-Bengio 95% CI)')\n"
            "ax.set_title('Stacking vs. base learners — Albania 2022 (school context)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "**Reading:** stacking (0.786) edges out the best single model, CatBoost (0.783), but the "
            "Nadeau-Bengio confidence intervals **overlap almost entirely** — the eye already suggests the "
            "gap is within noise. The next cell tests it properly, *paired* by fold."
        ),
        md(
            "## 3. Is the gain real? Paired Nadeau-Bengio test\n\n"
            "Because every model shares the same CV splits, fold $j$ is comparable across models and the "
            "per-fold AUC differences $d_j = \\text{AUC}^{\\text{stack}}_j - \\text{AUC}^{\\text{single}}_j$ "
            "can be paired. A naïve paired $t$-test is **anti-conservative** here: CV folds share training "
            "data, so the $d_j$ are positively correlated and their variance is underestimated. The "
            "**Nadeau-Bengio correction** inflates the variance to account for the train/test overlap,\n\n"
            "$$\\; t \\;=\\; \\frac{\\bar d}{\\sqrt{\\big(\\tfrac{1}{J} + \\tfrac{n_{\\text{test}}}"
            "{n_{\\text{train}}}\\big)\\, s_d^2}}, \\qquad \\tfrac{n_{\\text{test}}}{n_{\\text{train}}} "
            "= \\tfrac{1}{K-1}\\;$$\n\n"
            "replacing the naïve $1/J$ variance scaling (`src/models/evaluate.corrected_resampled_ttest`)."
        ),
        code(
            "pw = pd.read_csv('../outputs/results/stacking_pairwise_nb_2022.csv')\n"
            "pw"
        ),
        code(
            "# Effect size vs. statistical significance, side by side.\n"
            "row = pw[pw.model_b==best_single].iloc[0]\n"
            "cat = lb[lb.model==best_single].iloc[0]; stk = lb[lb.model=='stacking'].iloc[0]\n"
            "print(f\"Stacking vs {best_single}:\")\n"
            "print(f\"  AUC   {stk.roc_auc_mean:.4f} vs {cat.roc_auc_mean:.4f}  \"\n"
            "      f\"(delta {row.mean_diff:+.4f}, p={row.p_value:.3f})\")\n"
            "print(f\"  MCC   {stk.mcc_mean:.4f} vs {cat.mcc_mean:.4f}  \"\n"
            "      f\"(delta {stk.mcc_mean-cat.mcc_mean:+.4f})\")\n"
            "print(f\"  F1    {stk.f1_macro_mean:.4f} vs {cat.f1_macro_mean:.4f}  \"\n"
            "      f\"(delta {stk.f1_macro_mean-cat.f1_macro_mean:+.4f})\")"
        ),
        md(
            "**Reading:** the AUC gain is **statistically significant but practically negligible** "
            "(+0.003, p≈0.03) — and, decisively, stacking **regresses the operating-point metrics** "
            "(MCC and macro-F1, the quantities a real at-risk *screener* is scored on) while costing ~19× "
            "the compute of a single CatBoost. A win on the ranking metric that *loses* on the decision "
            "metrics is not a deployment case."
        ),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **The ~0.78 ceiling holds.** Stacking four base learners reaches AUC **0.786** vs single "
            "CatBoost **0.783** — a paired Nadeau-Bengio gain of **+0.003 (p≈0.03)**: real but negligible. "
            "The signal available in these features is essentially saturated; no combination unlocks a new "
            "tier.\n"
            "- **Worse where it counts.** Stacking *lowers* MCC (0.386 vs 0.393) and macro-F1 (0.683 vs "
            "0.693). For a screening tool the operating-point metrics matter more than AUC, so the "
            "ensemble is a net negative there.\n"
            "- **Cost/benefit fails.** ~19× the training time (516s vs 27s) and a far less interpretable, "
            "harder-to-deploy object, for a third of an AUC point that the CIs barely separate.\n"
            "- **Decision: single CatBoost remains the headline model.** Stacking is retained as a "
            "documented **ablation** — evidence that the ceiling is a property of the data, not a modelling "
            "shortfall — mirroring the earlier PV-stacking result (`stacking_ensemble_2022.csv`, "
            "`stacking_pairwise_nb_2022.csv`).\n"
            "- **Where headroom actually lives:** not in fancier learners but in *information* (richer "
            "school/teacher context, longitudinal linkage) and in the **decision layer** — threshold "
            "tuning + calibration (notebook: threshold-calibration) already buy more usable lift (MCC/F1, "
            "ECE) than stacking does. Method choice here is disciplined by effect size, the standard this "
            "whole project holds itself to."
        ),
    ]


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    build_notebook(_with_formulas(nb_01_eda_albania(), "01"), NB_DIR / "01_eda_albania.ipynb", force)
    build_notebook(_with_formulas(nb_02_comparative(), "02"), NB_DIR / "02_eda_comparative.ipynb", force)
    build_notebook(_with_formulas(nb_03_covariate_shift(), "03"), NB_DIR / "03_covariate_shift.ipynb", force)
    build_notebook(_with_formulas(nb_04_modeling(), "04"), NB_DIR / "04_modeling.ipynb", force)
    build_notebook(_with_formulas(nb_05_explainability(), "05"), NB_DIR / "05_explainability.ipynb", force)
    build_notebook(_with_formulas(nb_06_explainability_cases(), "06"), NB_DIR / "06_explainability_cases.ipynb", force)
    build_notebook(_with_formulas(nb_07_fairness(), "07"), NB_DIR / "07_fairness.ipynb", force)
    build_notebook(_with_formulas(nb_08_comparative(), "08"), NB_DIR / "08_comparative.ipynb", force)
    build_notebook(_with_formulas(nb_09_forecast(), "09"), NB_DIR / "09_forecast_2026.ipynb", force)
    build_notebook(nb_10_stacking(), NB_DIR / "10_stacking_ensemble.ipynb", force)
