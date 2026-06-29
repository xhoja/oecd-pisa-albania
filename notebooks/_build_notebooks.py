"""
Generator for the project's Jupyter notebooks.

Notebooks are thin narrative layers over the tested `src/` modules.
Run:  python notebooks/_build_notebooks.py
This writes the .ipynb files; execute them with jupyter or `jupyter nbconvert --execute`.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parent

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
        md("## 2. SES gradient by country (2022)\n\n"
           "Slope of math score on ESCS. A steeper slope = stronger socioeconomic determinism. "
           "Estonia's flatter slope shows it partly breaks the SES–achievement link."),
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
            "print('Shift-detection AUC:', round(res['shift_detection_auc'], 4))\n"
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
    ]


# Header for modeling notebooks: KMP env MUST be set before any OpenMP import.
MODEL_HEADER = """import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # macOS duplicate OpenMP guard
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
            "pd.DataFrame({m: {'test_AUC': d['roc_auc'], 'PR_AUC': d['pr_auc'], 'MCC': d['mcc']}\n"
            "              for m, d in oos['models'].items()}).T.sort_values('test_AUC', ascending=False)"
        ),
        md("**Takeaway:** out-of-sample AUC falls to ~0.64–0.67 for every model. The covariate shift "
           "(notebook 03, detection AUC 0.98) is real and degrades transfer, but no model fully "
           "collapses — the 2022 risk structure shifted yet remains partly predictable from history."),
    ]


# ===========================================================================
# Notebook 05 — Explainability (SHAP)
# ===========================================================================

def nb_05_explainability() -> list:
    return [
        md(
            "# 05 — Explainability: SHAP (Albania 2022)\n\n"
            "Global and local explanations for the LightGBM model on the Albania 2022 cohort. "
            "LightGBM is fit fresh in this kernel (boosters alone are OpenMP-safe), so SHAP runs live."
        ),
        code(MODEL_HEADER),
        code(
            "import shap\n"
            "from src.models.prepare import build_model_data, impute_median\n"
            "from src.models.registry import get_model\n"
            "from src.explainability.shap_analysis import compute_shap_values, global_feature_importance\n\n"
            "df = pd.read_parquet('../data/processed/alb_2022.parquet')\n"
            "feats = ['ESCS','HOMEPOS','GENDER','REPEAT','IMMIG','BELONG','TEACHSUP',\n"
            "         'ICTHOME','ICTSCH','ANXMAT','GRADE','HISCED','HISEI']\n"
            "data = build_model_data(df, feats, domain='math')\n"
            "(X,) = impute_median(data.X); y = data.y.values\n"
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
        md("**Reading:** `HOMEPOS` (home possessions) and `ANXMAT` (math anxiety) dominate, followed by "
           "parental education/occupation (`HISCED`, `HISEI`) and digital readiness. Immigration status "
           "and grade deviation contribute almost nothing."),
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
    ]


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    build_notebook(nb_01_eda_albania(), NB_DIR / "01_eda_albania.ipynb", force)
    build_notebook(nb_02_comparative(), NB_DIR / "02_eda_comparative.ipynb", force)
    build_notebook(nb_03_covariate_shift(), NB_DIR / "03_covariate_shift.ipynb", force)
    build_notebook(nb_04_modeling(), NB_DIR / "04_modeling.ipynb", force)
    build_notebook(nb_05_explainability(), NB_DIR / "05_explainability.ipynb", force)
