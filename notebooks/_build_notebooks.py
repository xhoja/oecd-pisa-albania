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


# --------------------------------------------------------------------------- #
# Notebook merging: several short notebooks are installments of one topic.     #
# `_assemble` stitches their bodies into a single coherent notebook - one      #
# intro + one header, each former notebook a "Part", its `##` sections demoted #
# to `###` so the merged notebook keeps a clean two-level outline.             #
# --------------------------------------------------------------------------- #
def _demote(cells: list) -> list:
    """Copy cells, demoting markdown `## ` headers to `### ` (keeps one H2 per
    Part in the merged notebook). Code cells pass through unchanged."""
    out = []
    for c in cells:
        if c.cell_type == "markdown":
            src = "\n".join(("#" + ln) if ln.startswith("## ") else ln
                            for ln in c.source.splitlines())
            out.append(md(src))
        else:
            out.append(c)
    return out


def _part(letter: str, title: str) -> nbf.NotebookNode:
    return md(f"---\n\n## Part {letter} — {title}")


def _combined_formulas(keys: list[str]) -> nbf.NotebookNode:
    """One Methods-&-formulas cell built from several notebooks' blocks; keeps a
    single header, concatenates the rest."""
    blocks = []
    for i, k in enumerate(keys):
        txt = FORMULAS[k]
        if i > 0:
            lines = txt.splitlines()
            if lines and lines[0].startswith("## Methods"):
                txt = "\n".join(lines[1:])
        blocks.append(txt.rstrip())
    return md("\n\n".join(blocks))


def _assemble(intro: str, header_src: str, formula_keys: list[str],
              parts: list[tuple[str, str, "callable"]]) -> list:
    """Build a merged notebook: intro, combined formulas, one header, then each
    part (former notebook) as a demoted section. Each ``fn`` is an existing
    ``nb_*`` builder; its intro (cell 0) and header (cell 1) are dropped."""
    cells = [md(intro)]
    fk = [k for k in formula_keys if k in FORMULAS]
    if fk:
        cells.append(_combined_formulas(fk))
    cells.append(code(header_src))
    for letter, title, fn in parts:
        cells.append(_part(letter, title))
        cells += _demote(fn()[2:])
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
        print(f"SKIP {path.name} - has outputs (use --force to overwrite, then re-execute)")
        return
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    nbf.write(nb, str(path))
    print(f"wrote {path.name} ({len(cells)} cells) - remember to execute it")


# ===========================================================================
# Notebook 01 - EDA: Albania
# ===========================================================================

def nb_01_eda_albania() -> list:
    return [
        md(
            "# 01 - Exploratory Data Analysis: Albania (PISA 2009–2022)\n\n"
            "**Project:** Predicting Albanian Student Low-Proficiency Risk - A Comparative ML Framework\n\n"
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
        md("## 0. Data dictionary & provenance\n\n"
           "Before any statistic, *what are we looking at?* Each analytic variable, its plain-English "
           "meaning, the underlying PISA source item, the direction of the scale, and the cycles in which "
           "it is available for Albania. The OECD background **indices** (ESCS, HOMEPOS, BELONG, …) are "
           "standardised to an OECD mean 0 / SD 1, so a value of −0.75 means ~0.75 SD *below* the OECD "
           "student average - already an interpretation aid before we compute anything."),
        code(
            "meta = {\n"
            "  'MATH_PV_MEAN': ('Math score (mean of 10 plausible values)', 'PV1-10MATH', 'higher = better'),\n"
            "  'AT_RISK_MATH': ('Low proficiency: math < 420 (below PISA Level 2)', 'derived', '1 = at risk'),\n"
            "  'ESCS':     ('Economic, social & cultural status index', 'ESCS', 'higher = advantaged'),\n"
            "  'HOMEPOS':  ('Home possessions (material + educational)', 'HOMEPOS', 'higher = more'),\n"
            "  'HISCED':   ('Highest parental education (ISCED level)', 'HISCED', 'higher = more'),\n"
            "  'HISEI':    ('Highest parental occupational status (ISEI)', 'HISEI', 'higher = higher status'),\n"
            "  'BELONG':   ('Sense of belonging at school', 'BELONG', 'higher = more belonging'),\n"
            "  'TEACHSUP': ('Perceived teacher support', 'TEACHSUP', 'higher = more support'),\n"
            "  'ANXMAT':   ('Mathematics anxiety', 'ANXMAT', 'higher = more anxious'),\n"
            "  'REPEAT':   ('Has repeated a grade', 'REPEAT', '1 = repeated'),\n"
            "  'IMMIG':    ('Immigrant background', 'IMMIG', '1 = 1st/2nd gen'),\n"
            "  'GRADE':    ('Grade relative to modal grade', 'GRADE', '0 = modal'),\n"
            "  'GENDER':   ('Student gender', 'ST004D01T', 'coded 0/1'),\n"
            "}\n"
            "rows = []\n"
            "for v,(desc,src,direction) in meta.items():\n"
            "    if v not in df.columns: continue\n"
            "    cyc = [int(c) for c in sorted(df.CYCLE.unique())\n"
            "           if df.loc[df.CYCLE==c, v].notna().any()]\n"
            "    rows.append({'variable':v,'meaning':desc,'PISA_source':src,\n"
            "                 'direction':direction,'cycles_available':cyc})\n"
            "pd.set_option('display.max_colwidth', 60)\n"
            "pd.DataFrame(rows).set_index('variable')"
        ),
        md("**Reading:** ESCS is genuinely absent for Albania in **2012 & 2015** (an OECD data gap, not a "
           "processing bug); HOMEPOS covers all five cycles and so is the safer longitudinal SES proxy. "
           "The attitudinal indices (BELONG, TEACHSUP, ANXMAT) arrive only in the cycles that fielded the "
           "relevant questionnaire rotation."),
        md("## 0b. Survey-weighted descriptive summary (2022)\n\n"
           "The population-level 'know your data' table for the crisis cohort: weighted mean, weighted SD, "
           "the raw range, the weighted median and - critically - **how much is missing**. High missingness "
           "on an index is itself a finding (it caps how much that feature can contribute downstream)."),
        code(
            "from src.data.weights import weighted_describe\n"
            "feats = ['MATH_PV_MEAN','ESCS','HOMEPOS','HISCED','HISEI','BELONG',\n"
            "         'TEACHSUP','ANXMAT','GRADE']\n"
            "weighted_describe(df[df.CYCLE==2022], feats)"
        ),
        md("**Reading:** ESCS/HOMEPOS sit well below 0 (Albania is ~0.8 SD below the OECD SES average). "
           "The attitudinal indices carry 30-40% missingness in 2022 - so any model leaning on TEACHSUP or "
           "ANXMAT is really fitting the ~60% who answered, a caveat that recurs in the modeling notebooks."),
        md("## 1. Target trajectory - the V-shaped crisis\n\n"
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
           "values contribute a non-trivial share of the total uncertainty - collapsing them to a single "
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
        md("## 4. Socioeconomic gradient - SES quintile × at-risk\n\n"
           "Weighted low-proficiency rate by HOMEPOS/ESCS quintile across cycles."),
        code(
            "from src.visualization.eda import plot_ses_quintile_heatmap\n"
            "fig = plot_ses_quintile_heatmap(df); plt.show()"
        ),
        md("## 4b. The SES gradient as a probability curve\n\n"
           "The quintile grid bins SES into five steps; here we show the *continuous* gradient - "
           "**P(low proficiency | ESCS)** - as a descriptive weighted-logistic curve, overlaid with the "
           "weighted at-risk rate in each SES decile (markers). Comparing **2018 vs 2022** asks whether the "
           "crisis merely lifted the whole curve or *steepened* it (hit the disadvantaged harder). This is a "
           "single-predictor descriptive gradient, not the multivariable models of notebooks 03/06."),
        code(
            "from src.visualization.eda import plot_ses_logistic_curve\n"
            "fig = plot_ses_logistic_curve(df, cycles=(2018, 2022)); plt.show()"
        ),
        md("**Reading:** the 2022 curve sits far above 2018 at *every* SES level - the crisis raised risk "
           "across the board, not only for the poor. The gradient (slope) persists, so disadvantage still "
           "matters, but a pure 'poverty shock' story would predict a steepening the data only partly shows - "
           "consistent with the feature-drift result below."),
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
        md("## 5c. Which features separate at-risk students? (effect-size ranking)\n\n"
           "The single most decision-relevant table in the EDA: every candidate predictor ranked by the "
           "*strength* of its weighted association with low proficiency, not just its significance (with "
           "n≈6k almost everything is 'significant'). Numeric features use weighted **Cohen's d** "
           "(at-risk vs proficient; sign = direction); categoricals use **Cramer's V** with the chi-square "
           "*p*. This is what a feature-importance plot later has to beat."),
        code(
            "from src.statistics.tests import feature_effect_sizes\n"
            "feature_effect_sizes(\n"
            "    d22,\n"
            "    numeric_features=['ESCS','HOMEPOS','HISCED','HISEI','BELONG','TEACHSUP','ANXMAT','GRADE'],\n"
            "    categorical_features=['GENDER','REPEAT','IMMIG'],\n"
            ")"
        ),
        md("**Reading:** HOMEPOS, ESCS and **math anxiety (ANXMAT)** are the strongest separators "
           "(|d| ≈ 0.5, medium); parental occupation (HISEI) follows. The demographic categoricals - gender, "
           "immigrant status - are *statistically* significant but trivially small (V < 0.1), so risk here is "
           "an SES-and-affect story far more than a demographic one. Immigrant status is not even significant "
           "(p ≈ 0.16), unusual internationally and worth a sentence in the write-up."),
        md("## 6. The 2022 crisis - feature drift (2018 → 2022)\n\n"
           "Standardized mean difference (Cohen's d) of each feature between the 2018 baseline and the 2022 "
           "crisis cohort. This previews the covariate-shift analysis in notebook 02 (Part B)."),
        code(
            "from src.visualization.eda import plot_feature_drift\n"
            "fig = plot_feature_drift(df, ['ESCS','HOMEPOS','BELONG','TEACHSUP','ANXMAT','GRADE','REPEAT','IMMIG'])\n"
            "plt.show()"
        ),
        md("**Takeaway:** Teacher support (TEACHSUP) shows the largest negative drift into 2022, while "
           "home possessions (HOMEPOS) shifts positively - the crisis is not a simple SES story."),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **V-shaped crisis confirmed.** Weighted low-proficiency (math < 420) fell steadily "
            "2009→2018 (**69.3% → 41.9%**), then reversed sharply to **75.3% in 2022** - the worst of "
            "all five cycles and the study's central puzzle.\n"
            "- **Not a pure SES story.** From 2018→2022 `TEACHSUP` shows the largest *negative* drift "
            "while `HOMEPOS` drifts *positively*; a simple 'poorer students' narrative does not fit.\n"
            "- **SES gradient real but partial.** Weighted Cohen's *d* for ESCS (at-risk vs proficient) "
            "≈ **−0.47** (medium); gender × at-risk is significant (χ² *p* < 0.001) but small "
            "(Cramér's V ≈ 0.07).\n"
            "- **Data-availability caveat.** Albania genuinely lacks ESCS in 2012 & 2015 (an OECD gap, "
            "not a bug) - longitudinal SES comparisons must skip those cycles.\n"
            "- **Next:** notebook 02 (Part B) formalises the 2018→2022 change as *covariate shift*."
        ),
    ]


# ===========================================================================
# Notebook 02 - Comparative EDA
# ===========================================================================

def nb_02_comparative() -> list:
    return [
        md(
            "# 02 - Comparative Analysis: Albania vs. Peer Countries (PISA 2022)\n\n"
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
        md("## 0. Sample composition & cross-country comparability (2022)\n\n"
           "Before ranking countries we ask *are these samples comparable?* For each 2022 cohort: the "
           "sample size, the survey-weighted mean math score, the weighted SES level (ESCS), the weighted "
           "low-proficiency rate, and how much ESCS is missing. Big gaps in N or in ESCS coverage would "
           "temper any head-to-head claim - an honest caveat stated up front."),
        code(
            "from src.data.weights import weighted_mean, weighted_proportion\n"
            "d22 = df[df.CYCLE==2022]\n"
            "rows = []\n"
            "for cnt, g in d22.groupby('COUNTRY'):\n"
            "    w = g['W_FSTUWT']\n"
            "    rows.append({'country':cnt, 'N':len(g),\n"
            "                 'mean_score':round(weighted_mean(g['MATH_PV_MEAN'], w),1),\n"
            "                 'mean_ESCS':round(weighted_mean(g['ESCS'], w),3),\n"
            "                 'at_risk_%':round(weighted_proportion(g['AT_RISK_MATH'], w)*100,1),\n"
            "                 'ESCS_missing_%':round(100*g['ESCS'].isna().mean(),1)})\n"
            "pd.DataFrame(rows).set_index('country').sort_values('at_risk_%', ascending=False)"
        ),
        md("**Reading:** the samples are broadly comparable in size and ESCS coverage (missingness ≤9%), so "
           "the ranking that follows is not an artefact of one country being thinly or selectively measured. "
           "Albania has the *highest* low-proficiency rate - yet Colombia and Mexico sit at a **lower** mean "
           "ESCS and still post *lower* risk. SES level alone therefore does not explain Albania's position; "
           "the gradient's *slope*, not just its level, is what §2 examines."),
        md("## 1. Low-proficiency ranking (2022)\n\n"
           "Where does Albania stand against peers?"),
        code(
            "from src.visualization.eda import plot_country_atrisk_ranking\n"
            "fig = plot_country_atrisk_ranking(df, cycle=2022); plt.show()"
        ),
        md("**Reading:** Albania has the highest low-proficiency rate (~75%) - above its Balkan neighbours "
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
        md("**Reading:** Albania's CI sits clearly above the Balkan peers' - the gap is not sampling "
           "noise. Estonia's non-overlapping low CI confirms the extremes are statistically distinct."),
        md("## 2. SES gradient by country (2022)\n\n"
           "Slope of math score on ESCS. A steeper slope = stronger socioeconomic determinism. "
           "**Albania has the *flattest* slope (~16 pts/SD) of all nine countries** - but this is a "
           "floor effect, not equity: nearly everyone scores low regardless of SES. Estonia and "
           "Finland have *steeper* gradients (~38–40) yet far higher overall levels."),
        code(
            "from src.visualization.eda import plot_ses_gradient_by_country\n"
            "fig = plot_ses_gradient_by_country(df, cycle=2022); plt.show()"
        ),
        md("## 3. Longitudinal trajectories - all countries\n\n"
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
            "- **Albania is the negative outlier.** 2022 low-proficiency ≈ **75%** - highest in the "
            "panel, above Balkan peers (N. Macedonia 67%, Montenegro 60%) and GDP-matched economies "
            "(Colombia 73%, Mexico 67%), and ~5× Estonia (**14%**).\n"
            "- **Flattest SES gradient - a floor effect, not equity.** Albania's math-on-ESCS slope "
            "(~16 pts/SD) is the *flattest* of the nine countries; even advantaged students score low "
            "(mean ~369). Estonia/Finland have *steeper* gradients (~38–40) but much higher levels - "
            "SES predicts more there simply because there is more score range to predict.\n"
            "- **Shared but uneven regression.** Several peers dipped in 2022, none as sharply as "
            "Albania - its spike is structurally distinct, not a common regional shock.\n"
            "- **Next:** is Albania's shift merely lower scores, or a change in *who* is at risk? "
            "→ Part B below."
        ),
    ]


# ===========================================================================
# Notebook 03 - Covariate shift / 2022 crisis deep-dive
# ===========================================================================

def nb_03_covariate_shift() -> list:
    return [
        md(
            "# 03 - The 2022 Crisis: Covariate Shift Analysis\n\n"
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
           "*structural* change in students' learning environment - not just lower scores. This is the "
           "mechanism we test against the predictive models in notebooks 03–04 (where tree models trained "
           "on 2009–2018 collapse out-of-sample on 2022)."),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **Structural shift, not just lower scores.** A domain classifier separates 2018 from 2022 "
            "students on background features alone with AUC **well above 0.5** - the *composition* of "
            "risk changed, corroborated independently by the MMD statistic.\n"
            "- **Learning-environment drivers.** `TEACHSUP` and `HOMEPOS` dominate the per-feature drift; "
            "the 2022 cohort differs most in reported teacher support and material resources - not in "
            "immigration or grade.\n"
            "- **Why it matters.** Covariate shift predicts that a model trained on the improving "
            "2009–2018 era should transfer *worse* to 2022 - tested directly in notebook 03.\n"
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
# Notebook 04 - Modeling: comparison + out-of-sample
# ===========================================================================

def nb_04_modeling() -> list:
    return [
        md(
            "# 04 - Predictive Modeling: Albania 2022 + Out-of-Sample\n\n"
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
            "                  'extra_trees','gradient_boosting','svm','mlp']\n"
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
           "strong, interpretable baseline (0.70). SVM (RBF) and a small MLP - added for "
           "completeness (Phase 5b) - sit mid-pack (~0.71), below LR and the boosters: neither a "
           "kernel margin nor a dense net beats gradient boosting on this tabular, weighted, "
           "~6k-row problem. Decision Tree trails badly (0.57)."),
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
        md("## 4. Rigorous PV evaluation - Rubin's rules\n\n"
           "The comparison above uses the PV-mean point target (cheap, standard for model *selection*). "
           "For the headline number we instead fit the winning model **once per plausible value** and "
           "combine the 10 estimates with Rubin's rules - the correct PISA methodology. The reported SE "
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
           "(notebook 02, detection AUC 0.98) is real and degrades transfer, but no model fully "
           "collapses - the 2022 risk structure shifted yet remains partly predictable from history."),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **Best in-sample model:** CatBoost (weighted CV AUC **0.731**), narrowly over Gradient "
            "Boosting (0.727) and - notably - Logistic Regression (0.727), a strong interpretable "
            "baseline. LightGBM (0.715) sits mid-pack; Decision Tree trails (0.568). (LightGBM's "
            "earlier `rc=-11` crash was an OpenMP import-order bug, now fixed, so it is back in the "
            "comparison.)\n"
            "- **PISA-correct headline.** Rubin's-rules per-PV evaluation gives an AUC close to the "
            "point-target CV number with a modest FMI - the ranking is not an artefact of collapsing "
            "the plausible values.\n"
            "- **Out-of-sample degrades but does not collapse.** Train 2009–2018 → test 2022 drops AUC "
            "to ~**0.62–0.67** (GBM/RF best at 0.674). The covariate shift is real and hurts transfer, "
            "yet 2022 risk stays partly predictable from history - a nuance vs. a 'total breakdown' "
            "story.\n"
            "- **Rigor:** metrics are **weighted** throughout (population estimates) with bootstrap/fold "
            "CIs; the decision threshold is tuned on train only."
        ),
    ]


# ===========================================================================
# Notebook 05 - Explainability (SHAP)
# ===========================================================================

def nb_05_explainability() -> list:
    return [
        md(
            "# 05 - Explainability: SHAP (Albania 2022)\n\n"
            "Global and local explanations for the LightGBM model on the Albania 2022 cohort, "
            "trained on the **school-context** feature set (student features + survey-weighted "
            "school-mean ESCS/HOMEPOS/ANXMAT/TEACHSUP + cohort size - the additions that lifted "
            "AUC ~+0.05). LightGBM is fit fresh in this kernel - its Homebrew libomp is imported "
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
            "ax.set_xlabel('Mean |SHAP value|'); ax.set_title('Global Feature Importance - Albania 2022 (LightGBM)')\n"
            "plt.show()"
        ),
        md("**Reading:** the top drivers are now dominated by **school-level context** - "
           "`SCH_MEAN_HOMEPOS` (school mean home possessions) is #1, with `SCH_MEAN_TEACHSUP`, "
           "`SCH_MEAN_ESCS` and `SCH_N` (cohort size) all in the top 7. The strongest *individual* "
           "factor is `ANXMAT` (math anxiety, #2), then `HISCED`/`HOMEPOS`. The model reads a "
           "student's risk more from the socioeconomic composition of their school than from their "
           "own resources - the compositional effect made explicit. Immigration and grade "
           "deviation still contribute almost nothing."),
        md("## 2. SHAP beeswarm - direction of effects"),
        code(
            "X_s = X.sample(min(2000, len(X)), random_state=42).reset_index(drop=True)\n"
            "shap.summary_plot(shap_vals, X_s[names], show=True, plot_size=(8,6))"
        ),
        md("## 3. SHAP dependence - does SES moderate math anxiety?"),
        code(
            "if 'ANXMAT' in names and 'ESCS' in names:\n"
            "    shap.dependence_plot('ANXMAT', shap_vals, X_s[names], interaction_index='ESCS', show=True)"
        ),
        md("**Interpretation:** higher math anxiety pushes predictions toward at-risk; the colouring shows "
           "whether socioeconomic status buffers or amplifies that effect - a policy-relevant interaction."),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **School composition dominates (mean |SHAP|):** `SCH_MEAN_HOMEPOS` > `ANXMAT` > "
            "`SCH_MEAN_TEACHSUP` > `SCH_MEAN_ESCS` > `HISCED` > `HOMEPOS` > `SCH_N`. Four of the "
            "top seven are school-level - the model reads risk more from the *school's* aggregate "
            "resources and support than from the student's own.\n"
            "- **Top individual factor:** `ANXMAT` (math anxiety) is the strongest student-level "
            "driver, followed by parental education / home resources (`HISCED`, `HOMEPOS`).\n"
            "- **Negligible:** immigration status and grade deviation add almost nothing - Albania's "
            "risk is a school-context-and-affect story, not a migration one.\n"
            "- **Policy read:** the highest-leverage levers are *structural* - the socioeconomic "
            "segregation of schools and school-level teacher support - alongside individual math "
            "anxiety. A pure individual-student intervention misses the dominant, compositional "
            "signal. SHAP explains the *model* - associational, not causal.\n"
            "- **Equity caveat:** because school-mean SES is now a top feature, the model partly "
            "encodes *where* a student goes to school; this is exactly the mechanism the fairness "
            "audit (notebook 07) probes for the SES gap.\n"
            "- **Caveat:** explanations are for the single all-data LightGBM fit; per-fold "
            "stability and per-country SHAP comparison come in Phase 8."
        ),
    ]


# ===========================================================================
# Notebook 06 - Local explainability: SHAP case studies + PDP/ICE
# ===========================================================================

def nb_06_explainability_cases() -> list:
    return [
        md(
            "# 06 - Local Explainability: SHAP Case Studies + PDP/ICE (Albania 2022)\n\n"
            "Notebook 05 gave the *global* picture (which features matter overall). This notebook is the "
            "*local* half of the story, mirroring `scripts/run_explainability_cases.py`:\n\n"
            "1. One representative **TP / TN / FP / FN** instance, each with a SHAP **waterfall** - why the "
            "model was confidently right, and confidently wrong.\n"
            "2. **PDP + centered-ICE** curves for the top drivers - the *shape* of each feature's effect and "
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
           "For each quadrant we pick the *most confident* instance - a confidently correct case (TP/TN) and "
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
           "The FP/FN cases expose where those signals mislead - e.g. a student in a low-resource "
           "school who is nonetheless proficient (FP), or a resourced-school student the model misses "
           "(FN) - the individual bucking their school context."),
        md("## 3. Partial dependence + centered ICE\n\n"
           "PDP = mean predicted at-risk probability as one feature varies (all else held). Centered ICE "
           "keeps one grey line per instance (anchored to the grid start) so heterogeneity - the fingerprint "
           "of interactions - is visible where the average curve hides it."),
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
            "same drivers notebook 04 flagged globally - **school-context** features (school-mean "
            "HOMEPOS/TEACHSUP/ESCS) plus individual math anxiety - so the compositional story isn't an "
            "averaging artefact.\n"
            "- **Where the model fails.** FP/FN cases show the failure mode: the model leans on school "
            "context + anxiety, so a student who defies their school's profile (proficient in a "
            "low-resource school, or at-risk in a resourced one) is misclassified. Risk is probabilistic, "
            "not deterministic, at the individual level.\n"
            "- **Effect shapes are monotone but heterogeneous.** PDP curves for the top drivers move risk "
            "smoothly and roughly monotonically; the ICE spread shows the *magnitude* varies by instance - "
            "evidence of interactions (e.g. anxiety's effect moderated by SES).\n"
            "- **Policy read:** the highest-leverage levers (math anxiety, material support) act on most "
            "students but not uniformly - targeting should be individualised, not blanket. Explanations are "
            "**associational** (they explain the model), not causal.\n"
            "- Figures mirror D3 (`D3_shap_local_cases`) and D4 (`D4_pdp_ice`); the case table matches "
            "`shap_local_cases_2022.csv`."
        ),
    ]


# ===========================================================================
# Notebook 07 - Fairness audit
# ===========================================================================

def nb_07_fairness() -> list:
    return [
        md(
            "# 07 - Fairness Audit: Albania 2022 (survey-weighted)\n\n"
            "Would a risk-screening model built on this data treat subgroups equitably? Mirroring "
            "`scripts/run_fairness_audit.py`, we generate **leakage-safe out-of-fold** predictions (engineered "
            "features fit per fold), then audit demographic parity, equal opportunity (TPR), FPR and "
            "calibration across three protected attributes - **GENDER**, **SES quintile** (from weighted "
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
        md("**Reading:** **SES quintile is the least fair axis** - parity gap ~**0.48** and FPR gap "
           "~**0.63**. The model flags almost every bottom-quintile student as at-risk (selection 0.91, "
           "FPR 0.83) while rarely flagging the top (0.43). With school-mean SES now a top feature the "
           "model has doubly learned the SES gradient and would *operationalise* it if deployed. "
           "**Immigrant status** shows a large FPR gap too (~0.59, small/unstable groups). By contrast the "
           "**gender gap shrank** vs. the student-only model (parity 0.18→0.11, FPR 0.14→0.05) - school "
           "context absorbed some of the gender signal."),
        md("## 3. Per-group detail - where the gap lives"),
        code(
            "ses = reports['SES_QUINTILE']\n"
            "pd.DataFrame({'selection_rate': ses.demographic_parity,\n"
            "              'TPR': ses.equal_opportunity,\n"
            "              'FPR': ses.fpr_by_group,\n"
            "              'mean_pred_prob': ses.calibration_by_group}).round(3)"
        ),
        md("## 4. Threshold sweep - can the operating point buy fairness?\n\n"
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
            "across SES quintiles - the bottom quintile is flagged at-risk ~91% of the time (selection "
            "rate 0.91, FPR 0.83) vs. ~43% for the top. Adding school-mean SES as a top feature does **not** "
            "fix this - school SES *is* SES, so the model doubly encodes the gradient and would act on it.\n"
            "- **Gender gap shrank.** Vs. the student-only model the gender gaps dropped (parity 0.18→0.11, "
            "FPR 0.14→0.05) - school context absorbed part of the gender signal. Girls (group 1) still get "
            "a somewhat higher selection rate and FPR.\n"
            "- **Immigrant status:** a large **FPR gap (~0.59)** - immigrant-background students are "
            "over-flagged among the truly proficient - though the immigrant groups are small and the "
            "FPR=1.0 cell is unstable; treat as a flag, not a precise estimate.\n"
            "- **No free threshold.** The sweep shows moving the operating point shifts the gaps around "
            "rather than eliminating them - fairness here is structural (in the risk signal), not a "
            "threshold artefact. Genuine mitigation needs reweighting/constraints or group-aware "
            "calibration, not just a knob.\n"
            "- **Deployment caveat:** a naïvely deployed screener would concentrate false alarms on "
            "low-SES and immigrant-background students, and the school-context features tie a student's "
            "score to *where* they study. Any real use needs an explicit fairness constraint. "
            "Matches `fairness_audit_2022.json` and figure E1."
        ),
    ]


# ===========================================================================
# Notebook 08 - Comparative modeling: Albania vs. peers (Phase 8)
# ===========================================================================

def nb_08_comparative() -> list:
    return [
        md(
            "# 08 - Comparative Modeling: Albania vs. Peers (PISA 2022)\n\n"
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
           "predict (AUC 0.77)** - when three of four students are at-risk there is little contrast to "
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
            "ax.set_title('SHAP importance rank by country - 2022 (1 = top driver)')\n"
            "fig.colorbar(im, ax=ax, label='rank (1 = most important, dark)'); plt.tight_layout(); plt.show()"
        ),
        md("**Reading:** **math anxiety (`ANXMAT`) is universal** - a top-3 driver in almost every "
           "country, including top-performer Estonia. **School composition** (`SCH_MEAN_HOMEPOS`, "
           "`SCH_MEAN_ESCS`) leads in most systems too. Albania's profile "
           "(school-HOMEPOS → anxiety → school-TEACHSUP → school-ESCS) is therefore **not idiosyncratic** - "
           "it shares the region's structure. Country-specific notes: grade repetition (`GRADE`) jumps to "
           "#2 in Colombia; individual `ESCS` is #1 in Finland (stronger individual-SES determinism)."),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **Albania = high prevalence, low separability, flat gradient.** Highest at-risk share "
            "(75%), one of the lowest model AUCs (0.77), and the flattest SES gradient (~17 pts/SD). The "
            "2022 crisis is **broad-based** - depressed across the whole SES distribution, not concentrated "
            "in the poor.\n"
            "- **Drivers are shared, not unique.** Math anxiety is a universal top factor and school "
            "socioeconomic composition dominates almost everywhere; Albania's risk structure mirrors its "
            "Balkan peers rather than standing apart. What sets Albania apart is the *level* and *breadth* "
            "of low proficiency, not a different mechanism.\n"
            "- **Policy read.** Because the gradient is flat and school composition dominates, "
            "SES-targeted transfers alone would miss most of Albania's at-risk students; system-wide levers "
            "(teacher support, math-anxiety reduction, raising the floor across all schools) are needed. "
            "Estonia's steep-but-high profile shows the opposite regime - a useful contrast, not a template "
            "to copy directly.\n"
            "- **Caveat.** Per-country AUCs are not directly comparable as 'model quality' - they partly "
            "reflect each country's at-risk prevalence (separability is hardest near 0/1 base rates). SHAP "
            "explains each *model*, associational not causal. Fits are single all-data LightGBM per "
            "country; figure `F1_shap_rank_matrix`."
        ),
    ]


# ===========================================================================
# Notebook 09 - Scenario forecast for the next PISA cycle (2026)
# ===========================================================================

def nb_09_forecast() -> list:
    return [
        md(
            "# 10 - Forecast: Albania's Low-Proficiency Rate for the Next Cycle (2026)\n\n"
            "After COVID delayed PISA 2021 to 2022, the assessment moved to a **4-year** cadence, so the "
            "next cycle is **2026**. Can we anticipate Albania's weighted "
            "low-proficiency (math < Level 2) rate? With only **five** cycles and a structural break in "
            "2022 (the COVID spike), a single point extrapolation is indefensible - so we forecast with "
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
           "- **persistence** - the 2022 crisis level holds.\n"
           "- **recovery** - the pre-COVID improving trend (2009–2018) resumes (2022 fully reverses).\n"
           "- **partial** - a 50/50 blend (the shock half-reverses).\n"
           "- **naive_linear** *(reference)* - a weighted line through all five cycles; the 2022 break "
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
            "cycles and a COVID break, that width is the honest message - not a single number.\n"
            "- **Central expectation leans high.** The 2022 spike (74%) reflects durable disruptions "
            "(learning loss, teacher-support drop seen in notebook 02's covariate shift) unlikely to fully "
            "reverse by 2026, so the **partial-reversion (~47%) to persistence (~74%)** band is the more "
            "defensible zone; the ~21% *recovery* scenario assumes the pre-COVID trend resumes untouched "
            "and is best read as an optimistic floor.\n"
            "- **The naive all-cycles line (~73%) is a trap** - it only looks confident (tight CI) because "
            "it ignores the structural break; shown to be discarded, not used.\n"
            "- **Caveats:** the persistence drift SD is a judgment parameter; the recovery interval is "
            "narrow because it propagates only the four pre-COVID SEs, not model-form uncertainty - treat "
            "its precision with suspicion. Forecast is of the **aggregate** rate (no 2026 microdata exists); "
            "it is a planning aid, not a prediction of a break.\n"
            "- **Policy read:** absent a strong, targeted recovery in school resources and teacher support "
            "(the notebook 02 drivers), Albania should plan for a 2026 low-proficiency rate still well "
            "above its 2018 low of 42%."
        ),
    ]


# ===========================================================================
# Notebook 10 - Stacking ensemble (Phase 9, Advanced)
# ===========================================================================

def nb_10_stacking() -> list:
    return [
        md(
            "# 10 - Stacking Ensemble: Can We Beat Single CatBoost? (Albania 2022)\n\n"
            "**Phase 9 (Advanced).** School context lifted the ceiling to ~0.78 AUC and made CatBoost the "
            "best single model. The natural next lever is **ensembling**: does a *stacked* combination of "
            "the base learners extract signal that no single model captures - or, as with the earlier "
            "PV-stacking experiment, is the gain marginal and not worth the complexity?\n\n"
            "We answer it with the same rigor as every other comparison in this project: identical "
            "weighted / leakage-safe CV folds, Nadeau-Bengio corrected intervals, and a **paired** "
            "significance test. Heavy fitting lives in `scripts/run_stacking_experiment.py`; this notebook "
            "loads its results and narrates.\n\n"
            "> **Honesty note.** A statistically significant win that is *practically* nil - or that "
            "regresses the metrics we actually deploy on - is **not** a headline. We report effect size, "
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
            "`fit`, and the scaler-wrapped models (LR/SVM) reject a *bare* weight kwarg - so the base set "
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
            "## 2. Leaderboard - stacking vs. its own base learners\n\n"
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
            "ax.set_title('Stacking vs. base learners - Albania 2022 (school context)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "**Reading:** stacking (0.786) edges out the best single model, CatBoost (0.783), but the "
            "Nadeau-Bengio confidence intervals **overlap almost entirely** - the eye already suggests the "
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
            "(+0.003, p≈0.03) - and, decisively, stacking **regresses the operating-point metrics** "
            "(MCC and macro-F1, the quantities a real at-risk *screener* is scored on) while costing ~19× "
            "the compute of a single CatBoost. A win on the ranking metric that *loses* on the decision "
            "metrics is not a deployment case."
        ),
        md(
            "## Conclusions & Interpretation\n\n"
            "- **The ~0.78 ceiling holds.** Stacking four base learners reaches AUC **0.786** vs single "
            "CatBoost **0.783** - a paired Nadeau-Bengio gain of **+0.003 (p≈0.03)**: real but negligible. "
            "The signal available in these features is essentially saturated; no combination unlocks a new "
            "tier.\n"
            "- **Worse where it counts.** Stacking *lowers* MCC (0.386 vs 0.393) and macro-F1 (0.683 vs "
            "0.693). For a screening tool the operating-point metrics matter more than AUC, so the "
            "ensemble is a net negative there.\n"
            "- **Cost/benefit fails.** ~19× the training time (516s vs 27s) and a far less interpretable, "
            "harder-to-deploy object, for a third of an AUC point that the CIs barely separate.\n"
            "- **Decision: single CatBoost remains the headline model.** Stacking is retained as a "
            "documented **ablation** - evidence that the ceiling is a property of the data, not a modelling "
            "shortfall - mirroring the earlier PV-stacking result (`stacking_ensemble_2022.csv`, "
            "`stacking_pairwise_nb_2022.csv`).\n"
            "- **Where headroom actually lives:** not in fancier learners but in *information* (richer "
            "school/teacher context, longitudinal linkage) and in the **decision layer** - threshold "
            "tuning + calibration (notebook: threshold-calibration) already buy more usable lift (MCC/F1, "
            "ECE) than stacking does. Method choice here is disciplined by effect size, the standard this "
            "whole project holds itself to."
        ),
    ]


# ===========================================================================
# Notebook 11 - Screener evaluation (decision curve + calibration) & multilevel model
# ===========================================================================

def nb_11_screener_multilevel() -> list:
    return [
        md(
            "# 11 - Is the Screener Useful, and Is the Ceiling Real? (Albania 2022)\n\n"
            "Two pre-submission strengthenings that answer the two hardest questions "
            "about a ~0.78-AUC model on a **75%-prevalence** problem:\n\n"
            "1. **Does the model add decision value** over the trivial 'screen everyone' / "
            "'screen no-one' policies, and are its probabilities trustworthy? → **decision-curve "
            "analysis + calibration** (`scripts/run_decision_curve.py`).\n"
            "2. **Is the 0.78 ceiling a modelling shortfall or a property of the data?** → a "
            "**multilevel (random-intercept) logistic model**, the statistically-correct spec for "
            "PISA's nested design, which also quantifies how much risk is a *school-level* "
            "phenomenon (`scripts/run_multilevel.py`).\n\n"
            "Both scripts do the heavy fitting (leakage-safe, OpenMP-isolated); this notebook "
            "loads their results and redraws the figures."
        ),
        code(HEADER),

        md("## Part A - Decision-curve analysis: does acting on the model help?\n\n"
           "With ~3 in 4 students at-risk, ROC-AUC is mechanically capped and says little about "
           "*use*. Net benefit asks the operational question directly: at a given risk tolerance "
           "$p_t$, does flagging students the model picks beat flagging everyone (or no-one)?"),
        code(
            "dca = pd.read_csv('../outputs/results/decision_curve_2022.csv')\n"
            "from src.visualization.style import apply_publication_style, PALETTE\n"
            "apply_publication_style()\n"
            "fig, ax = plt.subplots(figsize=(8,5))\n"
            "ax.plot(dca.threshold, dca.net_benefit_model, color=PALETTE['blue'], lw=2, label='Model (raw)')\n"
            "ax.plot(dca.threshold, dca.net_benefit_model_calibrated, color=PALETTE['green'], lw=2, label='Model (isotonic)')\n"
            "ax.plot(dca.threshold, dca.net_benefit_all, color=PALETTE['vermilion'], lw=1.5, ls='--', label='Screen everyone')\n"
            "ax.plot(dca.threshold, dca.net_benefit_none, color='0.4', lw=1.2, ls=':', label='Screen no-one')\n"
            "ax.set_ylim(bottom=min(-0.05, dca.net_benefit_model.min()))\n"
            "ax.set_xlabel('Threshold probability $p_t$ (risk tolerance)'); ax.set_ylabel('Net benefit (weighted)')\n"
            "ax.set_title('Decision-curve analysis - Albania 2022'); ax.legend(fontsize=9); plt.show()\n"
            "better = dca[(dca.net_benefit_model>dca.net_benefit_all)&(dca.net_benefit_model>0)]\n"
            "print(f'Model beats both references over p_t {better.threshold.min():.2f}-{better.threshold.max():.2f}')"
        ),
        md("**Reading:** below the prevalence (~0.75) 'screen everyone' is hard to beat - when "
           "most students are at-risk, indiscriminate screening is already good. The model earns "
           "its keep in the **selective regime** (higher $p_t$, ~0.51–0.95), where it holds net "
           "benefit while 'screen everyone' collapses below zero. A triage tool that must ration "
           "scarce support - the realistic case - operates exactly there."),

        md("## Part B - Calibration: do the probabilities mean what they say?\n\n"
           "Raw boosted-tree scores are typically miscalibrated; a weighted isotonic map (fit "
           "out-of-fold) fixes them. This matters because the decision threshold $p_t$ above is "
           "only meaningful if $\\hat p$ is a real probability."),
        code(
            "raw = pd.read_csv('../outputs/results/calibration_raw_2022.csv')\n"
            "iso = pd.read_csv('../outputs/results/calibration_isotonic_2022.csv')\n"
            "fig, ax = plt.subplots(figsize=(6,6))\n"
            "ax.plot([0,1],[0,1], color='0.5', ls='--', lw=1, label='Perfect calibration')\n"
            "ax.plot(raw.mean_predicted, raw.observed_fraction, '-o', color=PALETTE['blue'], lw=2, label='Raw')\n"
            "ax.plot(iso.mean_predicted, iso.observed_fraction, '-s', color=PALETTE['green'], lw=2, label='Isotonic')\n"
            "ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed at-risk fraction (weighted)')\n"
            "ax.set_title('Calibration - Albania 2022 (weighted, out-of-fold)'); ax.legend(fontsize=9, loc='upper left'); plt.show()"
        ),
        md("**Reading:** the raw model is over-confident (bows above the diagonal); isotonic "
           "recalibration lands it on the diagonal - weighted **ECE ~0.10 → ~0.01**. Calibrated "
           "probabilities are what make the decision-curve thresholds actionable."),

        md("## Part C - Multilevel model: how much risk is a *school* effect?\n\n"
           "A flat logistic treats students as independent; they are not - they are nested in "
           "schools. A random-intercept model gives each school its own baseline log-odds $u_j$ "
           "and reports the **ICC**: the share of risk variance that lives *between* schools."),
        code(
            "summ = pd.read_csv('../outputs/results/multilevel_summary_2022.csv').iloc[0]\n"
            "cv = pd.read_csv('../outputs/results/multilevel_cv_2022.csv')\n"
            "print(f\"ICC (null)        = {summ.icc_null:.3f}  -> ~{summ.icc_null*100:.0f}% of risk variance is BETWEEN schools\")\n"
            "print(f\"ICC (conditional) = {summ.icc_conditional:.3f}  (student features barely reduce it)\")\n"
            "print(f\"ICC (weighted PQL)= {summ.icc_weighted_pql:.3f}  (survey-weighted pseudo-likelihood, scaled wts; PQL mildly attenuates)\")\n"
            "print(f\"\\nCV mean AUC: multilevel {summ.auc_multilevel_mean:.4f} vs school-mean LightGBM {summ.auc_school_lgbm_mean:.4f}\")\n"
            "cv.round(4)"
        ),
        md("**Survey-weighted refinement.** The VB fit above ignores PISA sampling weights. A "
           "survey-weighted **pseudo-likelihood** fit (penalized quasi-likelihood, Schall 1991, "
           "with within-cluster scaled weights - Rabe-Hesketh & Skrondal) gives design-consistent "
           "fixed effects. The two agree closely, so the weighting does **not** change the story "
           "(it only sharpens it) - the odds ratios below are near-identical weighted vs unweighted."),
        code(
            "orc = pd.read_csv('../outputs/results/multilevel_weighted_vs_unweighted_2022.csv')\n"
            "orc = orc[orc.term!='Intercept'][['term','OR_unweighted_vb','OR_weighted_pql']]\n"
            "print('Max |OR_weighted - OR_unweighted| =', (orc.OR_weighted_pql-orc.OR_unweighted_vb).abs().max().round(3))\n"
            "orc.round(3)"
        ),
        md("**Within-school odds ratios** (per 1 SD). Risk-increasing factors in vermilion, "
           "protective in blue; the school baseline is absorbed by the random intercept."),
        code(
            "fe = pd.read_csv('../outputs/results/multilevel_fixed_effects_2022.csv')\n"
            "fe = fe[fe.term!='Intercept'].copy()\n"
            "fe['lo']=np.exp(fe.coef-1.96*fe.sd); fe['hi']=np.exp(fe.coef+1.96*fe.sd)\n"
            "fe = fe.sort_values('odds_ratio'); yp=np.arange(len(fe))\n"
            "colors=[PALETTE['vermilion'] if o>1 else PALETTE['blue'] for o in fe.odds_ratio]\n"
            "fig, ax = plt.subplots(figsize=(7,6))\n"
            "ax.errorbar(fe.odds_ratio, yp, xerr=[fe.odds_ratio-fe.lo, fe.hi-fe.odds_ratio], fmt='none', ecolor='0.5', elinewidth=1, capsize=3, zorder=1)\n"
            "ax.scatter(fe.odds_ratio, yp, color=colors, s=42, zorder=2)\n"
            "ax.axvline(1.0, color='0.4', ls='--', lw=1)\n"
            "ax.set_yticks(yp); ax.set_yticklabels(fe.term)\n"
            "ax.set_xlabel('Odds ratio per 1 SD (within-school, 95% CI)')\n"
            "ax.set_title(f\"Random-intercept logistic - ICC {summ.icc_conditional:.2f}\"); plt.show()"
        ),
        md("**Reading:** math anxiety (`ANXMAT`) is the strongest within-school risk factor; the "
           "SES cluster (`HOMEPOS`, `HISEI`, `ESCS`) is protective. `HISCED` flips positive - a "
           "*partial* effect under SES collinearity (the composite SES protection is already "
           "carried by HOMEPOS/HISEI/ESCS), not a real reversal; read the SES measures jointly."),

        md(
            "## Conclusions & Interpretation\n\n"
            "- **The model has genuine decision value - in the selective regime.** Net benefit "
            "beats both 'screen everyone' and 'screen no-one' for risk thresholds ~0.51–0.95. For "
            "a triage tool rationing scarce support that is the operating range that matters; the "
            "modest AUC is the wrong lens.\n"
            "- **Calibration is essentially fixed.** Weighted isotonic recalibration cuts ECE from "
            "~0.10 to ~0.01, making the probabilities (and hence the thresholds) trustworthy.\n"
            "- **The 0.78 ceiling is real, not a modelling shortfall.** The statistically-correct "
            "multilevel model - student features + a data-driven school random effect - matches "
            "the hand-crafted school-mean booster on identical folds (CV AUC ~0.78 either way). "
            "Two independent model families converge on the same ceiling.\n"
            "- **~31% of risk variance is between schools** (null ICC), barely reduced by student "
            "features (conditional ICC ~0.26). Risk is heavily a *school-level* phenomenon that "
            "student background cannot explain away - which motivated linking **school-level "
            "information** (the PISA school questionnaire: resources, staff, leadership). "
            "**Notebook 12 does that linkage and finds it adds no significant predictive signal** "
            "beyond school composition - so this ~0.78 ceiling is a genuine data limit, not an "
            "un-linked-file artefact.\n"
            "- **Survey weighting does not change the conclusions.** Beyond the unweighted VB fit, "
            "a survey-weighted pseudo-likelihood model (PQL with within-cluster scaled weights) "
            "gives design-consistent fixed effects; the odds ratios are near-identical (max shift "
            "~0.05) and the ICC lands in the same 0.22-0.31 band. The variance-partition and "
            "within-school risk story is robust to how sampling weights are handled."
        ),
    ]


def nb_12_school_questionnaire() -> list:
    return [
        md(
            "# 12 - Is the 0.78 Ceiling Features or Data? The School Questionnaire (Albania 2022)\n\n"
            "Notebook 11 left one lever open. The multilevel model showed **~31% of risk variance "
            "is between schools** (null ICC), barely dented by student features - so the natural "
            "hypothesis was that the ~0.78 AUC ceiling is a *feature-set* limit and the missing "
            "signal lives in **school-level information** we simply had not linked in yet.\n\n"
            "This notebook tests that hypothesis directly. We link the **PISA 2022 school "
            "questionnaire** (`CY08MSP_SCH_QQQ`, principal-reported: resources, staff shortage, "
            "class size, leadership, climate), joined onto students on `CNTSCHID`, and ask whether "
            "these *independent* school variables add predictive signal **beyond** the "
            "compositional `SCH_MEAN_*` features that already gave the 0.73→0.78 lift.\n\n"
            "Crucially these are **exogenous school attributes** (a real, known property of a "
            "student's school at prediction time), not student aggregates - so unlike the "
            "composition features they carry no leave-one-out subtlety and need no fold-safe "
            "recomputation. The heavy fitting is in "
            "`scripts/run_school_questionnaire_experiment.py`; this notebook loads its result.\n\n"
            "> **Distinction that matters.** `SCH_MEAN_ESCS` = *who attends* the school (composition, "
            "built from students). `SCHQ_EDUSHORT` = *what the school has* (inputs, reported by the "
            "principal). If the ceiling were a feature-set limit, the second should add signal the "
            "first cannot."
        ),
        code(HEADER),

        md("## 1. What we linked\n\n"
           "One row per school (274 Albanian schools, a 1:1 match to the sampled students), "
           "13 principal-reported variables. Missingness is low except the computer ratio."),
        code(
            "sch = pd.read_parquet('../data/processed/alb_2022_school.parquet')\n"
            "print(f'{len(sch)} schools x {sch.shape[1]-1} questionnaire vars')\n"
            "miss = (sch.drop(columns=['CNTSCHID']).isna().mean()*100).round(1).sort_values(ascending=False)\n"
            "miss.to_frame('missing_%')"
        ),

        md("## 2. The ablation - does the questionnaire add signal beyond composition?\n\n"
           "Three nested feature sets on **one shared** 5×4 repeated-stratified-CV loop (per-fold "
           "AUCs perfectly paired), weighted metrics, Nadeau-Bengio corrected resampled *t*-test:\n\n"
           "- **base** - 13 student features\n"
           "- **+composition** - base + survey-weighted `SCH_MEAN_*` + cohort size (current headline)\n"
           "- **+questionnaire** - +composition + `SCHQ_*` (the new lever)\n\n"
           "The test that matters is the **incremental** one: questionnaire *over* composition."),
        code(
            "ab = pd.read_csv('../outputs/results/school_questionnaire_ablation_2022.csv')\n"
            "ab.round(4)"
        ),
        code(
            "from src.visualization.style import apply_publication_style, PALETTE\n"
            "apply_publication_style()\n"
            "d = ab.sort_values('auc_questionnaire')\n"
            "yp = np.arange(len(d))\n"
            "fig, ax = plt.subplots(figsize=(8,5))\n"
            "ax.scatter(d.auc_base, yp, color='0.6', s=48, label='base (student-only)', zorder=3)\n"
            "ax.scatter(d.auc_composition, yp, color=PALETTE['blue'], s=48, label='+ composition (SCH_MEAN_*)', zorder=3)\n"
            "ax.scatter(d.auc_questionnaire, yp, color=PALETTE['green'], s=64, marker='D', label='+ questionnaire (SCHQ_*)', zorder=3)\n"
            "for i,(b,c,q) in enumerate(zip(d.auc_base,d.auc_composition,d.auc_questionnaire)):\n"
            "    ax.plot([b,q],[i,i], color='0.8', lw=1, zorder=1)\n"
            "ax.axvline(0.78, color=PALETTE['vermilion'], ls='--', lw=1, label='~0.78 composition ceiling')\n"
            "ax.set_yticks(yp); ax.set_yticklabels(d.model)\n"
            "ax.set_xlabel('Weighted ROC-AUC (5x4 repeated stratified CV)')\n"
            "ax.set_title('School questionnaire adds no signal beyond composition - Albania 2022')\n"
            "ax.legend(fontsize=8, loc='lower right'); plt.show()"
        ),
        md("**Reading:** the green diamonds (questionnaire) sit essentially on top of the blue "
           "dots (composition). The incremental lift is tiny - CatBoost **+0.004**, all models "
           "**< +0.005** - and **not significant for any model** (`p_q` all > 0.24). The big jump "
           "is base→composition (+0.05, highly significant); questionnaire→composition is noise."),

        md("## 3. Why composition already captures the school effect\n\n"
           "The compositional features and the questionnaire inputs are correlated - a school's "
           "intake (`SCH_MEAN_ESCS`) tracks its resources and staffing - so once composition is in "
           "the model, the questionnaire is largely redundant. The between-school variance the "
           "multilevel ICC flagged is real, but for *prediction* it is already spanned by who "
           "attends the school; the measured inputs add little on top."),
        code(
            "import numpy as np\n"
            "stu = pd.read_parquet('../data/processed/alb_2022.parquet')\n"
            "from src.features.engineer import add_school_aggregates, add_school_questionnaire\n"
            "tmp = add_school_aggregates(stu, cols=['ESCS'])\n"
            "tmp = add_school_questionnaire(tmp, sch, cols=['EDUSHORT','STAFFSHORT','STRATIO','PROATCE'])\n"
            "cc = tmp[['SCH_MEAN_ESCS','SCHQ_EDUSHORT','SCHQ_STAFFSHORT','SCHQ_STRATIO','SCHQ_PROATCE']].corr()\n"
            "cc['SCH_MEAN_ESCS'].round(3).to_frame('corr_with_SCH_MEAN_ESCS')"
        ),
        md("**Reading:** school-mean ESCS is materially correlated with the resource/staff "
           "measures - the composition feature is already a proxy for school inputs, which is why "
           "adding the raw inputs is redundant for prediction."),

        md(
            "## Conclusions & Interpretation\n\n"
            "- **The 0.78 ceiling is a genuine data limit, not a feature-set limit.** Linking the "
            "full PISA school questionnaire - the exact 'missing school-level information' Part A "
            "pointed to - adds **no significant predictive signal** beyond the compositional "
            "school-mean features (best incremental +0.004 AUC, all *p* > 0.24). The deferred "
            "data-linkage lever is now **tested and closed**.\n"
            "- **Composition subsumes inputs.** School-mean ESCS is correlated with staffing, "
            "resources and student-teacher ratio, so the compositional features already carry the "
            "school-level signal that matters for predicting individual low-proficiency risk.\n"
            "- **This strengthens the headline story rather than weakening it.** Three independent "
            "routes - a school-mean booster, a random-intercept multilevel model (Part A), and "
            "now a direct school-questionnaire linkage - converge on the same ~0.78 ceiling. The "
            "limit is the information questionnaire data carries about a 15-year-old's math risk, "
            "not the model or an un-linked file.\n"
            "- **Policy reading.** Since composition (who attends), not measured inputs (what the "
            "school has), is what predicts risk, interventions targeting *concentration of "
            "disadvantage* have more leverage here than uniform input top-ups - consistent with the "
            "compositional-effect literature.\n"
            "- **Caveat.** This is a within-Albania predictive test; a school input can matter "
            "causally yet add no *marginal predictive* signal once composition is known. The "
            "questionnaire remains valuable for explanation and cross-country contrast (a natural "
            "extension), just not for lifting this AUC."
        ),
    ]


def nb_13_decision_support() -> list:
    return [
        md(
            "# 08 - Decision Support: Uncertainty, Recourse, and Policy What-Ifs (Albania 2022)\n\n"
            "The screener is deployable and its ceiling is understood (notebook 06). "
            "This notebook adds three tools that turn a probability into something a "
            "policymaker can act on, each loading a pre-computed script result:\n\n"
            "1. **Conformal prediction** (`scripts/run_conformal.py`) - prediction *sets* with a "
            "finite-sample coverage guarantee: when is the model confident vs abstaining?\n"
            "2. **Counterfactual recourse** (`scripts/run_counterfactuals.py`) - the smallest "
            "realistic change to *actionable* levers that would un-flag an at-risk student.\n"
            "3. **Policy simulation** (`scripts/run_policy.py`) - population what-ifs: how far "
            "does shifting a lever nationally move the predicted at-risk rate?\n\n"
            "All three share the headline CatBoost school-context model and PISA survey weights."
        ),
        code(HEADER),

        md("## Part A - Conformal prediction sets: confident or abstaining?\n\n"
           "At risk level $\\alpha$ the set covers the true label with probability $\\ge 1-\\alpha$ "
           "(distribution-free, under exchangeability). A **singleton** set is a confident call; "
           "the **both-labels** set is the model declining to commit. Marginal and Mondrian "
           "(class-conditional) calibration, on a held-out test split, survey-weighted."),
        code(
            "conf = pd.read_csv('../outputs/results/conformal_2022.csv')\n"
            "conf.round(4)"
        ),
        code(
            "from src.visualization.style import apply_publication_style, PALETTE\n"
            "apply_publication_style()\n"
            "m = conf[~conf.mondrian]\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11,4.2))\n"
            "ax[0].plot(m.target_coverage, m.coverage, '-o', color=PALETTE['blue'], label='empirical')\n"
            "ax[0].plot([0.75,0.96],[0.75,0.96], ls='--', color='0.5', label='target')\n"
            "ax[0].set_xlabel('Target coverage (1 - alpha)'); ax[0].set_ylabel('Empirical (weighted) coverage')\n"
            "ax[0].set_title('Coverage guarantee holds'); ax[0].legend(fontsize=8)\n"
            "ax[1].bar(range(len(m)), m.share_singleton, color=PALETTE['green'], label='singleton (confident)')\n"
            "ax[1].bar(range(len(m)), m.share_ambiguous, bottom=m.share_singleton, color=PALETTE['vermilion'], label='both labels (abstain)')\n"
            "ax[1].set_xticks(range(len(m))); ax[1].set_xticklabels([f'{a:.2f}' for a in m.alpha])\n"
            "ax[1].set_xlabel('alpha'); ax[1].set_ylabel('share of students'); ax[1].set_title('Set-size mix'); ax[1].legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("**Reading:** empirical coverage sits at or above the target at every level - the "
           "guarantee holds under survey weighting. At the 90% level the model gives a **confident "
           "singleton for ~60%** of students and **abstains (both labels) on ~40%** - an honest "
           "'refer for human review' flag for the students where a 75%-prevalence screener is "
           "genuinely unsure. Mondrian calibration abstains more but balances coverage across the "
           "rare not-at-risk class."),

        md("## Part B - Counterfactual recourse: what would un-flag a student?\n\n"
           "For each flagged student we search for the smallest change to *actionable* levers "
           "(anxiety down; teacher support, belonging, ICT up) that pulls predicted risk below the "
           "operating threshold. Immutable attributes (gender, immigrant status, parental "
           "education, past repetition) are held fixed."),
        code(
            "cf = pd.read_csv('../outputs/results/counterfactuals_2022.csv')\n"
            "cfs = pd.read_csv('../outputs/results/counterfactuals_summary_2022.csv').iloc[0]\n"
            "print(f\"Recourse found for {cfs.recourse_rate*100:.0f}% of flagged students \"\n"
            "      f\"(median {cfs.median_features_changed:.0f} levers, {cfs.median_cost_sd:.2f} SD total)\")\n"
            "succ = cf[cf.success]\n"
            "from collections import Counter\n"
            "lever = Counter()\n"
            "for s in succ.recourse.dropna():\n"
            "    for part in str(s).split(';'):\n"
            "        tok = part.strip().split(' ')\n"
            "        if tok and tok[0] in ('ANXMAT','TEACHSUP','BELONG','ICTHOME','ICTSCH'): lever[tok[0]] += 1\n"
            "lev = pd.Series(dict(lever)).sort_values()\n"
            "fig, ax = plt.subplots(figsize=(7,3.6))\n"
            "ax.barh(lev.index, lev.values, color=PALETTE['blue'])\n"
            "ax.set_xlabel('times the lever appears in a successful recourse'); ax.set_title('Which levers un-flag students')\n"
            "plt.tight_layout(); plt.show()\n"
            "succ[['student_idx','p_original','p_counterfactual','recourse']].head(8)"
        ),
        md("**Reading:** recourse via soft levers exists for only a **minority** of flagged "
           "students - the rest are too deep in structural (SES / school) risk to flip by lowering "
           "anxiety or raising support within a plausible +-3 SD budget. Where recourse *does* "
           "exist, **math anxiety is the dominant lever**, consistent with the multilevel odds "
           "ratios. This is the individual-level echo of the project's thesis: the crisis is "
           "structural, not a fine-tuning problem."),

        md("## Part C - Policy simulation: national what-ifs\n\n"
           "Shift a lever across the whole population and re-read the weighted predicted at-risk "
           "rate. A *predictive* what-if under the model's associations, **not** a causal effect."),
        code(
            "pol = pd.read_csv('../outputs/results/policy_simulation_2022.csv')\n"
            "pol"
        ),
        code(
            "d = pol.sort_values('delta_at_risk_rate')\n"
            "colors = [PALETTE['blue'] if v < 0 else PALETTE['vermilion'] for v in d.delta_at_risk_rate]\n"
            "fig, ax = plt.subplots(figsize=(8,4.6))\n"
            "ax.barh(d.scenario, d.delta_at_risk_rate*100, color=colors)\n"
            "ax.axvline(0, color='0.4', lw=1)\n"
            "ax.set_xlabel('Change in predicted at-risk rate (percentage points)')\n"
            "ax.set_title('Policy what-ifs - Albania 2022 (baseline ~72% at risk)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md("**Reading:** even the **everything-at-once** bundle (lower anxiety + more support + "
           "belonging + higher SES + lifted school composition) only cuts the predicted at-risk "
           "rate from ~72% to ~60% - a real but bounded ~12 pp. **Math anxiety is the "
           "highest-leverage single factor** (~-7 pp at -1 SD); SES and school-composition shifts "
           "help modestly; belonging moves it slightly the *wrong* way, a non-causal model "
           "association (belonging co-varies with unobserved risk here), which is exactly why the "
           "causal caveat matters. The blunt message for policy: no soft lever rescues a "
           "75%-prevalence crisis - the gap is structural and needs system-level change."),

        md("## Part D - Malleable-lever ranking: of the things we can change, which matters?\n\n"
           "SHAP ranks features by how much the model *reads* them, but it mixes fixed endowments "
           "(SES, parental education, home possessions, immigrant status, gender, grade - a "
           "screener reads them but no policy moves them short-run) with malleable levers (math "
           "anxiety, teacher support, belonging, grade-repetition, ICT). Here we split the two and "
           "rank only the **actionable** levers by the population at-risk reduction from a "
           "standardized half-SD nudge in the beneficial direction, discovered on the smooth mean "
           "predicted probability (`src/models/levers.py`, `scripts/run_levers.py`)."),
        code(
            "lev = pd.read_csv('../outputs/results/lever_ranking_2022.csv')\n"
            "ceil = pd.read_csv('../outputs/results/lever_ceiling_2022.csv')\n"
            "base = float(ceil['baseline_at_risk_rate'].iloc[0])\n"
            "r = lev.sort_values('delta_at_risk_rate', ascending=True).copy()\n"
            "r['reduction_pp'] = -r['delta_at_risk_rate']*100\n"
            "lo = -r['ci_high']*100; hi = -r['ci_low']*100\n"
            "xerr = [(r['reduction_pp']-lo).values, (hi-r['reduction_pp']).values]\n"
            "fig, ax = plt.subplots(figsize=(8,4.2))\n"
            "ax.barh(r['description'], r['reduction_pp'], color=PALETTE['blue'], edgecolor='white',\n"
            "        xerr=xerr, error_kw=dict(ecolor='0.35', capsize=3, lw=1.2))\n"
            "for y,(v,h) in enumerate(zip(r['reduction_pp'], hi.values)):\n"
            "    ax.annotate(f'{v:+.1f} pp', xy=(max(v,h,0.0),y), xytext=(7,0),\n"
            "                textcoords='offset points', va='center', ha='left', fontsize=9)\n"
            "ax.axvline(0, color='0.6', lw=1)\n"
            "ax.set_xlim(min(lo.min()-0.3,-0.5), r['reduction_pp'].max()+1.2)\n"
            "ax.set_xlabel('Reduction in predicted at-risk rate (pp), 0.5-SD nudge (95% bootstrap CI)')\n"
            "ax.set_title(f'Actionable levers, ranked by reach (baseline {base*100:.0f}%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "lev[['lever','description','direction','delta_at_risk_rate','ci_low','ci_high']]"
        ),
        md("**Reading:** among the levers a school system can actually pull, **math anxiety "
           "dominates** (a half-SD reduction cuts the predicted at-risk rate ~5.1 pp, 95% "
           "bootstrap CI [4.5, 5.7]), with teacher support a distant second (~1.0 pp, [0.6, 1.5]). "
           "These are the only two levers whose interval excludes zero; belonging / ICT / "
           "grade-repetition are flat. The 95% intervals come from a population bootstrap (400 "
           "resamples of students, model held fixed), so they express finite-cohort sampling "
           "uncertainty, not model-fit uncertainty. The ranking matches the counterfactual-recourse "
           "and multilevel evidence: anxiety is the one soft lever with real reach."),
        code(
            "print(ceil.to_string(index=False))\n"
            "geo = pd.read_csv('../outputs/results/lever_ranking_by_geography_2022.csv')\n"
            "geo[geo.is_top][['group','n','lever','delta_at_risk_rate']]"
        ),
        md("**The structural floor, made explicit.** Moving *every* actionable lever half an SD at "
           "once takes the predicted at-risk rate from ~72% to ~66% (~7 pp) - and a *hypothetical* "
           "that lifts every fixed endowment by the same amount (which no short-run policy can "
           "deliver) buys only ~5 pp more. Both bundles leave a large majority at risk: the crisis "
           "is structural, not a tuning problem, the same conclusion the policy what-ifs, the "
           "covariate shift and the flat SES gradient reach from other directions. The anxiety "
           "lever is also **weaker where the crisis is deepest** - its reduction is far smaller in "
           "the rural band than the urban one - so lever-based remediation and the geographic "
           "targeting of notebook 02 Part C are complements: anxiety programmes help most in urban "
           "schools, while the rural/North floor needs system-level investment."),

        md(
            "## Conclusions & Interpretation\n\n"
            "- **Conformal sets add an honest abstention signal.** ~60% confident singletons, ~40% "
            "'refer for review' at 90% coverage - better than forcing a call on every student.\n"
            "- **Recourse exists for a minority, and it is mostly about anxiety.** The tool is "
            "useful for the students it *can* help, and its low hit-rate is itself a finding: most "
            "risk is not individually actionable through soft levers.\n"
            "- **Policy simulation and the lever ranking quantify the ceiling on interventions.** "
            "Aggressive population shifts buy ~7-12 pp; math anxiety is the single highest-reach "
            "actionable lever, but the structural core remains and is weakest to move where the "
            "crisis is deepest.\n"
            "- **All are model-associational, not causal.** They stress-test and communicate "
            "the fitted model; they are decision *support*, not effect estimates - read with the "
            "multilevel and school-questionnaire caveats."
        ),
    ]


def nb_14_causal_earthquake() -> list:
    return [
        md(
            "# 14 - A Natural Experiment That Does Not Hold: the 2019 Durres Earthquake\n\n"
            "Every result so far is *associational*. This notebook tests whether the one plausible "
            "**causal** design in reach - the 26 November 2019 M6.4 Durres earthquake as a natural "
            "experiment - can upgrade a slice of the paper. PISA's sampling `STRATUM` encodes a "
            "North / Center / South band; quake damage concentrated in the **Center** band (Durres + "
            "Tirana). A difference-in-differences compares the change in the at-risk rate in Center "
            "(treated) vs North+South (control) from pre-quake **2018** to post-quake **2022**.\n\n"
            "The honest result: **the design fails its own diagnostics.** The naive DiD looks "
            "significant, but the pre-quake placebo rejects parallel trends, the urbanicity "
            "triple-difference is ~0, and the effect is null under school-clustered inference. We "
            "therefore report a *well-identified null* - the 2022 collapse is national, not a "
            "localised shock - which strengthens the paper's structural-crisis reading. Results are "
            "pre-computed by `scripts/run_causal_did.py` (design-based BRR + Rubin SEs)."
        ),
        code(HEADER),
        code("import json\n"
             "from src.visualization.style import apply_publication_style, PALETTE\n"
             "apply_publication_style()\n"
             "es = pd.read_csv('../outputs/results/causal_event_study_2015_2022.csv')\n"
             "summ = json.load(open('../outputs/results/causal_did_summary.json'))\n"
             "lookup = pd.read_csv('../outputs/results/stratum_region_lookup.csv')"),

        md("## 0. Region encoding from the sampling stratum\n\n"
           "Albania has no subnational `REGION` in the public file, but the explicit sampling "
           "stratum labels every school `<Urbanicity> / <Region band> / <Sector>`. The band is "
           "stable across 2015/2018/2022 even though the numeric codes change; we parse the label "
           "text. Center = quake-affected treated group."),
        code("lookup[lookup.region.notna()][['CYCLE','STRATUM','region','urbanicity','sector']]"
             ".sort_values(['CYCLE','region']).head(12)"),

        md("## 1. Event study: at-risk rate by region band\n\n"
           "The parallel-trends assumption is visual here: do the bands move together *before* the "
           "quake (2015 to 2018)? Bars are design-based 95% CIs (BRR + plausible values)."),
        code("fig, ax = plt.subplots(figsize=(7.5,4.6))\n"
             "cmap = {'Center':PALETTE['vermilion'],'North':PALETTE['blue'],'South':PALETTE['green']}\n"
             "for band, s in es.groupby('region'):\n"
             "    s = s.sort_values('CYCLE')\n"
             "    lbl = band + (' (treated)' if band=='Center' else '')\n"
             "    ax.errorbar(s.CYCLE, s.at_risk, yerr=1.96*s.se, marker='o', capsize=3,\n"
             "                color=cmap[band], label=lbl, ls='--' if band=='Center' else '-')\n"
             "ax.axvspan(2019.9, 2022, color='0.5', alpha=0.08)\n"
             "ax.axvline(2019.9, color='0.5', ls=':', lw=1)\n"
             "ax.text(2020.05, 0.44, 'Nov 2019\\nM6.4 quake', fontsize=8, color='0.4')\n"
             "ax.set_xticks([2015,2018,2022]); ax.set_xlabel('PISA cycle')\n"
             "ax.set_ylabel('Share below Level 2 (math)')\n"
             "ax.set_title('Event study by region band (design-based 95% CI)')\n"
             "ax.legend(frameon=False); plt.tight_layout(); plt.show()\n"
             "es.round(3)"),
        md("**Reading:** the bands do **not** move in parallel before the quake. From 2015 to 2018 "
           "the Center band improved faster (fell ~15 pp) than North (~7 pp) or South (~8 pp) - it "
           "was already on a steeper trajectory. Then from 2018 to 2022 *every* band explodes "
           "upward by ~30-40 pp. The pre-quake divergence is the first warning that a DiD here will "
           "not cleanly isolate the earthquake."),

        md("## 2. Difference-in-differences and its placebo\n\n"
           "$\\text{DiD} = (Y^{\\text{Center}}_{\\text{post}}-Y^{\\text{Center}}_{\\text{pre}}) - "
           "(Y^{\\text{Control}}_{\\text{post}}-Y^{\\text{Control}}_{\\text{pre}})$. The placebo runs "
           "the identical estimator on the pre-quake window 2015 to 2018, where the true effect is "
           "zero by construction - a non-zero placebo means parallel trends is violated."),
        code("d, pl = summ['did_main_2018_2022'], summ['did_placebo_2015_2018']\n"
             "tab = pd.DataFrame({\n"
             "  'window':['main 2018->2022','placebo 2015->2018'],\n"
             "  'treated_change':[d['treated_change'], pl['treated_change']],\n"
             "  'control_change':[d['control_change'], pl['control_change']],\n"
             "  'DiD':[d['did'], pl['did']], 'se':[d['se'], pl['se']],\n"
             "  '95%CI':[f\"[{d['ci95_low']:.3f}, {d['ci95_high']:.3f}]\",\n"
             "           f\"[{pl['ci95_low']:.3f}, {pl['ci95_high']:.3f}]\"],\n"
             "  'p':[d['p_value'], pl['p_value']]})\n"
             "tab.round(4)"),
        md("**Reading:** the naive DiD is **+4.7 pp** (p < 0.001, design-based) - superficially a "
           "localised quake effect. But the **placebo is -7.0 pp (p < 0.001)**: in the pre-quake "
           "window the Center band was already diverging from control by *more* than the main "
           "estimate, in the opposite direction. Parallel trends is decisively rejected, so the "
           "main DiD is not a credible causal estimate - it is mostly the continuation of a "
           "pre-existing capital-region trend."),

        md("## 3. Triple-difference: netting out the capital-region trend\n\n"
           "The quake damage was concentrated in the **urban** cores of the Center band (Durres and "
           "Tirana cities). A real quake effect should widen the urban-minus-rural at-risk gap in "
           "Center more than in control. Differencing on urbanicity removes any Center-wide secular "
           "trend - the confound that broke the placebo."),
        code("ddd, dpl = summ['ddd_main_2018_2022'], summ['ddd_placebo_2015_2018']\n"
             "reg = summ['did_regression_clustered_2018_2022']\n"
             "pd.DataFrame({\n"
             "  'estimate':['DDD main 2018->2022','DDD placebo 2015->2018','DiD school-clustered'],\n"
             "  'value':[ddd['ddd'], dpl['ddd'], reg['did']],\n"
             "  'se':[ddd['se'], dpl['se'], reg['se']],\n"
             "  'p':[ddd['p_value'], dpl['p_value'], reg['p_value']]}).round(4)"),
        md("**Reading:** the triple-difference is **~-3 pp and not significant (p ~ 0.37)** - and "
           "the wrong sign for a quake story (urban Center did not worsen *more* than its rural "
           "counterpart). Once the capital-wide trend is differenced out, the apparent effect "
           "disappears. The school-clustered regression agrees: **+5 pp but p ~ 0.22**, null. The "
           "tight design-based SE on the naive DiD was an artefact of ignoring school-level "
           "clustering and the pre-trend."),

        md(
            "## Conclusions & Interpretation\n\n"
            "- **The natural experiment does not identify a causal earthquake effect.** A plausible "
            "DiD is rejected by its own placebo (parallel trends fail), collapses under an "
            "urbanicity triple-difference, and is null under school-clustered inference.\n"
            "- **The 2022 collapse is national, not localised.** At-risk jumps ~30-40 pp in *every* "
            "band and urbanicity cell - consistent with a country-wide COVID-plus-sample-coverage "
            "shock, not a Durres-region disaster signal.\n"
            "- **Why report a null?** It is the honest, well-identified answer, and it forecloses "
            "the natural over-claim that the earthquake caused the crisis. It reinforces the "
            "paper's thesis that the 2022 crisis is structural and national.\n"
            "- **Data ceiling.** The public file cannot separate Durres (epicentre) from Tirana, and "
            "only two pre-periods (2015, 2018) carry a decodable band - so no estimator can rescue "
            "identification here. Reported for transparency, not buried."
        ),
    ]


def nb_18_coverage_bounds() -> list:
    return [
        md(
            "# 18 - How Robust Is the Crisis to Coverage? (Manski bounds, 2022)\n\n"
            "Every rate in this project is computed among the students PISA *sampled* - but PISA "
            "2022 covered only about **79%** of Albania's 15-year-olds (Coverage Index 3 ~ 0.79; "
            "6 129 students representing ~28 400 of ~36 000, OECD Country Note). The ~21% uncovered "
            "- disproportionately out-of-school, disadvantaged youth - are unobserved. Rather than "
            "assume they resemble the sample, this notebook reports the **range of population "
            "at-risk rates consistent with the data** (Manski partial identification, "
            "`src/coverage/manski.py`, `scripts/run_coverage_bounds.py`)."
        ),
        code(HEADER),
        code("import json\n"
             "from src.visualization.style import apply_publication_style, PALETTE\n"
             "apply_publication_style()\n"
             "S = json.load(open('../outputs/results/coverage_bounds_2022.json'))\n"
             "sens = pd.read_csv('../outputs/results/coverage_sensitivity_2022.csv')\n"
             "S"),

        md("## 1. The bounds\n\n"
           "With coverage $c$, observed covered rate $p$, and unknown uncovered rate $u\\in[0,1]$, "
           "the population rate is $P=cp+(1-c)u$. Worst-case takes $u\\in[0,1]$; the monotone bound "
           "uses $u\\ge p$ (out-of-school youth are, if anything, more at risk)."),
        code("p=S['observed_at_risk_covered']; se=S['observed_se_design_based']\n"
             "wc=S['worst_case_bounds']; mono=S['monotone_bounds_uncovered_worse']; sc=S['scenarios']\n"
             "fig, ax = plt.subplots(figsize=(8,4.2))\n"
             "ax.axvspan(wc['lower'], wc['upper'], color=PALETTE['blue'], alpha=0.12, label='worst-case range')\n"
             "ax.axvspan(mono['lower'], mono['upper'], color=PALETTE['blue'], alpha=0.28, label='monotone (uncovered >= covered)')\n"
             "ax.errorbar([p],[1.0], xerr=[[1.96*se],[1.96*se]], fmt='o', color=PALETTE['black'], capsize=4, label='observed (covered) +/-95% CI')\n"
             "ax.scatter([sc['uncovered_like_poorest_quintile']],[1.0], marker='D', s=55, color=PALETTE['vermilion'], zorder=5)\n"
             "ax.annotate('uncovered ~ poorest quintile', (sc['uncovered_like_poorest_quintile'],1.0), xytext=(0,12), textcoords='offset points', ha='center', fontsize=8)\n"
             "ax.axvline(0.419, ls=':', color='0.5'); ax.annotate('2018 rate 0.42', (0.419,1.0), xytext=(0,-20), textcoords='offset points', ha='center', fontsize=8, color='0.4')\n"
             "ax.set_yticks([]); ax.set_ylim(0.8,1.35); ax.set_xlim(0.38,0.85)\n"
             "ax.set_xlabel('population at-risk rate (below Level 2, math)')\n"
             "ax.set_title('Manski coverage bounds: Albania 2022 (79% coverage)')\n"
             "ax.legend(loc='lower left', fontsize=8); plt.tight_layout(); plt.show()"),
        md("**Reading:** the observed covered rate is **0.74** (design-based SE 0.004 - sampling "
           "noise is tiny next to the coverage gap). Without any assumption the population rate is "
           "only pinned to **[0.58, 0.79]**. But the credible direction is one-sided: out-of-school "
           "15-year-olds are not *less* at risk than enrolled ones, so under the **monotone** "
           "assumption the rate is **[0.74, 0.79]** - coverage, if anything, means the headline "
           "*understates* the crisis. Anchoring the uncovered to the poorest covered quintile "
           "(reweighting) puts it at **~0.76**."),

        md("## 2. The crisis conclusion survives any coverage assumption\n\n"
           "The important robustness check: is 'Albania deteriorated sharply from 2018' an artefact "
           "of who got sampled?"),
        code("print(f\"Worst-case LOWER bound on 2022 rate: {wc['lower']:.3f}\")\n"
             "print(f\"Albania 2018 at-risk rate         : 0.419\")\n"
             "print(f\"Gap even in the most optimistic case: {wc['lower']-0.419:+.3f}\")\n"
             "fig, ax = plt.subplots(figsize=(7,4))\n"
             "ax.fill_between(sens.coverage, sens.wc_lower, sens.wc_upper, alpha=0.15, color=PALETTE['blue'], label='worst-case')\n"
             "ax.plot(sens.coverage, sens.mono_lower, color=PALETTE['vermilion'], label='monotone lower = observed')\n"
             "ax.axhline(0.419, ls=':', color='0.5'); ax.text(0.71,0.43,'2018 rate', fontsize=8, color='0.4')\n"
             "ax.set_xlabel('assumed coverage share'); ax.set_ylabel('bound on 2022 at-risk rate')\n"
             "ax.set_title('Bounds vs coverage assumption'); ax.legend(fontsize=8)\n"
             "plt.tight_layout(); plt.show()"),
        md("**Reading:** even the **worst-case lower bound (0.58)** sits ~16 pp above Albania's 2018 "
           "rate of 0.42, and stays above it across every plausible coverage share. No coverage "
           "assumption - however adversarial - can explain away the 2018->2022 deterioration. The "
           "*level* is uncertain by ~21 pp in the worst case; the *conclusion* is not."),

        md(
            "## Conclusions & Interpretation\n\n"
            "- **The reported rate is a covered-population rate.** At 79% coverage the sampling SE "
            "(0.004) badly understates total uncertainty; the coverage gap is the real one.\n"
            "- **Partial identification, honestly stated:** worst-case [0.58, 0.79]; under the "
            "credible monotone assumption [0.74, 0.79], i.e. the headline likely *understates* the "
            "crisis rather than overstating it.\n"
            "- **The comparative claim is robust.** The 2022 rate exceeds the 2018 rate under every "
            "coverage assumption, so the deterioration is not a sampling artefact - it is the one "
            "conclusion coverage cannot touch.\n"
            "- **Method, not hand-wringing:** bounds + a reweighting scenario turn a data-quality "
            "caveat into a quantified robustness result, the honest complement to the point "
            "estimates elsewhere in the project."
        ),
    ]


def nb_17_fairness_mitigation() -> list:
    return [
        md(
            "# 17 - Fairness Mitigation, Not Just Diagnosis (Albania 2022)\n\n"
            "The audit (Part A) found the screener **over-flags the poorest students**: at a "
            "single 0.5 threshold the bottom SES quintile carries a far higher false-positive rate "
            "than the top. Diagnosis is not enough - this notebook *fixes* it, post-hoc, with **no "
            "retraining** (`scripts/run_fairness_mitigation.py`): group-specific decision thresholds "
            "(Hardt et al. 2016) applied to frozen out-of-fold predictions, equalising the weighted "
            "FPR across quintiles, and the fairness-utility trade-off that buys."
        ),
        code(HEADER),
        code("import json\n"
             "from src.visualization.style import apply_publication_style, PALETTE\n"
             "apply_publication_style()\n"
             "S = json.load(open('../outputs/results/fairness_mitigation_summary.json'))\n"
             "fr = pd.read_csv('../outputs/results/fairness_mitigation_frontier_ses.csv')\n"
             "print('target FPR', S['target_fpr'], '| per-group thresholds', S['group_thresholds'])"),

        md("## 1. The harm: a single threshold over-flags the poorest\n\n"
           "Weighted false-positive rate by SES quintile (1 = poorest) at the 0.5 threshold, and "
           "after equalising FPR with group-specific thresholds."),
        code("b = S['baseline_fpr_by_group']; m = S['mitigated_fpr_by_group']\n"
             "q = sorted(b, key=float); x = np.arange(len(q))\n"
             "fig, ax = plt.subplots(figsize=(7.5,4.3))\n"
             "ax.bar(x-0.2, [b[k] for k in q], 0.4, label='single 0.5 threshold', color=PALETTE['vermilion'])\n"
             "ax.bar(x+0.2, [m[k] for k in q], 0.4, label='group-specific thresholds', color=PALETTE['blue'])\n"
             "ax.set_xticks(x); ax.set_xticklabels([f'Q{int(float(k))}' for k in q])\n"
             "ax.set_xlabel('SES quintile (1 = poorest)'); ax.set_ylabel('weighted false-positive rate')\n"
             "ax.set_title('FPR by SES quintile: before vs after mitigation'); ax.legend(fontsize=8)\n"
             "plt.tight_layout(); plt.show()\n"
             "print(f\"FPR gap {S['baseline']['fpr_gap']:.3f} -> {S['mitigated']['fpr_gap']:.3f}\")"),
        md("**Reading:** at a single 0.5 threshold the poorest quintile is flagged with an FPR of "
           "~0.83 versus ~0.20 for the richest - an equalized-odds gap of **0.63**. Because the "
           "model reads socioeconomic composition, a blanket threshold turns *being poor* into "
           "*being flagged*. Group-specific thresholds flatten every quintile's FPR to ~0.42, "
           "collapsing the gap to **0.003** - the disparity was an artefact of the operating point, "
           "and it is removable without touching the model."),

        md("## 2. What it costs: the fairness-utility trade-off\n\n"
           "Equalising FPR is not free - it changes who gets caught. The frontier sweeps the common "
           "target FPR; each point compares group-specific thresholds against a single global "
           "threshold matched to the same overall selection rate."),
        code("fig, ax = plt.subplots(figsize=(7,5))\n"
             "ax.plot(fr.grp_recall, fr.grp_fpr_gap, '-o', color=PALETTE['blue'], label='group-specific')\n"
             "ax.plot(fr.single_recall, fr.single_fpr_gap, '-s', color=PALETTE['vermilion'], label='single threshold')\n"
             "ax.set_xlabel('overall weighted recall (at-risk students caught)')\n"
             "ax.set_ylabel('FPR gap across SES quintiles')\n"
             "ax.set_title('Fairness-utility trade-off (2022)'); ax.legend(fontsize=8)\n"
             "plt.tight_layout(); plt.show()\n"
             "print(f\"At the equal-FPR operating point: recall {S['baseline']['overall_recall']:.3f} \"\n"
             "      f\"-> {S['mitigated']['overall_recall']:.3f}, accuracy \"\n"
             "      f\"{S['baseline']['overall_accuracy']:.3f} -> {S['mitigated']['overall_accuracy']:.3f}\")"),
        md("**Reading:** at matched utility the group-specific curve sits **far below** the single-"
           "threshold curve - for any given recall it delivers a much smaller FPR gap, so "
           "per-group thresholds dominate a blanket one. The equity is not free: equalising FPR "
           "moves overall recall from ~0.81 to ~0.75 (about 6 pp fewer at-risk students caught), "
           "because much of the poorest quintile's high FPR came bundled with genuinely catching "
           "its at-risk students. That trade - a small recall cost for near-zero disparity - is a "
           "policy choice the frontier makes explicit rather than hiding in a default 0.5."),

        md(
            "## Conclusions & Interpretation\n\n"
            "- **The disparity is fixable post-hoc.** Group-specific thresholds cut the SES "
            "false-positive gap from **0.63 to ~0.00** with no retraining, operating only on frozen "
            "out-of-fold predictions.\n"
            "- **Per-group thresholds dominate a single threshold** on the fairness-utility "
            "frontier - a strictly better operating curve for equalized odds.\n"
            "- **Equity has a price, and we name it:** ~6 pp of overall recall. Whether that trade "
            "is worth it is a policy decision, now explicit.\n"
            "- **Caveat:** applying different thresholds by socioeconomic group is itself an ethical "
            "and legal choice (disparate treatment to reduce disparate impact). We present it as a "
            "quantified option, not a default - the honest counterpart to the audit's diagnosis."
        ),
    ]


def nb_16_process_modality() -> list:
    return [
        md(
            "# 16 - Can a New Data Modality Move the 0.78 Ceiling? (CBA process data)\n\n"
            "Notebooks 11-15 argue the ~0.78 ceiling is a property of the *data*, not the feature "
            "set - but every feature so far comes from the same questionnaire modality. This "
            "notebook links a genuinely **new data source** the project has never used: PISA's "
            "computer-based-assessment **process data** (`scripts/run_process_modality.py`), joined "
            "1:1 to the student file by `CNTSTUID`:\n\n"
            "* **cognitive process** (2022 `STU_COG`, 2018 `STU_TTM`) - per-item response *time* and "
            "*visit* counts across 443 timed items;\n"
            "* **questionnaire timing** (2022 `STU_TIM`) - response latency per background-question "
            "screen.\n\n"
            "**Leakage discipline:** item *correctness* mechanically determines the plausible-value "
            "target, so it is never used. Only *behavioural* features enter - how long a student "
            "spent and how they navigated, not whether they were right (see "
            "`src/features/process.py`)."
        ),
        code(HEADER),
        code("from src.visualization.style import apply_publication_style, PALETTE\n"
             "apply_publication_style()\n"
             "abl = pd.read_csv('../outputs/results/process_modality_ablation.csv')\n"
             "imp = pd.read_csv('../outputs/results/process_feature_importance_2022.csv', index_col=0)\n"
             "abl"),

        md("## 1. The ceiling moves - a new modality lifts AUC where features could not\n\n"
           "Weighted 5x2 CV AUC, student and student+school baselines, +process features "
           "(Nadeau-Bengio corrected test)."),
        code("comb = abl[abl.modality=='cognitive+questionnaire']\n"
             "fig, axes = plt.subplots(1,2, figsize=(10,4.3), sharey=True)\n"
             "for ax, cyc in zip(axes,(2022,2018)):\n"
             "    t = comb[comb.cycle==cyc]; x=np.arange(len(t))\n"
             "    ax.bar(x-0.2, t.base_auc, 0.4, label='base', color=PALETTE['blue'])\n"
             "    ax.bar(x+0.2, t.process_auc, 0.4, label='+ process', color=PALETTE['vermilion'])\n"
             "    for i,r in enumerate(t.itertuples()):\n"
             "        ax.text(i, max(r.base_auc,r.process_auc)+0.006, f'{r.lift:+.3f}', ha='center', fontsize=8)\n"
             "    ax.set_xticks(x); ax.set_xticklabels(t.baseline, fontsize=8); ax.set_title(str(cyc)); ax.set_ylim(0.6,0.88)\n"
             "    if cyc==2022: ax.axhline(0.78, ls=':', color='0.5', lw=1)\n"
             "axes[0].set_ylabel('weighted 5x2 CV AUC'); axes[0].legend(frameon=False)\n"
             "plt.tight_layout(); plt.show()"),
        md("**Reading:** adding process features lifts the AUC at every baseline. At the headline "
           "student+school ceiling the 2022 AUC rises **0.773 -> 0.858 (+8.5 pp, p < 0.001)**, and "
           "0.696 -> 0.773 in 2018. So the ceiling that a decade of questionnaire features could not "
           "break is *not* an absolute limit - a different data modality carries real extra signal. "
           "The next section asks whether that signal is one we could ever actually screen with."),

        md("## 2. Decomposition: is the lift deployable, or an artefact of taking the test?\n\n"
           "The lift splits into two very different modalities. Cognitive process is measured *from "
           "the math test itself* - it is endogenous to the ability being predicted and only exists "
           "*after* a student sits the CBA. Questionnaire timing is a separate instrument, available "
           "without the cognitive test."),
        code("dec = abl[(abl.cycle==2022)&(abl.baseline=='student+school')]\n"
             "dec[['modality','base_auc','process_auc','lift','lift_p','n_proc_feats']].round(4)"),
        md("**Reading:** the +8.5 pp is almost entirely **cognitive process** (+7.8 pp on its own). "
           "The **questionnaire-timing** modality - the only slice usable *before* the test, i.e. "
           "the only one a real screener could deploy - adds just **+1.1 pp** (p = 0.019). So a new "
           "modality moves the *number*, but the movement is post-hoc process data that (a) is "
           "endogenous to proficiency and (b) cannot flag a student early. The **deployable "
           "screening ceiling essentially holds at ~0.78**."),

        md("## 3. What the model uses: pure timing\n\n"
           "LightGBM importance over the process features (explanatory fit, 2022 student+school)."),
        code("ip = imp[imp.index.str.startswith(('PROC_','QTIM_'))].head(10)\n"
             "fig, ax = plt.subplots(figsize=(7,3.8))\n"
             "ax.barh(ip.index[::-1], ip['gain'].values[::-1], color=PALETTE['blue'])\n"
             "ax.set_xlabel('LightGBM gain'); ax.set_title('Top process features (2022)')\n"
             "plt.tight_layout(); plt.show()"),
        md("**Reading:** total and median time-on-task, rapid-response fraction and pacing "
           "variability dominate - the model reads *how* a student worked through the test. These "
           "are exactly the endogenous, post-hoc signals from section 2: informative about ability, "
           "useless as an early-warning screen."),

        md(
            "## Conclusions & Interpretation\n\n"
            "- **A new modality does move the ceiling** - CBA process data lifts the 2022 AUC to "
            "**0.86**, the first thing in the project to break 0.78. The ceiling is not an absolute "
            "information limit.\n"
            "- **But the movement is not deployable.** ~90% of the lift is cognitive-test process "
            "data, which is endogenous to the proficiency it predicts and only observable after the "
            "assessment. It cannot screen a student in advance.\n"
            "- **The screening ceiling holds.** The only pre-test modality (questionnaire response "
            "latency) adds +1.1 pp - real but marginal. For the paper's actual use case - flagging "
            "at-risk students from background data - ~0.78 stands.\n"
            "- **This sharpens, not contradicts, the data-ceiling thesis.** The ceiling is about "
            "*what is knowable before the test*. Rich behavioural signal exists inside the test, but "
            "harvesting it requires administering the very assessment a screener is meant to "
            "anticipate. Reported honestly, including the large cognitive-process lift we cannot "
            "use."
        ),
    ]


def nb_15_ceiling_generalization() -> list:
    return [
        md(
            "# 15 - Does the Data-Ceiling Claim Generalise? (9 countries, PISA 2022)\n\n"
            "Albania's headline is that its ~0.78 predictive ceiling is a **property of the data, "
            "not a feature shortfall**: school socioeconomic *composition* gives a large lift, then "
            "accuracy plateaus, and a big share of the outcome variance sits *between* schools "
            "(notebook 06). This part asks whether that mechanism is Albanian or universal, "
            "by running the two data-only legs on all nine comparison countries "
            "(`scripts/run_ceiling_generalization.py`, survey-weighted 5x2 CV):\n\n"
            "1. **Composition lift** - CV AUC with student background features, then with "
            "leave-one-out school-mean composition added; is the lift real (Nadeau-Bengio "
            "corrected-resampled t-test)?\n"
            "2. **Between-school ICC** - a null multilevel model's share of variance between "
            "schools.\n\n"
            "The third Albanian leg (the linked school *questionnaire* adds nothing beyond "
            "composition) needs data only Albania has, so it stays a single-country result."
        ),
        code(HEADER),
        code("from src.visualization.style import apply_publication_style, PALETTE\n"
             "apply_publication_style()\n"
             "t = pd.read_csv('../outputs/results/ceiling_generalization_2022.csv')\n"
             "t.sort_values('icc')"),

        md("## 1. Composition lift is nearly universal - the ceiling regime differs\n\n"
           "Bars: student-only vs +school-composition CV AUC, countries ordered by between-school "
           "ICC. The dotted line is Albania's 0.78."),
        code("d = t.sort_values('icc'); x = np.arange(len(d))\n"
             "fig, ax = plt.subplots(figsize=(9,4.5))\n"
             "ax.bar(x-0.2, d.base_auc, 0.4, label='student features', color=PALETTE['blue'])\n"
             "ax.bar(x+0.2, d.school_auc, 0.4, label='+ school composition', color=PALETTE['vermilion'])\n"
             "ax.set_xticks(x); ax.set_xticklabels(d.country)\n"
             "ax.set_ylim(0.5,0.9); ax.axhline(0.78, ls=':', color='0.5', lw=1)\n"
             "ax.text(0,0.785,'Albania ceiling 0.78', fontsize=8, color='0.4')\n"
             "ax.set_ylabel('weighted 5x2 CV AUC')\n"
             "ax.set_title('Predictive ceiling: student vs +school composition (2022)')\n"
             "ax.legend(frameon=False); plt.tight_layout(); plt.show()"),
        md("**Reading:** school composition gives a **significant lift in 8 of 9 countries** - the "
           "Albanian mechanism (composition matters, then plateau) is not idiosyncratic. It repeats "
           "across the Balkans (+6 to +8 pp) and the GDP-matched pair (Colombia +3, Mexico +4 pp). "
           "The exception is the **equitable top performers**: Estonia's lift is +2.2 pp and *not "
           "significant* (p = 0.09), Finland's is +0.8 pp. Where schools are socially mixed, "
           "composition carries almost no extra signal."),

        md("## 2. The ceiling tracks between-school segregation (ICC)\n\n"
           "If the ceiling is really a data property, the achievable AUC should rise with how "
           "segregated the school system is - the ICC."),
        code("fig, ax = plt.subplots(figsize=(6.5,5))\n"
             "colors = {'Albania':PALETTE['vermilion'],'Balkan':PALETTE['blue'],\n"
             "          'Top performer':PALETTE['green'],'GDP-matched':PALETTE['orange']}\n"
             "for grp, s in t.groupby('group'):\n"
             "    ax.scatter(s.icc, s.school_auc, s=80, color=colors[grp], label=grp, zorder=3)\n"
             "for _, r in t.iterrows():\n"
             "    ax.annotate(r.country, (r.icc, r.school_auc), fontsize=8, xytext=(4,4), textcoords='offset points')\n"
             "corr = np.corrcoef(t.icc, t.school_auc)[0,1]\n"
             "ax.set_xlabel('between-school ICC (null multilevel model)')\n"
             "ax.set_ylabel('achievable CV AUC (+school composition)')\n"
             "ax.set_title(f'More between-school segregation -> higher ceiling (r={corr:.2f})')\n"
             "ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()"),
        md("**Reading:** across the nine systems the achievable AUC and the ICC correlate at "
           "**r = 0.80**. High-segregation systems (Bulgaria ICC 0.56, Colombia 0.45) are the most "
           "predictable; **Finland (ICC 0.10) is the least** - its schools are so socially uniform "
           "that between-school composition, the feature doing the heavy lifting everywhere else, "
           "has little to grip. The ceiling is not a modelling artefact; it is set by how much of "
           "the risk lives between schools versus within them."),

        md(
            "## Conclusions & Interpretation\n\n"
            "- **The data-ceiling mechanism generalises.** School socioeconomic composition lifts "
            "AUC significantly in 8 of 9 countries, and the achievable ceiling scales with the "
            "between-school ICC (r = 0.80). Albania's 0.78 is one point on a cross-national "
            "relationship, not a one-off.\n"
            "- **Finland is the boundary condition.** In an equitable, low-segregation system "
            "(ICC 0.10) composition adds almost nothing and the ceiling story weakens - the claim "
            "is about *segregated* education systems, and we say so rather than overreach.\n"
            "- **It is a data property, not a feature shortfall.** The lever that moves the ceiling "
            "is structural (how segregated schools are), not a cleverer feature set - consistent "
            "with the Albanian school-questionnaire null (Part B), which stays single-country "
            "because only Albania has the linked questionnaire.\n"
            "- **Policy read:** where risk is concentrated between schools, a screener rides that "
            "segregation to higher apparent accuracy; the equitable-system ceiling is lower because "
            "there is less between-school structure to exploit - a feature of fairness, not a bug."
        ),
    ]


# ===========================================================================
# Merged-notebook intros (18 short notebooks consolidated into 10 coherent ones)
# ===========================================================================
INTRO_CRISIS = (
    "# 02 - The 2022 Crisis in Context: Peers and Covariate Shift (PISA 2022)\n\n"
    "Notebook 01 established Albania's own V-shaped trajectory. This notebook places the 2022 "
    "collapse in context two ways: **Part A** benchmarks Albania against Balkan peers, top "
    "performers and GDP-matched economies (low-proficiency ranking and the SES gradient), and "
    "**Part B** characterises *how* the 2022 cohort differs from 2018 - a structural covariate "
    "shift, not merely lower scores. Together they frame the crisis the rest of the project models."
)
INTRO_MODELING = (
    "# 03 - Predictive Modeling: Comparison, Out-of-Sample, and Stacking (Albania 2022)\n\n"
    "**Part A** compares twelve models under weighted repeated-CV, runs the out-of-sample "
    "2009-2018 -> 2022 test and the rigorous per-plausible-value (Rubin's rules) evaluation. "
    "**Part B** asks whether a stacking ensemble beats the single best model - a model-selection "
    "coda that motivates keeping one interpretable headline model."
)
INTRO_EXPLAIN = (
    "# 04 - Explainability: Global, Local, and Partial Dependence (Albania 2022)\n\n"
    "How the headline model decides. **Part A** is the global view (SHAP importance, beeswarm, a "
    "dependence interaction); **Part B** drills into individual students (representative "
    "confusion-matrix cases, local SHAP waterfalls) and the shape of each effect (partial "
    "dependence + centered ICE)."
)
INTRO_XCOUNTRY = (
    "# 05 - Cross-Country: Risk Drivers and the Generalisable Ceiling (PISA 2022)\n\n"
    "Are Albania's risk drivers and its predictive ceiling idiosyncratic or universal? **Part A** "
    "fits the model per country and compares predictability, prevalence, the SES gradient and a "
    "SHAP importance rank matrix. **Part B** tests whether the 'data-ceiling' mechanism "
    "generalises across nine countries and where it breaks (the equitable-system boundary)."
)
INTRO_CEILING = (
    "# 06 - The 0.78 Ceiling: Real, Data-vs-Features, and Whether a New Modality Moves It "
    "(Albania 2022)\n\n"
    "The project's central methodological claim, in one place. **Part A** asks whether the screener "
    "is even useful (decision-curve + calibration) and how much risk is a *school* effect "
    "(multilevel ICC). **Part B** tests whether the ceiling is features or data by linking the "
    "school questionnaire. **Part C** links a genuinely new modality - CBA process data - and asks "
    "whether it can move the ceiling, and whether that movement is deployable."
)
INTRO_FAIR = (
    "# 07 - Fairness: Audit and Mitigation (Albania 2022)\n\n"
    "**Part A** audits the survey-weighted screener across gender, SES quintile and immigrant "
    "status, finding it over-flags the poorest students. **Part B** *fixes* that post-hoc with "
    "group-specific thresholds and traces the fairness-utility trade-off - diagnosis, then remedy."
)
INTRO_ROBUST = (
    "# 09 - How Robust Is the Crisis? Causal Test and Coverage Bounds (Albania)\n\n"
    "Two honesty checks on the crisis claim. **Part A** tests the 2019 Durres earthquake as a "
    "natural experiment (difference-in-differences, placebo, triple-difference) - a candidate "
    "causal story that does not survive its diagnostics. **Part B** puts partial-identification "
    "(Manski) bounds on the at-risk rate under PISA's ~79% coverage, showing the 2018->2022 "
    "deterioration is robust to any coverage assumption."
)


def nb_geographic() -> list:
    return [
        md(
            "# Geographic equity - where inside Albania the crisis concentrates\n\n"
            "The national at-risk rate hides *where* the crisis lives. Albania's sampling `STRATUM` "
            "decodes into region band (North / Center / South) x urbanicity (Urban / Rural) x sector "
            "(Public / Private), so we can map the below-Level-2 mathematics rate inside the country "
            "and across 2015 / 2018 / 2022. Every rate is survey-weighted with a PISA design-based "
            "standard error (BRR + Rubin over plausible values). Pre-computed by "
            "`scripts/run_geographic_equity.py` (`src/geography/equity.py`)."
        ),
        code(HEADER),
        code("from src.visualization.style import apply_publication_style, PALETTE, SEQUENTIAL_CMAP\n"
             "apply_publication_style()\n"
             "R = '../outputs/results/'\n"
             "region = pd.read_csv(R+'geographic_region_by_cycle.csv')\n"
             "urban  = pd.read_csv(R+'geographic_urbanicity_by_cycle.csv')\n"
             "cell   = pd.read_csv(R+'geographic_cell_by_cycle.csv')\n"
             "gaps   = pd.read_csv(R+'geographic_gaps.csv')"),

        md("## 1. The trajectory is not uniform: region and urbanicity\n\n"
           "Both panels show the same V shape - improvement to 2018, then the 2022 reversal - but "
           "the *levels* separate sharply. The **North band is highest in every cycle**, and "
           "**rural areas sit ~10-17 pp above urban** throughout. Error bars are design-based 95% "
           "CIs."),
        code("REG_C = {'North':PALETTE['vermilion'],'Center':PALETTE['blue'],'South':PALETTE['green']}\n"
             "URB_C = {'Rural':PALETTE['orange'],'Urban':PALETTE['blue']}\n"
             "fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.6),sharey=True)\n"
             "for b in ['North','Center','South']:\n"
             "    s=region[region.region==b].sort_values('CYCLE')\n"
             "    a1.errorbar(s.CYCLE,s.at_risk,yerr=1.96*s.se,marker='o',capsize=3,lw=2,\n"
             "                color=REG_C[b],label=b+(' (highest)' if b=='North' else ''))\n"
             "for u in ['Rural','Urban']:\n"
             "    s=urban[urban.urbanicity==u].sort_values('CYCLE')\n"
             "    a2.errorbar(s.CYCLE,s.at_risk,yerr=1.96*s.se,marker='s',capsize=3,lw=2,\n"
             "                color=URB_C[u],label=u)\n"
             "for ax,t in [(a1,'By region band'),(a2,'By urbanicity')]:\n"
             "    ax.set_xticks([2015,2018,2022]); ax.set_xlabel('PISA cycle')\n"
             "    ax.axvspan(2018,2022,color='0.5',alpha=0.06); ax.set_ylim(0.3,0.9)\n"
             "    ax.set_title(t); ax.legend(frameon=False)\n"
             "a1.set_ylabel('Share below Level 2 (math)')\n"
             "plt.tight_layout(); plt.show()"),

        md("## 2. Where the crisis is worst: region x urbanicity, 2022\n\n"
           "Crossing the two axes localises the load. Darker = higher at-risk share. **Rural North "
           "(83%) and rural Center (82%)** are the heaviest cells; even *urban* Center - which "
           "contains Tirana - sits at 69%, the lowest cell but still a majority at risk."),
        code("m = (cell[cell.CYCLE==2022].pivot(index='urbanicity',columns='region',values='at_risk')\n"
             "     .reindex(index=['Urban','Rural'],columns=['North','Center','South']))\n"
             "fig,ax=plt.subplots(figsize=(6.4,3.4)); d=m.to_numpy(float)\n"
             "im=ax.imshow(d,cmap=SEQUENTIAL_CMAP,vmin=0.5,vmax=0.9,aspect='auto')\n"
             "ax.set_xticks(range(3),m.columns); ax.set_yticks(range(2),m.index)\n"
             "ax.set_xlabel('Region band'); ax.set_ylabel('Urbanicity')\n"
             "for i in range(2):\n"
             "    for j in range(3):\n"
             "        ax.text(j,i,f'{d[i,j]*100:.0f}%',ha='center',va='center',fontsize=12,\n"
             "                fontweight='bold',color='white' if d[i,j]>0.72 else 'black')\n"
             "fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04).set_label('Share below Level 2')\n"
             "ax.set_title('At-risk rate by cell, 2022'); plt.tight_layout(); plt.show()"),

        md("## 3. Are the gaps real? Design-based significance\n\n"
           "Two overlapping marginal CIs do not test a gap; we estimate each contrast directly and "
           "attach a BRR + Rubin standard error to the *difference*. All three geographic gaps are "
           "significant in 2022, and the **public-minus-private gap (~25 pp) is the largest of "
           "all** - a structural divide larger than region or urbanicity."),
        code("g = gaps.copy()\n"
             "g['contrast']=g.level_hi+' - '+g.level_lo\n"
             "show=g[['cycle','group_col','contrast','gap','se','p_value','n_hi','n_lo']]\n"
             "show.round(4)"),
        md("**Reading:** rural areas run ~10 pp above urban and the North ~7 pp above the South in "
           "2022, both highly significant (p < 0.001). The public-vs-private gap is the widest and "
           "has *grown* (0.15 in 2018 to 0.26 in 2022), though private schools are a small, "
           "self-selected ~13% of students - a composition signal, not a like-for-like school "
           "effect. Together these are the actionable geography: the crisis is worst where it is "
           "rural, northern and public."),

        md("## 4. The 2018 to 2022 reversal, cell by cell\n\n"
           "The reversal was broad-based - every cell jumped ~28-37 pp - but it stacked *on top of* "
           "pre-existing disadvantage, so the rural bands ended highest. The dumbbell orders cells "
           "by their 2022 level; the number is the percentage-point jump from 2018."),
        code("piv=(cell.pivot_table(index=['region','urbanicity'],columns='CYCLE',values='at_risk')\n"
             "     .dropna(subset=[2018,2022]).copy())\n"
             "piv['jump']=piv[2022]-piv[2018]; piv=piv.sort_values(2022)\n"
             "labels=[f'{r} / {u}' for r,u in piv.index]; y=range(len(piv))\n"
             "fig,ax=plt.subplots(figsize=(8,4.6))\n"
             "for yi,(_,row) in zip(y,piv.iterrows()):\n"
             "    ax.plot([row[2018],row[2022]],[yi,yi],color='0.6',lw=2,zorder=1)\n"
             "    ax.annotate(f\"+{row['jump']*100:.0f}\",xy=(row[2022],yi),xytext=(6,0),\n"
             "                textcoords='offset points',va='center',fontsize=9,color=PALETTE['vermilion'])\n"
             "ax.scatter(piv[2018],list(y),s=70,color=PALETTE['sky_blue'],label='2018',zorder=2)\n"
             "ax.scatter(piv[2022],list(y),s=70,color=PALETTE['vermilion'],label='2022',zorder=2)\n"
             "ax.set_yticks(list(y),labels); ax.set_xlim(0.25,0.95)\n"
             "ax.set_xlabel('Share below Level 2 (math)')\n"
             "ax.legend(frameon=False,loc='lower right',title='PISA cycle')\n"
             "plt.tight_layout(); plt.show()"),
        md("**Policy read:** a national average of ~74% at-risk masks an 83% rural-North load and a "
           "25 pp public-private divide. Because the flat SES gradient (Part A, notebook 05) means "
           "income-targeted transfers alone miss most at-risk students, **geography is a sharper "
           "targeting axis than family SES** - the North, rural and public-school bands are where a "
           "fixed remediation budget buys the most reach."),
    ]


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    B = build_notebook
    B(_with_formulas(nb_01_eda_albania(), "01"), NB_DIR / "01_eda_albania.ipynb", force)
    B(_assemble(INTRO_CRISIS, HEADER, ["02", "03"],
                [("A", "Albania vs. peer countries", nb_02_comparative),
                 ("B", "The 2022 covariate shift", nb_03_covariate_shift),
                 ("C", "The geography of the collapse", nb_geographic)]),
      NB_DIR / "02_crisis_in_context.ipynb", force)
    B(_assemble(INTRO_MODELING, MODEL_HEADER, ["04"],
                [("A", "Model comparison, out-of-sample & Rubin's rules", nb_04_modeling),
                 ("B", "Stacking ensemble", nb_10_stacking)]),
      NB_DIR / "03_modeling.ipynb", force)
    B(_assemble(INTRO_EXPLAIN, MODEL_HEADER, ["05", "06"],
                [("A", "Global SHAP", nb_05_explainability),
                 ("B", "Local cases + partial dependence", nb_06_explainability_cases)]),
      NB_DIR / "04_explainability.ipynb", force)
    B(_assemble(INTRO_XCOUNTRY, HEADER, ["08"],
                [("A", "Per-country predictability & drivers", nb_08_comparative),
                 ("B", "Does the data-ceiling claim generalise?", nb_15_ceiling_generalization)]),
      NB_DIR / "05_cross_country.ipynb", force)
    B(_assemble(INTRO_CEILING, HEADER, ["11"],
                [("A", "Is the screener useful, and is the ceiling real?", nb_11_screener_multilevel),
                 ("B", "Features or data? The school questionnaire", nb_12_school_questionnaire),
                 ("C", "Can a new modality move it? CBA process data", nb_16_process_modality)]),
      NB_DIR / "06_the_ceiling.ipynb", force)
    B(_assemble(INTRO_FAIR, MODEL_HEADER, ["07"],
                [("A", "Fairness audit", nb_07_fairness),
                 ("B", "Mitigation", nb_17_fairness_mitigation)]),
      NB_DIR / "07_fairness.ipynb", force)
    B(nb_13_decision_support(), NB_DIR / "08_decision_support.ipynb", force)
    B(_assemble(INTRO_ROBUST, HEADER, [],
                [("A", "A natural experiment: the 2019 earthquake", nb_14_causal_earthquake),
                 ("B", "Coverage bounds (Manski)", nb_18_coverage_bounds)]),
      NB_DIR / "09_crisis_robustness.ipynb", force)
    B(_with_formulas(nb_09_forecast(), "09"), NB_DIR / "10_forecast_2026.ipynb", force)
