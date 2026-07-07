"""
Target #1 - can a genuinely new data MODALITY move the ~0.78 ceiling?

We link PISA computer-based-assessment process data (per-item response time,
visits, navigation; and questionnaire response latencies) - a data source the
project has never used - to the student file 1:1 by CNTSTUID, and ask whether
adding purely *behavioural* process features lifts the weighted CV AUC beyond
the established student + school-composition ceiling.

Modalities (all behavioural; item correctness is never used - see
src/features/process.py leakage note):
    2022  STU_COG  per cognitive item total time + visits (443 timed items)
    2022  STU_TIM  per questionnaire screen response time
    2018  STU_TTM  per cognitive item total time + visits (pre-crisis cycle)

For each cycle we run a paired ablation (weighted 5x2 CV, Nadeau-Bengio
corrected-resampled t-test) at two baselines:
    * student features only            -> + process
    * student + school composition     -> + process   (this is THE ceiling test)

Outputs:
    outputs/results/process_modality_ablation.csv
    outputs/results/process_feature_importance_2022.csv
    outputs/figures/process/F1_process_ablation.png
Process features are cached to data/processed/process_features_<cycle>.parquet
(gitignored, regenerated here) so the notebook need not re-read the SAV files.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import lightgbm  # noqa: F401  load libomp before sklearn (rc=-11 import-order fix)

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from threadpoolctl import threadpool_limits

from src.features.process import (
    BEHAVIOURAL_COLS,
    QTIM_COLS,
    aggregate_cog_process,
    link_process_features,
    load_cog_process,
    load_questionnaire_timing,
)
from src.models.evaluate import corrected_resampled_ttest, fit_with_sample_weight
from src.models.experiment import _wrap
from src.models.prepare import build_model_data
from src.models.registry import get_model
from src.utils.logging import configure_logging
from src.visualization.style import PALETTE, apply_publication_style, save_figure

FEATS = ["ESCS", "HOMEPOS", "GENDER", "REPEAT", "IMMIG", "BELONG", "TEACHSUP",
         "ICTHOME", "ICTSCH", "ANXMAT", "GRADE", "HISCED", "HISEI"]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
RES = ROOT / "outputs/results"
FIG = ROOT / "outputs/figures/process"

N_SPLITS, N_REPEATS = 5, 2
TEST_TRAIN_RATIO = 1.0 / (N_SPLITS - 1)


def _paired_cv(X_base, X_aug, y, w, seed: int = 42):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    b, a = [], []
    with threadpool_limits(limits=1):
        for tr, te in rskf.split(X_base, y):
            if len(np.unique(y.iloc[te])) < 2:
                continue
            wte = w.iloc[te].values
            for Xf, store in ((X_base, b), (X_aug, a)):
                pipe = _wrap(get_model("lightgbm"))
                fit_with_sample_weight(pipe, Xf.iloc[tr], y.iloc[tr], w.iloc[tr].values)
                p = pipe.predict_proba(Xf.iloc[te])[:, 1]
                store.append(roc_auc_score(y.iloc[te], p, sample_weight=wte))
    return np.array(b), np.array(a)


def _build_process_features_2022() -> pd.DataFrame:
    cog = aggregate_cog_process(load_cog_process(RAW / "CY08MSP_STU_COG.SAV"))
    qtim = load_questionnaire_timing(RAW / "CY08MSP_STU_TIM.SAV")
    feats = cog.merge(qtim, on="CNTSTUID", how="outer")
    feats.to_parquet(PROCESSED / "process_features_2022.parquet")
    return feats


def _build_process_features_2018() -> pd.DataFrame:
    cog = aggregate_cog_process(load_cog_process(RAW / "TTM/CY07_MSU_STU_TTM.SAV"))
    cog.to_parquet(PROCESSED / "process_features_2018.parquet")
    return cog


def _one_ablation(sub, proc_cols, cycle, baseline_label, modality, school, rows):
    base = build_model_data(sub, FEATS, domain="math", add_school_context=school)
    aug = build_model_data(sub, FEATS + proc_cols, domain="math", add_school_context=school)
    assert base.y.index.equals(aug.y.index)
    bf, af = _paired_cv(base.X, aug.X, base.y, base.weights)
    lift = float(af.mean() - bf.mean())
    _, _, p = corrected_resampled_ttest(af - bf, TEST_TRAIN_RATIO)
    rows.append({
        "cycle": cycle, "baseline": baseline_label, "modality": modality,
        "base_auc": round(float(bf.mean()), 4),
        "process_auc": round(float(af.mean()), 4),
        "lift": round(lift, 4), "lift_p": round(float(p), 4),
        "n": int(base.meta["n_samples"]), "n_proc_feats": len(proc_cols),
    })
    print(f"  {cycle} {baseline_label:14s} +{modality:24s}: "
          f"{bf.mean():.3f} -> {af.mean():.3f} lift={lift:+.3f} (p={p:.3f})", flush=True)


def _ablate(sub: pd.DataFrame, proc_cols: list[str], cycle: int, rows: list) -> None:
    for school in (False, True):
        label = "student+school" if school else "student"
        _one_ablation(sub, proc_cols, cycle, label, "cognitive+questionnaire", school, rows)


def main() -> None:
    configure_logging(level="WARNING")
    apply_publication_style()
    for d in (RES, FIG):
        d.mkdir(parents=True, exist_ok=True)

    proc22 = _build_process_features_2022()
    proc18 = _build_process_features_2018()
    proc22_cols = [c for c in BEHAVIOURAL_COLS + QTIM_COLS if c in proc22.columns]
    proc18_cols = [c for c in BEHAVIOURAL_COLS if c in proc18.columns]

    alb22 = link_process_features(pd.read_parquet(PROCESSED / "alb_2022.parquet"), proc22)
    alb18 = link_process_features(pd.read_parquet(PROCESSED / "alb_2018.parquet"), proc18)

    cog22 = [c for c in BEHAVIOURAL_COLS if c in proc22.columns]
    qtim22 = [c for c in QTIM_COLS if c in proc22.columns]

    rows: list[dict] = []
    print("=== 2022 (headline 0.78 cycle): +cognitive time/visits + questionnaire timing ===")
    _ablate(alb22, proc22_cols, 2022, rows)
    # decompose the lift at the ceiling baseline: which modality drives it?
    print("--- 2022 modality decomposition (at student+school ceiling) ---")
    _one_ablation(alb22, cog22, 2022, "student+school", "cognitive-only", True, rows)
    _one_ablation(alb22, qtim22, 2022, "student+school", "questionnaire-only", True, rows)
    print("=== 2018 (pre-crisis): +cognitive time/visits ===")
    _ablate(alb18, proc18_cols, 2018, rows)

    table = pd.DataFrame(rows)
    table.to_csv(RES / "process_modality_ablation.csv", index=False)
    print("\n" + table.to_string(index=False))

    # ---- process-feature importance (explanatory fit, 2022 student+school) ----
    base = build_model_data(alb22, FEATS + proc22_cols, domain="math", add_school_context=True)
    from src.features.transformers import EngineeredFeatureBuilder
    from src.models.prepare import impute_median
    Xe = EngineeredFeatureBuilder().fit_transform(base.X)
    (Xi,) = impute_median(Xe)
    model = get_model("lightgbm")
    model.fit(Xi, base.y.values, sample_weight=base.weights.values)
    imp = (pd.Series(model.feature_importances_, index=Xi.columns)
           .sort_values(ascending=False))
    imp_proc = imp[[c for c in imp.index if c.startswith(("PROC_", "QTIM_"))]]
    imp.to_frame("gain").to_csv(RES / "process_feature_importance_2022.csv")
    print("\nTop process features by LightGBM importance (2022):")
    print(imp_proc.head(10).round(1).to_string())

    # ---- figure (combined-modality rows only; decomposition shown in notebook) ----
    combined = table[table.modality == "cognitive+questionnaire"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, cyc in zip(axes, (2022, 2018)):
        t = combined[combined.cycle == cyc]
        x = np.arange(len(t))
        ax.bar(x - 0.2, t.base_auc, 0.4, label="base", color=PALETTE["blue"])
        ax.bar(x + 0.2, t.process_auc, 0.4, label="+ process", color=PALETTE["vermilion"])
        for i, r in enumerate(t.itertuples()):
            star = "*" if r.lift_p < 0.05 else "n.s."
            ax.text(i, max(r.base_auc, r.process_auc) + 0.005, f"{r.lift:+.3f}\n{star}",
                    ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(t.baseline, fontsize=8)
        ax.set_title(f"{cyc}"); ax.set_ylim(0.6, 0.86)
        if cyc == 2022:
            ax.axhline(0.78, ls=":", color="0.5", lw=1)
    axes[0].set_ylabel("weighted 5x2 CV AUC"); axes[0].legend(frameon=False)
    fig.suptitle("Does a new behavioural modality move the ceiling?")
    save_figure(fig, str(FIG / "F1_process_ablation"))


if __name__ == "__main__":
    main()
