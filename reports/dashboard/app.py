"""
Albania PISA low-proficiency risk screener - interactive dashboard.

A bilingual (English / Albanian) plain-language front-end over the project's
headline model (survey-weighted CatBoost with school socioeconomic composition,
isotonically calibrated). A student answers concrete questions - books at home,
parents' education and work, how they feel about maths - and the app maps those
answers onto the model's OECD indices, returns a calibrated probability of scoring
below PISA mathematics Level 2 (< 420), shows which factor GROUPS push the estimate
up or down, and can export a printable report card.

    streamlit run reports/dashboard/app.py

Bundle produced by ``scripts/export_dashboard_model.py``. This is an ASSOCIATIONAL
screener trained on one 2022 cross-section - see the caveats panel.
"""
from __future__ import annotations

import catboost  # noqa: F401  (import booster before sklearn: OpenMP order)
from catboost import Pool

import json
import sys
from datetime import datetime
from pathlib import Path

# Repo root on sys.path so the pickled pipeline (src.features.transformers...)
# unpickles; Streamlit only adds the script's own dir.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
import altair as alt
import streamlit as st

MODELS = ROOT / "outputs" / "models"
BUNDLE = MODELS / "dashboard_bundle.joblib"
META = MODELS / "dashboard_meta.json"

st.set_page_config(page_title="Albania PISA Risk Screener",
                   page_icon=None, layout="centered")

# ---- theme palettes (validated diverging pair blue<->red + neutrals) ---------
# Light & dark are SELECTED, not auto-flipped: each data pair is validated against
# its own surface (dataviz skill, validate_palette.js). Browser prefers-color-scheme
# can't reach Python-rendered Altair, so a Python toggle drives BOTH css and charts.
PALETTES = {
    "light": {
        "risk": "#d03b3b", "safe": "#2a78d6", "ink": "#0b0b0b", "second": "#52514e",
        "muted": "#8a9099", "track": "#e3e8ef", "grid": "#e9edf3",
        "bg": "#eef2f8", "card": "#ffffff", "border": "rgba(30,41,59,.08)",
        "shadow": "0 1px 4px rgba(30,41,59,.05)", "qlabel": "#33322f",
        "note_bg": "#eef4fc", "note_ink": "#33414f",
        "kpi_bg": "#f4f7fc", "kpi_border": "rgba(30,41,59,.07)",
    },
    "dark": {  # surfaces matched to Streamlit's stock dark theme (bg #0e1117)
        "risk": "#e85d5a", "safe": "#4a90e2", "ink": "#fafafa", "second": "#c3c8d0",
        "muted": "#8b929e", "track": "#2a2e38", "grid": "#262b34",
        "bg": "#0e1117", "card": "#1a1d26", "border": "rgba(255,255,255,.08)",
        "shadow": "0 1px 4px rgba(0,0,0,.4)", "qlabel": "#c9ccd2",
        "note_bg": "#172230", "note_ink": "#cdd4de",
        "kpi_bg": "#1e222b", "kpi_border": "rgba(255,255,255,.07)",
    },
}


def build_css(P: dict) -> str:
    return f"""
    <style>
      .stApp {{ background: {P['bg']}; }}
      /* header stays (holds the ⋮ menu for the theme switch) but goes transparent
         and content is padded below it so the toolbar never overlaps the radio */
      header[data-testid="stHeader"] {{ background: transparent; }}
      .block-container {{ max-width: 780px; padding-top: 3.4rem; padding-bottom: 4rem; }}
      footer {{ visibility: hidden; }}
      [data-testid="stToolbar"] {{ right: .4rem; }}
      h1,h2,h3 {{ letter-spacing:-0.01em; }}
      .app-title {{ text-align:center; font-weight:700; font-size:1.95rem; margin-bottom:.15rem; color:{P['ink']}; }}
      .app-sub {{ text-align:center; color:{P['second']}; font-size:1rem; margin:0 auto .8rem; max-width:35rem; }}
      .app-ctx {{ text-align:center; color:{P['muted']}; font-size:.8rem; letter-spacing:.02em;
                  text-transform:uppercase; margin:0 auto 1.1rem; }}
      .kpi-row {{ display:flex; gap:.7rem; margin:0 auto 1.3rem; }}
      .kpi {{ flex:1; background:{P['kpi_bg']}; border:1px solid {P['kpi_border']}; border-radius:14px;
              padding:.75rem .6rem; text-align:center; }}
      .kpi-val {{ font-weight:700; font-size:1.5rem; line-height:1.05; color:{P['ink']}; }}
      .kpi-lab {{ font-size:.72rem; color:{P['muted']}; margin-top:.2rem; line-height:1.25; }}
      div[data-testid="stVerticalBlockBorderWrapper"] {{
        background:{P['card']}; border:1px solid {P['border']}; border-radius:16px;
        padding:1.15rem 1.3rem; box-shadow:{P['shadow']}; margin-bottom:1rem;
      }}
      .card-h {{ font-weight:650; font-size:1.02rem; color:{P['ink']}; margin:.1rem 0 .6rem; }}
      .risk-num {{ text-align:center; font-size:3.6rem; font-weight:700; line-height:1; }}
      .risk-band {{ text-align:center; font-size:1.08rem; font-weight:600; margin-top:.35rem; }}
      .risk-pct {{ text-align:center; color:{P['second']}; font-size:.92rem; margin-top:.5rem; }}
      .risk-pct b {{ color:{P['ink']}; }}
      .risk-note {{ text-align:center; color:{P['muted']}; font-size:.85rem; margin-top:.3rem; }}
      .note-card {{ background:{P['note_bg']}; border-left:4px solid {P['safe']}; border-radius:10px;
                   padding:.85rem 1.05rem; color:{P['note_ink']}; font-size:.92rem; line-height:1.5; }}
      .stSelectbox label p {{ font-size:.9rem !important; color:{P['qlabel']} !important; }}
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
        color:{P['muted']} !important; }}
    </style>
    """


@st.cache_resource
def load_model():
    if not BUNDLE.exists():
        st.error("Model bundle not found. Run first:\n\n    python scripts/export_dashboard_model.py")
        st.stop()
    return joblib.load(BUNDLE), json.loads(META.read_text())


bundle, meta = load_model()
pipe = bundle["pipeline"]
iso = bundle["isotonic"]
feature_names = bundle["feature_names"]
shap_names = bundle["shap_feature_names"]
threshold = bundle["threshold"]
sliders = meta["sliders"]
base_rate = meta["base_rate_at_risk"]
# 101 survey-weighted quantile edges of the leak-free cohort risk distribution
risk_edges = np.asarray(meta.get("risk_percentile_edges", []), dtype=float)
ses_gradient = meta.get("ses_risk_gradient", [])          # mean risk per ESCS quintile
escs_q_edges = np.asarray(meta.get("escs_quintile_edges", []), dtype=float)


def escs_quintile(escs: float) -> int:
    """Which ESCS fifth (0..4, poorest..richest) a student's answers land in."""
    if escs_q_edges.size != 4:
        return -1
    return int(np.digitize([escs], escs_q_edges)[0])


def cohort_percentile(risk: float) -> int:
    """Where this estimate sits in the 2022 cohort's calibrated-risk distribution."""
    if risk_edges.size == 0:
        return -1
    return int(round(float(np.interp(risk, risk_edges, np.arange(risk_edges.size)))))


def anchor(feat: str, key: str) -> float:
    s = sliders[feat]
    return float({"min": s["min"], "p25": s["p25"], "p50": s["default"],
                  "p75": s["p75"], "max": s["max"]}[key])


# ---- question logic (assignments only; display text lives in TEXT below) -----
# each option = [(feature, anchor-key|None, raw-value|None), ...]
QMETA = [
    ("about", [
        ("sex",   [[("GENDER", None, 0.0)], [("GENDER", None, 1.0)]]),
        ("repeat", [[("REPEAT", None, 0.0), ("GRADE", "p50", None)],
                    [("REPEAT", None, 1.0), ("GRADE", "min", None)]]),
        ("immig", [[("IMMIG", None, 0.0)], [("IMMIG", None, 1.0)], [("IMMIG", None, 2.0)]]),
    ]),
    ("home", [
        ("books", [[("HOMEPOS", k, None)] for k in ("min", "p25", "p50", "p75", "max")]),
        ("paredu", [[("ESCS", k, None), ("HISCED", k, None)] for k in ("min", "p25", "p50", "max")]),
        ("paroc", [[("HISEI", k, None)] for k in ("min", "p25", "p50", "max")]),
        ("homeict", [[("ICTHOME", k, None)] for k in ("min", "p25", "p75", "max")]),
    ]),
    ("school", [
        ("anx", [[("ANXMAT", k, None)] for k in ("min", "p25", "p75", "max")]),
        ("belong", [[("BELONG", k, None)] for k in ("min", "p25", "p75", "max")]),
        ("tsup", [[("TEACHSUP", k, None)] for k in ("min", "p25", "p75", "max")]),
        ("schict", [[("ICTSCH", k, None)] for k in ("min", "p25", "p75", "max")]),
        ("schsize", [[("SCH_N", k, None)] for k in ("p25", "p50", "p75")]),
        ("schaff", [
            [("SCH_MEAN_ESCS", "min", None), ("SCH_MEAN_HOMEPOS", "min", None),
             ("SCH_MEAN_TEACHSUP", "p25", None), ("SCH_MEAN_ANXMAT", "max", None)],
            [("SCH_MEAN_ESCS", "p50", None), ("SCH_MEAN_HOMEPOS", "p50", None),
             ("SCH_MEAN_TEACHSUP", "p50", None), ("SCH_MEAN_ANXMAT", "p50", None)],
            [("SCH_MEAN_ESCS", "max", None), ("SCH_MEAN_HOMEPOS", "max", None),
             ("SCH_MEAN_TEACHSUP", "p75", None), ("SCH_MEAN_ANXMAT", "min", None)],
        ]),
    ]),
]

DEFAULT_IDX = {"sex": 0, "repeat": 0, "immig": 0, "books": 2, "paredu": 2, "paroc": 2,
               "homeict": 2, "anx": 1, "belong": 2, "tsup": 2, "schict": 2,
               "schsize": 1, "schaff": 1}

# Illustrative one-click composites (synthetic, not real students). Keys match the
# questionnaire; values are option indices. "under" = low resources / higher anxiety,
# "adv" = high resources / low anxiety; "avg" reuses the cohort-median defaults.
PRESETS = {
    "avg": DEFAULT_IDX,
    "under": {"sex": 0, "repeat": 1, "immig": 0, "books": 0, "paredu": 0, "paroc": 0,
              "homeict": 0, "anx": 3, "belong": 1, "tsup": 1, "schict": 0,
              "schsize": 1, "schaff": 0},
    "adv": {"sex": 0, "repeat": 0, "immig": 0, "books": 4, "paredu": 3, "paroc": 3,
            "homeict": 3, "anx": 0, "belong": 3, "tsup": 3, "schict": 3,
            "schsize": 2, "schaff": 2},
}


def apply_preset(name: str) -> None:
    """on_click: write a preset's option indices into widget session-state."""
    for qkey, idx in PRESETS[name].items():
        st.session_state[qkey] = idx

GROUP_KEYS = ["family", "home", "anx", "support", "schoolmix", "other"]
GROUP_FEATURES = {
    "family": ["ESCS", "HISCED", "HISEI", "SES_COMPLETE", "IMMIG"],
    "home": ["HOMEPOS", "MATERIAL_DEFICIT", "DIGITAL_READINESS", "DIGITAL_GAP", "ICTHOME", "ICTSCH"],
    "anx": ["ANXMAT", "SES_x_ANXIETY", "GENDER_x_ANXMAT"],
    "support": ["BELONG", "TEACHSUP", "BELONG_x_TEACHSUP"],
    "schoolmix": ["SCH_MEAN_ESCS", "SCH_MEAN_HOMEPOS", "SCH_MEAN_ANXMAT", "SCH_MEAN_TEACHSUP", "SCH_N"],
    "other": ["GENDER", "REPEAT", "GRADE", "GRADE_x_SES"],
}

# ---- translations -----------------------------------------------------------
TEXT = {
    "en": {
        "title": "Albania Student Maths-Risk Screener",
        "sub": ("Estimated chance a 15-year-old scores below the OECD basic-proficiency "
                "line in mathematics (PISA Level 2). Answer the questions; the "
                "estimate updates below."),
        "ctx": "PISA 2022 · Albania full cohort · survey-weighted CatBoost, calibrated",
        "kpi_cohort": "of the cohort scored below Level 2",
        "kpi_ceiling": "AUC, the ceiling background data can reach",
        "kpi_sample": "students in the 2022 sample",
        "pctile": "Higher estimated risk than <b>{p}%</b> of the 2022 cohort",
        "grad_title": "How risk tracks family background",
        "grad_cap": ("Average estimated risk by family socioeconomic status (ESCS), from the "
                     "poorest to the richest fifth of Albanian 15-year-olds. The gentle slope "
                     "IS the finding: in Albania family background predicts low proficiency "
                     "less strongly than in most PISA countries, opportunity is flatter, but "
                     "so is the ceiling. Your current answers sit in the highlighted fifth."),
        "grad_x": ["Poorest", "2nd", "Middle", "4th", "Richest"],
        "grad_you": "your answers",
        "grad_risk": "Estimated risk",
        "grad_group": "Family background (ESCS)",
        "perfactor_title": "Break the groups into individual factors",
        "perfactor_cap": ("The 12 strongest individual factors behind the grouped bars above, "
                          "for this profile. Same colours: blue lowers the estimate, red raises "
                          "it. Technical, these are model attributions, not causes."),
        "presets_title": "Quick-fill an illustrative profile",
        "presets_cap": "Illustrative composites for demonstration, not real students.",
        "preset_avg": "Cohort average", "preset_under": "Under-resourced", "preset_adv": "Advantaged",
        "calib_title": "Calibration, predicted vs actual",
        "calib_cap": ("Each dot is a band of students; the diagonal is perfect calibration, and "
                      "dot size ∝ the cohort share in that band. Sitting on the line is why the "
                      "reported probabilities can be read as real chances (weighted ECE "
                      "{er:.02f} → {ec:.02f} after calibration)."),
        "calib_pred": "Predicted risk", "calib_obs": "Actual share below Level 2",
        "footer": ("Source: OECD PISA 2022 (Albania). Associational screener from a single "
                   "cross-section, not causal, and it over-flags the poorest areas. No "
                   "individual student data is stored or shown; the profiles here are synthetic. "
                   "For triage and communication only."),
        "theme_hint": "Light or dark follows your device's system appearance setting.",
        "sec": {"about": "About the student", "home": "Home and family",
                "school": "School and feelings"},
        "q": {
            "sex": "Gender", "repeat": "Ever repeated a school year?",
            "immig": "Born abroad, or parents born abroad?",
            "books": "Books in the home", "paredu": "Highest parental education",
            "paroc": "Higher-earning parent's work",
            "homeict": "Quiet study space, computer, internet at home",
            "anx": "How the student feels about maths",
            "belong": "“I feel I belong at my school.”",
            "tsup": "“My maths teacher helps when I need it.”",
            "schict": "Computers & internet for students at school",
            "schsize": "School size", "schaff": "School & neighbourhood resources",
        },
        "opt": {
            "sex": ["Female", "Male"], "repeat": ["No", "Yes"],
            "immig": ["No, born in Albania", "One or both parents abroad", "Student born abroad"],
            "books": ["0-10", "11-25", "26-100", "101-200", "More than 200"],
            "paredu": ["Primary or less", "Lower secondary", "Upper secondary / vocational", "University degree+"],
            "paroc": ["Not currently working", "Manual or service work", "Skilled trade / clerical", "Professional / managerial"],
            "homeict": ["None of these", "One of these", "Two of these", "All three"],
            "anx": ["Relaxed and confident", "Mostly fine", "Sometimes anxious", "Often anxious and tense"],
            "belong": ["Strongly disagree", "Disagree", "Agree", "Strongly agree"],
            "tsup": ["Never", "Rarely", "Often", "Always"],
            "schict": ["Few or none", "Some", "Enough for most", "Plenty"],
            "schsize": ["Small", "Medium", "Large"],
            "schaff": ["Under-resourced / poorer area", "About average", "Well-resourced / wealthier area"],
        },
        "band_high": "Elevated risk, recommend extra support",
        "band_low": "Lower risk, monitor as usual",
        "note": "Cohort average {c}%. Screening line at {t}%. Calibrated estimate.",
        "drivers_title": "What is influencing this estimate",
        "drivers_cap": ("Each bar is a group of factors. Blue pulls the risk down, red "
                        "pushes it up; longer means stronger. The groups keep their order "
                        "so a single answer just moves its own bar."),
        "groups": {"family": "Family background", "home": "Home resources",
                   "anx": "Maths anxiety", "support": "Belonging & teacher support",
                   "schoolmix": "School's socioeconomic mix", "other": "Other background"},
        "lowers": "Lowers risk", "raises": "Raises risk",
        "ax_low": "lowers", "ax_high": "raises",
        "tt_group": "Factor group", "tt_dir": "Direction", "tt_str": "Strength",
        "howto_title": "How to read this",
        "howto": ("This tool flags students who may need help, it does not diagnose, "
                  "and it does not explain <em>why</em>. It learns patterns from one 2022 "
                  "survey, so an answer that lowers the estimate is <strong>not</strong> "
                  "proof that changing it would help a real student. Use it to widen "
                  "support, never to hold it back."),
        "tech_title": "Technical detail",
        "tech": ("- **Predicts:** probability of scoring below PISA mathematics Level 2 "
                 "(< {thr:.0f} points), the OECD low-performer line.\n"
                 "- **Model:** survey-weighted CatBoost on 13 student factors plus the "
                 "socioeconomic make-up of the school; isotonic-calibrated (ECE "
                 "{er:.02f} → {ec:.02f}).\n"
                 "- **Accuracy:** ~0.78 AUC, a real ceiling of what background "
                 "questions can tell us; treat borderline estimates as uncertain.\n"
                 "- **Fairness:** at this line the model over-flags students from the "
                 "poorest areas, use it only to add support."),
        "dl": "Download report card",
        "r_title": "Maths-Risk Screening Report",
        "r_gen": "Generated", "r_answers": "Answers", "r_infl": "Main influences",
        "r_result": "Estimated risk",
    },
    "sq": {
        "title": "Vlerësues i Riskut në Matematikë për Nxënësit Shqiptarë",
        "sub": ("Vlerësim i mundësisë që një 15-vjeçar të shënojë nën kufirin bazë "
                "të OECD-së në matematikë (PISA Niveli 2). Përgjigjuni pyetjeve; vlerësimi "
                "përditësohet më poshtë."),
        "ctx": "PISA 2022 · Kampioni i plotë shqiptar · CatBoost i peshuar, i kalibruar",
        "kpi_cohort": "e kampionit shënuan nën Nivelin 2",
        "kpi_ceiling": "AUC, kufiri që të dhënat e prejardhjes arrijnë",
        "kpi_sample": "nxënës në kampionin 2022",
        "pctile": "Risk i vlerësuar më i lartë se <b>{p}%</b> e kampionit 2022",
        "grad_title": "Si lidhet risku me prejardhjen familjare",
        "grad_cap": ("Risku mesatar i vlerësuar sipas statusit socio-ekonomik të familjes (ESCS), "
                     "nga e pesta më e varfër tek më e pasura e 15-vjeçarëve shqiptarë. Pjerrësia "
                     "e butë ËSHTË gjetja: në Shqipëri prejardhja familjare parashikon aftësinë e "
                     "ulët më dobët se në shumicën e vendeve PISA, mundësia është më e sheshtë, "
                     "por ashtu është edhe kufiri. Përgjigjet tuaja bien në të pestën e theksuar."),
        "grad_x": ["Më e varfër", "2-ta", "Mesatare", "4-ta", "Më e pasur"],
        "grad_you": "përgjigjet tuaja",
        "grad_risk": "Risku i vlerësuar",
        "grad_group": "Prejardhja familjare (ESCS)",
        "perfactor_title": "Ndaj grupet në faktorë individualë",
        "perfactor_cap": ("12 faktorët individualë më të fortë pas shtyllave të grupuara më sipër, "
                          "për këtë profil. Të njëjtat ngjyra: blu e ul vlerësimin, e kuqja e rrit. "
                          "Teknike, këto janë atribuime të modelit, jo shkaqe."),
        "presets_title": "Plotëso shpejt një profil ilustrues",
        "presets_cap": "Kombinime ilustruese për demonstrim, jo nxënës realë.",
        "preset_avg": "Mesatarja e kampionit", "preset_under": "Me pak burime", "preset_adv": "I favorizuar",
        "calib_title": "Kalibrimi, parashikuar kundrejt aktuales",
        "calib_cap": ("Çdo pikë është një brez nxënësish; diagonalja është kalibrim i përsosur, dhe "
                      "madhësia e pikës ∝ pjesa e kampionit në atë brez. Të qenit mbi vijë është "
                      "arsyeja pse probabilitetet mund të lexohen si shanse reale (ECE i peshuar "
                      "{er:.02f} → {ec:.02f} pas kalibrimit)."),
        "calib_pred": "Risku i parashikuar", "calib_obs": "Pjesa aktuale nën Nivelin 2",
        "footer": ("Burimi: OECD PISA 2022 (Shqipëri). Vlerësues asociativ nga një prerje e vetme, "
                   "jo shkakësor, dhe mbi-shënon zonat më të varfëra. Asnjë e dhënë individuale "
                   "nxënësi nuk ruhet apo shfaqet; profilet këtu janë sintetike. Vetëm për "
                   "triazh dhe komunikim."),
        "theme_hint": "E çelët ose e errët ndjek cilësimin e pamjes së sistemit të pajisjes suaj.",
        "sec": {"about": "Rreth nxënësit", "home": "Shtëpia dhe familja",
                "school": "Shkolla dhe ndjenjat"},
        "q": {
            "sex": "Gjinia", "repeat": "A ka përsëritur ndonjë vit shkollor?",
            "immig": "Lindur jashtë, apo prindërit lindur jashtë?",
            "books": "Libra në shtëpi", "paredu": "Arsimi më i lartë i prindërve",
            "paroc": "Puna e prindit me të ardhura më të larta",
            "homeict": "Vend i qetë studimi, kompjuter, internet në shtëpi",
            "anx": "Si ndihet nxënësi ndaj matematikës",
            "belong": "“Ndihem pjesë e shkollës sime.”",
            "tsup": "“Mësuesi i matematikës më ndihmon kur kam nevojë.”",
            "schict": "Kompjuterë & internet për nxënësit në shkollë",
            "schsize": "Madhësia e shkollës", "schaff": "Burimet e shkollës & lagjes",
        },
        "opt": {
            "sex": ["Femër", "Mashkull"], "repeat": ["Jo", "Po"],
            "immig": ["Jo, lindur në Shqipëri", "Një ose të dy prindërit jashtë", "Nxënësi lindur jashtë"],
            "books": ["0-10", "11-25", "26-100", "101-200", "Më shumë se 200"],
            "paredu": ["Fillor ose më pak", "Tetëvjeçar / i mesëm i ulët", "I mesëm / profesional", "Universitet e lart"],
            "paroc": ["Nuk punon aktualisht", "Punë manuale ose shërbimi", "Zeje / administratë", "Profesionist / drejtues"],
            "homeict": ["Asnjë prej tyre", "Një prej tyre", "Dy prej tyre", "Të tria"],
            "anx": ["I qetë dhe i sigurt", "Kryesisht mirë", "Ndonjëherë me ankth", "Shpesh me ankth e tension"],
            "belong": ["Nuk pajtohem fare", "Nuk pajtohem", "Pajtohem", "Pajtohem plotësisht"],
            "tsup": ["Kurrë", "Rrallë", "Shpesh", "Gjithmonë"],
            "schict": ["Pak ose aspak", "Disa", "Mjaft për shumicën", "Me bollëk"],
            "schsize": ["E vogël", "Mesatare", "E madhe"],
            "schaff": ["Me pak burime / zonë e varfër", "Mesatare", "Me shumë burime / zonë e pasur"],
        },
        "band_high": "Risk i lartë, rekomandohet mbështetje shtesë",
        "band_low": "Risk i ulët, monitoroni si zakonisht",
        "note": "Mesatarja e kampionit {c}%. Kufiri i vlerësimit në {t}%. Vlerësim i kalibruar.",
        "drivers_title": "Çfarë po ndikon në këtë vlerësim",
        "drivers_cap": ("Çdo shtyllë është një grup faktorësh. Blu e ul riskun, e kuqja e "
                        "rrit; sa më e gjatë, aq më e fortë. Grupet ruajnë renditjen që një "
                        "përgjigje të lëvizë vetëm shtyllën e vet."),
        "groups": {"family": "Prejardhja familjare", "home": "Burimet në shtëpi",
                   "anx": "Ankthi ndaj matematikës", "support": "Përkatësia & mbështetja e mësuesit",
                   "schoolmix": "Përbërja socio-ekonomike e shkollës", "other": "Prejardhje tjetër"},
        "lowers": "Ul riskun", "raises": "Rrit riskun",
        "ax_low": "ul", "ax_high": "rrit",
        "tt_group": "Grupi i faktorëve", "tt_dir": "Drejtimi", "tt_str": "Forca",
        "howto_title": "Si ta lexoni këtë",
        "howto": ("Ky mjet identifikon nxënës që mund të kenë nevojë për ndihmë, nuk "
                  "diagnostikon dhe nuk shpjegon <em>pse</em>. Mëson modele nga një anketë e "
                  "vetme e vitit 2022, prandaj një përgjigje që ul vlerësimin <strong>nuk</strong> "
                  "është provë se ndryshimi i saj do ta ndihmonte një nxënës real. Përdoreni "
                  "për të zgjeruar mbështetjen, kurrë për ta kufizuar."),
        "tech_title": "Detaje teknike",
        "tech": ("- **Parashikon:** probabilitetin e shënimit nën PISA Niveli 2 "
                 "(< {thr:.0f} pikë), kufirin e OECD-së për performuesit e ulët.\n"
                 "- **Modeli:** CatBoost i peshuar me peshat e anketës mbi 13 faktorë të "
                 "nxënësit plus përbërjen socio-ekonomike të shkollës; i kalibruar me izotoni "
                 "(ECE {er:.02f} → {ec:.02f}).\n"
                 "- **Saktësia:** ~0.78 AUC, një kufi real i asaj që pyetjet mbi "
                 "prejardhjen mund të tregojnë; vlerësimet kufitare janë të pasigurta.\n"
                 "- **Drejtësia:** në këtë kufi modeli mbi-shënon nxënësit nga zonat më të "
                 "varfëra, përdoreni vetëm për të shtuar mbështetje."),
        "dl": "Shkarko raportin",
        "r_title": "Raport i Vlerësimit të Riskut në Matematikë",
        "r_gen": "Krijuar", "r_answers": "Përgjigjet", "r_infl": "Ndikimet kryesore",
        "r_result": "Risku i vlerësuar",
    },
}


def build_row(T):
    """Render the bilingual questionnaire; return (model row, [(q, answer), ...])."""
    vals = {f: 0.0 for f in feature_names}
    for f, s in sliders.items():
        vals[f] = float(s["default"])

    # preset quick-fill row (mutates widget state before the selectboxes read it)
    with st.container(border=True):
        st.markdown(f"<div class='card-h'>{T['presets_title']}</div>", unsafe_allow_html=True)
        pcols = st.columns(3)
        for col, name in zip(pcols, ("avg", "under", "adv")):
            col.button(T[f"preset_{name}"], key=f"preset_btn_{name}", width="stretch",
                       on_click=apply_preset, args=(name,))
        st.caption(T["presets_cap"])

    selections: list[tuple[str, str]] = []
    for sec_id, qs in QMETA:
        with st.container(border=True):
            st.markdown(f"<div class='card-h'>{T['sec'][sec_id]}</div>", unsafe_allow_html=True)
            cols = st.columns(2)
            for i, (qkey, opts) in enumerate(qs):
                labels = T["opt"][qkey]
                st.session_state.setdefault(qkey, DEFAULT_IDX[qkey])
                with cols[i % 2]:
                    # value driven by session_state[qkey] (so presets can set it)
                    idx = st.selectbox(T["q"][qkey], range(len(opts)), key=qkey,
                                       format_func=lambda j, L=labels: L[j])
                for feat, akey, raw in opts[idx]:
                    if feat in vals:
                        vals[feat] = raw if raw is not None else anchor(feat, akey)
                selections.append((T["q"][qkey], labels[idx]))
    return pd.DataFrame([[vals[f] for f in feature_names]], columns=feature_names), selections


def predict(row):
    raw = float(pipe.predict_proba(row)[0, 1])
    return float(iso.predict([raw])[0])


# Human labels for the engineered features shown in the per-factor expander.
FEATURE_LABELS = {
    "en": {
        "ESCS": "Family socioeconomic status", "HOMEPOS": "Home possessions",
        "HISCED": "Parental education", "HISEI": "Parental occupation status",
        "SES_COMPLETE": "Family-background completeness", "IMMIG": "Immigration status",
        "MATERIAL_DEFICIT": "Material deficit", "DIGITAL_READINESS": "Digital readiness",
        "DIGITAL_GAP": "Home–school digital gap", "ICTHOME": "Devices at home",
        "ICTSCH": "Devices at school", "ANXMAT": "Maths anxiety",
        "SES_x_ANXIETY": "SES × anxiety", "GENDER_x_ANXMAT": "Gender × anxiety",
        "BELONG": "Sense of belonging", "TEACHSUP": "Teacher support",
        "BELONG_x_TEACHSUP": "Belonging × support", "SCH_MEAN_ESCS": "School-mean SES",
        "SCH_MEAN_HOMEPOS": "School-mean possessions", "SCH_MEAN_ANXMAT": "School-mean anxiety",
        "SCH_MEAN_TEACHSUP": "School-mean support", "SCH_N": "School sample size",
        "GENDER": "Gender", "REPEAT": "Repeated a grade", "GRADE": "Grade vs modal",
        "GRADE_x_SES": "Grade × SES",
    },
    "sq": {
        "ESCS": "Statusi socio-ekonomik", "HOMEPOS": "Zotërime në shtëpi",
        "HISCED": "Arsimi i prindërve", "HISEI": "Statusi profesional i prindërve",
        "SES_COMPLETE": "Plotësia e prejardhjes", "IMMIG": "Statusi i emigracionit",
        "MATERIAL_DEFICIT": "Mungesa materiale", "DIGITAL_READINESS": "Gatishmëria dixhitale",
        "DIGITAL_GAP": "Hendeku dixhital shtëpi–shkollë", "ICTHOME": "Pajisje në shtëpi",
        "ICTSCH": "Pajisje në shkollë", "ANXMAT": "Ankthi ndaj matematikës",
        "SES_x_ANXIETY": "SES × ankth", "GENDER_x_ANXMAT": "Gjini × ankth",
        "BELONG": "Ndjenja e përkatësisë", "TEACHSUP": "Mbështetja e mësuesit",
        "BELONG_x_TEACHSUP": "Përkatësi × mbështetje", "SCH_MEAN_ESCS": "SES mesatar i shkollës",
        "SCH_MEAN_HOMEPOS": "Zotërime mesatare të shkollës", "SCH_MEAN_ANXMAT": "Ankth mesatar i shkollës",
        "SCH_MEAN_TEACHSUP": "Mbështetje mesatare e shkollës", "SCH_N": "Madhësia e kampionit të shkollës",
        "GENDER": "Gjinia", "REPEAT": "Ka përsëritur klasë", "GRADE": "Klasa ndaj modales",
        "GRADE_x_SES": "Klasa × SES",
    },
}


def grouped_shap(row, T):
    """Return (grouped-driver frame, per-feature SHAP series) for one profile."""
    model = pipe.named_steps["model"]
    eng = pipe.named_steps["engineer"].transform(row)
    imp = pipe.named_steps["imputer"].transform(eng)
    sv = pd.Series(model.get_feature_importance(Pool(imp), type="ShapValues")[0][:-1],
                   index=shap_names)
    rows = [{"key": k, "group": T["groups"][k],
             "value": float(sv[[c for c in GROUP_FEATURES[k] if c in sv.index]].sum())}
            for k in GROUP_KEYS]
    return pd.DataFrame(rows), sv


def _tx(s: str) -> str:
    """Fold smart punctuation to ASCII so the PDF core font (cp1252) never chokes;
    Albanian ë/ç stay (they are in the encoding)."""
    for a, b in [("—", "-"), ("–", "-"), ("“", '"'), ("”", '"'),
                 ("‘", "'"), ("’", "'"), ("→", "->")]:
        s = s.replace(a, b)
    return s


def _rgb(hexstr: str):
    h = hexstr.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def build_report_pdf(T, risk, band_text, band_color, selections, drivers) -> bytes:
    """A one-page printable PDF report card (fpdf2, pure-Python, no system deps)."""
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    pdf.set_text_color(11, 11, 11)
    pdf.set_font("helvetica", "B", 18)
    # multi_cell leaves the x-cursor at the right of the cell; new_x=LMARGIN snaps
    # it back so the following full-width cell isn't computed with ~zero width and
    # pushed off the page.
    pdf.multi_cell(0, 9, _tx(T["r_title"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 6, f"{_tx(T['r_gen'])}: {datetime.now():%Y-%m-%d %H:%M}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, _tx(T["r_result"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*_rgb(band_color))
    pdf.set_font("helvetica", "B", 40)
    pdf.cell(0, 16, f"{risk*100:.0f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 7, _tx(band_text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_text_color(11, 11, 11)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, _tx(T["r_answers"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    with pdf.table(col_widths=(62, 38), text_align=("LEFT", "LEFT")) as table:
        for q, a in selections:
            r = table.row(); r.cell(_tx(q)); r.cell(_tx(a))
    pdf.ln(3)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, _tx(T["r_infl"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    dsorted = drivers.reindex(drivers["value"].abs().sort_values(ascending=False).index)
    with pdf.table(col_widths=(58, 27, 15), text_align=("LEFT", "LEFT", "RIGHT")) as table:
        for r_ in dsorted.itertuples():
            r = table.row()
            r.cell(_tx(r_.group))
            r.cell(_tx(T["raises"] if r_.value > 0 else T["lowers"]))
            r.cell(f"{abs(r_.value):.2f}")
    pdf.ln(3)

    caveat = (T["howto"].replace("<em>", "").replace("</em>", "")
              .replace("<strong>", "").replace("</strong>", ""))
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 4.5, _tx(caveat))
    return bytes(pdf.output())


# ---- language selector ------------------------------------------------------
lang_choice = st.radio("lang", ["English", "Shqip"], horizontal=True,
                       label_visibility="collapsed")
lang = "sq" if lang_choice == "Shqip" else "en"
T = TEXT[lang]

# Theme follows Streamlit's OWN light/dark (user switches via the top-right ☰
# menu, which themes native widgets too). We read the active theme and match the
# chart / card palette to it. Headless / first paint may report None -> light.
try:
    theme_type = st.context.theme.type or "light"
except Exception:
    theme_type = "light"

# bind active palette -> css + module-level chart colours (used by Altair below)
P = PALETTES["dark" if theme_type == "dark" else "light"]
st.markdown(build_css(P), unsafe_allow_html=True)
st.caption(T["theme_hint"])
C_RISK, C_SAFE = P["risk"], P["safe"]
C_INK, C_SECOND, C_MUTED, C_TRACK, C_GRID = (
    P["ink"], P["second"], P["muted"], P["track"], P["grid"])

# ---- header: title, provenance strip, KPI tiles -----------------------------
st.markdown(f"<div class='app-title'>{T['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='app-sub'>{T['sub']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='app-ctx'>{T['ctx']}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='kpi-row'>"
    f"<div class='kpi'><div class='kpi-val'>{base_rate*100:.0f}%</div>"
    f"<div class='kpi-lab'>{T['kpi_cohort']}</div></div>"
    f"<div class='kpi'><div class='kpi-val'>0.78</div>"
    f"<div class='kpi-lab'>{T['kpi_ceiling']}</div></div>"
    f"<div class='kpi'><div class='kpi-val'>{meta['n_samples']:,}</div>"
    f"<div class='kpi-lab'>{T['kpi_sample']}</div></div>"
    "</div>", unsafe_allow_html=True)

row, selections = build_row(T)
risk = predict(row)
flagged = risk >= threshold
band_color = C_RISK if flagged else C_SAFE
band_text = T["band_high"] if flagged else T["band_low"]

# ---- result card ------------------------------------------------------------
with st.container(border=True):
    pct = cohort_percentile(risk)
    pct_html = (f"<div class='risk-pct'>{T['pctile'].format(p=pct)}</div>"
                if pct >= 0 else "")
    st.markdown(
        f"<div class='risk-num' style='color:{band_color}'>{risk*100:.0f}%</div>"
        f"<div class='risk-band' style='color:{band_color}'>{band_text}</div>"
        f"{pct_html}"
        f"<div class='risk-note'>{T['note'].format(c=f'{base_rate*100:.0f}', t=f'{threshold*100:.0f}')}</div>",
        unsafe_allow_html=True)
    axis = alt.Axis(format="%", title=None, values=[0, .25, .5, .75, 1],
                    domain=False, ticks=False, labelColor=C_MUTED)
    base = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_bar(
        cornerRadius=7, color=C_TRACK, height=18).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0, 1]), axis=axis))
    fill = alt.Chart(pd.DataFrame({"x": [risk]})).mark_bar(
        cornerRadius=7, color=band_color, height=18).encode(x="x:Q")
    rule = alt.Chart(pd.DataFrame({"x": [threshold]})).mark_rule(
        color=C_SECOND, strokeDash=[4, 3], size=1.3).encode(x="x:Q")
    st.altair_chart((base + fill + rule).properties(height=44, background="transparent")
                    .configure_view(strokeWidth=0), use_container_width=True)

# ---- drivers (Altair, fixed order + fixed colours -> stable) ----------------
with st.container(border=True):
    st.markdown(f"<div class='card-h'>{T['drivers_title']}</div>", unsafe_allow_html=True)
    st.caption(T["drivers_cap"])
    df, sv = grouped_shap(row, T)
    order = [T["groups"][k] for k in GROUP_KEYS]
    dfx = df.assign(dir=np.where(df["value"] >= 0, T["raises"], T["lowers"]),
                    mag=df["value"].abs().round(2), v=df["value"].clip(-2.5, 2.5))
    label_expr = f"datum.value < 0 ? '{T['ax_low']}' : datum.value > 0 ? '{T['ax_high']}' : ''"
    bars = alt.Chart(dfx).mark_bar(cornerRadius=5, height=22).encode(
        x=alt.X("v:Q", scale=alt.Scale(domain=[-2.5, 2.5]),
                axis=alt.Axis(title=None, values=[-2, 0, 2], labelExpr=label_expr,
                              domain=False, ticks=False, grid=True,
                              gridColor=C_GRID, labelColor=C_MUTED)),
        y=alt.Y("group:N", sort=order, axis=alt.Axis(title=None, domain=False, ticks=False,
                labelColor=C_INK, labelFontSize=12, labelPadding=8)),
        color=alt.Color("dir:N", scale=alt.Scale(domain=[T["lowers"], T["raises"]],
                        range=[C_SAFE, C_RISK]), legend=None),
        tooltip=[alt.Tooltip("group:N", title=T["tt_group"]),
                 alt.Tooltip("dir:N", title=T["tt_dir"]),
                 alt.Tooltip("mag:Q", title=T["tt_str"])])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=C_SECOND, size=1).encode(x="x:Q")
    st.altair_chart((bars + zero).properties(height=250, background="transparent")
                    .configure_view(strokeWidth=0), use_container_width=True)

    # per-factor breakdown (technical): the individual features behind the groups
    with st.expander(T["perfactor_title"]):
        st.caption(T["perfactor_cap"])
        flabels = FEATURE_LABELS[lang]
        pf = pd.DataFrame({
            "feat": sv.index,
            "label": [flabels.get(f, f.replace("_", " ")) for f in sv.index],
            "value": sv.to_numpy(float),
        })
        pf = pf[pf["value"].abs() > 1e-4].copy()
        pf = pf.reindex(pf["value"].abs().sort_values(ascending=False).index).head(12)
        pf["dir"] = np.where(pf["value"] >= 0, T["raises"], T["lowers"])
        lim = float(pf["value"].abs().max()) * 1.15 or 1.0
        pbars = alt.Chart(pf).mark_bar(cornerRadius=4, height=16).encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[-lim, lim]), axis=alt.Axis(
                title=None, labelExpr=label_expr, domain=False, ticks=False, grid=True,
                gridColor=C_GRID, labelColor=C_MUTED)),
            y=alt.Y("label:N", sort=list(pf["label"]), axis=alt.Axis(
                title=None, domain=False, ticks=False, labelColor=C_INK, labelPadding=6)),
            color=alt.Color("dir:N", scale=alt.Scale(
                domain=[T["lowers"], T["raises"]], range=[C_SAFE, C_RISK]), legend=None),
            tooltip=[alt.Tooltip("label:N", title=T["tt_group"]),
                     alt.Tooltip("dir:N", title=T["tt_dir"]),
                     alt.Tooltip("value:Q", title=T["tt_str"], format="+.2f")])
        pzero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
            color=C_SECOND, size=1).encode(x="x:Q")
        st.altair_chart((pbars + pzero).properties(
            height=max(120, 26 * len(pf)), background="transparent").configure_view(
            strokeWidth=0), use_container_width=True)

# ---- SES-risk gradient (headline: Albania's is among the flattest) ----------
if ses_gradient and all(v is not None for v in ses_gradient):
    with st.container(border=True):
        st.markdown(f"<div class='card-h'>{T['grad_title']}</div>", unsafe_allow_html=True)
        st.caption(T["grad_cap"])
        your_q = escs_quintile(float(row["ESCS"].iloc[0]))
        gdf = pd.DataFrame({"label": T["grad_x"], "risk": ses_gradient,
                            "you": [i == your_q for i in range(5)]})
        gbars = alt.Chart(gdf).mark_bar(cornerRadius=4, size=40).encode(
            x=alt.X("label:N", sort=T["grad_x"], axis=alt.Axis(
                title=None, domain=False, ticks=False, labelAngle=0,
                labelColor=C_INK, labelFontSize=12, labelPadding=8)),
            y=alt.Y("risk:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(
                format="%", title=None, values=[0, .25, .5, .75, 1], domain=False,
                ticks=False, grid=True, gridColor=C_GRID, labelColor=C_MUTED)),
            color=alt.Color("you:N", scale=alt.Scale(
                domain=[False, True], range=[C_SECOND, C_SAFE]), legend=None),
            tooltip=[alt.Tooltip("label:N", title=T["grad_group"]),
                     alt.Tooltip("risk:Q", title=T["grad_risk"], format=".0%")])
        gtext = alt.Chart(gdf).mark_text(
            dy=-8, color=C_INK, fontSize=12, fontWeight=600).encode(
            x=alt.X("label:N", sort=T["grad_x"]), y="risk:Q",
            text=alt.Text("risk:Q", format=".0%"))
        st.altair_chart((gbars + gtext).properties(
            height=260, background="transparent").configure_view(strokeWidth=0),
            use_container_width=True)

# ---- how to read (neutral note card) ----------------------------------------
with st.container(border=True):
    st.markdown(f"<div class='card-h'>{T['howto_title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='note-card'>{T['howto']}</div>", unsafe_allow_html=True)
    with st.expander(T["tech_title"]):
        st.markdown(T["tech"].format(thr=meta["oecd_threshold_math"],
                                     er=meta["ece_raw"], ec=meta["ece_calibrated"]))
        rc = meta.get("reliability_curve", [])
        if rc:
            st.markdown(f"**{T['calib_title']}**")
            st.caption(T["calib_cap"].format(er=meta["ece_raw"], ec=meta["ece_calibrated"]))
            cdf = pd.DataFrame(rc)
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
                color=C_MUTED, strokeDash=[4, 3], size=1).encode(x="x:Q", y="y:Q")
            dots = alt.Chart(cdf).mark_circle(color=C_SAFE, opacity=0.85).encode(
                x=alt.X("pred:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(
                    format="%", title=T["calib_pred"], values=[0, .25, .5, .75, 1],
                    domain=False, ticks=False, gridColor=C_GRID, labelColor=C_MUTED,
                    titleColor=C_SECOND)),
                y=alt.Y("obs:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(
                    format="%", title=T["calib_obs"], values=[0, .25, .5, .75, 1],
                    domain=False, ticks=False, gridColor=C_GRID, labelColor=C_MUTED,
                    titleColor=C_SECOND)),
                size=alt.Size("frac:Q", scale=alt.Scale(range=[30, 500]), legend=None),
                tooltip=[alt.Tooltip("pred:Q", title=T["calib_pred"], format=".0%"),
                         alt.Tooltip("obs:Q", title=T["calib_obs"], format=".0%")])
            st.altair_chart((diag + dots).properties(
                height=300, background="transparent").configure_view(strokeWidth=0),
                use_container_width=True)

# ---- download report card ---------------------------------------------------
st.download_button(
    T["dl"],
    data=build_report_pdf(T, risk, band_text, band_color, selections, df),
    file_name=f"maths_risk_report_{lang}.pdf", mime="application/pdf",
    use_container_width=True)

# ---- provenance / license footer --------------------------------------------
st.markdown(
    f"<div class='app-ctx' style='margin-top:1.6rem; text-transform:none; "
    f"letter-spacing:0; line-height:1.5;'>{T['footer']}</div>",
    unsafe_allow_html=True)
