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

# ---- palette (validated diverging pair blue<->red + neutral) ----------------
C_RISK = "#d03b3b"      # raises risk
C_SAFE = "#2a78d6"      # lowers risk
C_INK = "#0b0b0b"
C_SECOND = "#52514e"
C_MUTED = "#8a9099"
C_TRACK = "#e3e8ef"

st.set_page_config(page_title="Albania PISA Risk Screener",
                   page_icon=None, layout="centered")

st.markdown(
    """
    <style>
      .stApp { background: #eef2f8; }              /* cool, bright */
      .block-container { max-width: 780px; padding-top: 1.4rem; padding-bottom: 4rem; }
      #MainMenu, footer, header { visibility: hidden; }
      h1,h2,h3 { letter-spacing:-0.01em; }
      .app-title { text-align:center; font-weight:700; font-size:1.95rem; margin-bottom:.15rem; color:#0b0b0b; }
      .app-sub { text-align:center; color:#52514e; font-size:1rem; margin:0 auto 1.3rem; max-width:35rem; }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background:#ffffff; border:1px solid rgba(30,41,59,.08); border-radius:16px;
        padding:1.15rem 1.3rem; box-shadow:0 1px 4px rgba(30,41,59,.05); margin-bottom:1rem;
      }
      .card-h { font-weight:650; font-size:1.02rem; color:#0b0b0b; margin:.1rem 0 .6rem; }
      .risk-num { text-align:center; font-size:3.6rem; font-weight:700; line-height:1; }
      .risk-band { text-align:center; font-size:1.08rem; font-weight:600; margin-top:.35rem; }
      .risk-note { text-align:center; color:#8a9099; font-size:.85rem; margin-top:.55rem; }
      .note-card { background:#eef4fc; border-left:4px solid #2a78d6; border-radius:10px;
                   padding:.85rem 1.05rem; color:#33414f; font-size:.92rem; line-height:1.5; }
      .stSelectbox label p { font-size:.9rem !important; color:#33322f !important; }
      div[data-testid="stRadio"] { display:flex; justify-content:flex-end; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        "band_high": "Elevated risk — recommend extra support",
        "band_low": "Lower risk — monitor as usual",
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
        "howto": ("This tool flags students who may need help — it does not diagnose, "
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
                 "- **Accuracy:** ~0.78 AUC — a real ceiling of what background "
                 "questions can tell us; treat borderline estimates as uncertain.\n"
                 "- **Fairness:** at this line the model over-flags students from the "
                 "poorest areas — use it only to add support."),
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
        "band_high": "Risk i lartë — rekomandohet mbështetje shtesë",
        "band_low": "Risk i ulët — monitoroni si zakonisht",
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
        "howto": ("Ky mjet identifikon nxënës që mund të kenë nevojë për ndihmë — nuk "
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
                 "- **Saktësia:** ~0.78 AUC — një kufi real i asaj që pyetjet mbi "
                 "prejardhjen mund të tregojnë; vlerësimet kufitare janë të pasigurta.\n"
                 "- **Drejtësia:** në këtë kufi modeli mbi-shënon nxënësit nga zonat më të "
                 "varfëra — përdoreni vetëm për të shtuar mbështetje."),
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
    selections: list[tuple[str, str]] = []
    for sec_id, qs in QMETA:
        with st.container(border=True):
            st.markdown(f"<div class='card-h'>{T['sec'][sec_id]}</div>", unsafe_allow_html=True)
            cols = st.columns(2)
            for i, (qkey, opts) in enumerate(qs):
                labels = T["opt"][qkey]
                with cols[i % 2]:
                    idx = st.selectbox(T["q"][qkey], range(len(opts)),
                                       index=DEFAULT_IDX[qkey], key=qkey,
                                       format_func=lambda j, L=labels: L[j])
                for feat, akey, raw in opts[idx]:
                    if feat in vals:
                        vals[feat] = raw if raw is not None else anchor(feat, akey)
                selections.append((T["q"][qkey], labels[idx]))
    return pd.DataFrame([[vals[f] for f in feature_names]], columns=feature_names), selections


def predict(row):
    raw = float(pipe.predict_proba(row)[0, 1])
    return float(iso.predict([raw])[0])


def grouped_shap(row, T):
    model = pipe.named_steps["model"]
    eng = pipe.named_steps["engineer"].transform(row)
    imp = pipe.named_steps["imputer"].transform(eng)
    sv = pd.Series(model.get_feature_importance(Pool(imp), type="ShapValues")[0][:-1],
                   index=shap_names)
    rows = [{"key": k, "group": T["groups"][k],
             "value": float(sv[[c for c in GROUP_FEATURES[k] if c in sv.index]].sum())}
            for k in GROUP_KEYS]
    return pd.DataFrame(rows)


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


# ---- language selector + header ---------------------------------------------
lang_choice = st.radio("lang", ["English", "Shqip"], horizontal=True,
                       label_visibility="collapsed")
lang = "sq" if lang_choice == "Shqip" else "en"
T = TEXT[lang]

st.markdown(f"<div class='app-title'>{T['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='app-sub'>{T['sub']}</div>", unsafe_allow_html=True)

row, selections = build_row(T)
risk = predict(row)
flagged = risk >= threshold
band_color = C_RISK if flagged else C_SAFE
band_text = T["band_high"] if flagged else T["band_low"]

# ---- result card ------------------------------------------------------------
with st.container(border=True):
    st.markdown(
        f"<div class='risk-num' style='color:{band_color}'>{risk*100:.0f}%</div>"
        f"<div class='risk-band' style='color:{band_color}'>{band_text}</div>"
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
    st.altair_chart((base + fill + rule).properties(height=44)
                    .configure_view(strokeWidth=0), use_container_width=True)

# ---- drivers (Altair, fixed order + fixed colours -> stable) ----------------
with st.container(border=True):
    st.markdown(f"<div class='card-h'>{T['drivers_title']}</div>", unsafe_allow_html=True)
    st.caption(T["drivers_cap"])
    df = grouped_shap(row, T)
    order = [T["groups"][k] for k in GROUP_KEYS]
    dfx = df.assign(dir=np.where(df["value"] >= 0, T["raises"], T["lowers"]),
                    mag=df["value"].abs().round(2), v=df["value"].clip(-2.5, 2.5))
    label_expr = f"datum.value < 0 ? '{T['ax_low']}' : datum.value > 0 ? '{T['ax_high']}' : ''"
    bars = alt.Chart(dfx).mark_bar(cornerRadius=5, height=22).encode(
        x=alt.X("v:Q", scale=alt.Scale(domain=[-2.5, 2.5]),
                axis=alt.Axis(title=None, values=[-2, 0, 2], labelExpr=label_expr,
                              domain=False, ticks=False, grid=True,
                              gridColor="#e9edf3", labelColor=C_MUTED)),
        y=alt.Y("group:N", sort=order, axis=alt.Axis(title=None, domain=False, ticks=False,
                labelColor=C_INK, labelFontSize=12, labelPadding=8)),
        color=alt.Color("dir:N", scale=alt.Scale(domain=[T["lowers"], T["raises"]],
                        range=[C_SAFE, C_RISK]), legend=None),
        tooltip=[alt.Tooltip("group:N", title=T["tt_group"]),
                 alt.Tooltip("dir:N", title=T["tt_dir"]),
                 alt.Tooltip("mag:Q", title=T["tt_str"])])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=C_SECOND, size=1).encode(x="x:Q")
    st.altair_chart((bars + zero).properties(height=250)
                    .configure_view(strokeWidth=0), use_container_width=True)

# ---- how to read (neutral note card) ----------------------------------------
with st.container(border=True):
    st.markdown(f"<div class='card-h'>{T['howto_title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='note-card'>{T['howto']}</div>", unsafe_allow_html=True)
    with st.expander(T["tech_title"]):
        st.markdown(T["tech"].format(thr=meta["oecd_threshold_math"],
                                     er=meta["ece_raw"], ec=meta["ece_calibrated"]))

# ---- download report card ---------------------------------------------------
st.download_button(
    T["dl"],
    data=build_report_pdf(T, risk, band_text, band_color, selections, df),
    file_name=f"maths_risk_report_{lang}.pdf", mime="application/pdf",
    use_container_width=True)
