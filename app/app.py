import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ObesoScan AI — Obesity Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 2. CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fraunces:opsz,wght@9..144,300;9..144,600;9..144,700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #F0F4F8 !important;
    color: #1A202C !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] .block-container { padding-top: 0 !important; }

.brand-wrap {
    background: linear-gradient(135deg, #1E3A5F 0%, #2D5986 100%);
    padding: 28px 24px 24px; margin: -1rem -1rem 0;
}
.brand-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.brand-name {
    font-family: 'Fraunces', serif !important;
    font-size: 22px !important; font-weight: 700 !important;
    color: #FFFFFF !important; line-height: 1.1;
}
.brand-tagline {
    font-size: 11px !important; color: rgba(255,255,255,0.55) !important;
    letter-spacing: 0.8px; text-transform: uppercase; margin-top: 2px;
}
.brand-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
    color: rgba(255,255,255,0.8); font-size: 10px; font-weight: 600;
    padding: 4px 10px; border-radius: 20px;
}
.sidebar-body { padding: 20px 16px; }
.info-block {
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 16px; margin-bottom: 14px;
}
.info-block-title {
    font-size: 11px; font-weight: 700; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px;
}
.info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid #F1F5F9; font-size: 12px;
}
.info-row:last-child { border-bottom: none; }
.info-row-label { color: #94A3B8; font-weight: 500; }
.info-row-value { color: #1E293B; font-weight: 700; }
.tag-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.tag { font-size: 11px; font-weight: 600; padding: 3px 9px; border-radius: 6px; }
.tag-blue  { background: #EFF6FF; color: #1D4ED8; }
.tag-green { background: #F0FDF4; color: #15803D; }
.tag-amber { background: #FFFBEB; color: #B45309; }
.tag-rose  { background: #FFF1F2; color: #BE123C; }
.sidebar-disclaimer {
    margin-top: 16px; padding: 12px 14px;
    background: #FFF7ED; border: 1px solid #FED7AA;
    border-radius: 10px; font-size: 11px; color: #92400E; line-height: 1.6;
}
.sidebar-footer { margin-top: 16px; text-align: center; font-size: 10px; color: #CBD5E1; }

/* ══ MAIN ══ */
.main .block-container { padding: 2.5rem 3rem !important; max-width: 1200px !important; }

/* ══ HERO ══ */
.hero {
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 50%, #0EA5E9 100%);
    border-radius: 20px; padding: 36px 40px; margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 240px; height: 240px; border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.hero-top { display: flex; align-items: center; gap: 18px; margin-bottom: 14px; }
.hero-icon {
    width: 56px; height: 56px; background: rgba(255,255,255,0.15);
    border-radius: 14px; display: flex; align-items: center; justify-content: center;
    font-size: 26px; border: 1px solid rgba(255,255,255,0.2); flex-shrink: 0;
}
.hero-title {
    font-family: 'Fraunces', serif !important; font-size: 30px !important;
    font-weight: 700 !important; color: #FFFFFF !important; line-height: 1.15 !important;
}
.hero-subtitle {
    font-size: 14px !important; color: rgba(255,255,255,0.7) !important;
    line-height: 1.55 !important; max-width: 600px;
}
.hero-chips { display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap; }
.hero-chip {
    font-size: 11px; font-weight: 600; padding: 5px 12px; border-radius: 20px;
    background: rgba(255,255,255,0.15); color: rgba(255,255,255,0.9);
    border: 1px solid rgba(255,255,255,0.2);
}

/* ══ SECTION CARDS ══ */
.section-card {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 16px; padding: 24px 26px; margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.sec-label {
    font-size: 11px; font-weight: 700; color: #64748B;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 16px; display: flex; align-items: center; gap: 8px;
}
.sec-label::after { content: ''; flex: 1; height: 1px; background: #E2E8F0; }

/* ══ INPUTS — labels foncés visibles ══ */
div[data-testid="stNumberInput"],
div[data-testid="stSelectbox"],
div[data-testid="stSlider"] {
    background: #F8FAFC !important;
    padding: 12px 14px !important; border-radius: 12px !important;
    border: 1.5px solid #E2E8F0 !important; margin-bottom: 10px !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stNumberInput"]:focus-within,
div[data-testid="stSelectbox"]:focus-within {
    border-color: #2563EB !important; background: #FFFFFF !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
.main [data-testid="stWidgetLabel"] p,
.main .stSlider label,
.main .stSelectbox label,
.main .stNumberInput label {
    color: #1E293B !important; font-size: 12px !important;
    font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
div[data-testid="stNumberInput"] input {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    color: #0F172A !important; border-radius: 8px !important;
    font-size: 15px !important; font-weight: 600 !important;
}
/* ══ SELECTBOX boîte — fond blanc, texte foncé ══ */
div[data-testid="stSelectbox"] * {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] label * {
    background-color: transparent !important;
    color: #1E293B !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] span,
div[data-testid="stSelectbox"] [data-baseweb="select"] div {
    background: #FFFFFF !important;
    color: #0F172A !important;
    font-weight: 600 !important;
}

/* ══ DROPDOWN liste déroulante — fond blanc ══ */
[data-baseweb="popover"] { background: #FFFFFF !important; }
[data-baseweb="popover"] * { background-color: #FFFFFF !important; color: #0F172A !important; }
[data-baseweb="menu"]    { background: #FFFFFF !important; border: 1.5px solid #E2E8F0 !important; border-radius: 10px !important; box-shadow: 0 8px 24px rgba(0,0,0,0.10) !important; }
[data-baseweb="menu"] *  { background-color: #FFFFFF !important; color: #0F172A !important; }

/* chaque option individuelle */
[role="option"]                        { background: #FFFFFF !important; color: #0F172A !important; font-size: 13px !important; font-weight: 500 !important; }
[role="option"] *                      { background: #FFFFFF !important; color: #0F172A !important; }
[role="option"]:hover                  { background: #EFF6FF !important; color: #1D4ED8 !important; }
[role="option"]:hover *                { background: #EFF6FF !important; color: #1D4ED8 !important; }
[role="option"][aria-selected="true"]  { background: #DBEAFE !important; color: #1D4ED8 !important; font-weight: 700 !important; }
li[role="option"]                      { background: #FFFFFF !important; color: #0F172A !important; }
li[role="option"]:hover                { background: #EFF6FF !important; color: #1D4ED8 !important; }
[data-testid="stSlider"] [data-testid="stSliderTrackFill"] {
    background: linear-gradient(90deg,#2563EB,#0EA5E9) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #2563EB !important; border: 2px solid #fff !important;
    box-shadow: 0 1px 4px rgba(37,99,235,0.4) !important;
}
[data-testid="stSlider"] p { color: #374151 !important; font-weight: 600 !important; }

/* ══ BMI CARD ══ */
.bmi-gauge-card {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 14px; padding: 20px 22px; height: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.bmi-gauge-title { font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }
.bmi-value-big { font-family: 'Fraunces', serif; font-size: 42px; font-weight: 700; line-height: 1; margin-bottom: 4px; }
.bmi-unit { font-size: 13px; color: #94A3B8; font-weight: 500; margin-bottom: 12px; }
.bmi-bar-track {
    width: 100%; height: 8px; border-radius: 99px;
    background: linear-gradient(90deg,#60A5FA 0%,#34D399 22%,#FBBF24 50%,#F87171 75%,#DC2626 100%);
    position: relative; margin-bottom: 6px;
}
.bmi-bar-label { display: flex; justify-content: space-between; font-size: 9px; color: #CBD5E1; font-weight: 600; }

/* ══ ACTIVITY CARD ══ */
.activity-card {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 14px; padding: 20px 22px; height: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ══ BOUTON ══ */
div.stButton > button {
    background: linear-gradient(135deg, #1E3A5F, #2563EB) !important;
    color: #FFFFFF !important; border-radius: 12px !important;
    padding: 15px 28px !important; font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important; font-size: 15px !important;
    border: none !important; width: 100% !important; margin-top: 22px !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important; transition: all 0.2s !important;
}
div.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(37,99,235,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ══ RESULT ══ */
.result-banner {
    border-radius: 16px; padding: 28px 32px; margin-top: 28px;
    display: flex; align-items: center; gap: 24px;
    position: relative; overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
}
.result-emoji { font-size: 44px; flex-shrink: 0; }
.result-lbl { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; opacity: 0.75; margin-bottom: 5px; }
.result-class { font-family: 'Fraunces', serif; font-size: 26px; font-weight: 700; line-height: 1.1; }
.result-note { font-size: 12px; opacity: 0.65; margin-top: 6px; }
.result-bmi-side {
    margin-left: auto; text-align: center; flex-shrink: 0;
    background: rgba(255,255,255,0.2); border-radius: 12px; padding: 12px 22px;
}
.result-bmi-num { font-family: 'Fraunces', serif; font-size: 46px; font-weight: 700; line-height: 1; }
.result-bmi-lbl { font-size: 11px; opacity: 0.65; font-weight: 600; margin-top: 2px; }

/* ══ RECO ══ */
.reco-grid { display: flex; gap: 12px; margin-top: 20px; flex-wrap: wrap; }
.reco-card {
    flex: 1; min-width: 160px; background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.reco-icon { font-size: 22px; margin-bottom: 8px; }
.reco-title { font-size: 12px; font-weight: 700; color: #1E293B; margin-bottom: 4px; }
.reco-text  { font-size: 11px; color: #64748B; line-height: 1.55; }

/* ══ SHAP ══ */
.shap-card {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 14px; padding: 22px 24px; margin-top: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.shap-title { font-size: 14px; font-weight: 700; color: #1E293B; margin-bottom: 6px; }
.shap-desc  { font-size: 12px; color: #64748B; line-height: 1.7; margin-bottom: 16px; }

/* misc */
#MainMenu, footer, header { visibility: hidden; }

/* ══ DROPDOWN OPTIONS — fond blanc, texte foncé ══ */
[data-baseweb="popover"] {
    background: #FFFFFF !important;
}
[data-baseweb="menu"] {
    background: #FFFFFF !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.10) !important;
}
[role="option"] {
    background: #FFFFFF !important;
    color: #1E293B !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background: #EFF6FF !important;
    color: #1D4ED8 !important;
}
li[role="option"] {
    background: #FFFFFF !important;
    color: #1E293B !important;
}
li[role="option"]:hover {
    background: #EFF6FF !important;
    color: #1D4ED8 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_tools():
    model = joblib.load('app/best_model.pkl')
    le    = joblib.load('app/label_encoder.pkl')
    cols  = joblib.load('app/feature_columns.pkl')
    return model, le, cols

model, le, feature_cols = load_tools()

# Noms lisibles des classes SHAP (ordre alphabétique du LabelEncoder)
CLASS_NAMES = {
    "Insufficient_Weight": "Class 0 — Insufficient Weight (Underweight)",
    "Normal_Weight":        "Class 1 — Normal Weight (Healthy)",
    "Obesity_Type_I":       "Class 2 — Obesity Type I (Moderate Obesity)",
    "Obesity_Type_II":      "Class 3 — Obesity Type II (Severe Obesity)",
    "Obesity_Type_III":     "Class 4 — Obesity Type III (Morbid Obesity)",
    "Overweight_Level_I":   "Class 5 — Overweight Level I (Pre-Obesity)",
    "Overweight_Level_II":  "Class 6 — Overweight Level II (Overweight)",
}

# ─────────────────────────────────────────────
# 4. SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-wrap">
        <div class="brand-row">
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                <rect width="40" height="40" rx="10" fill="rgba(255,255,255,0.15)"/>
                <rect x="1" y="1" width="38" height="38" rx="9" stroke="rgba(255,255,255,0.25)" stroke-width="1"/>
                <path d="M20 10v20M10 20h20" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
                <circle cx="20" cy="20" r="5" fill="white" fill-opacity="0.2"/>
            </svg>
            <div>
                <div class="brand-name">ObesoScan AI</div>
                <div class="brand-tagline">Obesity Risk Prediction</div>
            </div>
        </div>
        <div class="brand-pill">✦ Machine Learning Powered</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-body">', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block">
        <div class="info-block-title">⚙️ Model Information</div>
        <div class="info-row"><span class="info-row-label">Algorithm</span><span class="info-row-value">Random Forest</span></div>
        <div class="info-row"><span class="info-row-label">Accuracy</span><span class="info-row-value" style="color:#15803D">~94.8%</span></div>
        <div class="info-row"><span class="info-row-label">Classes</span><span class="info-row-value">7</span></div>
        <div class="info-row"><span class="info-row-label">Features</span><span class="info-row-value">16 inputs</span></div>
        <div class="info-row"><span class="info-row-label">Explainability</span><span class="info-row-value">SHAP</span></div>
    </div>
    <div class="info-block">
        <div class="info-block-title">📊 Output Classes</div>
        <div class="tag-row">
            <span class="tag tag-blue">Insufficient Weight</span>
            <span class="tag tag-green">Normal Weight</span>
            <span class="tag tag-amber">Overweight I</span>
            <span class="tag tag-amber">Overweight II</span>
            <span class="tag tag-rose">Obesity I</span>
            <span class="tag tag-rose">Obesity II</span>
            <span class="tag tag-rose">Obesity III</span>
        </div>
    </div>
    <div class="sidebar-disclaimer">
        <strong>⚠️ Medical Disclaimer</strong>
        For educational and research purposes only. Not a substitute for medical advice.
    </div>
    <div class="sidebar-footer">ObesoScan AI v3.0 · Ecole Centrale Casablanca</div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 5. HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-top">
        <div class="hero-icon">❤️</div>
        <div><div class="hero-title">Obesity Risk Prediction</div></div>
    </div>
    <div class="hero-subtitle">
        Enter the patient's clinical and lifestyle parameters below. The AI model will analyse
        the data and predict the obesity classification, with a full SHAP-based explanation.
    </div>
    <div class="hero-chips">
        <span class="hero-chip">🤖 Random Forest</span>
        <span class="hero-chip">📐 BMI Calculator</span>
        <span class="hero-chip">🔍 SHAP Explainability</span>
        <span class="hero-chip">⚡ Real-time Prediction</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 1 — Patient Demographics
# ══════════════════════════════════════════════
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sec-label">👤 Patient Demographics</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    age    = st.number_input("Age (years)", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    height = st.number_input("Height (m)", min_value=1.00, max_value=2.50,
                              value=1.75, step=0.01, format="%.2f")
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0,
                              value=70.0, step=0.5)

with col3:
    bmi = round(weight / (height ** 2), 1) if height > 0 else 0
    if   bmi < 18.5: bmi_color, bmi_cat = "#3B82F6", "Underweight"
    elif bmi < 25.0: bmi_color, bmi_cat = "#10B981", "Normal"
    elif bmi < 30.0: bmi_color, bmi_cat = "#F59E0B", "Overweight"
    else:            bmi_color, bmi_cat = "#EF4444", "Obese"
    needle_pct = min(max((bmi - 10) / 40 * 100, 0), 100)
    st.markdown(f"""
    <div class="bmi-gauge-card">
        <div class="bmi-gauge-title">📐 BMI — Auto Calculated</div>
        <div class="bmi-value-big" style="color:{bmi_color}">{bmi}</div>
        <div class="bmi-unit">kg/m² · <strong style="color:{bmi_color}">{bmi_cat}</strong></div>
        <div class="bmi-bar-track">
            <div style="position:absolute;top:-4px;left:calc({needle_pct}% - 6px);
                        width:12px;height:16px;background:{bmi_color};
                        border-radius:3px;border:2px solid white;
                        box-shadow:0 1px 4px rgba(0,0,0,0.25);"></div>
        </div>
        <div class="bmi-bar-label">
            <span>10</span><span>18.5</span><span>25</span><span>30</span><span>50</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 2 — Lifestyle & Genetics
# ══════════════════════════════════════════════
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sec-label">🧬 Lifestyle & Genetic Factors</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns([1, 1, 1])

with col4:
    family_history = st.selectbox(
        "Family History of Overweight",
        ["yes", "no"],
        help="Does any family member suffer or have suffered from overweight?")
    favc = st.selectbox(
        "Frequent High-Caloric Food",
        ["yes", "no"],
        help="Do you frequently eat high caloric food like fast food or fried food?")
    smoke = st.selectbox(
        "Do you Smoke?",
        ["no", "yes"])

with col5:
    faf = st.slider(
        "Physical Activity (days/week)",
        min_value=0, max_value=7, value=2,
        help="How many days per week do you do physical activity?")
    tue = st.slider(
        "Daily Screen Time (hours)",
        min_value=0.0, max_value=2.0, value=1.0, step=0.5,
        help="0 = less than 2h  ·  1 = 3 to 5h  ·  2 = more than 5h")
    scc = st.selectbox(
        "Do you Monitor your Calories?",
        ["no", "yes"],
        help="Do you keep track of the calories you eat daily?")

with col6:
    if   faf >= 5: act_color, act_icon, act_label = "#10B981", "🟢", "Very Active"
    elif faf >= 3: act_color, act_icon, act_label = "#3B82F6", "🔵", "Active"
    elif faf >= 1: act_color, act_icon, act_label = "#F59E0B", "🟡", "Moderate"
    else:          act_color, act_icon, act_label = "#EF4444", "🔴", "Sedentary"
    dots = "".join([
        f'<div style="width:28px;height:28px;border-radius:50%;'
        f'background:{"#2563EB" if i < faf else "#E2E8F0"};'
        f'display:inline-flex;align-items:center;justify-content:center;'
        f'font-size:10px;color:white;font-weight:700;margin:2px;">'
        f'{"✓" if i < faf else ""}</div>'
        for i in range(7)
    ])
    st.markdown(f"""
    <div class="activity-card">
        <div class="bmi-gauge-title">🏃 Activity Level</div>
        <div style="font-family:'Fraunces',serif;font-size:22px;font-weight:700;
                    color:{act_color};margin-bottom:10px;">{act_icon} {act_label}</div>
        <div style="display:flex;gap:4px;flex-wrap:wrap;">{dots}</div>
        <div style="font-size:11px;color:#94A3B8;margin-top:8px;">
            {faf} active day{'s' if faf!=1 else ''} per week
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 3 — Dietary Habits
# ══════════════════════════════════════════════
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="sec-label">🥗 Dietary Habits & Transport</div>', unsafe_allow_html=True)

col7, col8, col9 = st.columns([1, 1, 1])

with col7:
    fcvc = st.slider(
        "Vegetables in Meals",
        min_value=1.0, max_value=3.0, value=2.0, step=0.5,
        help="1 = Never  ·  2 = Sometimes  ·  3 = Always include vegetables")
    ncp = st.slider(
        "Number of Main Meals per Day",
        min_value=1.0, max_value=4.0, value=3.0, step=0.5,
        help="How many main meals do you have daily?")

with col8:
    ch2o = st.slider(
        "Daily Water Intake (Litres)",
        min_value=1.0, max_value=3.0, value=2.0, step=0.5,
        help="1 = Less than 1L  ·  2 = Between 1-2L  ·  3 = More than 2L")
    caec = st.selectbox(
        "Eating Between Meals (Snacking)",
        ["no", "Sometimes", "Frequently", "Always"],
        index=1,
        help="Do you eat any food between your main meals?")

with col9:
    calc = st.selectbox(
        "Alcohol Consumption",
        ["no", "Sometimes", "Frequently", "Always"],
        index=1,
        help="How often do you drink alcohol?")
    mtrans = st.selectbox(
        "Main Transportation Used",
        ["Public Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
        help="What is your main means of transportation?")

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PREDICT BUTTON
# ══════════════════════════════════════════════
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict = st.button("🔬  Analyse & Predict Obesity Risk")

# ─────────────────────────────────────────────
# 6. RESULTS
# ─────────────────────────────────────────────
if predict:

    mtrans_map = {
        "Public Transportation": "Public_Transportation",
        "Walking":    "Walking",
        "Automobile": "Automobile",
        "Motorbike":  "Motorbike",
        "Bike":       "Bike"
    }
    mtrans_val = mtrans_map[mtrans]

    input_data = pd.DataFrame({
        'Age':    [float(age)],
        'Height': [float(height)],
        'Weight': [float(weight)],
        'FCVC':   [float(fcvc)],
        'NCP':    [float(ncp)],
        'CH2O':   [float(ch2o)],
        'FAF':    [float(faf)],
        'TUE':    [float(tue)],
        'Gender_Male':                        [1 if gender == 'Male' else 0],
        'family_history_with_overweight_yes': [1 if family_history == 'yes' else 0],
        'FAVC_yes':        [1 if favc  == 'yes' else 0],
        'SMOKE_yes':       [1 if smoke == 'yes' else 0],
        'SCC_yes':         [1 if scc   == 'yes' else 0],
        'CAEC_Frequently': [1 if caec  == 'Frequently' else 0],
        'CAEC_Sometimes':  [1 if caec  == 'Sometimes'  else 0],
        'CAEC_Always':     [1 if caec  == 'Always'     else 0],
        'CAEC_no':         [1 if caec  == 'no'         else 0],
        'CALC_Frequently': [1 if calc  == 'Frequently' else 0],
        'CALC_Sometimes':  [1 if calc  == 'Sometimes'  else 0],
        'CALC_Always':     [1 if calc  == 'Always'     else 0],
        'CALC_no':         [1 if calc  == 'no'         else 0],
        'MTRANS_Automobile':            [1 if mtrans_val == 'Automobile'            else 0],
        'MTRANS_Bike':                  [1 if mtrans_val == 'Bike'                  else 0],
        'MTRANS_Motorbike':             [1 if mtrans_val == 'Motorbike'             else 0],
        'MTRANS_Public_Transportation': [1 if mtrans_val == 'Public_Transportation' else 0],
        'MTRANS_Walking':               [1 if mtrans_val == 'Walking'               else 0],
    })

    for col in feature_cols:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_cols]

    pred_encoded = model.predict(input_data)
    pred_class   = le.inverse_transform(pred_encoded)[0]
    label_clean  = pred_class.replace("_", " ")

    risk_map = {
        "Insufficient_Weight": ("#2563EB","#EFF6FF","#DBEAFE","⚠️",
            "Weight below healthy range. Consider increasing caloric intake with professional guidance."),
        "Normal_Weight":       ("#059669","#F0FDF4","#DCFCE7","✅",
            "Your weight is within the healthy range. Keep up your healthy habits!"),
        "Overweight_Level_I":  ("#D97706","#FFFBEB","#FEF3C7","⚠️",
            "Slightly above normal. Moderate diet adjustments and regular exercise recommended."),
        "Overweight_Level_II": ("#EA580C","#FFF7ED","#FFEDD5","⚠️",
            "Weight notably above normal. Consulting a nutritionist is strongly advised."),
        "Obesity_Type_I":      ("#DC2626","#FFF1F2","#FFE4E6","🔴",
            "Obesity Class I. Medical consultation and a structured weight-loss plan recommended."),
        "Obesity_Type_II":     ("#B91C1C","#FFF1F2","#FFE4E6","🚨",
            "Obesity Class II. Immediate medical and nutritional intervention strongly advised."),
        "Obesity_Type_III":    ("#7F1D1D","#FFF1F2","#FFE4E6","🚨",
            "Obesity Class III (Morbid). Urgent medical attention required."),
    }
    txt_c, bg_c, border_c, icon, advice = risk_map.get(
        pred_class, ("#2563EB","#EFF6FF","#DBEAFE","📋","Please consult a healthcare professional."))

    st.markdown(f"""
    <div class="result-banner" style="background:{bg_c};border:1.5px solid {border_c};">
        <div class="result-emoji">{icon}</div>
        <div>
            <div class="result-lbl" style="color:{txt_c}">Prediction Result</div>
            <div class="result-class" style="color:{txt_c}">{label_clean}</div>
            <div class="result-note" style="color:{txt_c}">{advice}</div>
        </div>
        <div class="result-bmi-side">
            <div class="result-bmi-num" style="color:{txt_c}">{bmi}</div>
            <div class="result-bmi-lbl" style="color:{txt_c}">BMI (kg/m²)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    recos = [
        ("🥗","Nutrition",        "Balanced diet: vegetables, lean protein, whole grains."),
        ("🏋️","Exercise",         "At least 150 min of moderate aerobic activity per week."),
        ("💧","Hydration",        "Drink 1.5–2L of water daily to support metabolism."),
        ("🩺","Medical Follow-up","Schedule a check-up with your GP to discuss your results."),
    ]
    reco_html = "".join([f"""
    <div class="reco-card">
        <div class="reco-icon">{r[0]}</div>
        <div class="reco-title">{r[1]}</div>
        <div class="reco-text">{r[2]}</div>
    </div>""" for r in recos])
    st.markdown(f'<div class="reco-grid">{reco_html}</div>', unsafe_allow_html=True)

    # ══ SHAP ══════════════════════════════════
    st.markdown("""
    <div class="shap-card">
        <div class="shap-title">🔍 AI Explanation — SHAP Feature Importance</div>
        <div class="shap-desc">
            The chart below shows which features most influenced this prediction.
            Each colour represents one obesity class. Longer bar = higher impact.
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # Ordre exact du LabelEncoder (alphabétique)
        # Chaque index correspond à une couleur dans le graphique SHAP
        shap_class_names = [
            'Insufficient_Weight  →  Underweight (BMI < 18.5)',
            'Normal_Weight        →  Healthy Weight (BMI 18.5–24.9)',
            'Obesity_Type_I       →  Moderate Obesity (BMI 30–34.9)',
            'Obesity_Type_II      →  Severe Obesity (BMI 35–39.9)',
            'Obesity_Type_III     →  Morbid Obesity (BMI ≥ 40)',
            'Overweight_Level_I   →  Pre-Obesity (BMI 25–27.4)',
            'Overweight_Level_II  →  Overweight (BMI 27.5–29.9)',
        ]

        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#F8FAFC')

        if isinstance(shap_values, list):
            shap.summary_plot(
                shap_values,
                input_data,
                plot_type="bar",
                class_names=shap_class_names,
                max_display=len(feature_cols),
                show=False
            )
        else:
            shap.summary_plot(
                shap_values,
                input_data,
                plot_type="bar",
                max_display=len(feature_cols),
                show=False
            )

        # Style du graphique
        for spine in ax.spines.values():
            spine.set_color('#E2E8F0')
        ax.tick_params(colors='#374151', labelsize=9)
        ax.xaxis.label.set_color('#64748B')
        ax.yaxis.label.set_color('#64748B')
        ax.set_title(
            "Feature Impact on Obesity Prediction",
            fontsize=11, color='#1E293B', pad=14, fontweight='bold'
        )

        # Agrandir la légende pour qu'elle soit lisible
        legend = ax.get_legend()
        if legend:
            legend.set_title("Obesity Classes", prop={'size': 9, 'weight': 'bold'})
            for text in legend.get_texts():
                text.set_fontsize(8)
                text.set_color('#1E293B')
            legend.get_frame().set_edgecolor('#E2E8F0')
            legend.get_frame().set_facecolor('#FFFFFF')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"SHAP Error: {e}")