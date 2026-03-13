import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')

# ─────────────────────────────────────────────
# 1. PAGE CONFIG & CSS (Ton design original)
# ─────────────────────────────────────────────
st.set_page_config(page_title="ObesoScan AI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fraunces:opsz,wght@9..144,300;9..144,600;9..144,700&display=swap');
html, body, .stApp { background-color: #F0F4F8 !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }
[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #E2E8F0 !important; }
.hero { background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 50%, #0EA5E9 100%); border-radius: 20px; padding: 36px; color: white; margin-bottom: 25px; }
.sec-label { font-size: 11px; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; margin: 20px 0 10px; display: flex; align-items: center; gap: 8px; }
.sec-label::after { content: ''; flex: 1; height: 1px; background: #E2E8F0; }
/* ... (Le reste de ton CSS est conservé ici) ... */
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 2. CHARGEMENT DES OUTILS
# ─────────────────────────────────────────────
@st.cache_resource
def load_tools():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(base_dir, 'app', 'best_model.pkl'))
    le    = joblib.load(os.path.join(base_dir, 'app', 'label_encoder.pkl'))
    cols  = joblib.load(os.path.join(base_dir, 'app', 'feature_columns.pkl'))
    return model, le, cols

model, le, feature_cols = load_tools()

# ─────────────────────────────────────────────
# 3. SIDEBAR & HERO (Design conservé)
# ─────────────────────────────────────────────
# [Ici ton bloc Sidebar avec Logo et Info-blocks...]

st.markdown("""
<div class="hero">
    <div style="font-size:30px; font-family:'Fraunces', serif; font-weight:700;">ObesoScan AI v2.0</div>
    <div style="opacity:0.8;">Analyse clinique complète basée sur les standards de l'OMS.</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 4. FORMULAIRE COMPLET (Toutes les colonnes du CSV)
# ─────────────────────────────────────────────

# --- SECTION 1 : BIOMÉTRIE ---
st.markdown('<div class="sec-label">1. Paramètres Biométriques</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Âge", 14, 100, 22)
    gender = st.selectbox("Sexe", ["Male", "Female"])
with c2:
    height = st.number_input("Taille (m)", 1.40, 2.20, 1.75, 0.01)
    weight = st.number_input("Poids (kg)", 40.0, 200.0, 70.0, 0.5)
with c3:
    bmi = round(weight / (height**2), 1)
    st.metric("IMC (Calculé)", f"{bmi} kg/m²")

# --- SECTION 2 : ALIMENTATION (FAVC, FCVC, NCP, CAEC, CH2O, SCC) ---
st.markdown('<div class="sec-label">2. Habitudes Alimentaires</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    favc = st.selectbox("Aliments hypercaloriques ?", ["yes", "no"])
    fcvc = st.slider("Consommation de légumes (1-3)", 1.0, 3.0, 2.0)
with c5:
    ncp = st.slider("Repas principaux par jour", 1.0, 4.0, 3.0)
    caec = st.selectbox("Grignotage entre les repas", ["Sometimes", "Frequently", "no", "Always"])
with c6:
    ch2o = st.slider("Eau par jour (Liters)", 1.0, 3.0, 2.0)
    scc = st.selectbox("Surveille ses calories ?", ["no", "yes"])

# --- SECTION 3 : STYLE DE VIE (GÉNÉTIQUE, FUMÉE, SPORT, ÉCRAN, ALCOOL, TRANSPORT) ---
st.markdown('<div class="sec-label">3. Mode de vie & Génétique</div>', unsafe_allow_html=True)
c7, c8, c9 = st.columns(3)
with c7:
    family_history = st.selectbox("Antécédents familiaux", ["yes", "no"])
    smoke = st.selectbox("Fumeur ?", ["no", "yes"])
with c8:
    faf = st.slider("Activité physique (0-3)", 0.0, 3.0, 1.0)
    tue = st.slider("Temps d'écran (0-2)", 0.0, 2.0, 1.0)
with c9:
    calc = st.selectbox("Consommation d'alcool", ["Sometimes", "no", "Frequently"])
    mtrans = st.selectbox("Transport principal", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

# ─────────────────────────────────────────────
# 5. PRÉDICTION & SHAP
# ─────────────────────────────────────────────
if st.button("🔬 Lancer l'Analyse Diagnostique"):
    # Construction du dictionnaire avec les EXACTS noms du dataset
    input_data = pd.DataFrame({
        'Gender': [gender], 'Age': [age], 'Height': [height], 'Weight': [weight],
        'family_history_with_overweight': [family_history], 'FAVC': [favc],
        'FCVC': [fcvc], 'NCP': [ncp], 'CAEC': [caec], 'SMOKE': [smoke],
        'CH2O': [ch2o], 'SCC': [scc], 'FAF': [faf], 'TUE': [tue],
        'CALC': [calc], 'MTRANS': [mtrans]
    })

    # Encodage One-Hot pour correspondre au modèle LightGBM
    input_encoded = pd.get_dummies(input_data)
    
    # Alignement des colonnes (très important !)
    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_cols]

    # Prédiction
    pred_idx = model.predict(input_encoded)[0]
    final_label = le.inverse_transform([pred_idx])[0]

    # Affichage stylisé du résultat
    st.markdown(f"""
    <div style="background:white; border-left:8px solid #2563EB; padding:25px; border-radius:15px; margin-top:20px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
        <div style="color:#64748B; font-size:12px; font-weight:700; text-transform:uppercase;">Diagnostic de l'IA</div>
        <div style="color:#1E3A5F; font-size:32px; font-family:'Fraunces',serif; font-weight:700;">{final_label.replace('_',' ')}</div>
        <div style="color:#64748B; font-size:14px; margin-top:10px;">Basé sur une précision de modèle de 96.7%.</div>
    </div>
    """, unsafe_allow_html=True)

    # SHAP Explainer
    with st.expander("🔍 Voir l'explication SHAP (Facteurs d'influence)"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_encoded)
        fig, ax = plt.subplots()
        # On affiche les SHAP values pour la classe prédite
        shap.summary_plot(shap_values[pred_idx], input_encoded, plot_type="bar", show=False)
        st.pyplot(fig)