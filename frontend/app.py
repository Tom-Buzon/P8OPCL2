# frontend/app.py

import json
import requests
import streamlit as st

API_URL = "http://localhost:8000"  # FastAPI

# 👉 Même listes que côté backend
TOP20_FEATURES = [
    "PAYMENT_RATE",
    "EXT_SOURCE_3",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "DAYS_BIRTH",
    "AMT_ANNUITY",
    "APPROVED_CNT_PAYMENT_MEAN",
    "DAYS_ID_PUBLISH",
    "INSTAL_DPD_MEAN",
    "AMT_CREDIT",
    "INSTAL_AMT_PAYMENT_SUM",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED_PERC",
    "DAYS_REGISTRATION",
    "PREV_CNT_PAYMENT_MEAN",
    "DAYS_EMPLOYED",
    "ACTIVE_DAYS_CREDIT_MAX",
    "INSTAL_DAYS_ENTRY_PAYMENT_MAX",
    "CODE_GENDER",
    "BURO_DAYS_CREDIT_MAX",
]

TOP40_FEATURES = [
    "PAYMENT_RATE",
    "EXT_SOURCE_3",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "DAYS_BIRTH",
    "AMT_ANNUITY",
    "APPROVED_CNT_PAYMENT_MEAN",
    "DAYS_ID_PUBLISH",
    "INSTAL_DPD_MEAN",
    "AMT_CREDIT",
    "INSTAL_AMT_PAYMENT_SUM",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED_PERC",
    "DAYS_REGISTRATION",
    "PREV_CNT_PAYMENT_MEAN",
    "DAYS_EMPLOYED",
    "ACTIVE_DAYS_CREDIT_MAX",
    "INSTAL_DAYS_ENTRY_PAYMENT_MAX",
    "CODE_GENDER",
    "BURO_DAYS_CREDIT_MAX",
    "ANNUITY_INCOME_PERC",
    "INCOME_CREDIT_PERC",
    "ACTIVE_DAYS_CREDIT_ENDDATE_MIN",
    "REGION_POPULATION_RELATIVE",
    "DAYS_LAST_PHONE_CHANGE",
    "ACTIVE_DAYS_CREDIT_ENDDATE_MEAN",
    "BURO_DAYS_CREDIT_ENDDATE_MAX",
    "INSTAL_PAYMENT_DIFF_MEAN",
    "PREV_APP_CREDIT_PERC_MEAN",
    "BURO_AMT_CREDIT_SUM_DEBT_MEAN",
    "BURO_AMT_CREDIT_SUM_MEAN",
    "INSTAL_DBD_SUM",
    "POS_MONTHS_BALANCE_MAX",
    "PREV_APP_CREDIT_PERC_MIN",
    "NAME_FAMILY_STATUS_Married",
    "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
    "APPROVED_AMT_ANNUITY_MEAN",
    "INSTAL_AMT_PAYMENT_MIN",
    "INSTAL_DAYS_ENTRY_PAYMENT_MEAN",
    "APPROVED_DAYS_DECISION_MAX",
]

FEATURES_BY_MODE = {
    "light": TOP20_FEATURES,
    "med": TOP40_FEATURES,
}

# ======================================
# Charger le feature_meta.json
# ======================================

try:
    with open("../feature_meta.json", "r") as f:
        FEATURE_META = json.load(f)
    meta_loaded = True
except FileNotFoundError:
    FEATURE_META = {}
    meta_loaded = False

# Helper pour récupérer les paramètres du slider
def get_slider_params(feat_name: str):
    """
    Retourne (slider_min, slider_max, default, dtype) pour une feature donnée,
    ou None si absente du feature_meta.json.
    """
    m = FEATURE_META.get(feat_name)
    if not m:
        return None

    slider_min = m.get("slider_min", m.get("min", 0.0))
    slider_max = m.get("slider_max", m.get("max", 1.0))
    default = m.get("default", (slider_min + slider_max) / 2)
    dtype = m.get("dtype", "float")

    return slider_min, slider_max, default, dtype


# ======================================
# UI Streamlit
# ======================================

st.set_page_config(page_title="OptiWeb Scoring", layout="wide")

st.sidebar.title("OptiWeb — Scoring")

mode_label = st.sidebar.radio(
    "Choix du modèle",
    options=["light", "med", "full"],
    format_func=lambda x: {
        "light": "Light (20 features)",
        "med": "Medium (40 features)",
        "full": "Full (en développement)",
    }[x],
)

st.sidebar.write("---")
st.sidebar.write(f"Mode sélectionné : **{mode_label}**")

if not meta_loaded:
    st.sidebar.warning("⚠️ feature_meta.json introuvable — sliders heuristiques utilisés.")

st.title("OptiWeb — Simulation de scoring")

if mode_label == "full":
    st.info(
        "Le mode **Full** est encore en développement. "
        "Utilise Light ou Medium pour l’instant."
    )
    st.stop()

features = FEATURES_BY_MODE[mode_label]

st.subheader(f"Formulaire ({len(features)} variables)")

# Formulaire dans deux colonnes pour être plus compact
col1, col2 = st.columns(2)

input_values = {}

for i, feat in enumerate(features):
    col = col1 if i % 2 == 0 else col2
    label = feat

    params = get_slider_params(feat)

    if params:
        slider_min, slider_max, default, dtype = params

        # Sécurité si min == max
        if slider_min == slider_max:
            slider_min = slider_min - 1
            slider_max = slider_max + 1

        if dtype == "int":
            slider_min = int(slider_min)
            slider_max = int(slider_max)
            default = int(default)
            step = max(1, (slider_max - slider_min) // 50)
            val = col.slider(
                label,
                min_value=slider_min,
                max_value=slider_max,
                value=default,
                step=step,
            )
        else:
            slider_min = float(slider_min)
            slider_max = float(slider_max)
            default = float(default)
            step = (slider_max - slider_min) / 100 if slider_max > slider_min else 0.01
            if step <= 0:
                step = 0.01
            val = col.slider(
                label,
                min_value=slider_min,
                max_value=slider_max,
                value=default,
                step=step,
            )
    else:
        # Fallback heuristique si la feature n'est pas dans feature_meta.json
        if "EXT_SOURCE" in feat:
            val = col.slider(label, 0.0, 1.0, 0.5, 0.01)
        elif "DAYS_" in feat:
            val = col.slider(label, -30000, 0, -10000, 100)
        elif "AMT_" in feat or "CREDIT" in feat or "ANNUITY" in feat:
            val = col.slider(label, 0.0, 1_000_000.0, 100_000.0, 1_000.0)
        elif "CNT_" in feat:
            val = col.slider(label, 0, 20, 2, 1)
        else:
            val = col.slider(label, -10.0, 10.0, 0.0, 0.1)

    input_values[feat] = float(val)

st.write("---")

if st.button("Prédire"):
    try:
        payload = {
            "mode": mode_label,
            "features": input_values,
        }
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if resp.status_code != 200:
            st.error(f"Erreur API ({resp.status_code}) : {resp.text}")
        else:
            data = resp.json()
            st.success(
                f"Probabilité de défaut (approx.) : **{data['probability']:.3f}**"
            )
            st.caption(f"Raw output: {data['raw_output']:.5f}")
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")
