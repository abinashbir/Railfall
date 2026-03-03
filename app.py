import streamlit as st
import joblib
import pandas as pd
import os

# --- Page Config ---
st.set_page_config(
    page_title="Rainfall Predictor",
    page_icon="cloud",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #111827;
    font-family: 'Inter', sans-serif;
    color: #e5e7eb;
}

#MainMenu, footer, header { visibility: hidden; }

/* --- Hero --- */
.hero {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 16px;
    padding: 2rem 1.5rem 1.8rem;
    text-align: center;
    margin-bottom: 1.8rem;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #f9fafb;
    margin: 0 0 .35rem;
}
.hero p {
    font-size: 1rem;
    color: #9ca3af;
    margin: 0 0 .8rem;
}
.chip-row { display: flex; justify-content: center; flex-wrap: wrap; gap: .4rem; }
.chip {
    background: #374151;
    border: 1px solid #4b5563;
    border-radius: 6px;
    padding: .25rem .7rem;
    font-size: .78rem;
    color: #d1d5db;
    font-weight: 500;
}

/* --- Section header --- */
.section-hdr {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f9fafb;
    margin-bottom: .8rem;
    padding-bottom: .4rem;
    border-bottom: 2px solid #374151;
}

/* --- Slider track --- */
.stSlider > div > div > div > div {
    background: #6366f1 !important;
}
.stSlider [data-testid="stThumbValue"] {
    color: #c7d2fe !important;
    font-weight: 600;
}

/* --- Labels --- */
.stSlider label, .stNumberInput label {
    color: #d1d5db !important;
    font-weight: 500 !important;
}

/* --- Number input field --- */
.stNumberInput input {
    background: #1f2937 !important;
    border: 1px solid #4b5563 !important;
    color: #f3f4f6 !important;
    border-radius: 8px !important;
}

/* --- Buttons --- */
.stButton > button {
    background: #6366f1 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .85rem 2.5rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    width: 100%;
    transition: background .2s ease, transform .15s ease !important;
}
.stButton > button:hover {
    background: #4f46e5 !important;
    transform: translateY(-1px) !important;
}

/* --- Tab styling --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #1f2937;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #374151;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #9ca3af;
    font-weight: 600;
    padding: .5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #6366f1 !important;
    color: #fff !important;
}

/* --- Result Cards --- */
.result-card {
    border-radius: 12px;
    padding: 1.1rem .9rem;
    text-align: center;
}
.result-yes {
    background: #064e3b;
    border: 1px solid #059669;
}
.result-no {
    background: #7f1d1d;
    border: 1px solid #dc2626;
}
.result-card .model-name {
    font-size: .78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .8px;
    color: #d1d5db;
    margin-bottom: .3rem;
}
.result-card .prediction {
    font-size: 1.4rem;
    font-weight: 800;
    color: #f9fafb;
}

/* --- Summary --- */
.summary-banner {
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
    margin-top: 1.2rem;
}
.summary-rain {
    background: #064e3b;
    border: 1px solid #059669;
}
.summary-no-rain {
    background: #78350f;
    border: 1px solid #d97706;
}
.summary-banner .verdict {
    font-size: 1.7rem;
    font-weight: 800;
    color: #f9fafb;
    margin: .2rem 0;
}
.summary-banner .detail {
    font-size: .9rem;
    color: #d1d5db;
}

/* --- Footer --- */
.app-footer {
    text-align: center;
    color: #6b7280;
    font-size: .8rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid #1f2937;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# --- Load Models ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    mdls = joblib.load(os.path.join(BASE_DIR, "all_models.pkl"))
    sc = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    cols = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))
    return mdls, sc, cols


models, scaler, columns = load_artifacts()

# --- Feature metadata ---
FEATURE_META = {
    "pressure ":     {"label": "Pressure (hPa)",     "min": 990.0,  "max": 1045.0, "default": 1013.0, "step": 0.5,  "help": "Atmospheric pressure in hectopascals"},
    "temparature":   {"label": "Temperature (C)",     "min": 0.0,    "max": 45.0,   "default": 24.0,   "step": 0.5,  "help": "Average temperature in Celsius"},
    "dewpoint":      {"label": "Dew Point (C)",       "min": -5.0,   "max": 30.0,   "default": 20.0,   "step": 0.5,  "help": "Dew point temperature in Celsius"},
    "humidity ":     {"label": "Humidity (%)",         "min": 20.0,   "max": 100.0,  "default": 80.0,   "step": 1.0,  "help": "Relative humidity percentage"},
    "cloud ":        {"label": "Cloud Cover (%)",     "min": 0.0,    "max": 100.0,  "default": 70.0,   "step": 1.0,  "help": "Cloud cover percentage"},
    "sunshine":      {"label": "Sunshine (hrs)",      "min": 0.0,    "max": 14.0,   "default": 4.4,    "step": 0.1,  "help": "Hours of sunshine"},
    "winddirection": {"label": "Wind Direction (deg)","min": 0.0,    "max": 360.0,  "default": 100.0,  "step": 5.0,  "help": "Wind direction in degrees (0-360)"},
    "windspeed":     {"label": "Wind Speed (km/h)",   "min": 0.0,    "max": 70.0,   "default": 21.0,   "step": 0.5,  "help": "Average wind speed in km/h"},
}

MODEL_LABELS = {
    "Logistic":     "Logistic Regression",
    "KNN":          "K-Nearest Neighbors",
    "DecisionTree": "Decision Tree",
    "RandomForest": "Random Forest",
    "XGBoost":      "XGBoost",
}

# --- Hero ---
st.markdown("""
<div class="hero">
    <h1>Rainfall Predictor</h1>
    <p>Enter weather conditions and let 5 ML models predict whether it will rain</p>
    <div class="chip-row">
        <span class="chip">Logistic Regression</span>
        <span class="chip">KNN</span>
        <span class="chip">Decision Tree</span>
        <span class="chip">Random Forest</span>
        <span class="chip">XGBoost</span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Input Section ---
st.markdown('<div class="section-hdr">Weather Parameters</div>', unsafe_allow_html=True)

tab_slider, tab_custom = st.tabs(["Sliders", "Custom Input"])

inputs = {}
feature_list = list(columns)
half = len(feature_list) // 2

# ---- Tab 1: Sliders ----
with tab_slider:
    col_l, col_r = st.columns(2, gap="large")
    for i, col_name in enumerate(feature_list):
        meta = FEATURE_META.get(col_name, {
            "label": col_name, "min": 0.0, "max": 100.0,
            "default": 50.0, "step": 1.0, "help": ""
        })
        parent = col_l if i < half else col_r
        with parent:
            inputs[col_name] = st.slider(
                meta["label"],
                min_value=meta["min"],
                max_value=meta["max"],
                value=meta["default"],
                step=meta["step"],
                help=meta["help"],
                key=f"slider_{col_name}",
            )

# ---- Tab 2: Custom number inputs ----
with tab_custom:
    st.caption("Type exact values for precise control.")
    col_l2, col_r2 = st.columns(2, gap="large")
    for i, col_name in enumerate(feature_list):
        meta = FEATURE_META.get(col_name, {
            "label": col_name, "min": 0.0, "max": 100.0,
            "default": 50.0, "step": 1.0, "help": ""
        })
        parent = col_l2 if i < half else col_r2
        with parent:
            inputs[col_name] = st.number_input(
                meta["label"],
                min_value=meta["min"],
                max_value=meta["max"],
                value=meta["default"],
                step=meta["step"],
                help=meta["help"],
                key=f"number_{col_name}",
            )

# --- Predict Button ---
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("Run Prediction", use_container_width=True)

# --- Results ---
if predict_clicked:
    input_values = [inputs[c] for c in columns]
    df = pd.DataFrame([input_values], columns=columns)

    results = {}
    for name, model in models.items():
        temp_df = df.copy()
        if name in ["Logistic", "KNN"]:
            temp_df = scaler.transform(temp_df)
        pred = model.predict(temp_df)[0]
        results[name] = "Yes" if pred == 1 else "No"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">Model Predictions</div>', unsafe_allow_html=True)

    result_cols = st.columns(len(results), gap="medium")
    for idx, (model_key, prediction) in enumerate(results.items()):
        label = MODEL_LABELS.get(model_key, model_key)
        css_class = "result-yes" if prediction == "Yes" else "result-no"
        text = "Rain" if prediction == "Yes" else "No Rain"

        with result_cols[idx]:
            st.markdown(f"""
            <div class="result-card {css_class}">
                <div class="model-name">{label}</div>
                <div class="prediction">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    # -- Majority Vote --
    yes_count = sum(1 for v in results.values() if v == "Yes")
    total = len(results)
    majority_rain = yes_count > total / 2

    verdict_text = "Rain Expected" if majority_rain else "No Rain Expected"
    verdict_class = "summary-rain" if majority_rain else "summary-no-rain"

    st.markdown(f"""
    <div class="summary-banner {verdict_class}">
        <div class="verdict">{verdict_text}</div>
        <div class="detail">{yes_count} out of {total} models predict rainfall</div>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="app-footer">
    Built by <strong>Abinash Bir</strong> &middot; Streamlit + scikit-learn + XGBoost
</div>
""", unsafe_allow_html=True)