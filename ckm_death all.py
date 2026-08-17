import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# -------------------- Page configuration --------------------
st.set_page_config(
    page_title="All-Cause Mortality Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Model path (relative to this script) --------------------
# If your .pkl files are in a subfolder, change MODEL_DIR to e.g., './model_artifacts'
MODEL_DIR = '.'

# Required model files
REQUIRED_FILES = ['rsf_model.pkl', 'scaler.pkl', 'encoders.pkl', 'num_medians.pkl', 'metadata.pkl']

# Check existence
missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(MODEL_DIR, f))]
if missing:
    st.error(f"❌ Missing model file(s): {', '.join(missing)}")
    st.info("Please ensure all .pkl files are in the same directory as this script, or adjust MODEL_DIR.")
    st.stop()

# -------------------- Load models (cached) --------------------
@st.cache_resource
def load_models():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'rsf_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        encoders = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))
        medians = joblib.load(os.path.join(MODEL_DIR, 'num_medians.pkl'))
        metadata = joblib.load(os.path.join(MODEL_DIR, 'metadata.pkl'))
        return model, scaler, encoders, medians, metadata
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

model, scaler, encoders, medians, metadata = load_models()

FEATURES = metadata['features']
CAT_COLS = metadata['cat_cols']

# -------------------- Preprocessing function (same as original) --------------------
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])
    # Categorical
    for col in CAT_COLS:
        if col in df.columns:
            val = str(df[col].iloc[0])
            if val.lower() in ['nan', 'none', '']:
                val = 'Missing'
            le = encoders[col]
            if val not in le.classes_:
                val = le.classes_[0]
            df[col] = le.transform([val])[0]
    # Numerical
    num_cols = [c for c in FEATURES if c not in CAT_COLS]
    for col in num_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            df[col] = np.nan
        df[col].fillna(medians.get(col, 0), inplace=True)
    X = df[FEATURES].values.astype(float)
    return scaler.transform(X)

# -------------------- Extract categorical options from encoders --------------------
cat_options = {}
for col in CAT_COLS:
    cat_options[col] = list(encoders[col].classes_) if col in encoders else ['Missing']

binary_choices = ['No', 'Yes']

# -------------------- UI Layout --------------------
st.title("🫀 All‑Cause Mortality Risk Prediction")
st.markdown("**Random Survival Forest model** – Enter patient characteristics to estimate risk at a given time.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📝 Patient Characteristics")

    with st.expander("🧑‍⚕️ Demographics", expanded=True):
        gender = st.selectbox("Sex", cat_options['GENDER'], key="gender")
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=65, step=1, key="age")
        pir = st.selectbox("PIR Group", cat_options['PIR_GROUP'], key="pir")

    with st.expander("🧪 Clinical & Laboratory", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            ckm = st.selectbox("CKM Stage", cat_options['CKM'], key="ckm")
            hba1c = st.number_input("HbA1c (%)", value=5.6, step=0.1, format="%.1f", key="hba1c")
            absi = st.number_input("A Body Shape Index", value=0.08, step=0.01, format="%.3f", key="absi")
            egfr = st.number_input("eGFR (mL/min/1.73m²)", value=90, step=1, key="egfr")
            rdw = st.number_input("RDW (%)", value=13.2, step=0.1, format="%.1f", key="rdw")
            shr = st.number_input("Stress Hyperglycemia Ratio", value=2.5, step=0.1, key="shr")
            mcv = st.number_input("MCV (fL)", value=92, step=1, key="mcv")
            glb = st.number_input("GLB (g/L)", value=28, step=1, key="glb")
        with c2:
            plt = st.number_input("PLT (×10⁹/L)", value=250, step=1, key="plt")
            crp = st.number_input("CRP (mg/L)", value=1.5, step=0.1, key="crp")
            sii = st.number_input("Systemic Immunity‑Inflammation Index (×10⁹/L)", value=500, step=10, key="sii")
            mon = st.number_input("MON (×10⁹/L)", value=0.4, step=0.1, key="mon")
            ast = st.number_input("AST (U/L)", value=22, step=1, key="ast")
            bmi = st.number_input("BMI (kg/m²)", value=24.5, step=0.1, key="bmi")
            tc = st.number_input("TC (mmol/L)", value=4.8, step=0.1, key="tc")

    with st.expander("🚬 Lifestyle & Medical History", expanded=True):
        smoke = st.selectbox("Smoking Status", cat_options['SMOKE'], key="smoke")
        activity = st.selectbox("Physical Activity", cat_options['ACTIVITY'], key="activity")
        lung_yn = st.selectbox("Pulmonary Disease", binary_choices, key="lung")
        cancer_yn = st.selectbox("Cancer History", binary_choices, key="cancer")
        lung = 1 if lung_yn == 'Yes' else 0
        cancer = 1 if cancer_yn == 'Yes' else 0

    # Time prediction
    st.subheader("⏱ Prediction Time")
    if 'time_years' not in st.session_state:
        st.session_state.time_years = 5.0

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.radio("Quick select", options=[1,3,5,10], index=2, horizontal=True, key="preset_time")
    with col_t2:
        cols_btn = st.columns(4)
        for i, y in enumerate([1,3,5,10]):
            with cols_btn[i]:
                if st.button(f"{y} yr", key=f"btn_{y}"):
                    st.session_state.time_years = float(y)
                    st.rerun()
        time_years = st.number_input("Custom years", min_value=0.5, max_value=30.0,
                                     value=st.session_state.time_years, step=0.5, key="time_input")
        if time_years != st.session_state.time_years:
            st.session_state.time_years = time_years

    predict_clicked = st.button("📊 Predict Mortality Risk", type="primary", use_container_width=True)

# -------------------- Results panel --------------------
with col_right:
    st.subheader("📊 Prediction Result")

    if predict_clicked:
        input_dict = {
            'GENDER': gender, 'AGE': age, 'PIR_GROUP': pir,
            'CKM': ckm, 'HBA1C': hba1c, 'ABSI': absi, 'EGFR': egfr,
            'RDW': rdw, 'SHR': shr, 'MCV': mcv, 'GLB': glb,
            'PLT': plt, 'CRP': crp, 'SII': sii, 'MON': mon,
            'AST': ast, 'BMI': bmi, 'TC': tc,
            'SMOKE': smoke, 'ACTIVITY': activity,
            'LUNG': lung, 'CANCER': cancer
        }
        try:
            X_scaled = preprocess_input(input_dict)
            surv_funcs = model.predict_survival_function(X_scaled)
            time_month = int(round(st.session_state.time_years * 12))
            if time_month < 1: time_month = 1
            surv_prob = surv_funcs[0](time_month)
            risk = 1 - surv_prob

            # Compare table
            preset_months = [12, 36, 60, 120]
            all_risks = {t: 1 - surv_funcs[0](t) for t in preset_months}

            st.markdown(f"**Risk at {st.session_state.time_years:.1f} years ({time_month} months)**")
            risk_pct = risk * 100
            surv_pct = surv_prob * 100

            # Risk level
            if risk < 0.2:
                level, color = "Low", "green"
            elif risk < 0.5:
                level, color = "Moderate", "orange"
            else:
                level, color = "High", "red"

            c_risk, c_surv = st.columns(2)
            with c_risk:
                st.metric("Mortality Risk", f"{risk_pct:.1f}%")
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>{level} Risk</span>", unsafe_allow_html=True)
            with c_surv:
                st.metric("Survival Probability", f"{surv_pct:.1f}%")

            st.markdown("#### 📈 Risk at multiple time points")
            df_compare = pd.DataFrame({
                "Months": [12, 36, 60, 120],
                "Years": [1, 3, 5, 10],
                "Risk (%)": [f"{all_risks[t]*100:.1f}" for t in [12,36,60,120]]
            })
            st.dataframe(df_compare, use_container_width=True, hide_index=True)
            st.caption("⚠️ This prediction is for research purposes only and does not constitute medical advice.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Please verify all inputs and model files.")
    else:
        st.info("👈 Fill in patient characteristics and click **Predict Mortality Risk**.")
        st.markdown("""
        **Instructions:**
        - All fields are required.
        - Categorical variables use drop‑downs.
        - Numerical values can be entered directly.
        - Prediction time can be set in years (including decimals).
        """)

# -------------------- Footer --------------------
st.divider()
st.caption(f"Model path: `{os.path.abspath(MODEL_DIR)}` · Random Survival Forest · v1.0")