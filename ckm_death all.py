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

# -------------------- Model path --------------------
MODEL_DIR = '.'
MODEL_FILE = 'rsf_model.pkl'

if not os.path.exists(os.path.join(MODEL_DIR, MODEL_FILE)):
    st.error(f"❌ Model file '{MODEL_FILE}' not found.")
    st.stop()

# -------------------- Load Pipeline (cached) --------------------
@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load(os.path.join(MODEL_DIR, MODEL_FILE))
        return pipeline
    except Exception as e:
        st.error(f"Failed to load Pipeline: {e}")
        st.stop()

pipeline = load_pipeline()

# Feature list (must match training)
feature_cols = [
    'AGE', 'CKM', 'HBA1C', 'ABSI', 'EGFR', 'RDW', 'SHR', 'MCV',
    'GLB', 'PLT', 'CRP', 'SII', 'MON', 'AST', 'SMOKE', 'BMI',
    'TC', 'ACTIVITY', 'PIR_GROUP', 'GENDER', 'LUNG', 'CANCER'
]

# -------------------- Mapping for CKM (numeric in training) --------------------
ckm_map = {
    'Stage 0': 0,
    'Stage 1': 1,
    'Stage 2': 2,
    'Stage 3': 3,
    'Stage 4': 4
}

# -------------------- UI options (updated) --------------------
binary_choices = ['No', 'Yes']
ckm_options = list(ckm_map.keys())
gender_options = ['Male', 'Female']
pir_options = ['<1.0', '1.0–2.9', '≥3.0']          # 3 groups
smoke_options = ['Never', 'Former', 'Current']
activity_options = ['No', 'Yes']                    # Moderate or vigorous activity? 0/1

# -------------------- UI Layout (all in English) --------------------
st.title("🫀 All‑Cause Mortality Risk Prediction")
st.markdown("**Random Survival Forest model** – Enter patient characteristics to estimate risk at a given time.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📝 Patient Characteristics")

    with st.expander("🧑‍⚕️ Demographics", expanded=True):
        gender = st.selectbox("Sex", gender_options, key="gender")
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=65, step=1, key="age")
        pir = st.selectbox("PIR Group", pir_options, key="pir")

    with st.expander("🧪 Clinical & Laboratory", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            ckm = st.selectbox("CKM Stage", ckm_options, key="ckm")
            hba1c = st.number_input("HbA1c (%)", value=5.6, step=0.1, format="%.1f", key="hba1c")
            absi = st.number_input("A Body Shape Index", value=0.08, step=0.01, format="%.3f", key="absi")
            egfr = st.number_input("eGFR (mL/min/1.73m²)", value=90, step=1, key="egfr")
            rdw = st.number_input("RDW (%)", value=13.2, step=0.1, format="%.1f", key="rdw")
            shr = st.number_input("Stress Hyperglycemia Ratio", value=2.5, step=0.1, key="shr")
            mcv = st.number_input("MCV (fL)", value=92, step=1, key="mcv")
            glb = st.number_input("GLB (g/L)", value=28, step=1, key="glb")
        with c2:
            plt = st.number_input("PLT (×10⁹/L)", value=250, step=1, key="plt")
            crp = st.number_input("CRP (mg/dL)", value=1.5, step=0.1, key="crp")          # unit: mg/dL
            sii = st.number_input("Systemic Immunity‑Inflammation Index (×10⁹/L)", value=500, step=10, key="sii")
            mon = st.number_input("MON (×10⁹/L)", value=0.4, step=0.1, key="mon")
            ast = st.number_input("AST (U/L)", value=22, step=1, key="ast")
            bmi = st.number_input("BMI (kg/m²)", value=24.5, step=0.1, key="bmi")
            tc = st.number_input("TC (mg/dL)", value=185, step=1, key="tc")              # unit: mg/dL

    with st.expander("🚬 Lifestyle & Medical History", expanded=True):
        smoke = st.selectbox("Smoking Status", smoke_options, key="smoke")
        activity = st.selectbox("Moderate or vigorous activity", activity_options, key="activity")   # 0/1
        lung_yn = st.selectbox("Pulmonary Disease", binary_choices, key="lung")
        cancer_yn = st.selectbox("Cancer History", binary_choices, key="cancer")
        lung = 1 if lung_yn == 'Yes' else 0
        cancer = 1 if cancer_yn == 'Yes' else 0
        # activity is already 0/1 from selectbox (since options are 'No'/'Yes')
        activity_val = 1 if activity == 'Yes' else 0

    # Prediction time
    st.subheader("⏱ Prediction Time")
    if 'time_years' not in st.session_state:
        st.session_state.time_years = 5.0

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.radio("Quick select", options=[1, 3, 5, 10], index=2, horizontal=True, key="preset_time")
    with col_t2:
        cols_btn = st.columns(4)
        for i, y in enumerate([1, 3, 5, 10]):
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
        # Build input dictionary
        input_dict = {
            'GENDER': gender,
            'AGE': age,
            'PIR_GROUP': pir,
            'CKM': ckm_map[ckm],
            'HBA1C': hba1c,
            'ABSI': absi,
            'EGFR': egfr,
            'RDW': rdw,
            'SHR': shr,
            'MCV': mcv,
            'GLB': glb,
            'PLT': plt,
            'CRP': crp,
            'SII': sii,
            'MON': mon,
            'AST': ast,
            'BMI': bmi,
            'TC': tc,
            'SMOKE': smoke,
            'ACTIVITY': activity_val,          # 0/1
            'LUNG': lung,
            'CANCER': cancer
        }

        try:
            X_raw = pd.DataFrame([input_dict])[feature_cols]
            surv_funcs = pipeline.predict_survival_function(X_raw)

            time_month = int(round(st.session_state.time_years * 12))
            if time_month < 1:
                time_month = 1
            surv_prob = surv_funcs[0](time_month)
            risk = 1 - surv_prob

            # ---------- Display results ----------
            st.markdown(f"**Risk at {st.session_state.time_years:.1f} years ({time_month} months)**")
            risk_pct = risk * 100
            surv_pct = surv_prob * 100

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

            # Multi‑time risk table
            st.markdown("#### 📈 Risk at multiple time points")
            preset_months = [12, 36, 60, 120]
            all_risks = {t: 1 - surv_funcs[0](t) for t in preset_months}
            df_compare = pd.DataFrame({
                "Months": [12, 36, 60, 120],
                "Years": [1, 3, 5, 10],
                "Risk (%)": [f"{all_risks[t]*100:.1f}" for t in preset_months]
            })
            st.dataframe(df_compare, use_container_width=True, hide_index=True)

            # ---------- Expert Recommendations based on risk level ----------
            st.markdown("#### 🩺 Clinical Recommendations")

            if risk < 0.2:
                st.success("""
                **Low Risk (< 20%) – Maintain optimal health**  
                - Adhere to **DASH/MIND diet** and ≥150 min/week moderate exercise.  
                - Control **BP < 130/80 mmHg**, **HbA1c < 7%** (if diabetic).  
                - Monitor **eGFR** and **UACR** annually (every 12–24 months).  
                - Continue routine primary care; avoid nephrotoxic agents.
                """)
            elif risk < 0.5:
                st.warning("""
                **Moderate Risk (20–50%) – Intensify cardio‑renal protection**  
                - Optimize **SGLT2i** (e.g., dapagliflozin) and/or **GLP‑1 RA** for heart/kidney benefits.  
                - Initiate or titrate **ACEi/ARB** and consider **non‑steroidal MRA** (e.g., finerenone) if eGFR >25.  
                - Strict BP target **< 130/80 mmHg**; consider **CAC scoring** or biomarkers (NT‑proBNP, hs‑cTn).  
                - Follow‑up every **3–6 months** with renal function and electrolytes.
                """)
            else:
                st.error("""
                **High Risk (> 50%) – Urgent multidisciplinary management**  
                - **Heart‑Kidney‑Metabolism team** consultation.  
                - Combine **RAAS blockade + SGLT2i + diuretics** (volume‑guided).  
                - Monitor **volume status**, **electrolytes**, and early injury biomarkers (KIM‑1, NGAL).  
                - Avoid **NSAIDs** and **contrast media**; assess for advanced heart failure or uremic complications.  
                - Consider **specialized inpatient** or day‑care intensive therapy.
                """)

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
st.caption(f"Model: Pipeline ({MODEL_FILE}) · Random Survival Forest · v2.1")