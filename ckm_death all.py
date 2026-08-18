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

# Feature list
feature_cols = [
    'AGE', 'CKM', 'HBA1C', 'ABSI', 'EGFR', 'RDW', 'SHR', 'MCV',
    'GLB', 'PLT', 'CRP', 'SII', 'MON', 'AST', 'SMOKE', 'BMI',
    'TC', 'ACTIVITY', 'PIR_GROUP', 'GENDER', 'LUNG', 'CANCER'
]

# CKM mapping
ckm_map = {'Stage 0': 0, 'Stage 1': 1, 'Stage 2': 2, 'Stage 3': 3, 'Stage 4': 4}

# UI options
binary_choices = ['No', 'Yes']
ckm_options = list(ckm_map.keys())
gender_options = ['Male', 'Female']
pir_options = ['<1.0', '1.0–2.9', '≥3.0']
smoke_options = ['Never', 'Former', 'Current']
activity_options = ['No', 'Yes']   # Moderate or vigorous activity

# -------------------- Title --------------------
st.title("🫀 All‑Cause Mortality Risk Prediction")
st.markdown("**Random Survival Forest** – Enter patient characteristics to estimate risk.")

# -------------------- Main layout: two columns (left for inputs, right for results) --------------------
col_left, col_right = st.columns([2, 1.2], gap="medium")

# ==================== LEFT COLUMN: INPUTS ====================
with col_left:
    # Use a form to batch inputs and reduce reruns
    with st.form(key="prediction_form"):
        st.subheader("📝 Patient Data")

        # ----- Row 1: Demographics (3 columns) -----
        r1 = st.columns(3)
        with r1[0]:
            gender = st.selectbox("Sex", gender_options, key="gender")
        with r1[1]:
            age = st.number_input("Age (years)", min_value=18, max_value=120, value=65, step=1, key="age")
        with r1[2]:
            pir = st.selectbox("PIR Group", pir_options, key="pir")

        # ----- Row 2: CKM & key labs (3 columns) -----
        r2 = st.columns(3)
        with r2[0]:
            ckm = st.selectbox("CKM Stage", ckm_options, key="ckm")
        with r2[1]:
            hba1c = st.number_input("HbA1c (%)", value=5.6, step=0.1, format="%.1f", key="hba1c")
        with r2[2]:
            egfr = st.number_input("eGFR (mL/min/1.73m²)", value=90, step=1, key="egfr")

        # ----- Row 3: Anthropometric (3 columns) -----
        r3 = st.columns(3)
        with r3[0]:
            bmi = st.number_input("BMI (kg/m²)", value=24.5, step=0.1, key="bmi")
        with r3[1]:
            absi = st.number_input("A Body Shape Index", value=0.08, step=0.01, format="%.3f", key="absi")
        with r3[2]:
            shr = st.number_input("Stress Hyperglycemia Ratio", value=2.5, step=0.1, key="shr")

        # ----- Row 4: Hematologic (3 columns) -----
        r4 = st.columns(3)
        with r4[0]:
            rdw = st.number_input("RDW (%)", value=13.2, step=0.1, format="%.1f", key="rdw")
        with r4[1]:
            mcv = st.number_input("MCV (fL)", value=92, step=1, key="mcv")
        with r4[2]:
            plt = st.number_input("PLT (×10⁹/L)", value=250, step=1, key="plt")

        # ----- Row 5: Biochemistry (3 columns) -----
        r5 = st.columns(3)
        with r5[0]:
            glb = st.number_input("GLB (g/L)", value=28, step=1, key="glb")
        with r5[1]:
            ast = st.number_input("AST (U/L)", value=22, step=1, key="ast")
        with r5[2]:
            crp = st.number_input("CRP (mg/dL)", value=1.5, step=0.1, key="crp")

        # ----- Row 6: Other labs (3 columns) -----
        r6 = st.columns(3)
        with r6[0]:
            sii = st.number_input("SII (×10⁹/L)", value=500, step=10, key="sii")
        with r6[1]:
            mon = st.number_input("MON (×10⁹/L)", value=0.4, step=0.1, key="mon")
        with r6[2]:
            tc = st.number_input("TC (mg/dL)", value=185, step=1, key="tc")

        # ----- Row 7: Lifestyle & History (3 columns) -----
        r7 = st.columns(3)
        with r7[0]:
            smoke = st.selectbox("Smoking Status", smoke_options, key="smoke")
        with r7[1]:
            activity = st.selectbox("Moderate or vigorous activity", activity_options, key="activity")
        with r7[2]:
            lung_yn = st.selectbox("Pulmonary Disease", binary_choices, key="lung")
            lung = 1 if lung_yn == 'Yes' else 0

        # Row 8: Cancer (only one item, but we can put it with other things or separate)
        r8 = st.columns(3)
        with r8[0]:
            cancer_yn = st.selectbox("Cancer History", binary_choices, key="cancer")
            cancer = 1 if cancer_yn == 'Yes' else 0
        # Leave other columns empty

        # ----- Prediction time -----
        st.subheader("⏱ Prediction Time")
        time_cols = st.columns([2, 1, 1, 1, 1])
        with time_cols[0]:
            st.radio("Quick select", options=[1, 3, 5, 10], index=2, horizontal=True, key="preset_time")
        with time_cols[1]:
            if st.button("1 yr", key="btn_1"):
                st.session_state.time_years = 1.0
        with time_cols[2]:
            if st.button("3 yr", key="btn_3"):
                st.session_state.time_years = 3.0
        with time_cols[3]:
            if st.button("5 yr", key="btn_5"):
                st.session_state.time_years = 5.0
        with time_cols[4]:
            if st.button("10 yr", key="btn_10"):
                st.session_state.time_years = 10.0

        if 'time_years' not in st.session_state:
            st.session_state.time_years = 5.0
        time_years = st.number_input("Custom years", min_value=0.5, max_value=30.0,
                                     value=st.session_state.time_years, step=0.5, key="time_input")
        if time_years != st.session_state.time_years:
            st.session_state.time_years = time_years

        # ----- Submit button -----
        submitted = st.form_submit_button("📊 Predict Mortality Risk", type="primary", use_container_width=True)

# ==================== RIGHT COLUMN: RESULTS ====================
with col_right:
    st.subheader("📊 Prediction Result")

    if submitted:
        # Build input dict
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
            'ACTIVITY': 1 if activity == 'Yes' else 0,
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

            # Display metrics
            risk_pct = risk * 100
            surv_pct = surv_prob * 100

            if risk < 0.2:
                level, color = "Low", "green"
            elif risk < 0.5:
                level, color = "Moderate", "orange"
            else:
                level, color = "High", "red"

            st.markdown(f"**Risk at {st.session_state.time_years:.1f} yrs**")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Mortality", f"{risk_pct:.1f}%")
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>{level}</span>", unsafe_allow_html=True)
            with c2:
                st.metric("Survival", f"{surv_pct:.1f}%")

            # Multi-time table
            st.markdown("#### 📈 Risk over time")
            preset_months = [12, 36, 60, 120]
            all_risks = {t: 1 - surv_funcs[0](t) for t in preset_months}
            df_compare = pd.DataFrame({
                "Year": ["1", "3", "5", "10"],
                "Risk (%)": [f"{all_risks[t]*100:.1f}" for t in preset_months]
            })
            st.dataframe(df_compare, use_container_width=True, hide_index=True)

            # Expert recommendations
            st.markdown("#### 🩺 Recommendations")
            if risk < 0.2:
                st.success("""
                **Low Risk** – Maintain lifestyle (DASH/MIND diet, 150 min/week exercise).  
                Control BP <130/80, HbA1c <7%. Monitor eGFR/UACR annually.
                """)
            elif risk < 0.5:
                st.warning("""
                **Moderate Risk** – Optimize SGLT2i/GLP‑1 RA, ACEi/ARB, consider MRA (finerenone).  
                Strict BP <130/80, consider CAC or biomarkers (NT‑proBNP, hs‑cTn).  
                Follow‑up every 3–6 months.
                """)
            else:
                st.error("""
                **High Risk** – Urgent multidisciplinary (heart‑kidney‑metabolism team).  
                Combine RAASi + SGLT2i + diuretics. Monitor volume, electrolytes, KIM‑1/NGAL.  
                Avoid NSAIDs/contrast. Consider inpatient intensive therapy.
                """)

            st.caption("⚠️ For research only – not medical advice.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Please verify inputs and model files.")
    else:
        st.info("👈 Enter patient data and click **Predict**.")
        st.markdown("""
        **Instructions:**
        - All fields required.
        - Use drop‑downs for categorical variables.
        - Numerical values direct entry.
        - Set prediction time in years.
        """)

# -------------------- Footer --------------------
st.divider()
st.caption(f"Model: Pipeline ({MODEL_FILE}) · Random Survival Forest · v2.2")