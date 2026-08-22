import streamlit as st
import pickle
import os
import sys
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import shap

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
MODEL_FILE = 'ENet_model.pkl'

if not os.path.exists(os.path.join(MODEL_DIR, MODEL_FILE)):
    st.error(f"❌ Model file '{MODEL_FILE}' not found. Please ensure it is in the same directory.")
    sys.exit(f"Error: Model file '{MODEL_FILE}' not found.")

# -------------------- Load Model Artifacts --------------------
@st.cache_resource
def load_artifacts():
    try:
        with open(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        return None

artifacts = load_artifacts()

if artifacts is None:
    st.error(f"Failed to load the model file ({MODEL_FILE}). It might be corrupted or incompatible.")
    sys.exit("Error: Failed to load model.")

model = artifacts.get('model')
scaler = artifacts.get('scaler')
label_encoders = artifacts.get('label_encoders')
FEATURES = artifacts.get('features')
CATEGORICAL_FEATURES = artifacts.get('categorical_features')
CONTINUOUS_FEATURES = artifacts.get('continuous_features')

# Default physiological reference values for initialization
default_cont_vals = {
    'AGE': 65.0, 'HBA1C': 5.6, 'ABSI': 0.08, 'EGFR': 90.0, 'RDW': 13.2,
    'SHR': 2.5, 'MCV': 92.0, 'GLB': 28.0, 'PLT': 250.0, 'CRP': 1.5,
    'SII': 500.0, 'MON': 0.4, 'AST': 22.0, 'BMI': 24.5, 'TC': 185.0,
    'UA': 5.0
}

# -------------------- UI Layout --------------------
st.title("🫀 All-Cause Mortality Risk Prediction")
st.markdown(
    "**ElasticNet Survival Model** – Enter patient characteristics to estimate risk and explore feature contributions (SHAP).")

col_left, col_right = st.columns([1, 1.2], gap="large")
input_data = {}

with col_left:
    st.subheader("📝 Patient Characteristics")

    # Categorical Features
    with st.expander("🏷️ Categorical Features", expanded=True):
        col1, col2 = st.columns(2)
        for i, col in enumerate(CATEGORICAL_FEATURES):
            options = [str(cls) for cls in label_encoders[col].classes_ if str(cls) != 'nan']
            with col1 if i % 2 == 0 else col2:
                input_data[col] = st.selectbox(f"{col}", options, key=f"cat_{col}")

    # Continuous Features
    with st.expander("📈 Continuous Features", expanded=True):
        col3, col4 = st.columns(2)
        for i, col in enumerate(CONTINUOUS_FEATURES):
            default_val = default_cont_vals.get(col, 50.0)
            display_label = f"{col}"
            if col in ['CRP', 'UA', 'TC']:
                display_label += " (mg/dL)"
            with col3 if i % 2 == 0 else col4:
                input_data[col] = st.number_input(display_label, value=float(default_val),
                                                  step=1.0 if default_val > 10 else 0.1, key=f"cont_{col}")

    # Prediction Time
    st.subheader("⏱ Prediction Time")
    if 'time_years' not in st.session_state:
        st.session_state.time_years = 5.0

    t1, t2 = st.columns([2, 1])
    with t1:
        st.radio("Quick select (Years)", options=[1, 3, 5, 10], index=2, horizontal=True, key="preset_time")
    with t2:
        time_years = st.number_input("Custom years", min_value=0.5, max_value=30.0, value=st.session_state.time_years,
                                     step=0.5, key="time_input")
        st.session_state.time_years = time_years

    predict_clicked = st.button("📊 Predict Mortality Risk & Show SHAP", type="primary", use_container_width=True)

# -------------------- Prediction & SHAP Results --------------------
with col_right:
    st.subheader("📊 Prediction & Interpretation")

    if predict_clicked:
        try:
            # 1. Prepare Data
            df_input = pd.DataFrame([input_data])
            cont_scaled = scaler.transform(df_input[CONTINUOUS_FEATURES])

            cat_encoded = []
            for col in CATEGORICAL_FEATURES:
                cat_encoded.append(label_encoders[col].transform([df_input[col].iloc[0]])[0])
            cat_encoded = np.array(cat_encoded).reshape(1, -1)

            X_final = np.hstack([cont_scaled, cat_encoded])
            final_feature_names = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

            # 2. Predict Survival
            surv_funcs = model.predict_survival_function(X_final)
            time_month = int(round(st.session_state.time_years * 12))

            max_train_month = int(surv_funcs[0].x[-1])
            if time_month > max_train_month:
                st.warning(
                    f"Requested time ({time_month} months) exceeds training follow-up. Capped to {max_train_month} months.")
                time_month = max_train_month

            surv_prob = surv_funcs[0](time_month)
            risk = 1 - surv_prob

            # Display Metrics
            st.markdown(f"### Risk at {time_month} Months (~{st.session_state.time_years:.1f} Years)")

            c_risk, c_surv = st.columns(2)
            if risk < 0.075:
                color, level = "green", "Low"
            elif risk < 0.20:
                color, level = "orange", "Moderate"
            else:
                color, level = "red", "High"

            with c_risk:
                st.metric("Mortality Risk", f"{risk * 100:.1f}%")
                st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:18px'>{level} Risk</span>",
                            unsafe_allow_html=True)
            with c_surv:
                st.metric("Survival Probability", f"{surv_prob * 100:.1f}%")

            # 3. Clinical Recommendations based on Risk Level (English version)
            st.markdown("### 🩺 Clinical Recommendations")

            if risk < 0.075:
                st.success("""
                **📉 Low Risk Group (Predicted Probability < 7.5%)**  
                **Core Strategy:** Primary prevention – maintain optimal cardiovascular health.

                **Clinical Guidance:**
                - **Continuous Monitoring:** Routine screening for metabolic risk factors and renal function is recommended for all adults.
                - **Healthy Lifestyle:** Emphasize maintaining normal weight, blood glucose, blood pressure, and lipids through lifestyle interventions (e.g., balanced diet, regular exercise).
                - **Assess Social Determinants of Health (SDOH):** Screen and intervene for unfavorable SDOH, which is a core component of holistic care.
                """)
            elif risk < 0.20:
                st.warning("""
                **📊 Moderate Risk Group (Predicted Probability 7.5% – < 20%)**  
                **Core Strategy:** Initiate pharmacological intervention, prioritizing medications with cardiorenal protective effects.

                **Clinical Guidance:**
                - **Initiate Cardiorenal Protective Medications:** For patients with type 2 diabetes or high cardiovascular risk, consider Sodium‑Glucose Cotransporter‑2 inhibitors (SGLT2i) or Glucagon‑Like Peptide‑1 Receptor Agonists (GLP‑1 RA).
                - **Intensive Lifestyle Intervention:** Combine lifestyle modifications with weight‑loss medications when appropriate.
                - **Assess CKM‑related Comorbidities:** Evaluate for pre‑heart failure, Metabolic Dysfunction‑Associated Steatotic Liver Disease (MASLD), Obstructive Sleep Apnea (OSA), etc.
                """)
            else:
                st.error("""
                **⚠️ High Risk Group (Predicted Probability ≥ 20%)**  
                **Core Strategy:** Intensify multidisciplinary comprehensive management – this risk level is one of the criteria for defining Stage 3 CKM syndrome.

                **Clinical Guidance:**
                - **Initiate Multidisciplinary Collaboration:** Requires coordinated care among cardiology, nephrology, endocrinology, and other specialists (MDT).
                - **Implement Guideline‑Directed Medical Therapy (GDMT):**
                  - *Blood Pressure:* Based on Renin‑Angiotensin System inhibitors (RASi).
                  - *Blood Glucose:* Prioritize SGLT2i and GLP‑1 RA.
                  - *Lipids:* Statin therapy as the cornerstone.
                - **Close Follow‑up:** Patients with CKM syndrome stages 2–4 should be followed at least 4 times per year.
                """)

            st.divider()

            # 4. SHAP Interpretation (Waterfall only)
            st.markdown("### 🔍 Model Interpretation (SHAP)")
            st.caption("How each feature contributes to the patient's individual log‑hazard score.")

            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = coefs[:, 0]

            contributions = X_final[0] * coefs

            display_data = []
            for col in CONTINUOUS_FEATURES:
                display_data.append(df_input[col].iloc[0])
            for col in CATEGORICAL_FEATURES:
                display_data.append(df_input[col].iloc[0])

            explanation = shap.Explanation(
                values=contributions,
                base_values=0.0,
                data=display_data,
                feature_names=final_feature_names
            )

            # SHAP Waterfall Plot
            st.markdown("#### SHAP Waterfall Plot")
            fig_waterfall, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(explanation, max_display=12, show=False)
            st.pyplot(fig_waterfall, bbox_inches='tight')
            plt.clf()

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Please verify your inputs and ensure 'ENet_model.pkl' is correctly generated.")
    else:
        st.info("👈 Fill in patient characteristics and click **Predict** to view Risk, Guidelines, and SHAP plots.")

st.divider()
st.caption(f"Model File: {MODEL_FILE} · Engine: ElasticNet Survival Analysis · Includes SHAP Interpretations")