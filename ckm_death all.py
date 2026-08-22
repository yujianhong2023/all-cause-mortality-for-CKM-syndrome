import streamlit as st
import pickle
import os
import sys
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components

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

# -------------------- Display name and unit mappings --------------------
# Continuous feature units
unit_map = {
    'AGE': 'years',
    'PLT': '×10⁹/L',
    'MCV': 'fL',
    'RDW': '%',
    'SII': '×10⁹/L',
    'ABSI': '',
    'HBA1C': '%',
    'GLB': 'g/L',
    'MON': '×10⁹/L',
    'EGFR': 'mL/min/1.73m²',
    'CRP': 'mg/dL',
    'UA': 'mg/dL',
    'SHR': '',
    'BMI': 'kg/m²',
    'TC': 'mg/dL',
    'AST': 'U/L'
}

# Display labels for categorical features (if different from original)
cat_display_label = {
    'GENDER': 'Sex',
    'CKM': 'CKM stage',
    'ACTIVITY': 'Moderate or vigorous activity',
    'PIR_GROUP': 'Poverty-income ratio category',
    'LUNG': 'Pulmonary disease',
    'EDU': 'Education'
}

# Option mapping for selected categorical features (display text -> encoded value)
cat_option_map = {
    'GENDER': {'Male': 1, 'Female': 2},
    'CKM': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4},
    'ACTIVITY': {'Yes': 1, 'No': 0},
    'PIR_GROUP': {'<1.0': 1, '1.0-3.0': 2, '>3.0': 3},
    'LUNG': {'Yes': 1, 'No': 0},
    'EDU': {'< High school': 1, 'High school': 2, 'Some college or above': 3}
}

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
            display_label = cat_display_label.get(col, col)
            if col in cat_option_map:
                options = list(cat_option_map[col].keys())
                selected_text = st.selectbox(display_label, options, key=f"cat_{col}")
                input_data[col] = cat_option_map[col][selected_text]
            else:
                options = [str(cls) for cls in label_encoders[col].classes_ if str(cls) != 'nan']
                selected_text = st.selectbox(display_label, options, key=f"cat_{col}")
                input_data[col] = selected_text  # will transform later

    # Continuous Features
    with st.expander("📈 Continuous Features", expanded=True):
        col3, col4 = st.columns(2)
        for i, col in enumerate(CONTINUOUS_FEATURES):
            default_val = default_cont_vals.get(col, 50.0)
            unit = unit_map.get(col, '')
            label = f"{col} ({unit})" if unit else col
            with col3 if i % 2 == 0 else col4:
                input_data[col] = st.number_input(label, value=float(default_val),
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

            # Encode categorical features
            cat_encoded = []
            for col in CATEGORICAL_FEATURES:
                if col in cat_option_map:
                    val = df_input[col].iloc[0]
                    cat_encoded.append(int(val))
                else:
                    val_str = df_input[col].iloc[0]
                    cat_encoded.append(label_encoders[col].transform([val_str])[0])
            cat_encoded = np.array(cat_encoded).reshape(1, -1)

            cont_scaled = scaler.transform(df_input[CONTINUOUS_FEATURES])
            X_final = np.hstack([cont_scaled, cat_encoded])
            final_feature_names = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

            # 2. Predict Survival
            surv_funcs = model.predict_survival_function(X_final)
            time_month = int(round(st.session_state.time_years * 12))
            max_train_month = int(surv_funcs[0].x[-1])
            if time_month > max_train_month:
                st.warning(f"Requested time ({time_month} months) exceeds training follow-up. Capped to {max_train_month} months.")
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

            # 3. Clinical Recommendations (English)
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

            # 4. SHAP Interpretation – Global & Local plots
            st.markdown("### 🔍 Model Interpretation (SHAP)")
            st.caption("Global feature importance (ordered by external test set, if available) and local explanation for this patient.")

            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = coefs[:, 0]

            contributions = X_final[0] * coefs
            base_value = 0.0

            # Build display data (user-friendly values)
            display_data = []
            for col in CONTINUOUS_FEATURES:
                display_data.append(df_input[col].iloc[0])
            for col in CATEGORICAL_FEATURES:
                if col in cat_option_map:
                    val = df_input[col].iloc[0]
                    inv_map = {v: k for k, v in cat_option_map[col].items()}
                    display_data.append(inv_map.get(val, str(val)))
                else:
                    display_data.append(df_input[col].iloc[0])

            explanation = shap.Explanation(
                values=contributions,
                base_values=base_value,
                data=display_data,
                feature_names=final_feature_names
            )

            # ---- Global Feature Importance Bar Plot ----
            # Priority: external test set -> training set -> local absolute values
            importance_vals = None
            importance_source = None

            if 'external_shap_importance' in artifacts and artifacts['external_shap_importance'] is not None:
                importance_vals = artifacts['external_shap_importance']
                importance_source = "External Test Cohort"
            elif 'global_shap_importance' in artifacts and artifacts['global_shap_importance'] is not None:
                importance_vals = artifacts['global_shap_importance']
                importance_source = "Training Cohort"
            else:
                # Fallback: use current patient's absolute SHAP values
                importance_vals = np.abs(contributions)
                importance_source = "Current Patient (local)"

            # Ensure importance_vals length matches number of features
            if len(importance_vals) != len(final_feature_names):
                st.warning("Importance values length mismatch. Falling back to local values.")
                importance_vals = np.abs(contributions)
                importance_source = "Current Patient (local)"

            # Sort descending
            sorted_idx = np.argsort(importance_vals)[::-1]
            sorted_names = [final_feature_names[i] for i in sorted_idx]
            sorted_vals = importance_vals[sorted_idx]

            fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
            ax_imp.barh(sorted_names, sorted_vals, color='#1f77b4')
            ax_imp.set_xlabel('mean(|SHAP value|)')
            ax_imp.set_title(f'Feature Importance ({importance_source})')
            st.pyplot(fig_imp, bbox_inches='tight')
            plt.clf()

            st.divider()

            # ---- Waterfall Plot ----
            st.markdown("#### SHAP Waterfall Plot (Local Interpretation)")
            fig_wf, ax_wf = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(explanation, max_display=12, show=False)
            st.pyplot(fig_wf, bbox_inches='tight')
            plt.clf()

            # ---- Force Plot ----
            st.markdown("#### SHAP Force Plot (Interactive)")
            try:
                shap_force = shap.plots.force(
                    explanation.base_values,
                    explanation.values,
                    explanation.data,
                    feature_names=explanation.feature_names,
                    matplotlib=False,
                    show=False
                )
                if hasattr(shap_force, 'html'):
                    force_html = shap_force.html()
                else:
                    force_html = str(shap_force)
                shap_html = f"<head>{shap.getjs()}</head><body>{force_html}</body>"
                components.html(shap_html, height=400, scrolling=True)
            except Exception as e:
                st.warning(f"Force plot could not be rendered: {e}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Please verify your inputs and ensure 'ENet_model.pkl' is correctly generated.")
    else:
        st.info("👈 Fill in patient characteristics and click **Predict** to view Risk, Guidelines, and SHAP plots.")

st.divider()
st.caption(f"Model File: {MODEL_FILE} · Engine: ElasticNet Survival Analysis · Includes SHAP Interpretations")