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
    initial_sidebar_state="collapsed"
)

# Ultra-compact CSS for A4-like fit
st.markdown("""
    <style>
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.3rem;
    }
    h1, h2, h3, .stMarkdown h3 {
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        font-size: 1.2rem !important;
    }
    div.stExpander {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    .stButton button {
        padding: 0.2rem 0.5rem;
        font-size: 0.9rem;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 0.2rem !important;
    }
    .stRadio > div {
        gap: 0.3rem;
    }
    .stRadio label {
        font-size: 0.8rem;
    }
    .stColumn {
        padding: 0 0.2rem !important;
    }
    .stMarkdown {
        font-size: 0.9rem;
    }
    .small-font {
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

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
unit_map = {
    'AGE': 'years', 'PLT': '×10⁹/L', 'MCV': 'fL', 'RDW': '%', 'SII': '×10⁹/L',
    'ABSI': '', 'HBA1C': '%', 'GLB': 'g/L', 'MON': '×10⁹/L', 'EGFR': 'mL/min/1.73m²',
    'CRP': 'mg/dL', 'UA': 'mg/dL', 'SHR': '', 'BMI': 'kg/m²', 'TC': 'mg/dL', 'AST': 'U/L'
}

cat_display_label = {
    'GENDER': 'Sex',
    'CKM': 'CKM stage',
    'ACTIVITY': 'Moderate or vigorous activity',
    'PIR_GROUP': 'Poverty-income ratio category',
    'LUNG': 'Pulmonary disease',
    'EDU': 'Education',
    'RACE': 'Race/Ethnicity',
    'MARITAL': 'Marital Status',
    'CANCER': 'Cancer History'
}

cat_option_map = {
    'GENDER': {'Male': 1, 'Female': 2},
    'CKM': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4},
    'ACTIVITY': {'Yes': 1, 'No': 0},
    'PIR_GROUP': {'<1.0': 1, '1.0-3.0': 2, '>3.0': 3},
    'LUNG': {'Yes': 1, 'No': 0},
    'EDU': {'< High school': 1, 'High school': 2, 'Some college or above': 3},
    'RACE': {
        'Mexican American': 1,
        'Other Hispanic': 2,
        'Non-Hispanic White': 3,
        'Non-Hispanic Black': 4,
        'Other, Including Multi-Racial': 5
    },
    'MARITAL': {'Married or living with partner': 1, 'Others': 2},
    'CANCER': {'Yes': 1, 'No': 0}
}

default_cont_vals = {
    'AGE': 65.0, 'HBA1C': 5.6, 'ABSI': 0.08, 'EGFR': 90.0, 'RDW': 13.2,
    'SHR': 2.5, 'MCV': 92.0, 'GLB': 28.0, 'PLT': 250.0, 'CRP': 1.5,
    'SII': 500.0, 'MON': 0.4, 'AST': 22.0, 'BMI': 24.5, 'TC': 185.0, 'UA': 5.0
}


# -------------------- Build display feature names --------------------
def get_display_feature_names():
    names = []
    for col in CONTINUOUS_FEATURES:
        unit = unit_map.get(col, '')
        names.append(f"{col} ({unit})" if unit else col)
    for col in CATEGORICAL_FEATURES:
        names.append(cat_display_label.get(col, col))
    return names


DISPLAY_FEATURE_NAMES = get_display_feature_names()

# -------------------- UI Layout --------------------
st.markdown("### 🫀 All-Cause Mortality Risk Prediction (ElasticNet)")

col_left, col_right = st.columns([1.1, 1.3], gap="small")

input_data = {}

with col_left:
    st.markdown("**📝 Patient Characteristics**")

    with st.expander("🏷️ Categorical Features", expanded=True):
        cat_cols = st.columns(3)
        for i, col in enumerate(CATEGORICAL_FEATURES):
            display_label = cat_display_label.get(col, col)
            with cat_cols[i % 3]:
                if col in cat_option_map:
                    options = list(cat_option_map[col].keys())
                    selected_text = st.selectbox(display_label, options, key=f"cat_{col}")
                    input_data[col] = cat_option_map[col][selected_text]
                else:
                    classes = label_encoders[col].classes_
                    options = [str(cls) for cls in classes if str(cls) != 'nan']
                    if not options:
                        options = [str(cls) for cls in classes]
                    selected_text = st.selectbox(display_label, options, key=f"cat_{col}")
                    input_data[col] = selected_text

    with st.expander("📈 Continuous Features", expanded=True):
        cont_cols = st.columns(4)
        for i, col in enumerate(CONTINUOUS_FEATURES):
            default_val = default_cont_vals.get(col, 50.0)
            unit = unit_map.get(col, '')
            label = f"{col} ({unit})" if unit else col
            with cont_cols[i % 4]:
                input_data[col] = st.number_input(label, value=float(default_val),
                                                  step=1.0 if default_val > 10 else 0.1,
                                                  key=f"cont_{col}",
                                                  format="%.1f" if default_val < 10 else "%.0f")

    st.markdown("**⏱ Prediction Time**")
    if 'time_years' not in st.session_state:
        st.session_state.time_years = 5.0

    t1, t2 = st.columns([2, 1])
    with t1:
        st.radio("Quick select (Years)", options=[1, 3, 5, 10], index=2,
                 horizontal=True, key="preset_time", label_visibility="collapsed")
    with t2:
        time_years = st.number_input("Custom years", min_value=0.5, max_value=30.0,
                                     value=st.session_state.time_years,
                                     step=0.5, key="time_input", label_visibility="collapsed")
        st.session_state.time_years = time_years

    predict_clicked = st.button("📊 Predict & Show SHAP", type="primary",
                                use_container_width=True)

# -------------------- Prediction & SHAP Results --------------------
with col_right:
    st.markdown("**📊 Results & Interpretation**")

    if predict_clicked:
        try:
            # 1. Prepare Data
            df_input = pd.DataFrame([input_data])

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

            # 2. Predict Survival
            surv_funcs = model.predict_survival_function(X_final)
            time_month = int(round(st.session_state.time_years * 12))
            max_train_month = int(surv_funcs[0].x[-1])
            if time_month > max_train_month:
                st.warning(f"Requested time exceeds training follow-up. Capped to {max_train_month} months.")
                time_month = max_train_month

            surv_prob = surv_funcs[0](time_month)
            risk = 1 - surv_prob
            t_years = st.session_state.time_years

            # Dynamic thresholds (based on 10‑year 7.5% and 20%)
            threshold_low = (0.075 / 10.0) * t_years
            threshold_high = (0.20 / 10.0) * t_years

            # Display Metrics
            st.markdown(f"**Risk at {time_month} Months (~{t_years:.1f} Years)**")
            c_risk, c_surv = st.columns(2)
            if risk < threshold_low:
                color, level = "green", "Low"
            elif risk < threshold_high:
                color, level = "orange", "Intermediate"
            else:
                color, level = "red", "High"

            with c_risk:
                st.metric("Mortality Risk", f"{risk * 100:.1f}%")
                st.markdown(
                    f"<span style='color:{color}; font-weight:bold; font-size:14px'>{level} Risk (thresholds adjusted for {t_years:.1f}y)</span>",
                    unsafe_allow_html=True)
            with c_surv:
                st.metric("Survival Probability", f"{surv_prob * 100:.1f}%")

            # 3. Evidence‑Based Clinical Recommendations (AHA CKM Guidelines)
            with st.expander("🩺 Evidence‑Based Clinical Recommendations", expanded=True):
                st.caption(
                    f"📝 *Based on AHA CKM Syndrome Guidelines. Risk thresholds dynamically adjusted for {t_years:.1f} years.*")

                if risk < threshold_low:
                    st.success(f"""
                    **📉 Low Risk (< {threshold_low * 100:.1f}%)** —— **Primary Prevention & Lifestyle Intervention**
                    - **Life's Essential 8**: Adopt Mediterranean or DASH diet; ≥150 min/week moderate‑intensity exercise; smoking cessation; maintain ideal weight and waist circumference.
                    - **Routine Screening**: Re‑evaluate blood pressure, lipids, fasting glucose/HbA1c, and kidney function (eGFR and UACR) every 1‑3 years.
                    - **Social Determinants (SDOH)**: Assess and address sleep quality, psychological stress, and other potential adverse factors.
                    """)
                elif risk < threshold_high:
                    st.warning(f"""
                    **📊 Intermediate Risk ({threshold_low * 100:.1f}% – < {threshold_high * 100:.1f}%)** —— **Shared Decision‑Making & Early Pharmacotherapy**
                    - **Cardiovascular‑Kidney Protection**: If type 2 diabetes or CKD coexists, initiate **SGLT2 inhibitors** or **GLP‑1 RA** as recommended.
                    - **Risk Factor Control**: Start moderate‑intensity statin; strictly control blood pressure (target <130/80 mmHg, preferably with ACEi/ARB).
                    - **Comorbidity Screening**: Actively screen for MASLD and obstructive sleep apnea (OSA).
                    """)
                else:
                    st.error(f"""
                    **⚠️ High Risk (≥ {threshold_high * 100:.1f}%)** —— **Multidisciplinary Care & Guideline‑Directed Medical Therapy (GDMT)**
                    - **MDT Management**: Strongly recommend a multidisciplinary team (cardiovascular, renal, endocrine) for individualized intervention.
                    - **Intensified GDMT**: Full application of cardio‑renal protective agents (SGLT2i, ACEi/ARB, GLP‑1 RA, or ns‑MRA); intensive lipid‑lowering (high‑intensity statin, ± PCSK9i if needed).
                    - **Close Follow‑up**: Monitor for CKM stages 3–4 (heart failure, advanced CKD); schedule target organ assessment every 3 months.
                    """)

            # 4. SHAP Interpretation (Local)
            st.markdown("**🔍 SHAP Interpretation (Local)**")
            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = coefs[:, 0]

            contributions = X_final[0] * coefs
            base_value = 0.0

            # Build display data
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
                feature_names=DISPLAY_FEATURE_NAMES
            )

            # Top 15 features
            importance_vals = np.abs(contributions)
            sorted_idx = np.argsort(importance_vals)[::-1][:15]
            sorted_names = [DISPLAY_FEATURE_NAMES[i] for i in sorted_idx]
            sorted_vals = importance_vals[sorted_idx]

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                fig_imp, ax_imp = plt.subplots(figsize=(4.5, 3.5))
                ax_imp.barh(sorted_names[::-1], sorted_vals[::-1], color='#1f77b4')
                ax_imp.set_xlabel('|SHAP|', fontsize=8)
                ax_imp.set_title('Top 15 Feature Importance', fontsize=9)
                ax_imp.tick_params(labelsize=7)
                plt.tight_layout()
                st.pyplot(fig_imp, bbox_inches='tight')
                plt.clf()

            with chart_col2:
                fig_wf, ax_wf = plt.subplots(figsize=(4.5, 3.5))
                shap.plots.waterfall(explanation, max_display=15, show=False)
                plt.tick_params(labelsize=7)
                plt.tight_layout()
                st.pyplot(fig_wf, bbox_inches='tight')
                plt.clf()

            # Force plot (compact)
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
                components.html(shap_html, height=160, scrolling=False)
            except Exception as e:
                st.warning(f"Force plot error: {e}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.info("👈 Fill in patient characteristics and click **Predict** to view results.")