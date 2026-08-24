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

st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    h1, h2, h3 {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
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
    with open(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        return pickle.load(f)

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
label_encoders = artifacts.get('label_encoders', {})
FEATURES = artifacts.get('features', [])
CATEGORICAL_FEATURES = artifacts.get('categorical_features', [])
CONTINUOUS_FEATURES = artifacts.get('continuous_features', [])

# Get scaler features (actual features used during training)
scaler_features = artifacts.get('scaler_features', None)
if scaler_features is None:
    scaler_features = CONTINUOUS_FEATURES[:scaler.n_features_in_]

if len(scaler_features) != scaler.n_features_in_:
    st.error(f"Feature count mismatch: scaler_features has {len(scaler_features)}, scaler expects {scaler.n_features_in_}")
    sys.exit("Feature count mismatch")

# Sidebar debug info (optional, can be removed)
st.sidebar.write(f"✅ Scaler expects: {scaler.n_features_in_} features")
st.sidebar.write(f"✅ Features used: {scaler_features}")

# -------------------- Display name mappings --------------------
unit_map = {
    'AGE': 'years', 'PLT': '×10⁹/L', 'MCV': 'fL', 'RDW': '%', 'SII': '×10⁹/L',
    'ABSI': '', 'HBA1C': '%', 'GLB': 'g/L', 'MON': '×10⁹/L', 'EGFR': 'mL/min/1.73m²',
    'CRP': 'mg/dL', 'UA': 'mg/dL', 'SHR': '', 'BMI': 'kg/m²', 'TC': 'mg/dL', 'AST': 'U/L'
}
cat_display_label = {
    'GENDER': 'Sex', 'CKM': 'CKM Stage', 'ACTIVITY': 'Physical Activity Level',
    'PIR_GROUP': 'Poverty-Income Ratio Category', 'LUNG': 'Pulmonary Disease',
    'EDU': 'Education Level', 'RACE': 'Race/Ethnicity', 'MARITAL': 'Marital Status',
    'CANCER': 'History of Cancer'
}
cat_option_map = {
    'GENDER': {'Male': 1, 'Female': 2},
    'CKM': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4},
    'ACTIVITY': {'Yes': 1, 'No': 0},
    'PIR_GROUP': {'<1.0': 1, '1.0-3.0': 2, '>3.0': 3},
    'LUNG': {'Yes': 1, 'No': 0},
    'EDU': {'< High School': 1, 'High School': 2, 'Some College or Above': 3},
    'RACE': {
        'Mexican American': 1,
        'Other Hispanic': 2,
        'Non-Hispanic White': 3,
        'Non-Hispanic Black': 4,
        'Other, Including Multi-Racial': 5
    },
    'MARITAL': {'Married/Living with Partner': 1, 'Others': 2},
    'CANCER': {'Yes': 1, 'No': 0}
}
default_cont_vals = {
    'AGE': 65.0, 'HBA1C': 5.6, 'ABSI': 0.08, 'EGFR': 90.0, 'RDW': 13.2,
    'SHR': 2.5, 'MCV': 92.0, 'GLB': 28.0, 'PLT': 250.0, 'CRP': 1.5,
    'SII': 500.0, 'MON': 0.4, 'AST': 22.0, 'BMI': 24.5, 'TC': 185.0, 'UA': 5.0
}

# Build display names for SHAP (continuous + categorical)
display_names = []
for col in scaler_features:
    unit = unit_map.get(col, '')
    display_names.append(f"{col} ({unit})" if unit else col)
for col in CATEGORICAL_FEATURES:
    display_names.append(cat_display_label.get(col, col))

# -------------------- UI Layout --------------------
st.markdown("### 🫀 All-Cause Mortality Risk Prediction (ElasticNet)")
col_left, col_right = st.columns([1.1, 1], gap="small")
input_data = {}

with col_left:
    st.markdown("**📝 Patient Characteristics**")

    # Categorical features
    with st.expander("🏷️ Categorical Features", expanded=True):
        cat_cols = st.columns(3)
        for i, col in enumerate(CATEGORICAL_FEATURES):
            display_label = cat_display_label.get(col, col)
            with cat_cols[i % 3]:
                if col in cat_option_map:
                    options = list(cat_option_map[col].keys())
                    selected = st.selectbox(display_label, options, key=f"cat_{col}")
                    input_data[col] = cat_option_map[col][selected]
                else:
                    options = [str(cls) for cls in label_encoders[col].classes_ if str(cls) != 'nan']
                    selected = st.selectbox(display_label, options, key=f"cat_{col}")
                    input_data[col] = selected

    # Continuous features (using scaler_features)
    with st.expander("📈 Continuous Features", expanded=True):
        cont_cols = st.columns(4)
        for i, col in enumerate(scaler_features):
            default_val = default_cont_vals.get(col, 50.0)
            unit = unit_map.get(col, '')
            label = f"{col} ({unit})" if unit else col
            with cont_cols[i % 4]:
                input_data[col] = st.number_input(label, value=float(default_val),
                                                  step=1.0 if default_val > 10 else 0.1,
                                                  key=f"cont_{col}")

    # Prediction Time
    st.markdown("**⏱ Prediction Time**")
    if 'time_years' not in st.session_state:
        st.session_state.time_years = 5.0
    t1, t2 = st.columns([2, 1])
    with t1:
        st.radio("Quick select (Years)", options=[1, 3, 5, 10], index=2, horizontal=True,
                 key="preset_time", label_visibility="collapsed")
    with t2:
        time_years = st.number_input("Custom years", min_value=0.5, max_value=30.0,
                                     value=st.session_state.time_years, step=0.5,
                                     key="time_input", label_visibility="collapsed")
        st.session_state.time_years = time_years

    predict_clicked = st.button("📊 Predict & Show SHAP", type="primary", use_container_width=True)

# -------------------- Prediction & Results --------------------
with col_right:
    st.markdown("**📊 Results & Interpretation**")
    if predict_clicked:
        try:
            # Build continuous array
            cont_values = []
            for col in scaler_features:
                if col in input_data:
                    cont_values.append(float(input_data[col]))
                else:
                    st.error(f"Missing continuous feature: {col}")
                    st.stop()
            cont_array = np.array(cont_values).reshape(1, -1)
            cont_scaled = scaler.transform(cont_array)

            # Build categorical array
            cat_values = []
            for col in CATEGORICAL_FEATURES:
                if col in input_data:
                    if col in cat_option_map:
                        cat_values.append(int(input_data[col]))
                    else:
                        val_str = str(input_data[col])
                        cat_values.append(label_encoders[col].transform([val_str])[0])
                else:
                    st.error(f"Missing categorical feature: {col}")
                    st.stop()
            cat_array = np.array(cat_values).reshape(1, -1)

            X_final = np.hstack([cont_scaled, cat_array])

            # --- Survival prediction ---
            surv_funcs = model.predict_survival_function(X_final)
            time_month = int(round(st.session_state.time_years * 12))
            max_month = int(surv_funcs[0].x[-1])
            if time_month > max_month:
                st.warning(f"Time capped to {max_month} months.")
                time_month = max_month
            surv_prob = surv_funcs[0](time_month)
            risk = 1 - surv_prob

            # Dynamic thresholds (10-year baseline: 7.5% and 20%)
            t_years = st.session_state.time_years
            threshold_low = (0.075 / 10.0) * t_years
            threshold_high = (0.20 / 10.0) * t_years

            if risk < threshold_low:
                color, level = "green", "Low"
            elif risk < threshold_high:
                color, level = "orange", "Moderate"
            else:
                color, level = "red", "High"

            # Display metrics
            st.markdown(f"**Risk at {time_month} Months (~{t_years:.1f} Years)**")
            c_risk, c_surv = st.columns(2)
            with c_risk:
                st.metric("Mortality Risk", f"{risk * 100:.1f}%")
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>{level} Risk</span>", unsafe_allow_html=True)
            with c_surv:
                st.metric("Survival Probability", f"{surv_prob * 100:.1f}%")

            # --- Clinical Recommendations (English) ---
            with st.expander("🩺 Evidence-Based Clinical Recommendations", expanded=False):
                if risk < threshold_low:
                    st.success(f"""
                    **📉 Low Risk** (< {threshold_low*100:.1f}%) – **Primary Prevention & Lifestyle Modification**
                    - Adopt a Mediterranean or DASH diet.
                    - Engage in ≥150 min/week of moderate-intensity exercise.
                    - Maintain ideal body weight and waist circumference.
                    - Screen blood pressure, lipids, glucose/HbA1c, and kidney function (eGFR, UACR) every 1–3 years.
                    - Address sleep quality and psychosocial stress.
                    """)
                elif risk < threshold_high:
                    st.warning(f"""
                    **📊 Intermediate Risk** ({threshold_low*100:.1f}% – < {threshold_high*100:.1f}%) – **Shared Decision-Making & Early Pharmacotherapy**
                    - For T2DM or CKD, consider initiating **SGLT2 inhibitors** or **GLP-1 receptor agonists**.
                    - Start moderate‑intensity statin therapy; target BP <130/80 mmHg (prefer ACEi/ARB).
                    - Screen for MASLD and obstructive sleep apnea.
                    - Discuss lifestyle intensification with patient.
                    """)
                else:
                    st.error(f"""
                    **⚠️ High Risk** (≥ {threshold_high*100:.1f}%) – **Multidisciplinary Team Management & Intensified GDMT**
                    - Involve cardiology, nephrology, and endocrinology specialists.
                    - Full implementation of cardiorenal protective agents (SGLT2i, ACEi/ARB, GLP-1 RA, ns-MRA).
                    - High‑intensity statin; consider PCSK9i if needed.
                    - Close follow‑up every 3 months; monitor for progression to CKM stages 3–4.
                    """)

            # --- SHAP Interpretation ---
            st.markdown("**🔍 SHAP Interpretation (Local)**")
            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = coefs[:, 0]
            contributions = X_final[0] * coefs
            base_value = 0.0

            # Build display data
            display_data = []
            for col in scaler_features:
                display_data.append(input_data[col])
            for col in CATEGORICAL_FEATURES:
                if col in cat_option_map:
                    inv = {v: k for k, v in cat_option_map[col].items()}
                    display_data.append(inv.get(input_data[col], str(input_data[col])))
                else:
                    display_data.append(input_data[col])

            explanation = shap.Explanation(
                values=contributions,
                base_values=base_value,
                data=display_data,
                feature_names=display_names
            )

            # --- Waterfall and Bar Plot (side by side) ---
            col_wf, col_bar = st.columns(2)

            with col_wf:
                st.markdown("**Waterfall Plot**")
                try:
                    shap.plots.waterfall(explanation, max_display=10, show=False)
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception as e:
                    st.warning(f"Waterfall plot error: {e}")

            with col_bar:
                st.markdown("**Top 10 Feature Importance**")
                importance = np.abs(contributions)
                sorted_idx = np.argsort(importance)[::-1][:10]
                fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
                ax_bar.barh([display_names[i] for i in sorted_idx[::-1]],
                            importance[sorted_idx[::-1]], color='#1f77b4')
                ax_bar.set_xlabel('|SHAP|')
                ax_bar.set_title('Top 10 Features')
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.clf()

            # --- Force Plot (full width, interactive) ---
            st.markdown("**SHAP Force Plot**")
            try:
                force_plot = shap.plots.force(
                    explanation.base_values,
                    explanation.values,
                    explanation.data,
                    feature_names=explanation.feature_names,
                    matplotlib=False,
                    show=False
                )
                if hasattr(force_plot, 'html'):
                    force_html = force_plot.html()
                else:
                    force_html = str(force_plot)
                shap_html = f"<head>{shap.getjs()}</head><body>{force_html}</body>"
                components.html(shap_html, height=200, scrolling=True)
            except Exception as e:
                st.warning(f"Force plot error: {e}")

            # --- Feature Summary Table (optional) ---
            with st.expander("📋 Input Features Summary", expanded=False):
                summary_df = pd.DataFrame({
                    'Feature': FEATURES,
                    'Type': ['Continuous' if f in scaler_features else 'Categorical' for f in FEATURES],
                    'Value': [input_data.get(f, 'N/A') for f in FEATURES]
                })
                st.dataframe(summary_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.info("Please check inputs.")
    else:
        st.info("👈 Fill in patient characteristics and click **Predict**.")