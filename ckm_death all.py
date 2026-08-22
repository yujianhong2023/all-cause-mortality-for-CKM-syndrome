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
# 使用宽屏模式，配合紧凑的列布局
st.set_page_config(
    page_title="All-Cause Mortality Risk Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"  # 默认收起侧边栏以留出更多空间
)

# 自定义CSS以减少全局内边距和组件间距，使得界面更紧凑
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
    'GENDER': 'Sex', 'CKM': 'CKM stage', 'ACTIVITY': 'Moderate or vigorous activity',
    'PIR_GROUP': 'Poverty-income ratio category', 'LUNG': 'Pulmonary disease', 'EDU': 'Education'
}

cat_option_map = {
    'GENDER': {'Male': 1, 'Female': 2},
    'CKM': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4},
    'ACTIVITY': {'Yes': 1, 'No': 0},
    'PIR_GROUP': {'<1.0': 1, '1.0-3.0': 2, '>3.0': 3},
    'LUNG': {'Yes': 1, 'No': 0},
    'EDU': {'< High school': 1, 'High school': 2, 'Some college or above': 3}
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

col_left, col_right = st.columns([1.1, 1], gap="small")
input_data = {}

with col_left:
    st.markdown("**📝 Patient Characteristics**")

    # Categorical Features (3 columns for compactness)
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
                    options = [str(cls) for cls in label_encoders[col].classes_ if str(cls) != 'nan']
                    selected_text = st.selectbox(display_label, options, key=f"cat_{col}")
                    input_data[col] = selected_text

    # Continuous Features (4 columns for compactness)
    with st.expander("📈 Continuous Features", expanded=True):
        cont_cols = st.columns(4)
        for i, col in enumerate(CONTINUOUS_FEATURES):
            default_val = default_cont_vals.get(col, 50.0)
            unit = unit_map.get(col, '')
            label = f"{col} ({unit})" if unit else col
            with cont_cols[i % 4]:
                input_data[col] = st.number_input(label, value=float(default_val),
                                                  step=1.0 if default_val > 10 else 0.1, key=f"cont_{col}")

    # Prediction Time
    st.markdown("**⏱ Prediction Time**")
    if 'time_years' not in st.session_state:
        st.session_state.time_years = 5.0

    t1, t2 = st.columns([2, 1])
    with t1:
        st.radio("Quick select (Years)", options=[1, 3, 5, 10], index=2, horizontal=True, key="preset_time",
                 label_visibility="collapsed")
    with t2:
        time_years = st.number_input("Custom years", min_value=0.5, max_value=30.0, value=st.session_state.time_years,
                                     step=0.5, key="time_input", label_visibility="collapsed")
        st.session_state.time_years = time_years

    predict_clicked = st.button("📊 Predict & Show SHAP", type="primary", use_container_width=True)

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

            # Display Metrics
            st.markdown(f"**Risk at {time_month} Months (~{st.session_state.time_years:.1f} Years)**")
            c_risk, c_surv = st.columns(2)
            if risk < 0.075:
                color, level = "green", "Low"
            elif risk < 0.20:
                color, level = "orange", "Moderate"
            else:
                color, level = "red", "High"

            with c_risk:
                st.metric("Mortality Risk", f"{risk * 100:.1f}%")
                st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:14px'>{level} Risk</span>",
                            unsafe_allow_html=True)
            with c_surv:
                st.metric("Survival Probability", f"{surv_prob * 100:.1f}%")

            # 3. Clinical Recommendations (collapsible, minimal text)
            with st.expander("🩺 Clinical Recommendations", expanded=False):
                if risk < 0.075:
                    st.success("**📉 Low Risk** – Routine screening. Maintain healthy lifestyle.")
                elif risk < 0.20:
                    st.warning("**📊 Moderate Risk** – Consider pharmacotherapy & evaluate comorbidities.")
                else:
                    st.error("**⚠️ High Risk** – Multidisciplinary intensive management. Follow-up regularly.")

            # 4. SHAP Interpretation
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

            # 移除 Global 选项，仅保留 Local
            importance_vals = np.abs(contributions)

            # Sort and plot compact bar chart
            sorted_idx = np.argsort(importance_vals)[::-1][:10]  # 仅显示Top 10以节省空间
            sorted_names = [DISPLAY_FEATURE_NAMES[i] for i in sorted_idx]
            sorted_vals = importance_vals[sorted_idx]

            # 安排在一行展示柱状图和瀑布图
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig_imp, ax_imp = plt.subplots(figsize=(4, 2.5))
                ax_imp.barh(sorted_names[::-1], sorted_vals[::-1], color='#1f77b4')
                ax_imp.set_xlabel('|SHAP|', fontsize=8)
                ax_imp.set_title('Top 10 Feature Importance', fontsize=9)
                ax_imp.tick_params(labelsize=7)
                st.pyplot(fig_imp, bbox_inches='tight')
                plt.clf()

            with chart_col2:
                fig_wf, ax_wf = plt.subplots(figsize=(4, 2.5))
                shap.plots.waterfall(explanation, max_display=6, show=False)
                plt.tick_params(labelsize=7)
                st.pyplot(fig_wf, bbox_inches='tight')
                plt.clf()

            # Force Plot (compact height)
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
                components.html(shap_html, height=150, scrolling=False)
            except Exception as e:
                st.warning(f"Force plot error: {e}")

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.info("👈 Fill in patient characteristics and click **Predict** to view results.")