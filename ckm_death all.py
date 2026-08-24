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

# 关键：获取scaler实际训练时使用的特征列表（如果保存了的话）
scaler_features = artifacts.get('scaler_features', None)
if scaler_features is None:
    # 如果模型中没有保存scaler_features，则从CONTINUOUS_FEATURES中取前scaler.n_features_in_个
    st.warning("模型未保存'scaler_features'，将使用CONTINUOUS_FEATURES的前N个特征。")
    scaler_features = CONTINUOUS_FEATURES[:scaler.n_features_in_]

# 验证特征数量
if len(scaler_features) != scaler.n_features_in_:
    st.error(f"特征数量不匹配: scaler_features有{len(scaler_features)}个，scaler期望{scaler.n_features_in_}个")
    sys.exit("特征数量不匹配")

# 显示调试信息
st.sidebar.write(f"✅ Scaler期望特征数: {scaler.n_features_in_}")
st.sidebar.write(f"✅ 使用的连续特征: {scaler_features}")

# -------------------- 显示名称映射 --------------------
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

# 构建显示特征名称列表（用于SHAP）
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

    # 分类特征
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

    # 连续特征：使用scaler_features（实际scaler训练用的特征）
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

    # 预测时间
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

# -------------------- 预测与结果 --------------------
with col_right:
    st.markdown("**📊 Results & Interpretation**")
    if predict_clicked:
        try:
            # 按scaler_features顺序构建连续特征数组
            cont_values = []
            for col in scaler_features:
                if col in input_data:
                    cont_values.append(float(input_data[col]))
                else:
                    st.error(f"Missing continuous feature: {col}")
                    st.stop()
            cont_array = np.array(cont_values).reshape(1, -1)
            cont_scaled = scaler.transform(cont_array)

            # 分类特征
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

            # 预测
            surv_funcs = model.predict_survival_function(X_final)
            time_month = int(round(st.session_state.time_years * 12))
            max_month = int(surv_funcs[0].x[-1])
            if time_month > max_month:
                st.warning(f"Time capped to {max_month} months.")
                time_month = max_month
            surv_prob = surv_funcs[0](time_month)
            risk = 1 - surv_prob

            # 风险阈值
            t_years = st.session_state.time_years
            threshold_low = (0.075 / 10.0) * t_years
            threshold_high = (0.20 / 10.0) * t_years

            if risk < threshold_low:
                color, level = "green", "Low"
            elif risk < threshold_high:
                color, level = "orange", "Moderate"
            else:
                color, level = "red", "High"

            st.markdown(f"**Risk at {time_month} Months (~{t_years:.1f} Years)**")
            c_risk, c_surv = st.columns(2)
            with c_risk:
                st.metric("Mortality Risk", f"{risk * 100:.1f}%")
                st.markdown(f"<span style='color:{color}; font-weight:bold;'>{level} Risk</span>", unsafe_allow_html=True)
            with c_surv:
                st.metric("Survival Probability", f"{surv_prob * 100:.1f}%")

            # 临床建议
            with st.expander("🩺 Evidence-Based Clinical Recommendations", expanded=False):
                if risk < threshold_low:
                    st.success("**Low Risk** — Primary Prevention & Lifestyle Modification...")
                elif risk < threshold_high:
                    st.warning("**Intermediate Risk** — Shared Decision-Making & Early Pharmacotherapy...")
                else:
                    st.error("**High Risk** — Multidisciplinary Team Management & Intensified GDMT...")

            # SHAP分析
            st.markdown("**🔍 SHAP Interpretation**")
            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = coefs[:, 0]
            contributions = X_final[0] * coefs
            base_value = 0.0

            # 显示数据
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

            # 重要性图
            import matplotlib.pyplot as plt
            importance = np.abs(contributions)
            sorted_idx = np.argsort(importance)[::-1][:10]
            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.barh([display_names[i] for i in sorted_idx[::-1]], importance[sorted_idx[::-1]], color='#1f77b4')
            ax.set_xlabel('|SHAP|')
            ax.set_title('Top 10 Features')
            st.pyplot(fig)
            plt.clf()

            # 特征汇总表
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