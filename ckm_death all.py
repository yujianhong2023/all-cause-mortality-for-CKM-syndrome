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
    sys.exit(f"Error: Model file '{MODEL_FILE}' not found.")  # 强制退出，防止 bare mode 崩溃


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

# 默认生理参考值（用于初始化连续变量输入框）
default_cont_vals = {
    'AGE': 65.0, 'HBA1C': 5.6, 'ABSI': 0.08, 'EGFR': 90.0, 'RDW': 13.2,
    'SHR': 2.5, 'MCV': 92.0, 'GLB': 28.0, 'PLT': 250.0, 'CRP': 1.5,
    'SII': 500.0, 'MON': 0.4, 'AST': 22.0, 'BMI': 24.5, 'TC': 185.0
}

# -------------------- UI Layout --------------------
st.title("🫀 All-Cause Mortality Risk Prediction")
st.markdown(
    "**ElasticNet Survival Model** – Enter patient characteristics to estimate risk and explore feature contributions (SHAP).")

col_left, col_right = st.columns([1, 1.2], gap="large")
input_data = {}

with col_left:
    st.subheader("📝 Patient Characteristics")

    # 分类变量输入区 (动态获取类别选项)
    with st.expander("🏷️ Categorical Features", expanded=True):
        col1, col2 = st.columns(2)
        for i, col in enumerate(CATEGORICAL_FEATURES):
            # 获取该特征在训练集中的所有合法分类
            options = [str(cls) for cls in label_encoders[col].classes_ if str(cls) != 'nan']
            with col1 if i % 2 == 0 else col2:
                input_data[col] = st.selectbox(f"{col}", options, key=f"cat_{col}")

    # 连续变量输入区
    with st.expander("📈 Continuous Features", expanded=True):
        col3, col4 = st.columns(2)
        for i, col in enumerate(CONTINUOUS_FEATURES):
            default_val = default_cont_vals.get(col, 50.0)
            with col3 if i % 2 == 0 else col4:
                input_data[col] = st.number_input(f"{col}", value=float(default_val),
                                                  step=1.0 if default_val > 10 else 0.1, key=f"cont_{col}")

    # 预测时间选择
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
            # 1. 构建 DataFrame 并预处理
            df_input = pd.DataFrame([input_data])

            # 标准化连续变量
            cont_scaled = scaler.transform(df_input[CONTINUOUS_FEATURES])

            # 编码分类变量
            cat_encoded = []
            for col in CATEGORICAL_FEATURES:
                cat_encoded.append(label_encoders[col].transform([df_input[col].iloc[0]])[0])
            cat_encoded = np.array(cat_encoded).reshape(1, -1)

            # 合并特征 (需严格按照训练时的拼接顺序: 先连续变量，再分类变量)
            X_final = np.hstack([cont_scaled, cat_encoded])
            final_feature_names = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

            # 2. 预测生存概率
            surv_funcs = model.predict_survival_function(X_final)
            time_month = int(round(st.session_state.time_years * 12))

            # 确保时间在模型允许的范围内
            max_train_month = int(surv_funcs[0].x[-1])
            if time_month > max_train_month:
                st.warning(
                    f"Requested time ({time_month} months) exceeds training follow-up. Capped to {max_train_month} months.")
                time_month = max_train_month

            surv_prob = surv_funcs[0](time_month)
            risk = 1 - surv_prob

            # 展示风险指标
            st.markdown(f"### Risk at {time_month} Months (~{st.session_state.time_years:.1f} Years)")
            c_risk, c_surv = st.columns(2)
            color = "green" if risk < 0.2 else ("orange" if risk < 0.5 else "red")
            level = "Low" if risk < 0.2 else ("Moderate" if risk < 0.5 else "High")

            with c_risk:
                st.metric("Mortality Risk", f"{risk * 100:.1f}%")
                st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:18px'>{level} Risk</span>",
                            unsafe_allow_html=True)
            with c_surv:
                st.metric("Survival Probability", f"{surv_prob * 100:.1f}%")

            st.divider()

            # 3. 核心功能：生成 SHAP 解释
            st.markdown("### 🔍 Model Interpretation (SHAP)")
            st.caption("How each feature contributes to the patient's individual log-hazard score.")

            # 提取 ElasticNet 模型的系数 (Beta)
            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = coefs[:, 0]

            # 基于线性模型特性计算各个特征的 SHAP 贡献值: Contribution = X_scaled * Beta
            contributions = X_final[0] * coefs

            # 构造原始数据用于图表展示
            display_data = []
            for col in CONTINUOUS_FEATURES:
                display_data.append(df_input[col].iloc[0])
            for col in CATEGORICAL_FEATURES:
                display_data.append(df_input[col].iloc[0])

            explanation = shap.Explanation(
                values=contributions,
                base_values=0.0,  # 线性中心化模型基准线为0
                data=display_data,
                feature_names=final_feature_names
            )

            # SHAP Waterfall Plot
            st.markdown("#### 1. SHAP Waterfall Plot")
            fig_waterfall, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(explanation, max_display=12, show=False)
            st.pyplot(fig_waterfall, bbox_inches='tight')
            plt.clf()

            st.write("")  # 增加一点空隙

            # SHAP Force Plot
            st.markdown("#### 2. SHAP Force Plot")
            shap_force = shap.plots.force(
                explanation.base_values,
                explanation.values,
                explanation.data,
                feature_names=explanation.feature_names
            )
            # 通过 HTML + JS 渲染交互式 Force 图
            shap_html = f"<head><script src='https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js'></script></head><body>{shap_force.html()}</body>"
            components.html(shap_html, height=150)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Please verify your inputs and ensure 'ENet_model.pkl' is correctly generated.")
    else:
        st.info("👈 Fill in patient characteristics and click **Predict** to view Risk and SHAP plots.")

st.divider()
st.caption(f"Model File: {MODEL_FILE} · Engine: ElasticNet Survival Analysis · Includes SHAP Interpretations")