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

# 特征列表（与训练时完全一致）
feature_cols = [
    'AGE', 'CKM', 'HBA1C', 'ABSI', 'EGFR', 'RDW', 'SHR', 'MCV',
    'GLB', 'PLT', 'CRP', 'SII', 'MON', 'AST', 'SMOKE', 'BMI',
    'TC', 'ACTIVITY', 'PIR_GROUP', 'GENDER', 'LUNG', 'CANCER'
]

# -------------------- 分类变量映射（因 Pipeline 训练时 CKM 为数值） --------------------
# 如果未来重新训练时将 CKM 纳入分类变量，可移除以下映射
ckm_map = {
    'Stage 0': 0,
    'Stage 1': 1,
    'Stage 2': 2,
    'Stage 3': 3,
    'Stage 4': 4
}

# UI 选项
binary_choices = ['No', 'Yes']
ckm_options = list(ckm_map.keys())
gender_options = ['Male', 'Female']
pir_options = ['<1.0', '1.0-1.9', '2.0-2.9', '>=3.0']
smoke_options = ['Never', 'Former', 'Current']
activity_options = ['Inactive', 'Moderate', 'Active']

# -------------------- UI Layout --------------------
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
            ckm = st.selectbox("CKM Stage", ckm_options, key="ckm")   # 字符串选项
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
        smoke = st.selectbox("Smoking Status", smoke_options, key="smoke")
        activity = st.selectbox("Physical Activity", activity_options, key="activity")
        lung_yn = st.selectbox("Pulmonary Disease", binary_choices, key="lung")
        cancer_yn = st.selectbox("Cancer History", binary_choices, key="cancer")
        lung = 1 if lung_yn == 'Yes' else 0
        cancer = 1 if cancer_yn == 'Yes' else 0

    # 预测时间设置（保留原代码）
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
        # 构建输入字典（CKM 映射为数值）
        input_dict = {
            'GENDER': gender,
            'AGE': age,
            'PIR_GROUP': pir,
            'CKM': ckm_map[ckm],          # 关键：字符串 → 数值
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
            'ACTIVITY': activity,
            'LUNG': lung,
            'CANCER': cancer
        }

        try:
            # 转换为 DataFrame（保持列顺序）
            X_raw = pd.DataFrame([input_dict])[feature_cols]

            # 直接使用 Pipeline 预测（内部自动进行预处理）
            surv_funcs = pipeline.predict_survival_function(X_raw)

            # 计算指定时间点的风险
            time_month = int(round(st.session_state.time_years * 12))
            if time_month < 1:
                time_month = 1
            surv_prob = surv_funcs[0](time_month)
            risk = 1 - surv_prob

            # ---------- 显示结果 ----------
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

            # 多时间点风险表
            st.markdown("#### 📈 Risk at multiple time points")
            preset_months = [12, 36, 60, 120]
            all_risks = {t: 1 - surv_funcs[0](t) for t in preset_months}
            df_compare = pd.DataFrame({
                "Months": [12, 36, 60, 120],
                "Years": [1, 3, 5, 10],
                "Risk (%)": [f"{all_risks[t]*100:.1f}" for t in preset_months]
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
st.caption(f"Model: Pipeline ({MODEL_FILE}) · Random Survival Forest · v2.0")