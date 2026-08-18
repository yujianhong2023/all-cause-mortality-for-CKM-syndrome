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
MODEL_FILE = 'rsf_model.pkl'   # 此文件应为完整的 Pipeline（包含预处理和模型）

# 检查模型文件是否存在
if not os.path.exists(os.path.join(MODEL_DIR, MODEL_FILE)):
    st.error(f"❌ Model file '{MODEL_FILE}' not found in '{MODEL_DIR}'")
    st.info("Please ensure the Pipeline model file is present.")
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

# 从 Pipeline 中提取特征列名（假设 Pipeline 的第一步是 ColumnTransformer）
# 如果您的 Pipeline 结构不同，可以手动定义 feature_cols
# 这里假设模型是通过前面训练脚本生成的 Pipeline，其预处理步骤包含特征列名
try:
    # 获取预处理器的特征名（如果支持）
    preprocessor = pipeline.named_steps['preprocessor']
    # 对于 ColumnTransformer，我们可以通过 get_feature_names_out 获取列名，但需要先 fit
    # 为了简化，我们直接手动定义特征列表（与训练时一致）
    feature_cols = [
        'AGE', 'CKM', 'HBA1C', 'ABSI', 'EGFR', 'RDW', 'SHR', 'MCV',
        'GLB', 'PLT', 'CRP', 'SII', 'MON', 'AST', 'SMOKE', 'BMI',
        'TC', 'ACTIVITY', 'PIR_GROUP', 'GENDER', 'LUNG', 'CANCER'
    ]
except:
    # 如果无法提取，手动定义
    feature_cols = [
        'AGE', 'CKM', 'HBA1C', 'ABSI', 'EGFR', 'RDW', 'SHR', 'MCV',
        'GLB', 'PLT', 'CRP', 'SII', 'MON', 'AST', 'SMOKE', 'BMI',
        'TC', 'ACTIVITY', 'PIR_GROUP', 'GENDER', 'LUNG', 'CANCER'
    ]

# 分类变量（仅用于 UI 选项，不再用于编码）
CAT_COLS = ['SMOKE', 'ACTIVITY', 'PIR_GROUP', 'GENDER', 'LUNG', 'CANCER']

# -------------------- UI 定义（与之前相同，但无需预处理器加载） --------------------
binary_choices = ['No', 'Yes']

st.title("🫀 All‑Cause Mortality Risk Prediction")
st.markdown("**Random Survival Forest model** – Enter patient characteristics to estimate risk at a given time.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📝 Patient Characteristics")

    with st.expander("🧑‍⚕️ Demographics", expanded=True):
        gender = st.selectbox("Sex", ['Male', 'Female'], key="gender")   # 假设类别为这些，您可以根据实际情况调整
        age = st.number_input("Age (years)", min_value=18, max_value=120, value=65, step=1, key="age")
        pir = st.selectbox("PIR Group", ['<1.0', '1.0-1.9', '2.0-2.9', '>=3.0'], key="pir")

    with st.expander("🧪 Clinical & Laboratory", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            ckm = st.selectbox("CKM Stage", ['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'], key="ckm")
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
        smoke = st.selectbox("Smoking Status", ['Never', 'Former', 'Current'], key="smoke")
        activity = st.selectbox("Physical Activity", ['Inactive', 'Moderate', 'Active'], key="activity")
        lung_yn = st.selectbox("Pulmonary Disease", binary_choices, key="lung")
        cancer_yn = st.selectbox("Cancer History", binary_choices, key="cancer")
        lung = 1 if lung_yn == 'Yes' else 0
        cancer = 1 if cancer_yn == 'Yes' else 0

    st.subheader("⏱ Prediction Time")
    # ... 时间选择部分保持不变（略，与原文相同）...
    # 为了简洁，此处省略时间选择代码，实际使用时请保留原代码

    predict_clicked = st.button("📊 Predict Mortality Risk", type="primary", use_container_width=True)

# -------------------- Results panel --------------------
with col_right:
    st.subheader("📊 Prediction Result")

    if predict_clicked:
        # 构建输入字典
        input_dict = {
            'GENDER': gender,
            'AGE': age,
            'PIR_GROUP': pir,
            'CKM': ckm,
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
            # 转换为 DataFrame（保持列顺序与训练时一致）
            X_raw = pd.DataFrame([input_dict])[feature_cols]

            # 直接使用 Pipeline 预测（内部自动进行预处理）
            surv_funcs = pipeline.predict_survival_function(X_raw)

            # 计算指定时间点的生存概率
            time_month = int(round(st.session_state.time_years * 12))
            if time_month < 1:
                time_month = 1
            surv_prob = surv_funcs[0](time_month)
            risk = 1 - surv_prob

            # 展示结果（与之前相同）
            # ... 此处保留原来的结果展示代码 ...

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Please verify all inputs and model files.")
    else:
        st.info("👈 Fill in patient characteristics and click **Predict Mortality Risk**.")
        st.markdown("""**Instructions:** ... """)

# -------------------- Footer --------------------
st.divider()
st.caption(f"Model: Pipeline ({MODEL_FILE}) · Random Survival Forest · v2.0")