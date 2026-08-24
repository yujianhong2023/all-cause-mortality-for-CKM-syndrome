import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# ================= 1. 固定参数与变量定义 =================
RANDOM_SEED = 123
TRAIN_RATIO = 0.7

# 自变量特征列表（共25个特征，已包含 UA，单位：mg/dL）
FEATURES = [
    'AGE', 'CKM', 'PLT', 'MCV', 'RDW', 'SII', 'PIR_GROUP', 'RACE',
    'ACTIVITY', 'GENDER', 'ABSI', 'HBA1C', 'GLB', 'MARITAL', 'MON',
    'LUNG', 'EGFR', 'CRP', 'CANCER', 'UA', 'SHR', 'BMI', 'TC', 'EDU', 'AST'
]

# 结局与时间变量
TARGET_EVENT = 'DEATH_ALL'
TARGET_TIME = 'PERMTH_INT'

# 划分分类变量与连续变量
CATEGORICAL_FEATURES = [
    'GENDER', 'CKM', 'ACTIVITY', 'PIR_GROUP', 'RACE',
    'MARITAL', 'LUNG', 'CANCER', 'EDU'
]
CONTINUOUS_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

# 模型文件路径（已更改为ENet_model.pkl）
MODEL_PATH = 'ENet_model.pkl'


# ================= 2. 训练模型函数（如果模型不存在） =================
def train_and_save_model():
    """训练并保存模型，如果模型文件不存在"""
    print("🔧 开始训练模型...")

    # 数据路径
    train_path = r"D:\R\AI_CKM\AI2026_US.csv"
    test_path = r"D:\R\AI_CKM\AI2026_ce.csv"

    # 加载数据
    df_internal = pd.read_csv(train_path)
    df_external = pd.read_csv(test_path)

    print(f"原始内部数据量: {len(df_internal)} 行")
    print(f"原始外部数据量: {len(df_external)} 行")

    # 数据清洗
    cols_to_numeric = CONTINUOUS_FEATURES + [TARGET_EVENT, TARGET_TIME]
    for col in cols_to_numeric:
        if col in df_internal.columns:
            df_internal[col] = pd.to_numeric(df_internal[col], errors='coerce')
        if col in df_external.columns:
            df_external[col] = pd.to_numeric(df_external[col], errors='coerce')

    for col in CATEGORICAL_FEATURES:
        if col in df_internal.columns:
            df_internal[col] = df_internal[col].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan})
        if col in df_external.columns:
            df_external[col] = df_external[col].astype(str).str.strip().replace({'': np.nan, 'nan': np.nan})

    # 清除缺失值
    cols_to_check = FEATURES + [TARGET_EVENT, TARGET_TIME]
    df_internal = df_internal.dropna(subset=cols_to_check).reset_index(drop=True)
    df_external = df_external.dropna(subset=cols_to_check).reset_index(drop=True)

    print(f"清洗后内部数据量: {len(df_internal)} 行")
    print(f"清洗后外部测试集数据量: {len(df_external)} 行")

    # 标签编码
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        combined_data = pd.concat([df_internal[col], df_external[col]]).astype(str)
        le.fit(combined_data)
        df_internal[col] = le.transform(df_internal[col].astype(str))
        df_external[col] = le.transform(df_external[col].astype(str))
        label_encoders[col] = le

    # 划分训练验证集
    X_internal = df_internal[FEATURES].copy()
    y_internal_event = df_internal[TARGET_EVENT].copy()
    y_internal_time = df_internal[TARGET_TIME].copy()

    X_train, X_val, y_train_event, y_val_event, y_train_time, y_val_time = train_test_split(
        X_internal, y_internal_event, y_internal_time,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=y_internal_event
    )

    # 外部测试集
    X_external = df_external[FEATURES].copy()
    y_ext_event = df_external[TARGET_EVENT].copy()
    y_ext_time = df_external[TARGET_TIME].copy()

    # 标准化
    scaler = StandardScaler()
    X_train_cont_scaled = scaler.fit_transform(X_train[CONTINUOUS_FEATURES])
    X_val_cont_scaled = scaler.transform(X_val[CONTINUOUS_FEATURES])
    X_ext_cont_scaled = scaler.transform(X_external[CONTINUOUS_FEATURES])

    X_train_final = np.hstack([X_train_cont_scaled, X_train[CATEGORICAL_FEATURES].values])
    X_val_final = np.hstack([X_val_cont_scaled, X_val[CATEGORICAL_FEATURES].values])
    X_ext_final = np.hstack([X_ext_cont_scaled, X_external[CATEGORICAL_FEATURES].values])

    # 构建生存数据
    y_train_surv = Surv.from_arrays(event=y_train_event.astype(bool), time=y_train_time)
    y_val_surv = Surv.from_arrays(event=y_val_event.astype(bool), time=y_val_time)
    y_ext_surv = Surv.from_arrays(event=y_ext_event.astype(bool), time=y_ext_time)

    # 训练模型（参数保持不变）
    enet_params = {
        'l1_ratio': 0.10407545001451728,
        'alphas': [0.017267726981514762],
        'fit_baseline_model': True
    }

    print("\n开始训练模型...")
    best_model = CoxnetSurvivalAnalysis(**enet_params)
    best_model.fit(X_train_final, y_train_surv)
    print("模型训练完成！")

    # 评估
    c_index_train = best_model.score(X_train_final, y_train_surv)
    c_index_val = best_model.score(X_val_final, y_val_surv)
    c_index_ext = best_model.score(X_ext_final, y_ext_surv)

    print("\n" + "=" * 50)
    print("模型性能评估:")
    print(f"内部训练集 C-index: {c_index_train:.4f}")
    print(f"内部验证集 C-index: {c_index_val:.4f}")
    print(f"外部测试集 C-index: {c_index_ext:.4f}")
    print("=" * 50)

    # 保存模型（使用新文件名ENet_model.pkl）
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'features': FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'continuous_features': CONTINUOUS_FEATURES,
        'target_event': TARGET_EVENT,
        'target_time': TARGET_TIME,
        'best_params': enet_params,
        'c_index_train': c_index_train,
        'c_index_val': c_index_val,
        'c_index_ext': c_index_ext
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_artifacts, f)

    print(f"\n✅ 模型已成功保存为 {MODEL_PATH}")
    return model_artifacts


# ================= 3. 加载或训练模型 =================
@st.cache_resource
def load_or_train_model():
    """加载已有模型，如果不存在则训练"""
    if Path(MODEL_PATH).exists():
        print(f"✅ 找到已有模型文件: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    else:
        print(f"⚠️ 模型文件 {MODEL_PATH} 不存在，开始训练...")
        return train_and_save_model()


# 获取模型
artifacts = load_or_train_model()
model = artifacts['model']
scaler = artifacts['scaler']
label_encoders = artifacts['label_encoders']
features = artifacts['features']
categorical_features = artifacts['categorical_features']
continuous_features = artifacts['continuous_features']
c_index_ext = artifacts.get('c_index_ext', 0.72)  # 从模型获取或使用默认值

# ================= 4. Web界面配置 =================
st.set_page_config(page_title="CKM 生存预测系统", layout="wide")
st.title("🫀 CKM 综合征生存预测")
st.markdown("---")

# 侧边栏 - 输入参数
st.sidebar.header("📋 患者特征输入")

# ---- 分类变量输入 ----
st.sidebar.subheader("分类特征")
gender = st.sidebar.selectbox("性别 (GENDER)", label_encoders['GENDER'].classes_)
ckm = st.sidebar.selectbox("CKM分期", label_encoders['CKM'].classes_)
activity = st.sidebar.selectbox("活动能力 (ACTIVITY)", label_encoders['ACTIVITY'].classes_)
pir_group = st.sidebar.selectbox("贫困收入比分组 (PIR_GROUP)", label_encoders['PIR_GROUP'].classes_)
race = st.sidebar.selectbox("种族 (RACE)", label_encoders['RACE'].classes_)
marital = st.sidebar.selectbox("婚姻状况 (MARITAL)", label_encoders['MARITAL'].classes_)
lung = st.sidebar.selectbox("肺部疾病 (LUNG)", label_encoders['LUNG'].classes_)
cancer = st.sidebar.selectbox("癌症史 (CANCER)", label_encoders['CANCER'].classes_)
edu = st.sidebar.selectbox("教育水平 (EDU)", label_encoders['EDU'].classes_)

# ---- 连续变量输入（包含UA） ----
st.sidebar.subheader("连续特征")
age = st.sidebar.number_input("年龄 AGE (岁)", min_value=20, max_value=120, value=65)
plt = st.sidebar.number_input("血小板 PLT (×10⁹/L)", min_value=0.0, max_value=1000.0, value=250.0)
mcv = st.sidebar.number_input("平均红细胞体积 MCV (fL)", min_value=50.0, max_value=150.0, value=92.0)
rdw = st.sidebar.number_input("红细胞分布宽度 RDW (%)", min_value=10.0, max_value=30.0, value=13.2)
sii = st.sidebar.number_input("全身免疫炎症指数 SII (×10⁹/L)", min_value=0.0, max_value=5000.0, value=500.0)
absi = st.sidebar.number_input("体型指数 ABSI", min_value=0.0, max_value=0.5, value=0.08)
hba1c = st.sidebar.number_input("糖化血红蛋白 HBA1C (%)", min_value=3.0, max_value=20.0, value=5.6)
glb = st.sidebar.number_input("球蛋白 GLB (g/L)", min_value=10.0, max_value=60.0, value=28.0)
mon = st.sidebar.number_input("单核细胞 MON (×10⁹/L)", min_value=0.0, max_value=5.0, value=0.4)
egfr = st.sidebar.number_input("估算肾小球滤过率 EGFR (mL/min/1.73m²)", min_value=5.0, max_value=200.0, value=90.0)
crp = st.sidebar.number_input("C反应蛋白 CRP (mg/dL)", min_value=0.0, max_value=50.0, value=22.0)
ua = st.sidebar.number_input("尿酸 UA (mg/dL)", min_value=0.0, max_value=20.0, value=5.0)  # ⭐ UA输入
shr = st.sidebar.number_input("应激性高血糖比率 SHR", min_value=0.0, max_value=5.0, value=1.0)
bmi = st.sidebar.number_input("体重指数 BMI (kg/m²)", min_value=10.0, max_value=60.0, value=24.5)
tc = st.sidebar.number_input("总胆固醇 TC (mg/dL)", min_value=50.0, max_value=400.0, value=185.0)
ast = st.sidebar.number_input("天冬氨酸转氨酶 AST (U/L)", min_value=0.0, max_value=500.0, value=22.0)

# ---- 预测时间选择 ----
st.sidebar.subheader("预测时间点")
predict_time = st.sidebar.selectbox("预测生存时间 (年)", [1, 3, 5, 10], index=2)


# ================= 5. 数据预处理函数 =================
def preprocess_input(input_dict):
    """将用户输入转换为模型可用的特征向量"""
    df_input = pd.DataFrame([input_dict])

    # 分类变量编码
    for col in categorical_features:
        if col in df_input.columns:
            le = label_encoders[col]
            df_input[col] = le.transform(df_input[col].astype(str))

    # 连续变量标准化
    cont_df = df_input[continuous_features].copy()
    cont_scaled = scaler.transform(cont_df)

    # 组合特征
    X_final = np.hstack([cont_scaled, df_input[categorical_features].values])
    return X_final


# ================= 6. 预测与结果展示 =================
if st.sidebar.button("🔍 预测生存概率", type="primary"):
    # 收集输入
    input_data = {
        'AGE': age, 'CKM': ckm, 'PLT': plt, 'MCV': mcv, 'RDW': rdw,
        'SII': sii, 'PIR_GROUP': pir_group, 'RACE': race,
        'ACTIVITY': activity, 'GENDER': gender, 'ABSI': absi,
        'HBA1C': hba1c, 'GLB': glb, 'MARITAL': marital, 'MON': mon,
        'LUNG': lung, 'EGFR': egfr, 'CRP': crp, 'CANCER': cancer,
        'UA': ua,  # ⭐ UA传递
        'SHR': shr, 'BMI': bmi, 'TC': tc, 'EDU': edu, 'AST': ast
    }

    try:
        # 预处理
        X_input = preprocess_input(input_data)

        # 预测生存概率
        surv_probs = model.predict_survival_function(X_input)

        # 提取数据
        times = surv_probs[0].x
        surv_probs_values = surv_probs[0].y

        # 插值获取指定时间的生存概率
        if predict_time in times:
            idx = np.where(times == predict_time)[0][0]
            surv_prob = surv_probs_values[idx]
        else:
            # 线性插值
            idx = np.searchsorted(times, predict_time)
            if idx == 0:
                surv_prob = surv_probs_values[0]
            elif idx >= len(times):
                surv_prob = surv_probs_values[-1]
            else:
                t0, t1 = times[idx - 1], times[idx]
                s0, s1 = surv_probs_values[idx - 1], surv_probs_values[idx]
                surv_prob = s0 + (s1 - s0) * (predict_time - t0) / (t1 - t0)

        # 计算死亡风险
        death_risk = 1 - surv_prob

        # 显示结果
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label=f"📊 {predict_time}年生存概率",
                value=f"{surv_prob:.2%}",
                delta=f"风险 {death_risk:.2%}"
            )
        with col2:
            st.metric(
                label="⚰️ 死亡风险",
                value=f"{death_risk:.2%}",
                delta="高风险" if death_risk > 0.5 else "低风险"
            )
        with col3:
            st.metric(
                label="📈 C-index (外部验证)",
                value=f"{c_index_ext:.3f}" if isinstance(c_index_ext, float) else c_index_ext
            )

        # 生存曲线图
        st.subheader("📉 生存曲线预测")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(times, surv_probs_values, where='post', label=f'预测生存曲线')
        ax.axvline(x=predict_time, color='red', linestyle='--', label=f'预测时间点 ({predict_time}年)')
        ax.axhline(y=surv_prob, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('生存时间 (年)')
        ax.set_ylabel('生存概率')
        ax.set_title('患者生存曲线预测')
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        st.info("💡 该预测基于Cox ElasticNet模型，输入特征包含25个变量（包括UA尿酸）。")

    except Exception as e:
        st.error(f"预测出错: {str(e)}")
        st.info("请检查所有输入是否完整有效。")

# ================= 7. 页脚说明 =================
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ 预测结果仅供参考，请结合临床判断")
st.sidebar.caption(f"🔄 模型包含UA (mg/dL) | 特征数: {len(FEATURES)}")

# 显示模型状态
if st.sidebar.checkbox("显示模型信息"):
    st.sidebar.info(f"""
    **模型信息**
    - 特征数: {len(FEATURES)}
    - 连续特征: {len(continuous_features)}个
    - 分类特征: {len(categorical_features)}个
    - 包含UA: ✅ (mg/dL)
    - 模型文件: {MODEL_PATH}
    """)