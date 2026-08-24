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

# ================= 1. Fixed Parameters and Variable Definitions =================
RANDOM_SEED = 123
TRAIN_RATIO = 0.7

# Feature list (25 features, including UA in mg/dL)
FEATURES = [
    'AGE', 'CKM', 'PLT', 'MCV', 'RDW', 'SII', 'PIR_GROUP', 'RACE',
    'ACTIVITY', 'GENDER', 'ABSI', 'HBA1C', 'GLB', 'MARITAL', 'MON',
    'LUNG', 'EGFR', 'CRP', 'CANCER', 'UA', 'SHR', 'BMI', 'TC', 'EDU', 'AST'
]

# Outcome and time variables
TARGET_EVENT = 'DEATH_ALL'
TARGET_TIME = 'PERMTH_INT'

# Categorical and continuous features
CATEGORICAL_FEATURES = [
    'GENDER', 'CKM', 'ACTIVITY', 'PIR_GROUP', 'RACE',
    'MARITAL', 'LUNG', 'CANCER', 'EDU'
]
CONTINUOUS_FEATURES = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]

# Model file path
MODEL_PATH = 'ENet_model.pkl'


# ================= 2. Model Training Function =================
def train_and_save_model():
    """Train and save model if model file does not exist"""
    print("🔧 Starting model training...")

    # Data paths
    train_path = r"D:\R\AI_CKM\AI2026_US.csv"  # Internal dataset
    test_path = r"D:\R\AI_CKM\AI2026_ce.csv"  # External dataset

    # Load data
    df_internal = pd.read_csv(train_path)
    df_external = pd.read_csv(test_path)

    print(f"Original internal data: {len(df_internal)} rows")
    print(f"Original external data: {len(df_external)} rows")

    # Data cleaning
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

    # Remove rows with missing values
    cols_to_check = FEATURES + [TARGET_EVENT, TARGET_TIME]
    df_internal = df_internal.dropna(subset=cols_to_check).reset_index(drop=True)
    df_external = df_external.dropna(subset=cols_to_check).reset_index(drop=True)

    print(f"Cleaned internal data: {len(df_internal)} rows")
    print(f"Cleaned external data: {len(df_external)} rows")

    # Label encoding
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        combined_data = pd.concat([df_internal[col], df_external[col]]).astype(str)
        le.fit(combined_data)
        df_internal[col] = le.transform(df_internal[col].astype(str))
        df_external[col] = le.transform(df_external[col].astype(str))
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} categories")

    # Split internal training and validation sets
    X_internal = df_internal[FEATURES].copy()
    y_internal_event = df_internal[TARGET_EVENT].copy()
    y_internal_time = df_internal[TARGET_TIME].copy()

    X_train, X_val, y_train_event, y_val_event, y_train_time, y_val_time = train_test_split(
        X_internal, y_internal_event, y_internal_time,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED,
        stratify=y_internal_event
    )

    # External test set
    X_external = df_external[FEATURES].copy()
    y_ext_event = df_external[TARGET_EVENT].copy()
    y_ext_time = df_external[TARGET_TIME].copy()

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"External test set: {len(X_external)} samples")

    # Feature standardization
    scaler = StandardScaler()
    X_train_cont_scaled = scaler.fit_transform(X_train[CONTINUOUS_FEATURES])
    X_val_cont_scaled = scaler.transform(X_val[CONTINUOUS_FEATURES])
    X_ext_cont_scaled = scaler.transform(X_external[CONTINUOUS_FEATURES])

    # Combine standardized continuous and categorical features
    X_train_final = np.hstack([X_train_cont_scaled, X_train[CATEGORICAL_FEATURES].values])
    X_val_final = np.hstack([X_val_cont_scaled, X_val[CATEGORICAL_FEATURES].values])
    X_ext_final = np.hstack([X_ext_cont_scaled, X_external[CATEGORICAL_FEATURES].values])

    print(f"Training feature matrix shape: {X_train_final.shape}")
    print(f"Validation feature matrix shape: {X_val_final.shape}")
    print(f"Test feature matrix shape: {X_ext_final.shape}")

    # Build survival data for sksurv
    y_train_surv = Surv.from_arrays(event=y_train_event.astype(bool), time=y_train_time)
    y_val_surv = Surv.from_arrays(event=y_val_event.astype(bool), time=y_val_time)
    y_ext_surv = Surv.from_arrays(event=y_ext_event.astype(bool), time=y_ext_time)

    # Train ElasticNet survival model
    enet_params = {
        'l1_ratio': 0.10407545001451728,
        'alphas': [0.017267726981514762],
        'fit_baseline_model': True
    }

    print("\nStarting model training...")
    best_model = CoxnetSurvivalAnalysis(**enet_params)
    best_model.fit(X_train_final, y_train_surv)
    print("Model training completed!")

    # Evaluate model C-index
    c_index_train = best_model.score(X_train_final, y_train_surv)
    c_index_val = best_model.score(X_val_final, y_val_surv)
    c_index_ext = best_model.score(X_ext_final, y_ext_surv)

    print("\n" + "=" * 50)
    print("Model Performance Evaluation:")
    print(f"Internal Training C-index: {c_index_train:.4f}")
    print(f"Internal Validation C-index: {c_index_val:.4f}")
    print(f"External Test C-index: {c_index_ext:.4f}")
    print("=" * 50)

    # Save model artifacts
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

    print(f"\n✅ Model successfully saved as {MODEL_PATH}")
    return model_artifacts


# ================= 3. Load or Train Model =================
@st.cache_resource
def load_or_train_model():
    """Load existing model or train if not exists"""
    if Path(MODEL_PATH).exists():
        print(f"✅ Found existing model: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    else:
        print(f"⚠️ Model file {MODEL_PATH} not found. Training new model...")
        return train_and_save_model()


# Load model
artifacts = load_or_train_model()
model = artifacts['model']
scaler = artifacts['scaler']
label_encoders = artifacts['label_encoders']
features = artifacts['features']
categorical_features = artifacts['categorical_features']
continuous_features = artifacts['continuous_features']
c_index_ext = artifacts.get('c_index_ext', 0.72)

# ================= 4. Web Interface Configuration =================
st.set_page_config(page_title="CKM Syndrome Survival Prediction", layout="wide")
st.title("🫀 CKM Syndrome Survival Prediction")
st.markdown("---")

# Sidebar - Input Parameters
st.sidebar.header("📋 Patient Characteristics")

# ---- Categorical Features Input ----
st.sidebar.subheader("Categorical Features")
gender = st.sidebar.selectbox("Gender", label_encoders['GENDER'].classes_)
ckm = st.sidebar.selectbox("CKM Stage", label_encoders['CKM'].classes_)
activity = st.sidebar.selectbox("Physical Activity Level", label_encoders['ACTIVITY'].classes_)
pir_group = st.sidebar.selectbox("Poverty-Income Ratio Category", label_encoders['PIR_GROUP'].classes_)
race = st.sidebar.selectbox("Race/Ethnicity", label_encoders['RACE'].classes_)
marital = st.sidebar.selectbox("Marital Status", label_encoders['MARITAL'].classes_)
lung = st.sidebar.selectbox("Pulmonary Disease", label_encoders['LUNG'].classes_)
cancer = st.sidebar.selectbox("History of Cancer", label_encoders['CANCER'].classes_)
edu = st.sidebar.selectbox("Education Level", label_encoders['EDU'].classes_)

# ---- Continuous Features Input (including UA) ----
st.sidebar.subheader("Continuous Features")
age = st.sidebar.number_input("Age (years)", min_value=20, max_value=120, value=65)
plt = st.sidebar.number_input("Platelet Count PLT (×10⁹/L)", min_value=0.0, max_value=1000.0, value=250.0)
mcv = st.sidebar.number_input("Mean Corpuscular Volume MCV (fL)", min_value=50.0, max_value=150.0, value=92.0)
rdw = st.sidebar.number_input("Red Cell Distribution Width RDW (%)", min_value=10.0, max_value=30.0, value=13.2)
sii = st.sidebar.number_input("Systemic Immune-Inflammation Index SII (×10⁹/L)", min_value=0.0, max_value=5000.0,
                              value=500.0)
absi = st.sidebar.number_input("A Body Shape Index ABSI", min_value=0.0, max_value=0.5, value=0.08)
hba1c = st.sidebar.number_input("Glycated Hemoglobin HBA1C (%)", min_value=3.0, max_value=20.0, value=5.6)
glb = st.sidebar.number_input("Globulin GLB (g/L)", min_value=10.0, max_value=60.0, value=28.0)
mon = st.sidebar.number_input("Monocyte Count MON (×10⁹/L)", min_value=0.0, max_value=5.0, value=0.4)
egfr = st.sidebar.number_input("eGFR (mL/min/1.73m²)", min_value=5.0, max_value=200.0, value=90.0)
crp = st.sidebar.number_input("C-Reactive Protein CRP (mg/dL)", min_value=0.0, max_value=50.0, value=22.0)
ua = st.sidebar.number_input("Uric Acid UA (mg/dL)", min_value=0.0, max_value=20.0, value=5.0)  # ⭐ UA input
shr = st.sidebar.number_input("Stress-Hyperglycemia Ratio SHR", min_value=0.0, max_value=5.0, value=1.0)
bmi = st.sidebar.number_input("Body Mass Index BMI (kg/m²)", min_value=10.0, max_value=60.0, value=24.5)
tc = st.sidebar.number_input("Total Cholesterol TC (mg/dL)", min_value=50.0, max_value=400.0, value=185.0)
ast = st.sidebar.number_input("AST (U/L)", min_value=0.0, max_value=500.0, value=22.0)

# ---- Prediction Time Selection ----
st.sidebar.subheader("Prediction Time")
predict_time = st.sidebar.selectbox("Prediction Time (years)", [1, 3, 5, 10], index=2)


# ================= 5. Data Preprocessing Function =================
def preprocess_input(input_dict):
    """Convert user input to model-ready feature vector"""
    df_input = pd.DataFrame([input_dict])

    # Encode categorical features
    for col in categorical_features:
        if col in df_input.columns:
            le = label_encoders[col]
            df_input[col] = le.transform(df_input[col].astype(str))

    # Standardize continuous features
    cont_df = df_input[continuous_features].copy()
    cont_scaled = scaler.transform(cont_df)

    # Combine features
    X_final = np.hstack([cont_scaled, df_input[categorical_features].values])
    return X_final


# ================= 6. Prediction and Results Display =================
if st.sidebar.button("🔍 Predict Survival Probability", type="primary"):
    # Collect all 25 features
    input_data = {
        'AGE': age, 'CKM': ckm, 'PLT': plt, 'MCV': mcv, 'RDW': rdw,
        'SII': sii, 'PIR_GROUP': pir_group, 'RACE': race,
        'ACTIVITY': activity, 'GENDER': gender, 'ABSI': absi,
        'HBA1C': hba1c, 'GLB': glb, 'MARITAL': marital, 'MON': mon,
        'LUNG': lung, 'EGFR': egfr, 'CRP': crp, 'CANCER': cancer,
        'UA': ua,  # ⭐ UA included
        'SHR': shr, 'BMI': bmi, 'TC': tc, 'EDU': edu, 'AST': ast
    }

    try:
        # Preprocess input
        X_input = preprocess_input(input_data)

        # Predict survival probability
        surv_probs = model.predict_survival_function(X_input)

        # Extract survival data
        times = surv_probs[0].x
        surv_probs_values = surv_probs[0].y

        # Interpolate to get survival probability at specified time
        if predict_time in times:
            idx = np.where(times == predict_time)[0][0]
            surv_prob = surv_probs_values[idx]
        else:
            # Linear interpolation
            idx = np.searchsorted(times, predict_time)
            if idx == 0:
                surv_prob = surv_probs_values[0]
            elif idx >= len(times):
                surv_prob = surv_probs_values[-1]
            else:
                t0, t1 = times[idx - 1], times[idx]
                s0, s1 = surv_probs_values[idx - 1], surv_probs_values[idx]
                surv_prob = s0 + (s1 - s0) * (predict_time - t0) / (t1 - t0)

        # Calculate death risk
        death_risk = 1 - surv_prob

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label=f"📊 {predict_time}-Year Survival Probability",
                value=f"{surv_prob:.2%}",
                delta=f"Risk {death_risk:.2%}"
            )
        with col2:
            st.metric(
                label="⚰️ Death Risk",
                value=f"{death_risk:.2%}",
                delta="High Risk" if death_risk > 0.5 else "Low Risk"
            )
        with col3:
            st.metric(
                label="📈 Model C-index (External Validation)",
                value=f"{c_index_ext:.3f}" if isinstance(c_index_ext, float) else c_index_ext
            )

        # Survival curve plot
        st.subheader("📉 Survival Curve Prediction")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(times, surv_probs_values, where='post', label='Predicted Survival Curve')
        ax.axvline(x=predict_time, color='red', linestyle='--', label=f'Prediction Time ({predict_time} years)')
        ax.axhline(y=surv_prob, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Survival Time (years)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Patient Survival Curve Prediction')
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        # Feature summary table
        st.subheader("📋 Input Features Summary")
        feature_summary = pd.DataFrame({
            'Feature': FEATURES,
            'Type': ['Continuous' if f in CONTINUOUS_FEATURES else 'Categorical' for f in FEATURES],
            'Value': [input_data[f] for f in FEATURES]
        })
        st.dataframe(feature_summary, use_container_width=True)

        st.info("💡 This prediction is based on a Cox ElasticNet model with 25 features including UA (uric acid).")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Please check that all inputs are complete and valid.")

# ================= 7. Footer Information =================
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Prediction results are for reference only. Please combine with clinical judgment.")
st.sidebar.caption(f"🔄 Model includes UA (mg/dL) | {len(FEATURES)} features total")

# Display model information
if st.sidebar.checkbox("Show Model Information"):
    st.sidebar.info(f"""
    **Model Information**
    - Total Features: {len(FEATURES)}
    - Continuous Features: {len(continuous_features)}
    - Categorical Features: {len(categorical_features)}
    - Includes UA: ✅ (mg/dL)
    - Model File: {MODEL_PATH}
    - Model Type: Cox ElasticNet Survival Analysis
    - Random Seed: {RANDOM_SEED}
    - Train/Validation Split: {TRAIN_RATIO:.0%}
    """)

    # Show feature list
    st.sidebar.write("**Feature List:**")
    for i, feat in enumerate(FEATURES, 1):
        feat_type = "Continuous" if feat in CONTINUOUS_FEATURES else "Categorical"
        st.sidebar.write(f"{i}. {feat} ({feat_type})")