# 🫀 All‑Cause Mortality Risk Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A web‑based tool that predicts an individual’s **all‑cause mortality risk** using an **ElasticNet Cox Proportional Hazards** model.  
Built with [Streamlit](https://streamlit.io), it provides an intuitive interface for clinicians, researchers, and data scientists to explore risk estimates based on **25 clinical and demographic features**.

---

## ✨ Features

- **25 clinical features**, including demographics, laboratory values, lifestyle factors, and medical history.
- **Predict risk at any time** from 0.5 to 30 years (with quick presets: 1, 3, 5, 10 years).
- **Two‑panel layout**: input patient characteristics on the left, view results instantly on the right.
- **Clear risk interpretation**: mortality risk percentage, survival probability, and risk level (Low / Moderate / High).
- **Comparison table** showing risk at 1, 3, 5, and 10 years for a broader perspective.
- **Fully responsive** – works on desktop, tablet, and mobile.

---

## 🧬 Features Used

The model uses the following **25 features**, all commonly accessible in routine clinical practice:

| Feature | Description | Type |
|---------|-------------|------|
| `AGE` | Age (years) | Numeric |
| `CKM stage` | Cardiovascular‑Kidney‑Metabolic stage (0–4) | Categorical |
| `PLT` | Platelet count (×10⁹/L) | Numeric |
| `MCV` | Mean corpuscular volume (fL) | Numeric |
| `RDW` | Red cell distribution width (%) | Numeric |
| `SII` | Systemic immune‑inflammation index (×10⁹/L) | Numeric |
| `PIR_GROUP` | Poverty‑income ratio category | Categorical |
| `RACE` | Race/ethnicity (Mexican American, Other Hispanic, Non‑Hispanic White, Non‑Hispanic Black, Other/Multi‑Racial) | Categorical |
| `ACTIVITY` | Physical activity level | Categorical |
| `GENDER` | Sex | Categorical |
| `ABSI` | A Body Shape Index | Numeric |
| `HBA1C` | Glycated hemoglobin (%) | Numeric |
| `GLB` | Globulin (g/L) | Numeric |
| `Marital status` | Married/living with partner or Others | Categorical |
| `MON` | Monocyte count (×10⁹/L) | Numeric |
| `Pulmonary disease` | Presence of pulmonary disease (Yes/No) | Binary |
| `EGFR` | Estimated glomerular filtration rate (mL/min/1.73m²) | Numeric |
| `CRP` | C‑reactive protein (mg/L) | Numeric |
| `CANCER` | History of cancer (Yes/No) | Binary |
| `UA` | Uric acid (µmol/L) | Numeric |
| `SHR` | Stress‑hyperglycemia ratio | Numeric |
| `BMI` | Body mass index (kg/m²) | Numeric |
| `TC` | Total cholesterol (mmol/L) | Numeric |
| `EDU` | Education level | Categorical |
| `AST` | Aspartate aminotransferase (U/L) | Numeric |

> **Note**: All variables are preprocessed exactly as in the training pipeline. Missing values are handled by median imputation.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (package manager)

### Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/yourusername/all-cause-mortality-prediction.git
   cd all-cause-mortality-prediction
