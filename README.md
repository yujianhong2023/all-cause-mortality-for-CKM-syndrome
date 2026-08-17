# 🫀 All‑Cause Mortality Risk Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A web‑based tool that predicts an individual’s **all‑cause mortality risk** using a **Random Survival Forest** model.  
Built with [Streamlit](https://streamlit.io), it provides an intuitive interface for clinicians, researchers, and data scientists to explore risk estimates based on 22 clinical and demographic features.

---

## ✨ Features

- **22 clinical features**, including demographics, laboratory values, lifestyle factors, and medical history.
- **Predict risk at any time** from 0.5 to 30 years (with quick presets: 1, 3, 5, 10 years).
- **Two‑panel layout**: input patient characteristics on the left, view results instantly on the right.
- **Clear risk interpretation**: mortality risk percentage, survival probability, and risk level (Low / Moderate / High).
- **Comparison table** showing risk at 1, 3, 5, and 10 years for a broader perspective.
- **Fully responsive** – works on desktop, tablet, and mobile.

---

## 🧬 Features Used

The model uses the following 22 features:

| Feature | Description | Type |
|---------|-------------|------|
| `AGE` | Age (years) | Numeric |
| `GENDER` | Sex | Categorical |
| `PIR_GROUP` | Poverty‑income ratio group | Categorical |
| `CKM` | CKM Stage | Categorical |
| `HBA1C` | Glycated hemoglobin (%) | Numeric |
| `ABSI` | A Body Shape Index | Numeric |
| `EGFR` | Estimated glomerular filtration rate (mL/min/1.73m²) | Numeric |
| `RDW` | Red cell distribution width (%) | Numeric |
| `SHR` | Stress hyperglycemia ratio | Numeric |
| `MCV` | Mean corpuscular volume (fL) | Numeric |
| `GLB` | Globulin (g/L) | Numeric |
| `PLT` | Platelet count (×10⁹/L) | Numeric |
| `CRP` | C‑reactive protein (mg/L) | Numeric |
| `SII` | Systemic immunity‑inflammation index (×10⁹/L) | Numeric |
| `MON` | Monocyte count (×10⁹/L) | Numeric |
| `AST` | Aspartate aminotransferase (U/L) | Numeric |
| `BMI` | Body mass index (kg/m²) | Numeric |
| `TC` | Total cholesterol (mmol/L) | Numeric |
| `SMOKE` | Smoking status | Categorical |
| `ACTIVITY` | Physical activity level | Categorical |
| `LUNG` | Pulmonary disease (0/1) | Binary |
| `CANCER` | Cancer history (0/1) | Binary |

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
