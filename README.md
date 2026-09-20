# Heart Failure Detection System Using Machine Learning
### Clinical Decision-Support System | College Capstone Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)

---

## 👥 Project Team Members

| Roll No. | Student Name | Role / Contribution |
|:---:|:---|:---|
| **7** | **Adhithyan JS** | Model Tuning, Hyperparameter Optimization & Cross-Validation |
| **29** | **Evin Saj Abraham** | Pipeline Architecture, Data Preprocessing & System Integration |
| **30** | **Farhana H** | Exploratory Data Analysis (EDA), Feature Selection & Reporting |
| **53** | **Ridhin Krishna M** | Streamlit Web Application Development, Inference API & UI/UX |

---

## 📌 Project Abstract & Objectives

Cardiovascular diseases (CVDs) remain the leading cause of death globally, accounting for nearly 18 million lives each year according to the World Health Organization (WHO). Heart failure occurs when the heart muscle cannot pump blood as effectively as normal. Early diagnosis and timely intervention are crucial in mitigating mortality and hospitalization rates.

The **Heart Failure Detection System** is an end-to-end machine learning decision-support tool engineered to analyze routine patient clinical indicators (demographics, metabolic biomarkers, and exercise electrocardiography) and estimate the likelihood of heart failure.

> [!IMPORTANT]
> **Clinical Disclaimer:** This tool is designed strictly as an assistive **clinical decision-support system (CDSS)** for healthcare professionals. It does **not** replace formal clinical assessment, physician judgment, coronary angiography, or echocardiography.

---

## 🏗️ System Architecture & Methodology

```
┌─────────────────────────┐
│     Raw Patient Data    │ (Demographics, Vitals, Labs, ECG)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  ClinicalDataCleaner    │ -> Detects & flags 172 zero Cholesterol entries (Cholesterol_missing)
│                         │ -> Replaces invalid 0s with NaN (RestingBP, Cholesterol)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   ColumnTransformer     │ -> Numerical: Median Imputer + StandardScaler
│                         │ -> Categorical: OneHotEncoder(handle_unknown='ignore')
│                         │ -> Binary flags: Passthrough (leakage-free)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Stratified 5-Fold CV    │ -> Grid search across 4 classifiers
│  Hyperparameter Tuning  │    (Logistic Regression, Decision Tree, Random Forest, SVM)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Clinical Model Selection│ -> Prioritize Recall & F1-Score (minimize False Negatives)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Streamlit Web App     │ -> Interactive patient risk assessment, risk gauge & model demo
└─────────────────────────┘
```

### 1. Data Ingestion & Quality Remediation
- **Dataset Source:** Kaggle Heart Failure Prediction Dataset by fedesoriano ([kaggle.com/datasets/fedesoriano/heart-failure-prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)), consisting of 918 patient observations and 11 clinical features.
- **Cholesterol Zero Remediation:** 172 rows (18.7% of the dataset) contained `Cholesterol = 0`. Because serum cholesterol cannot biologically be zero in a living human, these represent missing laboratory measurements. The pipeline converts 0 to `NaN`, imputes using the **median of the training fold**, and adds a binary feature `Cholesterol_missing` (1 if unmeasured, 0 otherwise) to capture hospital admission context without data leakage.
- **RestingBP Zero Remediation:** 1 row had `RestingBP = 0 mm Hg`, an invalid physiological reading, which is converted to `NaN` and imputed with the training median.
- **Target Distribution:** 508 positive (55.3%) vs. 410 negative (44.7%) cases. The dataset is balanced (~55/45); hence synthetic resampling (SMOTE) is unnecessary.
- **Schema Flexibility:** Column schema definitions are maintained in `src/config.py`, enabling adaptation to other cardiology datasets with minimal alterations.

### 2. Feature Engineering & Selection
All 11 features are evaluated using:
1. Pearson Correlation with Target
2. Mutual Information Scores (`mutual_info_classif`)
3. Random Forest Gini Feature Importances

**Decision:** All 11 features contribute distinct, clinically proven prognostic signal (e.g., ST slope, chest pain classification, maximum heart rate, exercise angina, and ST depression), and are retained in the production model.

### 3. Machine Learning Models & Cross-Validation
Four diverse classification paradigms are trained and compared:
1. **Logistic Regression:** Linear probabilistic baseline (`L2` regularization).
2. **Decision Tree Classifier:** Non-linear decision rules (`Gini` and `Entropy` criteria).
3. **Random Forest Classifier:** Bagged ensemble of decorrelated trees.
4. **Support Vector Machine (SVM):** Maximum-margin kernel separator (`RBF` and `Linear` kernels, `probability=True`).

All models are tuned via `GridSearchCV` using **5-Fold Stratified Cross-Validation** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).

### 4. Clinical Model Selection Criterion
In cardiology, a **False Negative** (failing to identify a patient with heart disease) can result in untreated myocardial infarction or sudden cardiac death. In contrast, a **False Positive** leads to secondary non-invasive testing. Consequently, our evaluation strictly prioritizes **Recall (Sensitivity)** and **F1-Score** over raw accuracy.

---

## 📊 Experimental Results & Model Comparison

Evaluated on the held-out stratified test set (20% partition, $N = 184$ patients):

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | CV F1 (Mean ± Std) | False Negatives |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | *To be populated* | *To be populated* | *To be populated* | *To be populated* | *To be populated* | *To be populated* | *To be populated* |
| **Support Vector Machine** | *To be populated* | *To be populated* | *To be populated* | *To be populated* | *To be populated* | *To be populated* | *To be populated* |
| **Logistic Regression** | *To be populated* | *To be populated* | *To be populated* | *To be populated* | *To be populated* | *To be populated* | *To be populated* |
| **Decision Tree** | *To be populated* | *To be populated* | *To be populated* | *To be populated* | *To be populated* | *To be populated* | *To be populated* |

*(The complete benchmark metrics table is generated automatically in `reports/model_comparison_metrics.md` upon executing the training pipeline).*

---

## 📁 Project Directory Structure

```
heart-failure-detection/
├── .venv/                             # Isolated Python virtual environment
├── .gitignore                         # Version control ignore definitions
├── data/
│   └── heart.csv                      # Fedesoriano Kaggle dataset (918 records)
├── notebooks/
│   └── eda_and_modeling.ipynb         # Full EDA, visualization & modeling experiments
├── src/
│   ├── __init__.py
│   ├── config.py                      # Schema definitions, paths, random seed, grids
│   ├── preprocess.py                  # Clinical cleaner, ColumnTransformer, pipeline
│   ├── evaluate.py                    # Metric computations & diagnostic plotting
│   ├── train.py                       # 5-fold CV grid tuning, comparison & export
│   └── predict.py                     # Single-patient clinical inference engine
├── models/
│   └── best_pipeline.joblib           # Serialized winning pipeline
├── reports/
│   ├── eda_target_distribution.png    # Class balance chart
│   ├── eda_numeric_distributions.png  # Continuous feature histograms with KDE
│   ├── eda_categorical_by_target.png  # Categorical features stratified by diagnosis
│   ├── eda_correlation_matrix.png     # Pearson correlation heatmap
│   ├── feature_importance.png         # Model feature importances & mutual info
│   ├── model_roc_curves.png           # Multi-model ROC curves overlay
│   ├── best_model_confusion_matrix.png# Confusion matrix of best model
│   ├── model_comparison_metrics.csv   # Performance metrics table (CSV)
│   └── model_comparison_metrics.md    # Performance metrics table (Markdown)
├── app.py                             # Interactive Streamlit Web Application
├── requirements.txt                   # Pinned dependency specifications
└── README.md                          # Project documentation and guide
```

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10 or higher
- Git version control installed

### 1. Clone the Repository
```bash
git clone <repo-url>
cd Heart_failure
```

### 2. Create and Activate Virtual Environment
Using a virtual environment ensures that the project dependencies do not conflict with your other projects.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models and Generate Reports
Executes end-to-end data validation, EDA visualization generation, 5-Fold cross-validation hyperparameter search across all 4 models, test set evaluation, and saves the winning pipeline to `models/best_pipeline.joblib`:
```bash
python src/train.py
```

### 5. Run Single Patient Inference (CLI Test)
Tests the saved production pipeline on archetypal low-risk and high-risk patients:
```bash
python src/predict.py
```

### 6. Launch the Streamlit Web Application
Starts the interactive clinical decision-support app in your local browser:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Web Application Features

The interactive web application provides three distinct interfaces:
1. **🩺 Patient Risk Assessment:**
   - Input forms for all 11 patient parameters with physiological ranges.
   - Quick-load demo archetypes: **Low-Risk Patient (Healthy)** and **High-Risk Patient (Cardiac Distress)**.
   - Immediate risk categorization: **Low Risk (<30%)**, **Moderate Risk (30-60%)**, and **High Risk (>60%)**.
   - Color-coded risk meter and actionable clinical interpretations.
   - Top contributing risk factors breakdown for each patient prediction.
2. **📊 Model Comparison & Demo:**
   - Full benchmark metrics table highlighting best scores.
   - High-resolution ROC Curves comparison and Confusion Matrix plots.
   - College viva presentation talking points and clinical selection rationale.
3. **ℹ️ Dataset & Project Information:**
   - Complete dataset documentation, data cleaning notes, team member roster, and clinical limitations.

---

## ⚠️ Limitations & Ethical Considerations

1. **Cohort Size:** The dataset contains 918 patient observations. While substantial for academic exploration, modern clinical AI models typically require hundreds of thousands of diverse electronic health records.
2. **Dataset Aggregation:** The records are synthesized from 5 historical source datasets (Cleveland, Hungarian, Switzerland, Long Beach VA, and Statlog) collected in the 1980s and 1990s.
3. **Missing Biomarkers:** The dataset lacks key contemporary cardiovascular predictors:
   - Body Mass Index (BMI) and waist-to-hip ratio
   - Smoking history (pack-years)
   - High-sensitivity Troponin and Brain Natriuretic Peptide (BNP / NT-proBNP)
   - Detailed lipid subfractions (LDL, HDL, triglycerides)
   - `FastingBS` is used as a binary threshold proxy (>120 mg/dl) rather than continuous HbA1c.
4. **Binary Target Definition:** The outcome reflects the presence of coronary heart disease (angiographic disease status $>50\%$ diameter narrowing in at least one major coronary artery), which acts as a precursor/proxy for heart failure risk.

---

## 🔮 Future Work

1. **EHR / EMR Integration:** Integrate with hospital standards like HL7 / FHIR for real-time risk scoring during triage.
2. **Wearable & Continuous Telemetry:** Incorporate time-series photoplethysmography (PPG) and ECG data from smartwatches for continuous ambulatory monitoring.
3. **Deep Learning & Survival Analysis:** Implement Cox Proportional Hazards and DeepSurv models to predict time-to-event (e.g., 1-year or 5-year heart failure decompensation risk).
4. **Explainable AI (XAI):** Deploy patient-specific SHAP (SHapley Additive exPlanations) waterfall plots and counterfactual explanations for clinician trust.
