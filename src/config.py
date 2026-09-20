"""
Configuration file for the Heart Failure Detection System.
Provides flexible schema definitions, project filepaths, random seeds,
and model hyperparameter grids for easy adaptation and reproducibility.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure essential directories exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset file paths
DATA_PATH = DATA_DIR / "heart.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_pipeline.joblib"
METRICS_CSV_PATH = REPORTS_DIR / "model_comparison_metrics.csv"
METRICS_MD_PATH = REPORTS_DIR / "model_comparison_metrics.md"

# Dataset Schema Configurations (Schema-Flexible)
TARGET_COL = "HeartDisease"

# Numerical continuous/discrete features
NUMERIC_COLS = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak",
]

# Categorical features requiring encoding
CATEGORICAL_COLS = [
    "Sex",
    "ChestPainType",
    "FastingBS",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]

# Clinical data quality remediation targets
# 0 is medically invalid for serum cholesterol (biological impossibility) and resting blood pressure
COLS_WITH_INVALID_ZEROS = ["Cholesterol", "RestingBP"]
CHOLESTEROL_FLAG_COL = "Cholesterol_missing"

# Cross-Validation & Splitting Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# Human-readable labels for UI presentation
FEATURE_LABELS = {
    "Age": "Age (years)",
    "Sex": "Sex",
    "ChestPainType": "Chest Pain Type",
    "RestingBP": "Resting Blood Pressure (mm Hg)",
    "Cholesterol": "Serum Cholesterol (mg/dl)",
    "FastingBS": "Fasting Blood Sugar > 120 mg/dl",
    "RestingECG": "Resting Electrocardiogram",
    "MaxHR": "Maximum Heart Rate Achieved (bpm)",
    "ExerciseAngina": "Exercise-Induced Angina",
    "Oldpeak": "ST Depression ('Oldpeak')",
    "ST_Slope": "Slope of Peak Exercise ST Segment",
}

CHEST_PAIN_DESCRIPTIONS = {
    "ATA": "Atypical Angina (ATA)",
    "NAP": "Non-Anginal Pain (NAP)",
    "ASY": "Asymptomatic (ASY)",
    "TA": "Typical Angina (TA)",
}

RESTING_ECG_DESCRIPTIONS = {
    "Normal": "Normal",
    "ST": "ST-T Wave Abnormality",
    "LVH": "Left Ventricular Hypertrophy (LVH)",
}

ST_SLOPE_DESCRIPTIONS = {
    "Up": "Upsloping (Up)",
    "Flat": "Flat (Flat)",
    "Down": "Downsloping (Down)",
}

# Hyperparameter search grids for GridSearchCV (stratified 5-fold CV)
PARAM_GRIDS = {
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__penalty": ["l2"],
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__max_iter": [1000],
    },
    "Decision Tree": {
        "classifier__max_depth": [3, 5, 7, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [4, 6, 8, 12, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    },
    "Support Vector Machine": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__kernel": ["linear", "rbf"],
        "classifier__gamma": ["scale", "auto"],
    },
}
