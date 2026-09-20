"""
Preprocessing and pipeline construction for the Heart Failure Detection System.
Implements custom data quality remediation transformers to handle medically invalid zeros
and packages everything into a leak-free scikit-learn Pipeline with ColumnTransformer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLS,
    CHOLESTEROL_FLAG_COL,
    COLS_WITH_INVALID_ZEROS,
    NUMERIC_COLS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)


class ClinicalDataCleaner(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to remediate clinical recording anomalies:
    1. Generates a binary flag feature `Cholesterol_missing` (1 if 0 or NaN, else 0).
       Clinical rationale: Absence of lipid profiling often correlates with admission context.
    2. Replaces invalid 0 values with NaN in `Cholesterol` and `RestingBP`.
       Clinical rationale: Serum cholesterol = 0 or resting blood pressure = 0 are biological
       impossibilities in living individuals and represent missing or corrupt entries.
    3. Guarantees zero data leakage by allowing downstream imputers to compute medians
       strictly on the training fold during cross-validation.
    """

    def __init__(self, cols_with_invalid_zeros=None, flag_col=CHOLESTEROL_FLAG_COL):
        self.cols_with_invalid_zeros = (
            cols_with_invalid_zeros
            if cols_with_invalid_zeros is not None
            else COLS_WITH_INVALID_ZEROS
        )
        self.flag_col = flag_col

    def fit(self, X, y=None):
        # Stateless transformer; no parameters to estimate from training fold
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_out = X.copy()

        # Generate binary missingness indicator for Cholesterol
        if "Cholesterol" in X_out.columns:
            chol_vals = pd.to_numeric(X_out["Cholesterol"], errors="coerce")
            X_out[self.flag_col] = ((chol_vals == 0) | (chol_vals.isna())).astype(int)

        # Convert invalid zero readings to NaN
        for col in self.cols_with_invalid_zeros:
            if col in X_out.columns:
                series = pd.to_numeric(X_out[col], errors="coerce")
                X_out[col] = series.replace(0, np.nan)

        return X_out


def build_preprocessor():
    """
    Builds the ColumnTransformer for the processed dataset.
    - Numerical features: Median Imputation -> StandardScaler
    - Categorical features: OneHotEncoder(handle_unknown='ignore')
    - Binary flag features (Cholesterol_missing): SimpleImputer (most_frequent) -> Passthrough
    """
    # Numeric pipeline
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: handle categorical string fields
    cat_cols_to_encode = [c for c in CATEGORICAL_COLS if c != "FastingBS"]
    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    # Binary indicators pipeline (already 0/1, just ensure no NaNs)
    binary_cols = ["FastingBS", CHOLESTEROL_FLAG_COL]
    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, cat_cols_to_encode),
            ("bin", binary_pipeline, binary_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def build_full_pipeline(classifier):
    """
    Constructs an end-to-end scikit-learn Pipeline incorporating data cleaning,
    feature engineering, transformations, and the specified classifier.
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline(
        steps=[
            ("cleaner", ClinicalDataCleaner()),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def load_and_split_data(data_path, target_col=TARGET_COL, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Loads raw CSV dataset and performs a stratified train/test split.
    Checks for duplicates and logs basic structural metrics.
    """
    df = pd.read_csv(data_path)

    # Check and report duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"[PREPROCESS] Found and dropped {dup_count} duplicate rows.")
        df = df.drop_duplicates()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, df
