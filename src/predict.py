"""
Inference module for Heart Failure Detection System.
Loads the trained production pipeline and performs patient risk assessment,
probability scoring, clinical risk stratification, and feature contribution analysis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
)


class HeartFailurePredictor:
    """
    Inference service for assessing heart disease risk in individual clinical cases.
    """

    def __init__(self, model_path=BEST_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Train the model first via src/train.py"
            )
        self.pipeline = joblib.load(self.model_path)

    def _prepare_df(self, patient_data):
        """Converts input dictionary or dataframe to structured DataFrame."""
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        elif isinstance(patient_data, pd.DataFrame):
            df = patient_data.copy()
        else:
            raise TypeError("patient_data must be a dict or a pandas DataFrame")

        # Verify presence of expected fields
        expected_cols = set(NUMERIC_COLS + CATEGORICAL_COLS)
        missing_cols = expected_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Input data is missing required clinical fields: {missing_cols}")

        return df

    def predict(self, patient_data):
        """
        Generates risk prediction, probability percentage, risk category,
        and contributing factor analysis.
        """
        df = self._prepare_df(patient_data)

        # Raw prediction and probability
        pred_class = int(self.pipeline.predict(df)[0])

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(df)[0]
            risk_prob = float(probs[1])
        else:
            risk_prob = float(pred_class)

        # Risk Stratification: Low (<30%), Moderate (30-60%), High (>60%)
        if risk_prob < 0.30:
            risk_level = "Low"
            risk_color = "#27ae60"  # Green
            clinical_advice = "Patient displays low cardiac risk profile based on provided metrics."
        elif risk_prob <= 0.60:
            risk_level = "Moderate"
            risk_color = "#f39c12"  # Amber/Orange
            clinical_advice = "Patient displays moderate cardiac risk. Lifestyle intervention and regular monitoring suggested."
        else:
            risk_level = "High"
            risk_color = "#c0392b"  # Red
            clinical_advice = "Patient displays high risk of heart disease. Immediate cardiology review and diagnostic tests indicated."

        diagnosis_label = "Heart Disease Detected" if pred_class == 1 else "Normal (No Heart Disease Detected)"

        # Contributing factors breakdown
        contributions = self._compute_contributions(df)

        return {
            "prediction": pred_class,
            "diagnosis": diagnosis_label,
            "probability": risk_prob,
            "probability_percentage": f"{risk_prob * 100:.1f}%",
            "risk_level": risk_level,
            "risk_color": risk_color,
            "clinical_advice": clinical_advice,
            "contributions": contributions,
        }

    def _compute_contributions(self, df):
        """
        Identifies top contributing factors to risk score using model weights
        or tree importances relative to sample values.
        """
        try:
            preprocessor = self.pipeline.named_steps["preprocessor"]
            cleaner = self.pipeline.named_steps["cleaner"]
            classifier = self.pipeline.named_steps["classifier"]

            cleaned_df = cleaner.transform(df)
            transformed = preprocessor.transform(cleaned_df)[0]

            cat_cols_to_encode = [c for c in CATEGORICAL_COLS if c != "FastingBS"]
            encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_encoded_names = list(encoder.get_feature_names_out(cat_cols_to_encode))
            all_feature_names = NUMERIC_COLS + cat_encoded_names + ["FastingBS", "Cholesterol_missing"]

            if hasattr(classifier, "coef_"):
                weights = classifier.coef_[0]
                feature_effects = weights * transformed
            elif hasattr(classifier, "feature_importances_"):
                # Weight by magnitude of transformed feature
                feature_effects = classifier.feature_importances_ * np.abs(transformed)
            else:
                return {}

            contrib_dict = dict(zip(all_feature_names, feature_effects))
            # Sort top 5 driving factors
            sorted_contrib = sorted(contrib_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            return {k: round(float(v), 3) for k, v in sorted_contrib}
        except Exception:
            return {}


def get_sample_patients():
    """Returns archetypal low-risk and high-risk patients for testing and UI presets."""
    low_risk_patient = {
        "Age": 38,
        "Sex": "F",
        "ChestPainType": "ATA",
        "RestingBP": 115,
        "Cholesterol": 185,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 172,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up",
    }

    high_risk_patient = {
        "Age": 62,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 155,
        "Cholesterol": 0,  # Demonstrates zero cholesterol handling
        "FastingBS": 1,
        "RestingECG": "LVH",
        "MaxHR": 110,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.5,
        "ST_Slope": "Flat",
    }

    return low_risk_patient, high_risk_patient


if __name__ == "__main__":
    print("=" * 60)
    print(" Heart Failure Prediction - Single Patient Inference Test")
    print("=" * 60)

    try:
        predictor = HeartFailurePredictor()
        low_p, high_p = get_sample_patients()

        print("\n--- Testing Low-Risk Archetype ---")
        res_low = predictor.predict(low_p)
        print(f"Risk Probability: {res_low['probability_percentage']} | Level: {res_low['risk_level']}")
        print(f"Diagnosis: {res_low['diagnosis']}")

        print("\n--- Testing High-Risk Archetype ---")
        res_high = predictor.predict(high_p)
        print(f"Risk Probability: {res_high['probability_percentage']} | Level: {res_high['risk_level']}")
        print(f"Diagnosis: {res_high['diagnosis']}")
        print(f"Top Contributing Factors: {res_high['contributions']}")

    except Exception as e:
        print(f"Prediction test failed: {e}")
        sys.exit(1)
