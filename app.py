"""
Streamlit Web Application: Heart Failure Detection System
Decision-support machine learning interface for clinicians and college project presentation.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

# Set page configuration with medical styling
st.set_page_config(
    page_title="CardioSense AI - Heart Failure Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for crisp, high-contrast clinical UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 0.85rem 1.1rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.92rem;
        color: #991b1b;
    }
    .card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 1.2rem;
        font-weight: 700;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Lazy import predictor to handle missing model gracefully
@st.cache_resource
def get_predictor():
    from src.predict import HeartFailurePredictor
    try:
        return HeartFailurePredictor()
    except Exception as e:
        return None


# Archetype presets for rapid demonstration
PRESETS = {
    "Select Preset": None,
    "Archetype: Low-Risk Patient (Healthy)": {
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
    },
    "Archetype: High-Risk Patient (Cardiac Distress)": {
        "Age": 62,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 155,
        "Cholesterol": 0,  # Demonstrates zero imputation + indicator handling
        "FastingBS": 1,
        "RestingECG": "LVH",
        "MaxHR": 110,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.5,
        "ST_Slope": "Flat",
    },
}

# Top Header
st.markdown('<div class="main-title">❤️ CardioSense AI: Heart Failure Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Machine Learning Clinical Decision-Support System | College Project Demonstration</div>',
    unsafe_allow_html=True,
)

# Mandatory Clinical Disclaimer
st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ Clinical Decision-Support Notice:</strong>
        This machine learning system is an investigational academic prototype designed for clinical decision-support only.
        It is <strong>not</strong> a certified medical device and does <strong>not</strong> substitute for professional medical consultation,
        clinical judgment, or formal diagnostic evaluation.
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🩺 Patient Risk Assessment",
    "📊 Model Comparison & Demo",
    "ℹ️ Dataset & Project Information",
])

# ---------------------------------------------------------------------------------
# TAB 1: PATIENT RISK ASSESSMENT
# ---------------------------------------------------------------------------------
with tab1:
    st.markdown("### Patient Clinical Profile Input")
    st.markdown("Enter patient physiological measurements or select an archetype preset to assess cardiac risk.")

    # Preset Selector
    preset_choice = st.selectbox(
        "⚡ Quick Load Patient Archetype (For Demo):",
        list(PRESETS.keys()),
        index=0,
    )
    preset_data = PRESETS[preset_choice]

    # Initialize form default values
    def get_val(key, default):
        return preset_data[key] if preset_data and key in preset_data else default

    with st.form("patient_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 👤 Demographics & Symptoms")
            age = st.slider("Age (years)", min_value=18, max_value=90, value=int(get_val("Age", 54)), step=1)

            sex_opts = ["Male", "Female"]
            sex_default_idx = 0 if get_val("Sex", "M") == "M" else 1
            sex_choice = st.selectbox("Biological Sex", sex_opts, index=sex_default_idx)
            sex = "M" if sex_choice == "Male" else "F"

            cp_mapping = {
                "Typical Angina (TA)": "TA",
                "Atypical Angina (ATA)": "ATA",
                "Non-Anginal Pain (NAP)": "NAP",
                "Asymptomatic (ASY)": "ASY",
            }
            inv_cp = {v: k for k, v in cp_mapping.items()}
            default_cp_str = inv_cp.get(get_val("ChestPainType", "ASY"), "Asymptomatic (ASY)")
            cp_choice = st.selectbox("Chest Pain Type", list(cp_mapping.keys()), index=list(cp_mapping.keys()).index(default_cp_str))
            chest_pain_type = cp_mapping[cp_choice]

            ex_angina_opts = ["No", "Yes"]
            ex_default_idx = 1 if get_val("ExerciseAngina", "N") == "Y" else 0
            ex_angina_choice = st.selectbox("Exercise-Induced Angina", ex_angina_opts, index=ex_default_idx)
            exercise_angina = "Y" if ex_angina_choice == "Yes" else "N"

            st.markdown("#### 🧪 Metabolic Profile")
            fasting_bs_default = bool(get_val("FastingBS", 0) == 1)
            fasting_bs = st.checkbox(
                "Fasting Blood Sugar > 120 mg/dl (Hyperglycemia / Diabetes indicator)",
                value=fasting_bs_default,
            )
            fasting_bs_val = 1 if fasting_bs else 0

        with col2:
            st.markdown("#### 💓 Hemodynamics & Electrocardiogram")
            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=0,
                max_value=240,
                value=int(get_val("RestingBP", 130)),
                step=1,
                help="Systolic BP at rest. Entering 0 will trigger automated median imputation.",
            )

            cholesterol = st.number_input(
                "Serum Cholesterol (mg/dl) [Enter 0 if unmeasured / unknown]",
                min_value=0,
                max_value=700,
                value=int(get_val("Cholesterol", 220)),
                step=1,
                help="Zero represents missing clinical recording. It will be imputed via training median and flagged with a missingness feature.",
            )

            ecg_mapping = {
                "Normal": "Normal",
                "ST-T Wave Abnormality": "ST",
                "Left Ventricular Hypertrophy (LVH)": "LVH",
            }
            inv_ecg = {v: k for k, v in ecg_mapping.items()}
            default_ecg_str = inv_ecg.get(get_val("RestingECG", "Normal"), "Normal")
            ecg_choice = st.selectbox("Resting ECG Result", list(ecg_mapping.keys()), index=list(ecg_mapping.keys()).index(default_ecg_str))
            resting_ecg = ecg_mapping[ecg_choice]

            max_hr = st.slider(
                "Maximum Heart Rate Achieved (MaxHR - bpm)",
                min_value=50,
                max_value=230,
                value=int(get_val("MaxHR", 140)),
                step=1,
            )

            oldpeak = st.number_input(
                "ST Depression ('Oldpeak' - mm)",
                min_value=-3.0,
                max_value=7.0,
                value=float(get_val("Oldpeak", 0.0)),
                step=0.1,
                format="%.1f",
                help="ST depression induced by exercise relative to rest.",
            )

            st_slope_mapping = {
                "Upsloping (Up)": "Up",
                "Flat (Flat)": "Flat",
                "Downsloping (Down)": "Down",
            }
            inv_st = {v: k for k, v in st_slope_mapping.items()}
            default_st_str = inv_st.get(get_val("ST_Slope", "Flat"), "Flat (Flat)")
            st_choice = st.selectbox("Slope of Peak Exercise ST Segment", list(st_slope_mapping.keys()), index=list(st_slope_mapping.keys()).index(default_st_str))
            st_slope = st_slope_mapping[st_choice]

        submit_btn = st.form_submit_button("🩺 Compute Cardiac Risk Assessment", use_container_width=True)

    if submit_btn:
        patient_dict = {
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain_type,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs_val,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope,
        }

        predictor = get_predictor()
        if predictor is None:
            st.error("❌ Trained model pipeline not found! Please run `python src/train.py` first.")
        else:
            with st.spinner("Analyzing clinical markers..."):
                result = predictor.predict(patient_dict)

            st.markdown("---")
            st.markdown("### 📋 Risk Assessment Results")

            r_col1, r_col2, r_col3 = st.columns([1.5, 1.5, 2])

            with r_col1:
                st.markdown(
                    f"""
                    <div style="background-color: {result['risk_color']}; color: white; padding: 1.2rem; border-radius: 8px; text-align: center;">
                        <h4 style="margin:0; color: white;">Risk Stratification</h4>
                        <h2 style="margin: 0.5rem 0; color: white; font-size: 2.2rem;">{result['risk_level'].upper()} RISK</h2>
                        <p style="margin:0; font-size: 1rem;">{result['diagnosis']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r_col2:
                st.markdown(
                    f"""
                    <div style="border: 2px solid {result['risk_color']}; padding: 1.2rem; border-radius: 8px; text-align: center;">
                        <h4 style="margin:0; color: #4b5563;">Predicted Probability</h4>
                        <h2 style="margin: 0.5rem 0; color: {result['risk_color']}; font-size: 2.4rem;">{result['probability_percentage']}</h2>
                        <p style="margin:0; font-size: 0.85rem; color: #6b7280;">Decision Threshold: 50% | High Risk: >60%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with r_col3:
                st.markdown("#### Clinical Interpretation")
                st.info(result["clinical_advice"])
                if cholesterol == 0:
                    st.warning("ℹ️ **Data Note:** Cholesterol was recorded as 0. Pipeline automatically imputed median value and marked missingness indicator.")

            # Probability Progress Meter
            st.markdown("#### Cardiac Risk Probability Meter")
            st.progress(min(result["probability"], 1.0))

            # Contributing factors breakdown
            if result.get("contributions"):
                st.markdown("#### 🔍 Top Contributing Risk Factors for this Patient")
                st.markdown("Positive weights drive higher cardiac risk; negative weights indicate protective factors:")
                contrib_df = pd.DataFrame(
                    list(result["contributions"].items()),
                    columns=["Clinical Factor", "Relative Contribution Weight"],
                )
                st.dataframe(contrib_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------------
# TAB 2: MODEL COMPARISON & DEMO
# ---------------------------------------------------------------------------------
with tab2:
    st.markdown("### 🏆 Machine Learning Model Comparison & Evaluation")
    st.markdown(
        "Four distinct classification architectures were evaluated with **5-Fold Stratified Cross-Validation** "
        "and optimized using Grid Search. Given the medical setting, **Recall (Sensitivity)** and **F1-Score** "
        "were prioritized to minimize False Negatives."
    )

    metrics_csv = Path("reports/model_comparison_metrics.csv")
    if metrics_csv.exists():
        metrics_df = pd.read_csv(metrics_csv)
        st.markdown("#### 📈 Benchmark Metrics Table (Test Set N=184)")
        st.dataframe(
            metrics_df.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                color="#d1fae5",
            ).highlight_min(
                subset=["False Negatives"],
                color="#d1fae5",
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### 🎯 Clinical Selection Rationale")
        best_row = metrics_df.sort_values(by=["Recall", "F1-Score"], ascending=[False, False]).iloc[0]
        st.success(
            f"**Selected Model: {best_row['Model']}**\n\n"
            f"- **Recall (Sensitivity):** {best_row['Recall']:.3f} — accurately flags true positive cardiac cases.\n"
            f"- **F1-Score:** {best_row['F1-Score']:.3f} — harmonic balance preventing excessive false alarms.\n"
            f"- **Missed Diagnoses (False Negatives):** Only {int(best_row['False Negatives'])} out of test cohort, "
            f"minimizing life-threatening clinical oversight."
        )
    else:
        st.info("Run `python src/train.py` to generate model benchmark metrics.")

    st.markdown("---")
    st.markdown("### 📊 Diagnostic Visualizations")
    img_col1, img_col2 = st.columns(2)

    roc_path = Path("reports/model_roc_curves.png")
    cm_path = Path("reports/best_model_confusion_matrix.png")
    fi_path = Path("reports/feature_importance.png")

    with img_col1:
        if roc_path.exists():
            st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)
        else:
            st.warning("ROC curve plot not found.")

    with img_col2:
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix of Winning Model", use_container_width=True)
        else:
            st.warning("Confusion matrix plot not found.")

    if fi_path.exists():
        st.markdown("#### 🌲 Feature Importance & Predictive Markers")
        st.image(str(fi_path), caption="Top Predictive Clinical Features in Pipeline", use_container_width=True)


# ---------------------------------------------------------------------------------
# TAB 3: DATASET & PROJECT INFORMATION
# ---------------------------------------------------------------------------------
with tab3:
    st.markdown("### 📖 Project & Dataset Overview")

    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.markdown(
            """
            #### 🗂️ Dataset Architecture
            - **Source:** [Kaggle Heart Failure Prediction Dataset by fedesoriano](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
            - **Records:** 918 patients (synthesized across 5 historical cardiac datasets: Cleveland, Hungarian, Switzerland, Long Beach VA, and Statlog).
            - **Features:** 11 clinical features + 1 binary target (`HeartDisease`).
            - **Class Balance:** 508 Heart Disease (55.3%) vs. 410 Normal (44.7%) — balanced cohort, no synthetic resampling required.

            #### 🛠️ Data Quality Remediation
            1. **Invalid Serum Cholesterol Zeros:**
               - 172 records contained `Cholesterol = 0` (physiologically impossible in living humans).
               - Addressed inside the pipeline via median imputation strictly computed on the training split, coupled with a binary missingness flag `Cholesterol_missing`.
            2. **Invalid Blood Pressure Zero:**
               - 1 record contained `RestingBP = 0`. Handled via median imputation.
            3. **Leakage Prevention:**
               - All scalers, encoders, and imputers fit solely on training folds via scikit-learn `Pipeline` and `ColumnTransformer`.
            """
        )

    with col_info2:
        st.markdown(
            """
            #### 👥 Project Team Members
            - **Adhithyan JS** (Roll No. 7)
            - **Evin Saj Abraham** (Roll No. 29)
            - **Farhana H** (Roll No. 30)
            - **Ridhin Krishna M** (Roll No. 53)

            #### ⚠️ Limitations & Considerations
            - **Cohort Size:** 918 records is modest for deep clinical generalization.
            - **Missing Biomarkers:** Does not include Body Mass Index (BMI), smoking pack-years, HbA1c, or Troponin levels. `FastingBS` is used as a proxy for glucose tolerance.
            - **Decision Support:** Designed solely to assist triage, not to issue definitive prescriptions.
            """
        )
