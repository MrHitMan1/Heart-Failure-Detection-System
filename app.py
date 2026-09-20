"""
Streamlit Web Application: Heart Failure Detection System
Modern Clinical Decision Support Interface
Clinical decision-support machine learning interface for clinicians and college project presentation.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

# ── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense - SCT Edition",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── External Design System & Background Assets ─────────────────────────────────
def load_assets():
    css_path = Path("assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    bg_path = Path("assets/background.html")
    if bg_path.exists():
        st.iframe(bg_path.read_text(encoding="utf-8"), height=1)

load_assets()

# ── Model Predictor Singleton ──────────────────────────────────────────────────
@st.cache_resource
def get_predictor():
    from src.predict import HeartFailurePredictor
    try:
        return HeartFailurePredictor()
    except Exception:
        return None

# ── Archetype Presets ───────────────────────────────────────────────────────────
PRESETS = {
    "— Select Patient Archetype —": None,
    "🟢  Archetype: Low-Risk Patient (Healthy Vitals)": {
        "Age": 38, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 115,
        "Cholesterol": 185, "FastingBS": 0, "RestingECG": "Normal",
        "MaxHR": 172, "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Up",
    },
    "🔴  Archetype: High-Risk Patient (Cardiac Distress)": {
        "Age": 62, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 155,
        "Cholesterol": 0, "FastingBS": 1, "RestingECG": "LVH",
        "MaxHR": 110, "ExerciseAngina": "Y", "Oldpeak": 2.5, "ST_Slope": "Flat",
    },
}

# ── Hero Telemetry Deck ────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-card">
        <div class="telemetry-status-row">
            <div class="telemetry-beacon-chip">
                <span class="beacon-dot"></span>
                <span>BIO-TELEMETRY INFERENCE ENGINE // ONLINE</span>
            </div>
            <div class="telemetry-specs">
                <span>COHORT N=918</span> · <span>5 CLINICAL REGISTRIES</span> · <span>DIAGNOSTIC RECALL: 93.1%</span>
            </div>
        </div>
        <div class="hero-title">
            <span class="heart-emoji">🫀</span>
            <span class="title-text">CardioSense</span>
        </div>
        <p class="hero-subtitle">
            Precision clinical decision-support telemetry predicting heart failure risk from 11 multivariable biomarkers.
            Tuned with Stratified 5-Fold Cross-Validation, prioritizing diagnostic sensitivity to eliminate false negatives.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Medical Notice ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="clinical-notice">
        <span>⚠️</span>
        <div>
            <strong>Clinical Protocol Notice:</strong>
            This software is an investigational decision-support prototype calibrated strictly for clinician risk triage.
            It operates as an adjunct to, and does not replace, 12-lead ECG telemetry, serum troponin assay, or coronary angiography.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Navigation Tabs ────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  🩺  01 // PATIENT TRIAGE  ",
    "  📊  02 // BENCHMARK MATRIX  ",
    "  ℹ️  03 // CLINICAL PROTOCOL & TEAM  ",
])

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — PATIENT RISK ASSESSMENT                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<div class="section-header"><span>📋</span> CLINICAL PROFILE INPUT</div>', unsafe_allow_html=True)

    # Archetype Selector
    st.markdown(
        '<div class="section-header" style="margin-bottom: 0.5rem;">'
        '<span>⚡</span> QUICK PRESET ARCHETYPES (DEMO & VIVA PRESENTATION)'
        '</div>',
        unsafe_allow_html=True,
    )
    preset_choice = st.selectbox(
        "⚡ Quick Preset Archetypes:",
        list(PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
    )

    preset_data = PRESETS[preset_choice]
    def _val(key, default):
        return preset_data[key] if preset_data and key in preset_data else default

    with st.form("patient_clinical_form"):
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown("##### 👤 Demographics & Symptoms")
            age = st.slider("Age (years)", 18, 90, int(_val("Age", 54)))

            sex_opts = ["Male", "Female"]
            sex_choice = st.selectbox(
                "Biological Sex",
                sex_opts,
                index=0 if _val("Sex", "M") == "M" else 1,
            )
            sex = "M" if sex_choice == "Male" else "F"

            cp_mapping = {
                "Typical Angina (TA)": "TA",
                "Atypical Angina (ATA)": "ATA",
                "Non-Anginal Pain (NAP)": "NAP",
                "Asymptomatic (ASY)": "ASY",
            }
            inv_cp = {v: k for k, v in cp_mapping.items()}
            default_cp_str = inv_cp.get(_val("ChestPainType", "ASY"), "Asymptomatic (ASY)")
            cp_choice = st.selectbox(
                "Chest Pain Type",
                list(cp_mapping.keys()),
                index=list(cp_mapping.keys()).index(default_cp_str),
            )
            chest_pain_type = cp_mapping[cp_choice]

            ex_opts = ["No", "Yes"]
            ex_choice = st.selectbox(
                "Exercise-Induced Angina",
                ex_opts,
                index=1 if _val("ExerciseAngina", "N") == "Y" else 0,
            )
            exercise_angina = "Y" if ex_choice == "Yes" else "N"

            st.markdown("##### 🧪 Metabolic Markers")
            fasting_bs = st.checkbox(
                "Fasting Blood Sugar > 120 mg/dl (Hyperglycemia Indicator)",
                value=bool(_val("FastingBS", 0) == 1),
            )
            fasting_bs_val = 1 if fasting_bs else 0

        with col_right:
            st.markdown("##### 💓 Hemodynamics & Electrocardiogram")
            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                0, 240, int(_val("RestingBP", 130)),
                step=1,
                help="Systolic blood pressure at rest. Entering 0 will trigger automated training median imputation.",
            )

            cholesterol = st.number_input(
                "Serum Cholesterol (mg/dl) [Enter 0 if unmeasured]",
                0, 700, int(_val("Cholesterol", 220)),
                step=1,
                help="Serum cholesterol. If entered as 0, the pipeline automatically imputes training median and sets the Cholesterol_missing flag.",
            )

            ecg_mapping = {
                "Normal": "Normal",
                "ST-T Wave Abnormality": "ST",
                "Left Ventricular Hypertrophy (LVH)": "LVH",
            }
            inv_ecg = {v: k for k, v in ecg_mapping.items()}
            default_ecg_str = inv_ecg.get(_val("RestingECG", "Normal"), "Normal")
            ecg_choice = st.selectbox(
                "Resting Electrocardiogram (ECG)",
                list(ecg_mapping.keys()),
                index=list(ecg_mapping.keys()).index(default_ecg_str),
            )
            resting_ecg = ecg_mapping[ecg_choice]

            max_hr = st.slider(
                "Maximum Heart Rate Achieved (MaxHR - bpm)",
                50, 230, int(_val("MaxHR", 140)),
            )

            oldpeak = st.number_input(
                "ST Depression ('Oldpeak' - mm)",
                -3.0, 7.0, float(_val("Oldpeak", 0.0)),
                step=0.1, format="%.1f",
                help="ST depression induced by exercise relative to rest.",
            )

            slope_mapping = {
                "Upsloping (Up)": "Up",
                "Flat (Flat)": "Flat",
                "Downsloping (Down)": "Down",
            }
            inv_slope = {v: k for k, v in slope_mapping.items()}
            default_slope_str = inv_slope.get(_val("ST_Slope", "Flat"), "Flat (Flat)")
            slope_choice = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                list(slope_mapping.keys()),
                index=list(slope_mapping.keys()).index(default_slope_str),
            )
            st_slope = slope_mapping[slope_choice]

        submit_assessment = st.form_submit_button("⚡  EXECUTE MULTIVARIABLE RISK INFERENCE", width="stretch")

    # ── Assessment Results (Clinical Dashboard) ──
    if submit_assessment:
        patient_dict = {
            "Age": age, "Sex": sex, "ChestPainType": chest_pain_type,
            "RestingBP": resting_bp, "Cholesterol": cholesterol,
            "FastingBS": fasting_bs_val, "RestingECG": resting_ecg,
            "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak, "ST_Slope": st_slope,
        }
        predictor = get_predictor()
        if predictor is None:
            st.error("❌ Production pipeline not found. Please execute `python src/train.py` first.")
        else:
            with st.spinner("Processing clinical markers with Logistic Regression pipeline…"):
                res = predictor.predict(patient_dict)

            prob = res["probability"]
            pct = prob * 100
            lvl = res["risk_level"]
            diag = res["diagnosis"]

            # Clinical Telemetry Palette per Risk Tier
            RISK_TIERS = {
                "Low": {
                    "primary": "#00E676",      # Bioluminescent Emerald
                    "bg": "rgba(0, 230, 118, 0.10)",
                    "border": "rgba(0, 230, 118, 0.40)",
                    "tag": "NORMAL // LOW RISK",
                },
                "Moderate": {
                    "primary": "#FFB300",      # Solar Amber
                    "bg": "rgba(255, 179, 0, 0.10)",
                    "border": "rgba(255, 179, 0, 0.40)",
                    "tag": "ELEVATED // CAUTION",
                },
                "High": {
                    "primary": "#FF1E44",      # Arterial Crimson
                    "bg": "rgba(255, 30, 68, 0.14)",
                    "border": "rgba(255, 30, 68, 0.50)",
                    "tag": "CRITICAL // HIGH RISK",
                },
            }
            tier = RISK_TIERS[lvl]

            st.markdown("---")
            st.markdown('<div class="section-header"><span>📊</span> RISK ASSESSMENT DASHBOARD</div>', unsafe_allow_html=True)

            # 3-Column Diagnostic Widget Row
            w1, w2, w3 = st.columns([1.2, 1.2, 1.8], gap="medium")

            with w1:
                st.markdown(
                    f"""
                    <div class="health-widget" style="border-color: {tier['border']}; background: {tier['bg']};">
                        <div class="telemetry-beacon-chip" style="background: rgba(255,255,255,0.06); border-color: {tier['border']}; color: {tier['primary']}; margin-bottom: 0.6rem;">
                            {tier['tag']}
                        </div>
                        <div class="widget-value" style="color: {tier['primary']}; font-family: var(--font-display);">{lvl.upper()}</div>
                        <p class="widget-caption" style="color: #F1F5F9; font-weight: 500;">{diag}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with w2:
                # Circular Probability Ring SVG
                circ = 364.42
                stroke_dashoffset = circ - (prob * circ)

                st.markdown(
                    f"""
                    <div class="health-widget">
                        <div class="widget-label">CARDIAC RISK PROBABILITY</div>
                        <div class="ring-container">
                            <svg width="152" height="152" viewBox="0 0 152 152">
                                <circle cx="76" cy="76" r="58" stroke="rgba(255, 255, 255, 0.08)" stroke-width="12" fill="none" />
                                <circle cx="76" cy="76" r="58" stroke="{tier['primary']}" stroke-width="12" fill="none"
                                        stroke-linecap="round"
                                        stroke-dasharray="{circ}"
                                        stroke-dashoffset="{stroke_dashoffset}"
                                        style="transition: stroke-dashoffset 0.8s cubic-bezier(0.16, 1, 0.3, 1); filter: drop-shadow(0 0 8px {tier['primary']});" />
                            </svg>
                            <div class="ring-center-content">
                                <div class="ring-percent" style="color: {tier['primary']}; font-family: var(--font-mono);">{pct:.1f}<span style="font-size: 1.1rem; font-weight: 700;">%</span></div>
                                <div class="ring-subtext">PROBABILITY</div>
                            </div>
                        </div>
                        <p class="widget-caption" style="font-size: 0.72rem; color: #8290A4; font-family: var(--font-mono);">DECISION BOUNDARY: 50% | SENSITIVITY: 93.1%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with w3:
                st.markdown(
                    f"""
                    <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center; margin: 0;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.6rem;">
                            <span style="background: rgba(0, 242, 254, 0.12); border: 1px solid rgba(0, 242, 254, 0.35); border-radius: 6px; padding: 3px 10px; font-family: var(--font-mono); font-size: 0.72rem; font-weight: 700; color: #00F2FE; letter-spacing: 0.06em;">// CLINICAL ACTION PROTOCOL</span>
                        </div>
                        <p style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.65; margin: 0;">
                            {res['clinical_advice']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if cholesterol == 0:
                st.warning("ℹ️ **Data Fix Applied:** Serum cholesterol was entered as 0. The pipeline safely imputed the training median and marked the binary missingness indicator (`Cholesterol_missing=1`).")

            # ── Top Contributing Factors (Metric Trends) ──
            contribs = res.get("contributions", {})
            if contribs:
                st.markdown("<br>", unsafe_allow_html=True)
                max_val = max(abs(v) for v in contribs.values()) or 1.0
                rows_html = []
                for feat, weight in contribs.items():
                    bar_pct = min(int((abs(weight) / max_val) * 100), 100)
                    is_risk = weight > 0
                    color = "#FF375F" if is_risk else "#30D158"
                    badge_bg = "rgba(255, 55, 95, 0.15)" if is_risk else "rgba(48, 209, 88, 0.15)"
                    badge_border = "rgba(255, 55, 95, 0.35)" if is_risk else "rgba(48, 209, 88, 0.35)"
                    badge_label = "↑ Increases Risk" if is_risk else "↓ Protective Factor"
                    rows_html.append(
                        f'<div class="trend-row">'
                        f'<span class="trend-name">{feat}</span>'
                        f'<div class="trend-bar-bg"><div class="trend-bar-fill" style="width: {bar_pct}%; background: {color};"></div></div>'
                        f'<span class="trend-badge" style="background: {badge_bg}; border: 1px solid {badge_border}; color: {color};">'
                        f'{badge_label} ({weight:+.3f})'
                        f'</span>'
                        f'</div>'
                    )
                trend_content = "".join(rows_html)
                st.markdown(
                    f'<div class="glass-card" style="padding: 1.6rem 1.8rem; margin-bottom: 1rem;">'
                    f'<div class="section-header" style="margin-bottom: 0.35rem;"><span>📈</span> PHYSIOLOGICAL RISK FACTOR CONTRIBUTIONS</div>'
                    f'<p style="color:#8E8E93; font-size:0.85rem; margin-top:0; margin-bottom:1rem;">'
                    f'Derived from the trained Logistic Regression model coefficients scaled by this patient\'s transformed biomarkers.'
                    f'</p>'
                    f'{trend_content}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — MODEL COMPARISON & COLLEGE DEMO                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="section-header"><span>🏆</span> BENCHMARK PERFORMANCE & VALIDATION</div>', unsafe_allow_html=True)
    st.caption("Four machine learning classifiers tuned with Stratified 5-Fold Cross-Validation. Evaluated on unseen test set (N = 184).")

    metrics_csv = Path("reports/model_comparison_metrics.csv")
    if metrics_csv.exists():
        mdf = pd.read_csv(metrics_csv)
        best_row = mdf.sort_values(by=["Recall", "F1-Score"], ascending=[False, False]).iloc[0]

        # Summary Metric Cells
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-cell" style="border-color: rgba(255, 30, 68, 0.45); background: rgba(255, 30, 68, 0.12) !important;">
                    <div class="cell-label" style="color: #FF1E44;">CHAMPION MODEL</div>
                    <div class="cell-value" style="color: #FF1E44; font-size: 1.05rem;">{best_row['Model']}</div>
                </div>
                <div class="summary-cell">
                    <div class="cell-label">ACCURACY</div>
                    <div class="cell-value">{best_row['Accuracy']:.1%}</div>
                </div>
                <div class="summary-cell" style="border-color: rgba(0, 230, 118, 0.35); background: rgba(0, 230, 118, 0.08) !important;">
                    <div class="cell-label" style="color: #00E676;">RECALL (SENSITIVITY)</div>
                    <div class="cell-value" style="color: #00E676;">{best_row['Recall']:.1%}</div>
                </div>
                <div class="summary-cell">
                    <div class="cell-label">F1-SCORE</div>
                    <div class="cell-value">{best_row['F1-Score']:.3f}</div>
                </div>
                <div class="summary-cell">
                    <div class="cell-label">ROC-AUC</div>
                    <div class="cell-value">{best_row['ROC-AUC']:.3f}</div>
                </div>
                <div class="summary-cell" style="border-color: rgba(255, 179, 0, 0.35); background: rgba(255, 179, 0, 0.08) !important;">
                    <div class="cell-label" style="color: #FFB300;">MISSED CASES</div>
                    <div class="cell-value" style="color: #FFB300;">{int(best_row['False Negatives'])}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Detailed Metric Comparison Table")
        disp_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "CV F1 (Mean)", "False Negatives"]
        st.dataframe(
            mdf[disp_cols].style.format({
                "Accuracy": "{:.3f}", "Precision": "{:.3f}",
                "Recall": "{:.3f}", "F1-Score": "{:.3f}",
                "ROC-AUC": "{:.3f}", "CV F1 (Mean)": "{:.3f}",
            }).highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                color="rgba(48, 209, 88, 0.25)",
            ).highlight_min(
                subset=["False Negatives"],
                color="rgba(48, 209, 88, 0.25)",
            ),
            width="stretch",
            hide_index=True,
        )

        st.success(
            f"🎯 **Model Selection Rationale:** **{best_row['Model']}** won the benchmark evaluation by achieving "
            f"the highest Recall ({best_row['Recall']:.3f}) tied with Random Forest, highest overall F1-Score ({best_row['F1-Score']:.3f}), "
            f"and lowest False Negatives ({int(best_row['False Negatives'])} missed cases out of 102 diseased patients), "
            f"while providing full linear transparency for clinical interpretability."
        )
    else:
        st.info("Run `python src/train.py` to generate the benchmark comparison matrix.")

    st.markdown("---")
    st.markdown('<div class="section-header"><span>📊</span> DIAGNOSTIC VISUALIZATIONS</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    roc_img = Path("reports/model_roc_curves.png")
    cm_img = Path("reports/best_model_confusion_matrix.png")
    fi_img = Path("reports/feature_importance.png")

    with c1:
        if roc_img.exists():
            st.image(str(roc_img), caption="Multi-Model ROC Curves Overlay (Test Set)", width="stretch")
        else:
            st.warning("ROC curves plot not found.")

    with c2:
        if cm_img.exists():
            st.image(str(cm_img), caption="Normalized Confusion Matrix (Winning Pipeline)", width="stretch")
        else:
            st.warning("Confusion matrix plot not found.")

    if fi_img.exists():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span>🌲</span> TOP CLINICAL PREDICTIVE MARKERS</div>', unsafe_allow_html=True)
        st.image(str(fi_img), caption="Top Clinical Predictive Markers in the Pipeline", width="stretch")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — DATASET & PROJECT TEAM                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<div class="section-header"><span>ℹ️</span> PROJECT SPECIFICATION & TEAM</div>', unsafe_allow_html=True)

    info_left, info_right = st.columns(2, gap="large")

    with info_left:
        st.markdown(
            '<div class="glass-card">'
            '<h4 style="margin-top:0; color:#FFFFFF;">🗂️ Dataset Architecture</h4>'
            '<ul style="color:#AEAEB2; line-height:1.7; font-size:0.92rem; padding-left:1.2rem;">'
            '<li><strong>Dataset Source:</strong> Kaggle Heart Failure Prediction Dataset by <em>fedesoriano</em></li>'
            '<li><strong>Total Cohort:</strong> 918 patients synthesized across 5 clinical registries (Cleveland, Hungarian, Switzerland, Long Beach VA, Statlog)</li>'
            '<li><strong>Features:</strong> 11 clinical indicators + 1 binary target (<code>HeartDisease</code>)</li>'
            '<li><strong>Class Distribution:</strong> 508 Positive (55.3%) vs. 410 Negative (44.7%) — balanced cohort</li>'
            '</ul>'
            '<h4 style="margin-top:1.4rem; color:#FFFFFF;">🛠️ Data Quality Remediation</h4>'
            '<ul style="color:#AEAEB2; line-height:1.7; font-size:0.92rem; padding-left:1.2rem;">'
            '<li><strong>Cholesterol Zeros (172 records):</strong> Biologically impossible serum values converted to <code>NaN</code>, imputed with training median inside the Pipeline, and flagged via <code>Cholesterol_missing</code>.</li>'
            '<li><strong>RestingBP Zero (1 record):</strong> Handled safely via training median imputation.</li>'
            '<li><strong>Leakage Prevention:</strong> Scalers and encoders fit exclusively on training folds via scikit-learn <code>ColumnTransformer</code>.</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True,
        )

    with info_right:
        st.markdown(
            '<div class="glass-card">'
            '<h4 style="margin-top:0; color:#FFFFFF;">👥 Project Team Members</h4>'
            '<div class="team-grid" style="margin-top: 1rem; margin-bottom: 1.4rem;">'
            '<div class="team-member">'
            '<div class="avatar">AJ</div>'
            '<div class="team-info">'
            '<div class="name">Adhithyan JS</div>'
            '<div class="roll">Roll No. 7 · ML Tuning</div>'
            '</div>'
            '</div>'
            '<div class="team-member">'
            '<div class="avatar">EA</div>'
            '<div class="team-info">'
            '<div class="name">Evin Saj Abraham</div>'
            '<div class="roll">Roll No. 29 · Architecture</div>'
            '</div>'
            '</div>'
            '<div class="team-member">'
            '<div class="avatar">FH</div>'
            '<div class="team-info">'
            '<div class="name">Farhana H</div>'
            '<div class="roll">Roll No. 30 · EDA & Reporting</div>'
            '</div>'
            '</div>'
            '<div class="team-member">'
            '<div class="avatar">RK</div>'
            '<div class="team-info">'
            '<div class="name">Ridhin Krishna M</div>'
            '<div class="roll">Roll No. 53 · UI/UX & API</div>'
            '</div>'
            '</div>'
            '</div>'
            '<h4 style="margin-top:0; color:#FFFFFF;">⚠️ Clinical Limitations</h4>'
            '<ul style="color:#AEAEB2; line-height:1.7; font-size:0.92rem; padding-left:1.2rem; margin-bottom:0;">'
            '<li><strong>Cohort Size:</strong> 918 observations is an academic exploratory sample.</li>'
            '<li><strong>Missing Modern Biomarkers:</strong> Lacks Body Mass Index (BMI), smoking pack-years, and hs-Troponin. <code>FastingBS</code> acts as a binary glucose surrogate.</li>'
            '<li><strong>Decision-Support Only:</strong> Designed strictly as an adjunctive triage aid.</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True,
        )
