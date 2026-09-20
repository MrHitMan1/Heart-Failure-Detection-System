"""
Streamlit Web Application: Heart Failure Detection System
Decision-support machine learning interface for clinicians and college project presentation.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

# ── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense AI — Heart Failure Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Import a clean sans-serif Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Top Header Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d1f3d 50%, #1a1f2e 100%);
        border: 1px solid rgba(231,76,60,0.25);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.2rem;
        text-align: center;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #e74c3c 0%, #f39c12 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin: 0;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: rgba(231,76,60,0.08);
        border: 1px solid rgba(231,76,60,0.30);
        border-left: 4px solid #e74c3c;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.5rem;
        font-size: 0.88rem;
        color: #fca5a5;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid rgba(231,76,60,0.3);
        letter-spacing: 0.3px;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(26, 31, 46, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Risk Result Cards ── */
    .risk-card {
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .risk-card:hover { transform: translateY(-2px); }

    .risk-card h3 {
        margin: 0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.85;
    }
    .risk-card h1 {
        margin: 0.4rem 0 0.25rem 0;
        font-size: 2.4rem;
        font-weight: 900;
        letter-spacing: -1px;
    }
    .risk-card p {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* ── Probability Donut (CSS only) ── */
    .prob-ring {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        margin: 0 auto 0.6rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.9rem;
        font-weight: 900;
        color: #fafafa;
    }

    /* ── Contribution Bar ── */
    .contrib-bar-wrap {
        margin-bottom: 0.55rem;
    }
    .contrib-label {
        font-size: 0.82rem;
        color: #cbd5e1;
        margin-bottom: 2px;
        font-weight: 500;
    }
    .contrib-track {
        background: rgba(255,255,255,0.06);
        border-radius: 6px;
        height: 22px;
        overflow: hidden;
        position: relative;
    }
    .contrib-fill {
        height: 100%;
        border-radius: 6px;
        display: flex;
        align-items: center;
        padding-left: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        color: white;
        min-width: 32px;
        transition: width 0.6s ease;
    }

    /* ── Metrics card row ── */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        margin-bottom: 1rem;
    }
    .metric-pill {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.9rem 0.7rem;
        text-align: center;
    }
    .metric-pill .label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 600;
    }
    .metric-pill .value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f1f5f9;
        margin-top: 2px;
    }

    /* ── Team card ── */
    .team-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.8rem;
    }
    .team-member {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .team-member .name {
        font-size: 1rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .team-member .roll {
        font-size: 0.82rem;
        color: #94a3b8;
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Better form buttons ── */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        border: none !important;
        letter-spacing: 0.3px;
        transition: all 0.2s ease !important;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(231,76,60,0.35) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Predictor cache ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_predictor():
    from src.predict import HeartFailurePredictor
    try:
        return HeartFailurePredictor()
    except Exception:
        return None

# ── Presets ──────────────────────────────────────────────────────────────────────
PRESETS = {
    "— Select a preset —": None,
    "🟢  Low-Risk Patient (Healthy Profile)": {
        "Age": 38, "Sex": "F", "ChestPainType": "ATA", "RestingBP": 115,
        "Cholesterol": 185, "FastingBS": 0, "RestingECG": "Normal",
        "MaxHR": 172, "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Up",
    },
    "🔴  High-Risk Patient (Cardiac Distress)": {
        "Age": 62, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 155,
        "Cholesterol": 0, "FastingBS": 1, "RestingECG": "LVH",
        "MaxHR": 110, "ExerciseAngina": "Y", "Oldpeak": 2.5, "ST_Slope": "Flat",
    },
}

# ── Hero Banner ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🫀 CardioSense AI</div>
        <div class="hero-subtitle">HEART FAILURE DETECTION SYSTEM &nbsp;·&nbsp; ML-POWERED CLINICAL DECISION SUPPORT</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong>
        This is an academic research prototype for <em>clinical decision-support only</em>.
        It is <strong>not</strong> a certified medical device and does <strong>not</strong> replace
        professional medical consultation, clinical judgment, or formal diagnostic evaluation.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🩺  Risk Assessment",
    "📊  Model Comparison",
    "ℹ️  About & Team",
])

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — PATIENT RISK ASSESSMENT                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<div class="section-header">Patient Clinical Profile</div>', unsafe_allow_html=True)
    st.caption("Enter physiological measurements or load a demo preset, then press **Predict**.")

    preset_choice = st.selectbox("⚡ Quick Preset", list(PRESETS.keys()), index=0)
    preset_data = PRESETS[preset_choice]
    def _v(key, default):
        return preset_data[key] if preset_data and key in preset_data else default

    with st.form("patient_form"):
        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            st.markdown("##### 👤 Demographics & Symptoms")
            age = st.slider("Age (years)", 18, 90, int(_v("Age", 54)))

            sex_choice = st.selectbox("Biological Sex", ["Male", "Female"],
                                      index=0 if _v("Sex", "M") == "M" else 1)
            sex = "M" if sex_choice == "Male" else "F"

            cp_map = {"Typical Angina (TA)": "TA", "Atypical Angina (ATA)": "ATA",
                      "Non-Anginal Pain (NAP)": "NAP", "Asymptomatic (ASY)": "ASY"}
            inv_cp = {v: k for k, v in cp_map.items()}
            cp_choice = st.selectbox("Chest Pain Type", list(cp_map.keys()),
                                     index=list(cp_map.keys()).index(
                                         inv_cp.get(_v("ChestPainType", "ASY"), "Asymptomatic (ASY)")))
            chest_pain_type = cp_map[cp_choice]

            ex_choice = st.selectbox("Exercise-Induced Angina", ["No", "Yes"],
                                     index=1 if _v("ExerciseAngina", "N") == "Y" else 0)
            exercise_angina = "Y" if ex_choice == "Yes" else "N"

            st.markdown("##### 🧪 Metabolic")
            fasting_bs = st.checkbox("Fasting Blood Sugar > 120 mg/dl",
                                     value=bool(_v("FastingBS", 0)))
            fasting_bs_val = 1 if fasting_bs else 0

        with col_r:
            st.markdown("##### 💓 Hemodynamics & ECG")
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 0, 240,
                                         int(_v("RestingBP", 130)),
                                         help="Enter 0 if unknown – will be auto-imputed.")
            cholesterol = st.number_input("Serum Cholesterol (mg/dl)", 0, 700,
                                          int(_v("Cholesterol", 220)),
                                          help="Enter 0 if unmeasured – handled automatically.")

            ecg_map = {"Normal": "Normal", "ST-T Wave Abnormality": "ST",
                       "LV Hypertrophy (LVH)": "LVH"}
            inv_ecg = {v: k for k, v in ecg_map.items()}
            ecg_choice = st.selectbox("Resting ECG", list(ecg_map.keys()),
                                      index=list(ecg_map.keys()).index(
                                          inv_ecg.get(_v("RestingECG", "Normal"), "Normal")))
            resting_ecg = ecg_map[ecg_choice]

            max_hr = st.slider("Max Heart Rate (bpm)", 50, 230, int(_v("MaxHR", 140)))
            oldpeak = st.number_input("ST Depression — Oldpeak (mm)", -3.0, 7.0,
                                      float(_v("Oldpeak", 0.0)), step=0.1, format="%.1f")

            slope_map = {"Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"}
            inv_slope = {v: k for k, v in slope_map.items()}
            slope_choice = st.selectbox("Peak ST Slope", list(slope_map.keys()),
                                        index=list(slope_map.keys()).index(
                                            inv_slope.get(_v("ST_Slope", "Flat"), "Flat")))
            st_slope = slope_map[slope_choice]

        submit = st.form_submit_button("🫀  Predict Cardiac Risk", use_container_width=True)

    # ── Results ──
    if submit:
        patient = {
            "Age": age, "Sex": sex, "ChestPainType": chest_pain_type,
            "RestingBP": resting_bp, "Cholesterol": cholesterol,
            "FastingBS": fasting_bs_val, "RestingECG": resting_ecg,
            "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak, "ST_Slope": st_slope,
        }
        predictor = get_predictor()
        if predictor is None:
            st.error("❌  Model not found. Run `python src/train.py` first.")
        else:
            with st.spinner("Analyzing clinical markers…"):
                res = predictor.predict(patient)

            prob = res["probability"]
            pct  = prob * 100
            lvl  = res["risk_level"]
            diag = res["diagnosis"]

            # Colour palette per risk tier
            COLORS = {
                "Low":      {"bg": "#064e3b", "border": "#10b981", "accent": "#34d399"},
                "Moderate": {"bg": "#78350f", "border": "#f59e0b", "accent": "#fbbf24"},
                "High":     {"bg": "#7f1d1d", "border": "#ef4444", "accent": "#f87171"},
            }
            c = COLORS[lvl]

            st.markdown("---")
            st.markdown('<div class="section-header">Risk Assessment Results</div>',
                        unsafe_allow_html=True)

            rc1, rc2, rc3 = st.columns([1.3, 1.3, 2], gap="medium")

            with rc1:
                st.markdown(
                    f"""
                    <div class="risk-card" style="background:{c['bg']}; border:2px solid {c['border']};">
                        <h3 style="color:{c['accent']};">Risk Level</h3>
                        <h1 style="color:{c['accent']};">{lvl.upper()}</h1>
                        <p style="color:#e2e8f0;">{diag}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with rc2:
                # CSS conic-gradient ring
                filled_deg = int(prob * 360)
                st.markdown(
                    f"""
                    <div class="risk-card" style="background:rgba(255,255,255,0.03);
                                border:1px solid rgba(255,255,255,0.08);">
                        <h3 style="color:#94a3b8;">Probability</h3>
                        <div class="prob-ring"
                             style="background: conic-gradient({c['accent']} {filled_deg}deg,
                                    rgba(255,255,255,0.06) {filled_deg}deg);">
                            <span style="background:#0e1117; width:100px; height:100px;
                                         border-radius:50%; display:flex; align-items:center;
                                         justify-content:center; color:{c['accent']};">
                                {pct:.1f}%
                            </span>
                        </div>
                        <p style="color:#94a3b8; font-size:0.78rem;">
                            Low &lt;30% · Moderate 30-60% · High &gt;60%
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with rc3:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <h4 style="color:#f1f5f9; margin-top:0;">📋 Clinical Interpretation</h4>
                        <p style="color:#cbd5e1; line-height:1.6;">{res['clinical_advice']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if cholesterol == 0:
                    st.warning("ℹ️ Cholesterol = 0 → auto-imputed with training median; "
                               "missingness indicator flag set to 1.")

            # ── Contributing Factors (horizontal bars) ──
            contribs = res.get("contributions", {})
            if contribs:
                st.markdown("---")
                st.markdown('<div class="section-header">🔍 Top Contributing Risk Factors</div>',
                            unsafe_allow_html=True)
                max_abs = max(abs(v) for v in contribs.values()) or 1
                bar_colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6"]
                for idx, (feat, val) in enumerate(contribs.items()):
                    width = int(abs(val) / max_abs * 100)
                    color = bar_colors[idx % len(bar_colors)]
                    direction = "⬆ risk" if val > 0 else "⬇ protective"
                    st.markdown(
                        f"""
                        <div class="contrib-bar-wrap">
                            <div class="contrib-label">{feat}  <span style="color:{color};
                                font-weight:700;">{val:+.3f}</span>
                                <span style="font-size:0.72rem; color:#64748b;"> ({direction})</span>
                            </div>
                            <div class="contrib-track">
                                <div class="contrib-fill"
                                     style="width:{max(width, 5)}%; background:{color};">
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — MODEL COMPARISON & DEMO                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="section-header">🏆 Model Benchmark Comparison</div>',
                unsafe_allow_html=True)
    st.caption("Four classifiers were tuned with 5-Fold Stratified CV. "
               "**Recall** and **F1-Score** are prioritized (minimising missed cardiac cases).")

    metrics_csv = Path("reports/model_comparison_metrics.csv")
    if metrics_csv.exists():
        mdf = pd.read_csv(metrics_csv)

        # ── Summary metric pills for the winning model ──
        best = mdf.sort_values(["Recall", "F1-Score"], ascending=False).iloc[0]
        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-pill">
                    <div class="label">Winner</div>
                    <div class="value" style="color:#e74c3c; font-size:1.05rem;">{best['Model']}</div>
                </div>
                <div class="metric-pill">
                    <div class="label">Accuracy</div>
                    <div class="value">{best['Accuracy']:.1%}</div>
                </div>
                <div class="metric-pill">
                    <div class="label">Recall</div>
                    <div class="value" style="color:#34d399;">{best['Recall']:.1%}</div>
                </div>
                <div class="metric-pill">
                    <div class="label">F1-Score</div>
                    <div class="value">{best['F1-Score']:.3f}</div>
                </div>
                <div class="metric-pill">
                    <div class="label">ROC-AUC</div>
                    <div class="value">{best['ROC-AUC']:.3f}</div>
                </div>
                <div class="metric-pill">
                    <div class="label">Missed Cases</div>
                    <div class="value" style="color:#fbbf24;">{int(best['False Negatives'])}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Full Benchmark Table (Test Set N = 184)")
        display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score",
                        "ROC-AUC", "CV F1 (Mean)", "False Negatives"]
        st.dataframe(
            mdf[display_cols].style.format({
                "Accuracy": "{:.3f}", "Precision": "{:.3f}",
                "Recall": "{:.3f}", "F1-Score": "{:.3f}",
                "ROC-AUC": "{:.3f}", "CV F1 (Mean)": "{:.3f}",
            }).highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                color="rgba(16,185,129,0.25)",
            ).highlight_min(subset=["False Negatives"], color="rgba(16,185,129,0.25)"),
            use_container_width=True, hide_index=True,
        )

        st.success(
            f"**Selected: {best['Model']}** — Highest Recall ({best['Recall']:.3f}) "
            f"and F1-Score ({best['F1-Score']:.3f}) with only "
            f"{int(best['False Negatives'])} false negatives."
        )
    else:
        st.info("Run `python src/train.py` to generate benchmark metrics.")

    # ── Diagnostic Plots ──
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Diagnostic Visualizations</div>',
                unsafe_allow_html=True)

    roc_path = Path("reports/model_roc_curves.png")
    cm_path  = Path("reports/best_model_confusion_matrix.png")
    fi_path  = Path("reports/feature_importance.png")

    pc1, pc2 = st.columns(2, gap="medium")
    with pc1:
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Curves — All Models",
                     use_container_width=True)
        else:
            st.warning("ROC curve plot not found.")
    with pc2:
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix — Best Model",
                     use_container_width=True)
        else:
            st.warning("Confusion matrix plot not found.")

    if fi_path.exists():
        st.markdown("##### 🌲 Feature Importance")
        st.image(str(fi_path), caption="Top Predictive Clinical Factors",
                 use_container_width=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — ABOUT & TEAM                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<div class="section-header">📖 Project Overview</div>',
                unsafe_allow_html=True)

    about_l, about_r = st.columns(2, gap="large")

    with about_l:
        st.markdown(
            """
            ##### 🗂️ Dataset
            - **Source:** [Kaggle — Heart Failure Prediction by fedesoriano](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
            - **Records:** 918 patients (Cleveland, Hungarian, Switzerland, Long Beach VA, Statlog)
            - **Features:** 11 clinical features + 1 binary target (`HeartDisease`)
            - **Class Balance:** 508 positive (55.3%) vs 410 normal (44.7%) — balanced, no SMOTE needed

            ##### 🛠️ Data Quality Fixes
            1. **Cholesterol = 0** (172 records): imputed with training-set median + `Cholesterol_missing` flag
            2. **RestingBP = 0** (1 record): imputed with training-set median
            3. **No data leakage**: all transforms fit only on training folds inside the `Pipeline`
            """
        )

    with about_r:
        st.markdown("##### 👥 Project Team")
        st.markdown(
            """
            <div class="team-grid">
                <div class="team-member">
                    <div class="name">Adhithyan JS</div>
                    <div class="roll">Roll No. 7</div>
                </div>
                <div class="team-member">
                    <div class="name">Evin Saj Abraham</div>
                    <div class="roll">Roll No. 29</div>
                </div>
                <div class="team-member">
                    <div class="name">Farhana H</div>
                    <div class="roll">Roll No. 30</div>
                </div>
                <div class="team-member">
                    <div class="name">Ridhin Krishna M</div>
                    <div class="roll">Roll No. 53</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            ##### ⚠️ Limitations
            - **918 records** — modest for clinical generalisation
            - **No BMI, smoking, HbA1c, or Troponin** — `FastingBS` used as glucose proxy
            - **Academic prototype** — not for standalone diagnostic use
            """
        )
