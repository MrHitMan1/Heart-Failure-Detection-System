"""
Streamlit Web Application: Heart Failure Detection System
Apple Human Interface Guidelines (HIG) Edition
Clinical decision-support machine learning interface for clinicians and college project presentation.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

# ── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense — Apple Health Edition",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Apple HIG Design System CSS ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Apple SF Pro Typography Stack ── */
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "SF Pro",
                     "Helvetica Neue", Helvetica, Arial, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        color: #F5F5F7;
    }

    /* ── App Container Adjustments ── */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1200px !important;
        position: relative !important;
        z-index: 1 !important;
    }

    /* ── Apple Glassmorphic Card (Translucent Acrylic / Vibrancy) ── */
    .apple-card {
        background: rgba(20, 20, 26, 0.50) !important;
        backdrop-filter: blur(30px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(30px) saturate(190%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.18),
                    0 16px 40px rgba(0, 0, 0, 0.45) !important;
        border-radius: 22px !important;
        padding: 1.8rem 2rem;
        margin-bottom: 1.4rem;
        transition: transform 0.25s cubic-bezier(0.2, 0.8, 0.2, 1),
                    border-color 0.25s ease,
                    box-shadow 0.25s ease;
    }
    .apple-card:hover {
        border-color: rgba(255, 255, 255, 0.22) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.25),
                    0 20px 48px rgba(0, 0, 0, 0.55) !important;
    }

    /* ── Apple Health Hero Header ── */
    .apple-hero {
        background: linear-gradient(180deg, rgba(28, 28, 36, 0.54) 0%, rgba(14, 14, 18, 0.60) 100%) !important;
        backdrop-filter: blur(35px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(35px) saturate(190%) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2),
                    0 24px 50px rgba(0, 0, 0, 0.55) !important;
        border-radius: 26px !important;
        padding: 2.2rem 2.6rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .apple-hero::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -20%;
        width: 140%;
        height: 200%;
        background: radial-gradient(circle at 30% 20%, rgba(255, 45, 85, 0.14) 0%, transparent 60%);
        pointer-events: none;
    }
    .apple-pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 45, 85, 0.16);
        border: 1px solid rgba(255, 45, 85, 0.4);
        border-radius: 999px;
        padding: 4px 13px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #FF375F;
        margin-bottom: 0.7rem;
    }
    .apple-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #FFFFFF;
        margin: 0 0 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .apple-title .heart-emoji {
        font-size: 2.6rem;
        display: inline-block;
        filter: drop-shadow(0 4px 12px rgba(255, 45, 85, 0.4));
    }
    .apple-title .title-text {
        color: #FFFFFF;
        background: linear-gradient(180deg, #FFFFFF 40%, #D1D1D6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .apple-subtitle {
        font-size: 1rem;
        color: #98989D;
        font-weight: 400;
        margin: 0;
        letter-spacing: -0.01em;
        line-height: 1.5;
    }

    /* ── Apple Medical Notice ── */
    .apple-notice {
        background: rgba(255, 69, 58, 0.1) !important;
        backdrop-filter: blur(35px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(35px) saturate(190%) !important;
        border: 1px solid rgba(255, 69, 58, 0.28) !important;
        border-radius: 16px !important;
        padding: 0.95rem 1.4rem;
        margin-bottom: 1.6rem;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.88rem;
        color: #FF8A8A;
        box-shadow: 0 8px 24px rgba(255, 69, 58, 0.1);
    }
    .apple-notice strong {
        color: #FF453A;
        font-weight: 600;
    }

    /* ── Apple Segmented Control (Tabs) ── */
    div[data-testid="stTabs"] {
        background: transparent !important;
        margin-bottom: 1.8rem !important;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-list"],
    .stTabs [data-baseweb="tab-list"],
    [data-baseweb="tab-list"] {
        background: rgba(118, 118, 128, 0.24) !important;
        backdrop-filter: blur(45px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(45px) saturate(200%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 16px !important;
        padding: 5px !important;
        gap: 6px !important;
        border-bottom: none !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.15),
                    0 10px 28px rgba(0, 0, 0, 0.4) !important;
        width: fit-content !important;
        max-width: 100% !important;
    }

    /* Remove ALL default underlines, borders, and red highlight lines from Streamlit */
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"],
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"],
    [data-baseweb="tab-highlight"],
    [data-baseweb="tab-border"] {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        width: 0 !important;
        opacity: 0 !important;
        visibility: hidden !important;
        border: none !important;
        background: transparent !important;
        position: absolute !important;
        pointer-events: none !important;
    }

    /* Base Tab Buttons */
    div[data-testid="stTabs"] button[data-baseweb="tab"],
    .stTabs button[data-baseweb="tab"],
    button[data-baseweb="tab"] {
        background: transparent !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        border-bottom: none !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
        border-radius: 11px !important;
        padding: 0.55rem 1.4rem !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        color: #8E8E93 !important;
        outline: none !important;
        box-shadow: none !important;
        transition: all 0.22s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover,
    .stTabs button[data-baseweb="tab"]:hover,
    button[data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
        background: rgba(255, 255, 255, 0.08) !important;
        background-color: rgba(255, 255, 255, 0.08) !important;
        border-bottom: none !important;
    }

    /* Selected Tab: Apple Floating Frosted Capsule (No Red Underline!) */
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"],
    .stTabs button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] {
        background: rgba(255, 255, 255, 0.18) !important;
        background-color: rgba(255, 255, 255, 0.18) !important;
        backdrop-filter: blur(25px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(25px) saturate(180%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 11px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
    }

    /* Strip any child border elements */
    div[data-testid="stTabs"] button[data-baseweb="tab"] *,
    button[data-baseweb="tab"] * {
        border-bottom: none !important;
        border-bottom-color: transparent !important;
        border-bottom-width: 0 !important;
        text-decoration: none !important;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"]::after,
    button[data-baseweb="tab"]::after,
    div[data-testid="stTabs"] button[data-baseweb="tab"]::before,
    button[data-baseweb="tab"]::before {
        display: none !important;
        content: none !important;
        border: none !important;
    }

    /* ── Form Container Frosted Glass ── */
    div[data-testid="stForm"] {
        background: rgba(18, 18, 24, 0.48) !important;
        backdrop-filter: blur(30px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(30px) saturate(190%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 24px !important;
        padding: 2.2rem !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.18),
                    0 20px 50px rgba(0, 0, 0, 0.5) !important;
        margin-bottom: 1.5rem !important;
    }

    /* Form Column Cards (Apple Inset Grouped style) */
    div[data-testid="stForm"] div[data-testid="stColumn"] {
        background: rgba(255, 255, 255, 0.035) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 18px !important;
        padding: 1.4rem 1.6rem !important;
    }

    /* ── Streamlit Form Input Styling (Apple Elevated Surfaces) ── */
    /* Labels */
    div[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] {
        color: #F5F5F7 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        margin-bottom: 0.4rem !important;
        letter-spacing: -0.01em !important;
    }

    /* ── Number Input & Text Input Boxes ── */
    div[data-testid="stNumberInputContainer"],
    div[data-testid="stTextInput"] [data-baseweb="input"] > div {
        background-color: #22222a !important;
        background: #22222a !important;
        border: 1.5px solid rgba(255, 255, 255, 0.28) !important;
        border-radius: 12px !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.35), 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        min-height: 44px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease !important;
    }

    div[data-testid="stNumberInputContainer"]:hover {
        background-color: #282832 !important;
        background: #282832 !important;
        border-color: rgba(255, 255, 255, 0.45) !important;
    }

    div[data-testid="stNumberInputContainer"]:focus-within {
        background-color: #262630 !important;
        background: #262630 !important;
        border-color: #FF2D55 !important;
        box-shadow: 0 0 0 3px rgba(255, 45, 85, 0.28), inset 0 2px 4px rgba(0, 0, 0, 0.35) !important;
    }

    div[data-testid="stNumberInputContainer"] input {
        color: #FFFFFF !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding-left: 12px !important;
        background: transparent !important;
        border: none !important;
    }

    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        margin: 4px 3px !important;
        height: 32px !important;
        width: 32px !important;
        color: #E5E5EA !important;
        transition: all 0.15s ease !important;
    }
    button[data-testid="stNumberInputStepDown"]:hover,
    button[data-testid="stNumberInputStepUp"]:hover {
        background: rgba(255, 255, 255, 0.20) !important;
        color: #FFFFFF !important;
    }

    /* ── Dropdown Selectboxes (Streamlit 1.64+ React-Aria & Fallbacks) ── */
    div[data-testid="stSelectbox"] .react-aria-Group,
    div[data-testid="stSelectbox"] div[role="group"],
    div[data-testid="stSelectbox"] div[class*="e1fp86qc0"],
    div.stSelectbox .react-aria-Group,
    div.stSelectbox div[role="group"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-baseweb="select"] > div {
        background-color: #22222a !important;
        background: #22222a !important;
        border: 1.5px solid rgba(255, 255, 255, 0.28) !important;
        border-radius: 12px !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.35), 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        min-height: 44px !important;
        height: 44px !important;
        display: flex !important;
        align-items: center !important;
        padding: 0 6px !important;
        box-sizing: border-box !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease !important;
    }

    /* Hover State for Selectboxes */
    div[data-testid="stSelectbox"] .react-aria-Group:hover,
    div[data-testid="stSelectbox"] div[role="group"]:hover,
    div[data-testid="stSelectbox"] div[class*="e1fp86qc0"]:hover,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
    div[data-baseweb="select"] > div:hover {
        background-color: #282832 !important;
        background: #282832 !important;
        border-color: rgba(255, 255, 255, 0.45) !important;
    }

    /* Focus State: Apple Halo Glow (React-Aria & BaseWeb) */
    div[data-testid="stSelectbox"] .react-aria-Group[data-focus-within],
    div[data-testid="stSelectbox"] .react-aria-Group:focus-within,
    div[data-testid="stSelectbox"] div[class*="e1fp86qc0"][data-focus-within],
    div[data-testid="stSelectbox"] div[class*="e1fp86qc0"]:focus-within,
    div[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div,
    div[data-baseweb="select"]:focus-within > div {
        background-color: #262630 !important;
        background: #262630 !important;
        border-color: #FF2D55 !important;
        box-shadow: 0 0 0 3px rgba(255, 45, 85, 0.28), inset 0 2px 4px rgba(0, 0, 0, 0.35) !important;
    }

    /* Selectbox input text formatting */
    div[data-testid="stSelectbox"] .react-aria-Input,
    div[data-testid="stSelectbox"] input,
    div[data-testid="stSelectbox"] [class*="e1fp86qc1"] {
        background: transparent !important;
        background-color: transparent !important;
        color: #FFFFFF !important;
        font-size: 0.98rem !important;
        font-weight: 500 !important;
        border: none !important;
        box-shadow: none !important;
        padding-left: 8px !important;
    }

    /* Selectbox dropdown chevron button */
    div[data-testid="stSelectbox"] button,
    div[data-testid="stSelectbox"] [class*="e1fp86qc2"] {
        background: transparent !important;
        border: none !important;
        color: #D1D1D6 !important;
    }
    div[data-testid="stSelectbox"] svg {
        fill: #D1D1D6 !important;
        color: #D1D1D6 !important;
    }

    /* Dropdown Virtual Popup List (stSelectboxVirtualDropdown & BaseWeb Popovers) */
    div[data-testid="stSelectboxVirtualDropdown"],
    div[class*="e1fp86qc4"],
    div[data-baseweb="popover"] > div,
    ul[data-baseweb="menu"] {
        background-color: #22222a !important;
        background: #22222a !important;
        border: 1.5px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 14px !important;
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.75) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        padding: 6px !important;
    }

    /* Dropdown list items */
    div[data-testid="stSelectboxVirtualDropdown"] li,
    div[data-testid="stSelectboxVirtualDropdown"] [role="option"],
    div[class*="e1fp86qc7"],
    li[data-baseweb="menu-item"] {
        background: transparent !important;
        color: #FFFFFF !important;
        font-size: 0.94rem !important;
        border-radius: 8px !important;
        margin: 2px 0 !important;
        padding: 8px 12px !important;
        transition: background-color 0.15s ease !important;
    }

    div[data-testid="stSelectboxVirtualDropdown"] li[data-hovered],
    div[data-testid="stSelectboxVirtualDropdown"] li[data-focused],
    div[data-testid="stSelectboxVirtualDropdown"] li[aria-selected="true"],
    div[class*="e1fp86qc7"][data-hovered],
    div[class*="e1fp86qc7"][data-focused],
    li[data-baseweb="menu-item"]:hover,
    li[data-baseweb="menu-item"][aria-selected="true"] {
        background-color: rgba(255, 45, 85, 0.3) !important;
        background: rgba(255, 45, 85, 0.3) !important;
        color: #FFFFFF !important;
    }

    /* Completely hide 'Press Enter to submit form' helper prompt */
    div[data-testid="InputInstructions"],
    [data-testid="InputInstructions"],
    [data-testid="stInputInstructions"],
    small[data-testid="InputInstructions"] {
        display: none !important;
        opacity: 0 !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        font-size: 0 !important;
        line-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Slider Container Distinction */
    div[data-testid="stSlider"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 14px !important;
        padding: 0.65rem 1rem 0.4rem 1rem !important;
        margin-bottom: 0.8rem !important;
    }

    /* Checkbox Container Distinction */
    div[data-testid="stCheckbox"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
    }

    /* ── Section Title (Apple Style) ── */
    .apple-section-header {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #8E8E93;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Apple Health Widget Cards ── */
    .health-widget {
        background: rgba(18, 18, 24, 0.52) !important;
        backdrop-filter: blur(30px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(30px) saturate(190%) !important;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 22px;
        padding: 1.6rem 1.4rem;
        text-align: center;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.16),
                    0 16px 36px rgba(0, 0, 0, 0.45);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    .health-widget .widget-label {
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8E8E93;
        margin-bottom: 0.4rem;
    }
    .health-widget .widget-value {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    .health-widget .widget-caption {
        font-size: 0.85rem;
        color: #AEAEB2;
        margin: 0;
    }

    /* ── Apple Ring Activity Gauge ── */
    .ring-container {
        position: relative;
        width: 148px;
        height: 148px;
        margin: 0.4rem auto;
    }
    .ring-container svg {
        transform: rotate(-90deg);
    }
    .ring-center-content {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        pointer-events: none;
    }
    .ring-percent {
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .ring-subtext {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8E8E93;
        margin-top: 2px;
    }

    /* ── Apple Metric Trends (Factor Contributions) ── */
    .trend-row {
        background: rgba(26, 26, 34, 0.50) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.75rem 1.1rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        transition: background 0.2s ease, border-color 0.2s ease;
    }
    .trend-row:hover {
        background: rgba(36, 36, 44, 0.65) !important;
        border-color: rgba(255, 255, 255, 0.16);
    }
    .trend-name {
        font-size: 0.88rem;
        font-weight: 600;
        color: #F5F5F7;
        flex: 1;
    }
    .trend-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 999px;
    }
    .trend-bar-bg {
        width: 140px;
        height: 6px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        overflow: hidden;
        position: relative;
    }
    .trend-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.6s cubic-bezier(0.2, 0.8, 0.2, 1);
    }

    /* ── Apple System Button (Form Submit) ── */
    .stFormSubmitButton > button {
        background: linear-gradient(180deg, #FF375F 0%, #E00034 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        border-radius: 14px !important;
        padding: 0.8rem 2.2rem !important;
        border: none !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.3),
                    0 8px 24px rgba(255, 45, 85, 0.38) !important;
        letter-spacing: -0.01em !important;
        transition: all 0.2s cubic-bezier(0.2, 0.8, 0.2, 1) !important;
    }
    .stFormSubmitButton > button:hover {
        transform: scale(1.015) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.35),
                    0 12px 30px rgba(255, 45, 85, 0.5) !important;
    }
    .stFormSubmitButton > button:active {
        transform: scale(0.985) !important;
    }

    /* ── Summary Metric Pills Row (Tab 2) ── */
    .apple-summary-grid {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 10px;
        margin-bottom: 1.4rem;
    }
    .apple-summary-cell {
        background: rgba(20, 20, 26, 0.50) !important;
        backdrop-filter: blur(25px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(25px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        padding: 0.9rem 0.6rem;
        text-align: center;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.14),
                    0 8px 20px rgba(0, 0, 0, 0.3);
    }
    .apple-summary-cell .cell-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #8E8E93;
        margin-bottom: 4px;
    }
    .apple-summary-cell .cell-value {
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F5F5F7;
    }

    /* ── Apple DataFrame Container ── */
    div[data-testid="stDataFrame"] {
        background: rgba(18, 18, 24, 0.50) !important;
        backdrop-filter: blur(28px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(190%) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 12px !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.14),
                    0 16px 40px rgba(0, 0, 0, 0.45) !important;
        margin-bottom: 1.2rem !important;
    }

    /* ── Apple Alert Callouts (st.success, st.warning, st.info) ── */
    div[data-testid="stAlert"] {
        backdrop-filter: blur(25px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(25px) saturate(190%) !important;
        border-radius: 16px !important;
        border-width: 1px !important;
    }

    /* ── Apple Team Grid ── */
    .apple-team-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
    }
    .apple-team-member {
        background: rgba(24, 24, 32, 0.52) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px;
        padding: 1.2rem;
        display: flex;
        align-items: center;
        gap: 14px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .apple-team-member:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    .apple-avatar {
        width: 46px;
        height: 46px;
        border-radius: 50%;
        background: linear-gradient(135deg, #FF2D55 0%, #5E5CE6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        font-weight: 800;
        color: #FFFFFF;
        box-shadow: 0 4px 14px rgba(255, 45, 85, 0.35);
    }
    .apple-team-info .name {
        font-size: 0.98rem;
        font-weight: 700;
        color: #F5F5F7;
        margin: 0;
    }
    .apple-team-info .roll {
        font-size: 0.78rem;
        color: #8E8E93;
        margin-top: 2px;
    }

    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Interactive Apple Ambient Glow Background (Safe WebGL/Canvas) ──────────────
# Uses st.iframe() to isolate the script, resting in a fixed fullscreen plane
# behind all content. The mouse creates soft glowing attraction waves.
_APPLE_FLUID_BG = """
<style>
    html, body { margin:0; padding:0; overflow:hidden; background:transparent; }
    canvas { display:block; }
</style>
<canvas id="c"></canvas>
<script>
(function(){
    var f = window.frameElement;
    if (f) {
        f.style.cssText =
            'position:fixed!important;top:0!important;left:0!important;' +
            'width:100vw!important;height:100vh!important;z-index:0!important;' +
            'pointer-events:none!important;border:none!important;background:transparent!important;';
    }

    var cv = document.getElementById('c');
    var ctx = cv.getContext('2d');
    var W, H, mx = -9999, my = -9999, orbs = [];

    function resize(){ W = cv.width = window.innerWidth; H = cv.height = window.innerHeight; }
    window.addEventListener('resize', resize); resize();

    try {
        window.parent.document.addEventListener('mousemove', function(e){ mx=e.clientX; my=e.clientY; });
        window.parent.document.addEventListener('mouseleave', function(){ mx=-9999; my=-9999; });
    } catch(e){
        document.addEventListener('mousemove', function(e){ mx=e.clientX; my=e.clientY; });
    }

    /* Apple Health/VisionOS Palette: Rose, Violet, Cyan, Coral */
    var colors = [
        {r:255, g:45,  b:85},   /* Apple Heart Pink/Red */
        {r:94,  g:92,  b:230},  /* Apple Purple/Iris */
        {r:255, g:149, b:0},   /* Apple Coral/Orange */
        {r:10,  g:132, b:255}   /* Apple System Blue */
    ];

    for(var i=0; i<85; i++){
        var clr = colors[i % colors.length];
        orbs.push({
            x: Math.random()*2000,
            y: Math.random()*2000,
            baseRadius: Math.random()*3.5 + 1.8,
            radius: Math.random()*3.5 + 1.8,
            vx: (Math.random()-0.5)*0.35,
            vy: (Math.random()-0.5)*0.35,
            color: clr,
            alpha: 0.38 + Math.random()*0.38
        });
    }

    function render(){
        ctx.clearRect(0,0,W,H);

        for(var i=0; i<orbs.length; i++){
            var o = orbs[i];
            var dx = mx - o.x;
            var dy = my - o.y;
            var dist = Math.sqrt(dx*dx + dy*dy) || 1;

            if(dist < 280){
                var f = (280 - dist)/280 * 0.018;
                o.vx += dx/dist * f;
                o.vy += dy/dist * f;
                o.radius = o.baseRadius + (280 - dist)/280 * 3.0;
            } else {
                o.radius += (o.baseRadius - o.radius) * 0.05;
            }

            o.vx *= 0.992;
            o.vy *= 0.992;
            o.x += o.vx;
            o.y += o.vy;

            if(o.x < -20) o.x = W + 20;
            if(o.x > W + 20) o.x = -20;
            if(o.y < -20) o.y = H + 20;
            if(o.y > H + 20) o.y = -20;

            /* Soft glowing orb */
            ctx.beginPath();
            ctx.arc(o.x, o.y, o.radius, 0, Math.PI*2);
            ctx.fillStyle = 'rgba('+o.color.r+','+o.color.g+','+o.color.b+','+o.alpha+')';
            ctx.fill();
        }

        /* Ambient connection mesh */
        for(var i=0; i<orbs.length; i++){
            for(var j=i+1; j<orbs.length; j++){
                var a = orbs[i], b = orbs[j];
                var d = Math.hypot(a.x-b.x, a.y-b.y);
                if(d < 140){
                    var md = Math.min(Math.hypot(a.x-mx, a.y-my), Math.hypot(b.x-mx, b.y-my));
                    var alpha = (1 - d/140) * (md < 240 ? 0.28 : 0.07);
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.strokeStyle = 'rgba(255, 45, 85, ' + alpha + ')';
                    ctx.lineWidth = 0.75;
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(render);
    }
    render();
})();
</script>
"""
st.iframe(_APPLE_FLUID_BG, height=0)

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

# ── Apple Health Hero Banner ───────────────────────────────────────────────────
st.markdown(
    """
    <div class="apple-hero">
        <div class="apple-pill-badge">
            <span></span> CARDIOVASCULAR INTELLIGENCE · DECISION SUPPORT
        </div>
        <div class="apple-title">
            <span class="heart-emoji">🫀</span>
            <span class="title-text">CardioSense</span>
        </div>
        <p class="apple-subtitle">
            Machine learning decision-support system predicting heart failure risk from physiological biomarkers.
            Tuned with Stratified 5-Fold Cross-Validation, prioritizing clinical recall.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Medical Notice ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="apple-notice">
        <span>⚠️</span>
        <div>
            <strong>Clinical Decision-Support Notice:</strong>
            This software is an investigational academic prototype designed strictly to assist clinicians with risk triage.
            It does not replace comprehensive medical evaluation, ECG telemetry, or coronary angiography.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Apple Segmented Navigation ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  🩺  Risk Assessment  ",
    "  📊  Model Comparison  ",
    "  ℹ️  About & Team  ",
])

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — PATIENT RISK ASSESSMENT                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<div class="apple-section-header"><span>📋</span> CLINICAL PROFILE INPUT</div>', unsafe_allow_html=True)

    # Archetype Selector wrapped in Frosted Glass Panel
    st.markdown(
        """
        <div class="apple-card" style="padding: 1.1rem 1.6rem; margin-bottom: 1.4rem;">
            <div class="apple-section-header" style="margin-bottom: 0.35rem;">
                <span>⚡</span> QUICK PRESET ARCHETYPES (DEMO & VIVA PRESENTATION)
            </div>
        """,
        unsafe_allow_html=True,
    )
    preset_choice = st.selectbox(
        "⚡ Quick Preset Archetypes:",
        list(PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

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

        submit_assessment = st.form_submit_button("🫀  Analyze Cardiac Risk", width="stretch")

    # ── Assessment Results (Apple Health Dashboard) ──
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

            # Apple HIG Theme Palette per Risk Tier
            APPLE_TIERS = {
                "Low": {
                    "primary": "#30D158",      # Apple Health Green
                    "bg": "rgba(48, 209, 88, 0.14)",
                    "border": "rgba(48, 209, 88, 0.38)",
                    "tag": "NORMAL PROFILE",
                },
                "Moderate": {
                    "primary": "#FF9F0A",      # Apple Amber
                    "bg": "rgba(255, 159, 10, 0.14)",
                    "border": "rgba(255, 159, 10, 0.38)",
                    "tag": "ELEVATED RISK",
                },
                "High": {
                    "primary": "#FF2D55",      # Apple Health Red
                    "bg": "rgba(255, 45, 85, 0.16)",
                    "border": "rgba(255, 45, 85, 0.42)",
                    "tag": "HIGH RISK ALERT",
                },
            }
            tier = APPLE_TIERS[lvl]

            st.markdown("---")
            st.markdown('<div class="apple-section-header"><span>📊</span> RISK ASSESSMENT DASHBOARD</div>', unsafe_allow_html=True)

            # 3-Column Apple Health Widget Row
            w1, w2, w3 = st.columns([1.2, 1.2, 1.8], gap="medium")

            with w1:
                st.markdown(
                    f"""
                    <div class="health-widget" style="border-color: {tier['border']}; background: {tier['bg']};">
                        <div class="apple-pill-badge" style="background: rgba(255,255,255,0.1); border-color: {tier['border']}; color: {tier['primary']}; margin-bottom: 0.5rem;">
                            {tier['tag']}
                        </div>
                        <div class="widget-value" style="color: {tier['primary']};">{lvl.upper()}</div>
                        <p class="widget-caption" style="color: #FFFFFF; font-weight: 500;">{diag}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with w2:
                # Apple Activity Ring SVG
                circ = 364.42
                stroke_dashoffset = circ - (prob * circ)

                st.markdown(
                    f"""
                    <div class="health-widget">
                        <div class="widget-label">CARDIAC RISK PROBABILITY</div>
                        <div class="ring-container">
                            <svg width="148" height="148" viewBox="0 0 148 148">
                                <circle cx="74" cy="74" r="58" stroke="rgba(255, 255, 255, 0.08)" stroke-width="12" fill="none" />
                                <circle cx="74" cy="74" r="58" stroke="{tier['primary']}" stroke-width="12" fill="none"
                                        stroke-linecap="round"
                                        stroke-dasharray="{circ}"
                                        stroke-dashoffset="{stroke_dashoffset}"
                                        style="transition: stroke-dashoffset 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);" />
                            </svg>
                            <div class="ring-center-content">
                                <div class="ring-percent" style="color: {tier['primary']};">{pct:.1f}<span style="font-size: 1.1rem; font-weight: 700;">%</span></div>
                                <div class="ring-subtext">PROBABILITY</div>
                            </div>
                        </div>
                        <p class="widget-caption" style="font-size: 0.74rem; color: #8E8E93;">Threshold: 50% | High Risk: &gt;60%</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with w3:
                st.markdown(
                    f"""
                    <div class="apple-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center; margin: 0;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.6rem;">
                            <span style="background: rgba(10, 132, 255, 0.18); border: 1px solid rgba(10, 132, 255, 0.4); border-radius: 8px; padding: 3px 8px; font-size: 0.75rem; font-weight: 700; color: #0A84FF;"> CLINICAL INSIGHT</span>
                        </div>
                        <p style="color: #E5E5EA; font-size: 0.95rem; line-height: 1.6; margin: 0;">
                            {res['clinical_advice']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if cholesterol == 0:
                st.warning("ℹ️ **Data Fix Applied:** Serum cholesterol was entered as 0. The pipeline safely imputed the training median and marked the binary missingness indicator (`Cholesterol_missing=1`).")

            # ── Top Contributing Factors (Apple Health Metric Trends) ──
            contribs = res.get("contributions", {})
            if contribs:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="apple-card" style="padding: 1.6rem 1.8rem; margin-bottom: 1rem;">
                        <div class="apple-section-header" style="margin-bottom: 0.35rem;">
                            <span>📈</span> PHYSIOLOGICAL RISK FACTOR CONTRIBUTIONS
                        </div>
                        <p style="color:#8E8E93; font-size:0.85rem; margin-top:0; margin-bottom:1rem;">
                            Derived from the trained Logistic Regression model coefficients scaled by this patient's transformed biomarkers.
                        </p>
                    """,
                    unsafe_allow_html=True,
                )

                max_val = max(abs(v) for v in contribs.values()) or 1.0

                for feat, weight in contribs.items():
                    bar_pct = min(int((abs(weight) / max_val) * 100), 100)
                    is_risk = weight > 0
                    color = "#FF375F" if is_risk else "#30D158"
                    badge_bg = "rgba(255, 55, 95, 0.15)" if is_risk else "rgba(48, 209, 88, 0.15)"
                    badge_border = "rgba(255, 55, 95, 0.35)" if is_risk else "rgba(48, 209, 88, 0.35)"
                    badge_label = "↑ Increases Risk" if is_risk else "↓ Protective Factor"

                    st.markdown(
                        f"""
                        <div class="trend-row">
                            <span class="trend-name">{feat}</span>
                            <div class="trend-bar-bg">
                                <div class="trend-bar-fill" style="width: {bar_pct}%; background: {color};"></div>
                            </div>
                            <span class="trend-badge" style="background: {badge_bg}; border: 1px solid {badge_border}; color: {color};">
                                {badge_label} ({weight:+.3f})
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — MODEL COMPARISON & COLLEGE DEMO                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<div class="apple-section-header"><span>🏆</span> BENCHMARK PERFORMANCE & VALIDATION</div>', unsafe_allow_html=True)
    st.caption("Four machine learning classifiers tuned with Stratified 5-Fold Cross-Validation. Evaluated on unseen test set (N = 184).")

    metrics_csv = Path("reports/model_comparison_metrics.csv")
    if metrics_csv.exists():
        mdf = pd.read_csv(metrics_csv)
        best_row = mdf.sort_values(by=["Recall", "F1-Score"], ascending=[False, False]).iloc[0]

        # Apple Summary Metric Cells
        st.markdown(
            f"""
            <div class="apple-summary-grid">
                <div class="apple-summary-cell" style="border-color: rgba(255, 45, 85, 0.4); background: rgba(255, 45, 85, 0.1) !important;">
                    <div class="cell-label" style="color: #FF375F;">TOP MODEL</div>
                    <div class="cell-value" style="color: #FF375F; font-size: 1.05rem;">{best_row['Model']}</div>
                </div>
                <div class="apple-summary-cell">
                    <div class="cell-label">TEST ACCURACY</div>
                    <div class="cell-value">{best_row['Accuracy']:.1%}</div>
                </div>
                <div class="apple-summary-cell">
                    <div class="cell-label" style="color: #30D158;">RECALL (SENSITIVITY)</div>
                    <div class="cell-value" style="color: #30D158;">{best_row['Recall']:.1%}</div>
                </div>
                <div class="apple-summary-cell">
                    <div class="cell-label">F1-SCORE</div>
                    <div class="cell-value">{best_row['F1-Score']:.3f}</div>
                </div>
                <div class="apple-summary-cell">
                    <div class="cell-label">ROC-AUC</div>
                    <div class="cell-value">{best_row['ROC-AUC']:.3f}</div>
                </div>
                <div class="apple-summary-cell" style="border-color: rgba(255, 159, 10, 0.4); background: rgba(255, 159, 10, 0.1) !important;">
                    <div class="cell-label" style="color: #FF9F0A;">MISSED DIAGNOSES</div>
                    <div class="cell-value" style="color: #FF9F0A;">{int(best_row['False Negatives'])}</div>
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
    st.markdown('<div class="apple-section-header"><span>📊</span> DIAGNOSTIC VISUALIZATIONS</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    roc_img = Path("reports/model_roc_curves.png")
    cm_img = Path("reports/best_model_confusion_matrix.png")
    fi_img = Path("reports/feature_importance.png")

    with c1:
        st.markdown('<div class="apple-card" style="padding: 1.2rem; margin-bottom: 0;">', unsafe_allow_html=True)
        if roc_img.exists():
            st.image(str(roc_img), caption="Multi-Model ROC Curves Overlay (Test Set)", width="stretch")
        else:
            st.warning("ROC curves plot not found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="apple-card" style="padding: 1.2rem; margin-bottom: 0;">', unsafe_allow_html=True)
        if cm_img.exists():
            st.image(str(cm_img), caption="Normalized Confusion Matrix (Winning Pipeline)", width="stretch")
        else:
            st.warning("Confusion matrix plot not found.")
        st.markdown('</div>', unsafe_allow_html=True)

    if fi_img.exists():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="apple-card" style="padding: 1.4rem;">', unsafe_allow_html=True)
        st.markdown("##### 🌲 Feature Importance & Predictive Markers")
        st.image(str(fi_img), caption="Top Clinical Predictive Markers in the Pipeline", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — DATASET & PROJECT TEAM                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<div class="apple-section-header"><span>ℹ️</span> PROJECT SPECIFICATION & TEAM</div>', unsafe_allow_html=True)

    info_left, info_right = st.columns(2, gap="large")

    with info_left:
        st.markdown(
            """
            <div class="apple-card">
                <h4 style="margin-top:0; color:#FFFFFF;">🗂️ Dataset Architecture</h4>
                <ul style="color:#AEAEB2; line-height:1.7; font-size:0.92rem; padding-left:1.2rem;">
                    <li><strong>Dataset Source:</strong> Kaggle Heart Failure Prediction Dataset by <em>fedesoriano</em></li>
                    <li><strong>Total Cohort:</strong> 918 patients synthesized across 5 clinical registries (Cleveland, Hungarian, Switzerland, Long Beach VA, Statlog)</li>
                    <li><strong>Features:</strong> 11 clinical indicators + 1 binary target (<code>HeartDisease</code>)</li>
                    <li><strong>Class Distribution:</strong> 508 Positive (55.3%) vs. 410 Negative (44.7%) — balanced cohort</li>
                </ul>

                <h4 style="margin-top:1.4rem; color:#FFFFFF;">🛠️ Data Quality Remediation</h4>
                <ul style="color:#AEAEB2; line-height:1.7; font-size:0.92rem; padding-left:1.2rem;">
                    <li><strong>Cholesterol Zeros (172 records):</strong> Biologically impossible serum values converted to <code>NaN</code>, imputed with training median inside the Pipeline, and flagged via <code>Cholesterol_missing</code>.</li>
                    <li><strong>RestingBP Zero (1 record):</strong> Handled safely via training median imputation.</li>
                    <li><strong>Leakage Prevention:</strong> Scalers and encoders fit exclusively on training folds via scikit-learn <code>ColumnTransformer</code>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info_right:
        st.markdown(
            """
            <div class="apple-card">
                <h4 style="margin-top:0; color:#FFFFFF;">👥 Project Team Members</h4>
                <div class="apple-team-grid" style="margin-top: 1rem; margin-bottom: 1.4rem;">
                    <div class="apple-team-member">
                        <div class="apple-avatar">AJ</div>
                        <div class="apple-team-info">
                            <div class="name">Adhithyan JS</div>
                            <div class="roll">Roll No. 7 · ML Tuning</div>
                        </div>
                    </div>
                    <div class="apple-team-member">
                        <div class="apple-avatar">EA</div>
                        <div class="apple-team-info">
                            <div class="name">Evin Saj Abraham</div>
                            <div class="roll">Roll No. 29 · Architecture</div>
                        </div>
                    </div>
                    <div class="apple-team-member">
                        <div class="apple-avatar">FH</div>
                        <div class="apple-team-info">
                            <div class="name">Farhana H</div>
                            <div class="roll">Roll No. 30 · EDA & Reporting</div>
                        </div>
                    </div>
                    <div class="apple-team-member">
                        <div class="apple-avatar">RK</div>
                        <div class="apple-team-info">
                            <div class="name">Ridhin Krishna M</div>
                            <div class="roll">Roll No. 53 · UI/UX & API</div>
                        </div>
                    </div>
                </div>

                <h4 style="margin-top:0; color:#FFFFFF;">⚠️ Clinical Limitations</h4>
                <ul style="color:#AEAEB2; line-height:1.7; font-size:0.92rem; padding-left:1.2rem; margin-bottom:0;">
                    <li><strong>Cohort Size:</strong> 918 observations is an academic exploratory sample.</li>
                    <li><strong>Missing Modern Biomarkers:</strong> Lacks Body Mass Index (BMI), smoking pack-years, and hs-Troponin. <code>FastingBS</code> acts as a binary glucose surrogate.</li>
                    <li><strong>Decision-Support Only:</strong> Designed strictly as an adjunctive triage aid.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
