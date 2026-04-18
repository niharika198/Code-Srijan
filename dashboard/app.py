import os
import sys
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import EnergyPredictor, PredictConfig
from src.sensor_mapper import get_latest_metrics, get_recent_history, map_sensors_to_energy


ARTIFACTS_DIR = "artifacts"
MODELS_DIR = "models"


@st.cache_resource
def load_predictor() -> EnergyPredictor:
    return EnergyPredictor(PredictConfig(artifacts_dir=ARTIFACTS_DIR, models_dir=MODELS_DIR))


@st.cache_data
def load_daily_home() -> pd.DataFrame:
    path = os.path.join(ARTIFACTS_DIR, "daily_home.pkl")
    return pd.read_pickle(path)


@st.cache_data
def load_daily_appliance() -> pd.DataFrame:
    path = os.path.join(ARTIFACTS_DIR, "daily_appliance.pkl")
    return pd.read_pickle(path)


@st.cache_data
def load_peak_hour_usage() -> Optional[dict[str, Any]]:
    path = os.path.join(ARTIFACTS_DIR, "peak_hour_usage.pkl")
    if not os.path.isfile(path):
        return None
    return pd.read_pickle(path)


def _avg_kwh_by_appliance_global(daily_appliance: pd.DataFrame) -> pd.Series:
    """Mean daily kWh per appliance type across the whole dataset (baseline for insights)."""
    return daily_appliance.groupby("Appliance Type", observed=True)["kwh_day"].mean()


def _avg_kwh_by_home_appliance(daily_appliance: pd.DataFrame, home_id: str) -> pd.Series:
    """Mean daily kWh for this home + appliance type (personalized baseline)."""
    sub = daily_appliance[daily_appliance["Home ID"].astype(str) == str(home_id)]
    return sub.groupby("Appliance Type", observed=True)["kwh_day"].mean()


def _build_optimization_suggestions(
    top3: list,
    temp_used: float,
) -> list[str]:
    """Action-style tips: cost windows, temperature drivers (heuristic)."""
    lines: list[str] = []
    app_names_lower = [a.lower() for a, _ in top3]

    # Temperature-driven load
    hot_threshold = 26.0
    cold_threshold = 12.0
    if temp_used >= hot_threshold and any("air" in n or "conditioning" in n for n in app_names_lower):
        lines.append("🌡️ **Hot Day Alert**: High AC load. Raise your setpoint by 1-2°C to save big.")
    if temp_used <= cold_threshold and any("heater" in n for n in app_names_lower):
        lines.append("❄️ **Cold Day Alert**: High heating load. Check for drafts to keep the warmth in.")

    for app, _kwh in top3:
        al = app.lower()
        if "washing" in al or "dishwasher" in al or "dryer" in al:
            lines.append(f"🧺 **{app}**: Use after 10 PM for the lowest rates.")
        elif "oven" in al or "microwave" in al:
            lines.append(f"🍲 **{app}**: Batch cook your meals to avoid multiple heat-ups.")
        elif "computer" in al or "tv" in al:
            lines.append(f"🖥️ **{app}**: Switch to standby when not in use.")
        elif "lights" in al or "light" in al:
            lines.append(f"💡 **{app}**: Turn off in empty rooms to save instantly.")

    # De-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    if not out:
        out.append("✨ **Tip**: Shift heavy use to late night for better savings.")
    return out



def _build_efficiency_insights(
    appliance_preds: dict,
    top3: list,
    avg_global: pd.Series,
    avg_home: pd.Series,
) -> list[str]:
    """Compare predicted kWh to historical averages (dataset + this home)."""
    insights: list[str] = []
    for app, pred_kwh in top3:
        g = float(avg_global.get(app, float("nan")))
        h = float(avg_home.get(app, float("nan")))

        if np.isfinite(g) and g > 1e-6:
            pct_vs_global = (pred_kwh - g) / g * 100.0
            if pct_vs_global >= 5:
                insights.append(f"⚠️ **{app}** is using **{pct_vs_global:.0f}% more** than typical neighbors.")
            elif pct_vs_global <= -5:
                insights.append(f"🌟 **{app}** is **{abs(pct_vs_global):.0f}% more efficient** than average!")
            else:
                insights.append(f"✅ **{app}** usage is exactly on track with typical averages.")
        else:
            insights.append(f"📊 **{app}**: Monitoring your new usage pattern.")


        if np.isfinite(h) and h > 1e-6:
            pct_vs_home = (pred_kwh - h) / h * 100.0
            if abs(pct_vs_home) >= 5:
                insights.append(
                    f"🔹 Compared to **this home’s own past** daily average for **{app}** (~{h:.2f} kWh/day), "
                    f"today’s prediction is **{pct_vs_home:+.0f}%**."
                )

    return insights


def _build_peak_hour_insights(
    home_id: str,
    top3: list,
    peak_data: Optional[dict[str, Any]],
) -> list[str]:
    """
    Use row-level `Time` from the raw dataset (via preprocess) to estimate
    share of kWh in evening peak (6–10 PM) and after 8 PM.
    """
    if peak_data is None:
        return [
            "🔹 **Peak-hour usage (from `Time`):** run `python src/preprocess.py` to build `peak_hour_usage.pkl`, then restart the app."
        ]

    ha = peak_data.get("home_appliance")
    gl = peak_data.get("appliance_global")
    if ha is None or gl is None or ha.empty:
        return [
            "🔹 Peak-hour stats are missing or empty — rerun preprocessing on the CSV that includes **Time**."
        ]

    lines: list[str] = []
    ha = ha.copy()
    ha["Home ID"] = ha["Home ID"].astype(str)

    for app, _pred in top3:
        sub = ha[(ha["Home ID"] == str(home_id)) & (ha["Appliance Type"] == app)]
        if not sub.empty and float(sub.iloc[0]["total_kwh"]) >= 1e-9:
            r = sub.iloc[0]
            label = "From **this home’s** recorded start times"
        else:
            subg = gl[gl["Appliance Type"] == app]
            if subg.empty:
                continue
            r = subg.iloc[0]
            label = "From **dataset-wide** recorded start times"

        ps = float(r["peak_share"])
        a8 = float(r["after8_share"])

        if ps >= 0.30:
            lines.append(
                f"🔹 {label}, **~{ps * 100:.0f}%** of historical **{app}** kWh falls in **peak hours (6–10 PM)** — "
                f"shift flexible use **after 8 PM** where possible to reduce peak charges."
            )
        elif a8 >= 0.45:
            lines.append(
                f"🔹 {label}, **~{a8 * 100:.0f}%** of **{app}** kWh is already logged **after 8 PM** — "
                f"prioritize **shorter cycles / eco modes** over time-shifting."
            )
        else:
            lines.append(
                f"🔹 {label}: **{app}** — **{ps * 100:.0f}%** of kWh in **peak 6–10 PM**, **{a8 * 100:.0f}%** **after 8 PM**."
            )

    if not lines:
        lines.append(
            "🔹 No peak-time breakdown for these appliances — check `peak_hour_usage.pkl` after preprocessing."
        )
    return lines


def apply_style():
    """Apply unified high-fidelity styling across all pages."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@400;600;700;800&display=swap');
    
    /* Global Styles */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 10rem !important; /* Extra space for sticky footer */
    }
    
    .stApp, .stApp > header {
        background: radial-gradient(circle at center, #1e3a8a 0%, #060e20 100%) !important;
        font-family: 'Inter', sans-serif !important;
        color: #ffffff !important;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6, .hero-title, .feat-title, .mission-title {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    
    /* Ensure Streamlit Icons remain as icons and don't turn into text */
    [data-testid="stHeader"] *, [data-testid="stSidebarNav"] *, .st-emotion-cache-16idsys p, [data-testid="stSidebarCollapseButton"] * {
        font-family: inherit !important;
    }
    
    /* Absolute Fix for Sidebar Icon: Hide all internal text and inject a pure Material Icon */
    [data-testid="stSidebarCollapseButton"] button {
        font-size: 0 !important;
        color: transparent !important;
    }
    [data-testid="stSidebarCollapseButton"] button * {
        display: none !important;
    }
    [data-testid="stSidebarCollapseButton"] button::before {
        content: "chevron_right" !important;
        font-family: 'Material Icons' !important;
        font-size: 24px !important;
        color: #60a5fa !important;
        display: block !important;
        visibility: visible !important;
    }




    
    h1 { font-size: 2.2rem !important; font-weight: 800 !important; line-height: 1.1 !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 1.5rem !important; margin-bottom: 0.5rem !important; }
    h3 { font-size: 1.25rem !important; }
    
    p, span, div, label, .feature-text {
        font-family: 'Inter', sans-serif !important;
        font-weight: 400;
    }
    
    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(to right, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        font-weight: 800;
    }
    
    /* Cards (Glassmorphism) */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1.5rem;
        padding: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(96, 165, 250, 0.4);
        transform: translateY(-6px);
        box-shadow: 0 20px 40px -20px rgba(0, 0, 0, 0.7);
    }
    
    /* Metric Styling */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.3) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 1rem !important;
        border-radius: 1.25rem !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 0.85rem !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-family: 'Outfit', sans-serif !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(to right, #3b82f6, #8b5cf6) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.5) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Sidebar Toggle Button Styling */
    [data-testid="stSidebarCollapseButton"] button {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 50% !important;
        color: #60a5fa !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover {
        background: rgba(96, 165, 250, 0.2) !important;
        border-color: #60a5fa !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060e20 0%, #0d1b3e 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: #60a5fa !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input, 
    .stSelectbox > div > div > div,
    .stDateInput > div > div > input {
        background-color: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 0.75rem !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Slider & Toggle Styling */
    .stSlider [data-testid="stTickBar"] {
        display: none;
    }
    .stSlider [data-baseweb="slider"] > div > div {
        background: linear-gradient(to right, #3b82f6, #8b5cf6) !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #ffffff !important;
        border: 2px solid #3b82f6 !important;
    }
    
    [data-testid="stCheckbox"] div div {
        background-color: #3b82f6 !important;
    }
    
    /* Toggle (st.toggle) */
    .stToggle div[data-baseweb="toggle"] > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    .stToggle div[data-baseweb="toggle"][aria-checked="true"] > div {
        background: linear-gradient(to right, #3b82f6, #8b5cf6) !important;
    }

    /* Plots */
    .js-plotly-plot .plotly .bg, .js-plotly-plot .plotly .paper-bg {
        fill: transparent !important;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #060e20; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #334155; }

    /* Custom Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background-color: rgba(30, 41, 59, 0.4) !important;
        padding: 0.25rem 0.75rem !important;
        border-radius: 0.75rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        margin-bottom: 1rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px !important;
        background-color: transparent !important;
        border: none !important;
        color: #94a3b8 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] {
        color: #60a5fa !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #60a5fa !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_nav(key_prefix: str):
    """Render the unified navigation bar."""
    c1, c2, c3, c4, c5, c6 = st.columns([3, 1, 1, 1, 1, 3])
    with c1:
        st.markdown("<h3 style='margin:0; padding-top:5px; font-family: Outfit;'>⚡ SmartVolt AI</h3>", unsafe_allow_html=True)
    
    pages = [
        ("Home", "landing"),
        ("Features", "features"),
        ("About Us", "about"),
        ("Predictor", "dashboard")
    ]
    
    cols = [c2, c3, c4, c5]
    for col, (label, target) in zip(cols, pages):
        with col:
            if st.button(label, use_container_width=True, key=f"{key_prefix}_nav_{target}"):
                st.session_state.page = target
                st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

def render_live_updates(tariff_per_kwh):
    # This fragment will rerun every 2 seconds automatically
    live_metrics = get_latest_metrics()
    
    if live_metrics:
        try:
            last_sync = datetime.fromisoformat(live_metrics['timestamp'])
            time_diff = datetime.now() - last_sync
            status_color = "🟢" if time_diff.total_seconds() < 60 else "🟠"
            sync_text = last_sync.strftime('%H:%M:%S')
        except (ValueError, KeyError):
            status_color = "🟢"
            sync_text = live_metrics.get('timestamp', 'N/A')
            
        st.write(f"{status_color} **Auto-Sync Active**: Updating every 2s (Last: {sync_text})")
        
        l1, l2, l3, l4, l5 = st.columns(5)
        l1.metric("Lighting", f"{live_metrics['live_lighting_kw']:.2f} kW")
        l2.metric("Washing Machine", f"{live_metrics['live_appliance_kw']:.2f} kW")
        l3.metric("AC Load", f"{live_metrics.get('live_ac_kw', 0):.2f} kW")
        l4.metric("Base Load", f"{live_metrics['live_base_kw']:.2f} kW")
        l5.metric("Total Load", f"{live_metrics['total_live_kw']:.2f} kW")
        
        st.header("THERMAL INTELLIGENCE")
        t1, t2 = st.columns(2)
        t1.metric("Estimated Room Temp", f"{live_metrics.get('inferred_room_temp', 25):.1f} °C")
        t2.metric("Optimal AC Setpoint", f"{live_metrics.get('recommended_setpoint', 24):.0f} °C")

        st.header("LIVE INSIGHTS")
        
        # Simple baseline for insights if forecast not run
        hourly_baseline = 0.4 
        actual_load = live_metrics['total_live_kw']
        
        wasted_kw = 0.0
        waste_reason = ""
        
        if live_metrics['raw_lux'] > 400 and live_metrics['live_lighting_kw'] > 0.05:
            wasted_kw = live_metrics['live_lighting_kw']
            waste_reason = "Daylight Harvesting: Lights are active during peak natural light."
        elif actual_load > (hourly_baseline * 2.5):
            wasted_kw = actual_load - hourly_baseline
            waste_reason = f"Consumption Anomaly: Current load is high compared to typical base load."
        elif actual_load > 0.5 and live_metrics['raw_lux'] < 20 and live_metrics['raw_noise'] < -60:
            wasted_kw = actual_load - 0.15 
            waste_reason = "Idle Load: High consumption detected in an unoccupied room."

        cw1, cw2 = st.columns([1, 2])
        with cw1:
            st.metric("Total Waste Stream", f"{wasted_kw:.3f} kW", delta=f"{wasted_kw:+.3f} kW", delta_color="inverse")
        with cw2:
            if wasted_kw > 0.01:
                st.error(f"⚠️ **Waste Detected**: {waste_reason}")
            else:
                st.success("✅ **Status: Optimal**. Real-time efficiency targets met.")

        st.markdown("### 🎯 OPTIMAL ACTION PLAN")
        suggestions = []
        if live_metrics.get('inferred_room_temp', 0) > 25:
            suggestions.append(f"🌡️ **Cooling Optimization**: Room is ~{live_metrics['inferred_room_temp']:.1f}°C. Set AC to **24°C**.")
        if live_metrics['raw_lux'] > 300:
            suggestions.append("💡 **Turn off your lights**: Ambient light is sufficient.")
        if live_metrics['live_appliance_kw'] > 0.5:
            suggestions.append("⏲️ **Wait until after 10 PM**: Large appliances are cheaper to run later.")
        
        if not suggestions:
            suggestions.append("🌟 **Everything looks perfect!**")
        
        for s in suggestions:
            st.markdown(f"- {s}")
    else:
        st.info("Waiting for live sensor data... Connect your device.")

# Fragment decorator must be applied after function definition to use it as a fragment
render_live_updates = st.fragment(run_every=2)(render_live_updates)


def render_dashboard():
    apply_style()
    render_nav("dash")
    
    daily_home = load_daily_home()
    home_ids = sorted(daily_home["Home ID"].astype(str).unique().tolist())
    min_dt = pd.to_datetime(daily_home["Date"].min()).date()
    max_dt = pd.to_datetime(daily_home["Date"].max()).date()

    # Layout: Navigation Tabs (75%) | Controls (25%)
    # Background Defaults
    home_id = home_ids[0]
    target_date = max_dt
    month_start = max_dt.replace(day=1)
    expected_temp = None
    predictor = load_predictor()

    col_nav, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        tariff_per_kwh = st.slider("TARIFF (₹/kWh)", 0.0, 50.0, 12.0, 0.5)
        st.markdown("<div style='text-align: right; margin-top: -1.5rem;'>", unsafe_allow_html=True)
        st.markdown("<div style='background: rgba(105, 246, 184, 0.1); padding: 0.3rem 0.6rem; border-radius: 2rem; border: 1px solid rgba(105, 246, 184, 0.2); color: #69f6b8; display: inline-block; font-family: Inter, sans-serif; font-size: 0.6rem; letter-spacing: 0.05em; font-weight: 600;'><span style='display:inline-block; width: 4px; height: 4px; background: #69f6b8; border-radius: 50%; margin-right: 5px; box-shadow: 0 0 8px #69f6b8;'></span>LIVE</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


    with col_nav:
        tab_live, tab_today, tab_month = st.tabs(["Live Updates", "Daily Forecast", "Monthly Analytics"])

    with tab_live:
        st.markdown("<h1>LIVE TELEMETRY</h1>", unsafe_allow_html=True)
        render_live_updates(tariff_per_kwh)

    with tab_today:
        st.markdown("<h1>TODAY'S ENERGY FORECAST</h1>", unsafe_allow_html=True)
        compute_btn_today = st.button("⚡ Generate Daily Analysis", type="primary", use_container_width=True, key="btn_today")
        
        if not compute_btn_today:
            st.markdown("<div style='text-align: center; padding: 4rem; background: #0d1835; border-radius: 0.5rem; margin-top: 1rem; box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4); border: 1px solid rgba(163, 170, 196, 0.1);'><h3 style='color: #ffffff; margin-bottom: 1rem;'>No Forecast Data Generated</h3><p>Click <span style='color: #ff8f76; font-weight: 600;'>Generate Daily Analysis</span> to see results.</p></div>", unsafe_allow_html=True)
        else:
            # Predict day
            appliance_preds = predictor.predict_appliance_day(home_id=str(home_id), target_date=target_date, expected_temp=expected_temp)
            day_info = predictor.predict_home_and_building_day(home_id=str(home_id), target_date=target_date, expected_temp=expected_temp)

            home_kwh_day = float(day_info["home_kwh_day"])
            home_cost_day = home_kwh_day * float(tariff_per_kwh)
            temp_used = day_info["avg_temp"]

            # Metrics
            c1, c2 = st.columns(2)
            c1.metric("Predicted Home Energy", f"{home_kwh_day:.2f} kWh")
            c2.metric("Estimated Daily Cost", f"₹{home_cost_day:.2f}")

            # Filter for requested appliances
            target_appliances = ["Air Conditioning", "Washing Machine", "Lights", "Fridge"]
            filtered_preds = {k: v for k, v in appliance_preds.items() if k in target_appliances}

            # Appliance-wise chart
            if filtered_preds:
                df_app = pd.DataFrame({"Appliance Type": list(filtered_preds.keys()), "Predicted kWh": list(filtered_preds.values())})
                df_app = df_app.sort_values("Predicted kWh", ascending=False)
                fig_app = px.bar(df_app, x="Appliance Type", y="Predicted kWh", title="Core Appliance Forecast (Focused View)")
                fig_app.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_app, use_container_width=True)

                # Insights using filtered data
                daily_appliance_df = load_daily_appliance()
                avg_global = _avg_kwh_by_appliance_global(daily_appliance_df)
                avg_home = _avg_kwh_by_home_appliance(daily_appliance_df, str(home_id))
                top3 = sorted(filtered_preds.items(), key=lambda kv: kv[1], reverse=True)[:3]

                st.markdown("### 1. Optimization suggestions")
                for line in _build_optimization_suggestions(top3, temp_used):
                    st.markdown(f"- {line}")

                st.markdown("### 2. Efficiency insights")
                for line in _build_efficiency_insights(appliance_preds, top3, avg_global, avg_home):
                    st.markdown(f"- {line}")
            
    with tab_month:
        st.markdown("<h1>MONTHLY CONSUMPTION ANALYTICS</h1>", unsafe_allow_html=True)
        compute_btn_month = st.button("⚡ Generate Monthly Analysis", type="primary", use_container_width=True, key="btn_month")

        if not compute_btn_month:
             st.markdown("<div style='text-align: center; padding: 4rem; background: #0d1835; border-radius: 0.5rem; margin-top: 1rem; box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4); border: 1px solid rgba(163, 170, 196, 0.1);'><h3 style='color: #ffffff; margin-bottom: 1rem;'>No Forecast Data Generated</h3><p>Click <span style='color: #ff8f76; font-weight: 600;'>Generate Monthly Analysis</span> to see results.</p></div>", unsafe_allow_html=True)
        else:
            # Predict monthly totals
            month_total_home_kwh = predictor.predict_home_month_kwh(home_id=str(home_id), month_date=month_start, expected_temp=expected_temp)
            home_cost_month = month_total_home_kwh * float(tariff_per_kwh)

            # Monthly totals
            mc1, mc2 = st.columns(2)
            mc1.metric("Home total (kWh/month)", f"{month_total_home_kwh:.2f}")
            mc2.metric("Home total cost/month", f"₹{home_cost_month:.2f}")



        # End of Monthly Predicted Energy





def render_features_page():
    apply_style()
    render_nav("feat")
    
    st.markdown("<div class='feat-title' style='text-align:center;'>System <span class='gradient-text'>Capabilities</span></div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto 3rem auto; text-align: center;'>Our end-to-end pipeline leverages AI and IoT to revolutionize building management.</p>", unsafe_allow_html=True)
    
    _, center_col, _ = st.columns([1, 4, 1])
    with center_col:
        f1, f2 = st.columns(2)
        with f1:
            st.markdown('''
            <div class="glass-card">
                <div style="font-size: 2.5rem; margin-bottom: 1.5rem; color: #3b82f6;">📱</div>
                <h3 style="margin-bottom: 0.75rem;">Live Sensor Telemetry</h3>
                <div class="feature-text" style="color: #94a3b8; line-height: 1.6;">Connect the Sensor Logger app to stream real-time environmental data (lux, noise) to dynamically calculate live infrastructure power draw.</div>
            </div>
            ''', unsafe_allow_html=True)
        with f2:
            st.markdown('''
            <div class="glass-card">
                <div style="font-size: 2.5rem; margin-bottom: 1.5rem; color: #8b5cf6;">🤖</div>
                <h3 style="margin-bottom: 0.75rem;">Predictive Load Modeling</h3>
                <div class="feature-text" style="color: #94a3b8; line-height: 1.6;">Leverage historical datasets and advanced machine learning models to forecast daily and monthly energy consumption with pinpoint accuracy.</div>
            </div>
            ''', unsafe_allow_html=True)
            
        f3, f4 = st.columns(2)
        with f3:
            st.markdown('''
            <div class="glass-card">
                <div style="font-size: 2.5rem; margin-bottom: 1.5rem; color: #10b981;">☀️</div>
                <h3 style="margin-bottom: 0.75rem;">Daylight Harvesting</h3>
                <div class="feature-text" style="color: #94a3b8; line-height: 1.6;">Automatically adjust your lighting loads in real-time based on ambient sunlight availability to maximize efficiency without sacrificing comfort.</div>
            </div>
            ''', unsafe_allow_html=True)
        with f4:
            st.markdown('''
            <div class="glass-card">
                <div style="font-size: 2.5rem; margin-bottom: 1.5rem; color: #f97316;">📊</div>
                <h3 style="margin-bottom: 0.75rem;">Peak-Hour Analytics</h3>
                <div class="feature-text" style="color: #94a3b8; line-height: 1.6;">Gain actionable insights into appliance-level consumption patterns and identify high-cost demand spikes during the critical 6 PM - 10 PM peak window.</div>
            </div>
            ''', unsafe_allow_html=True)


def render_about_page():
    apply_style()
    render_nav("abt")
    
    # Statistical Impact Row
    st.markdown("""
    <div style='display: flex; justify-content: space-around; gap: 2rem; margin-bottom: 3.5rem; text-align: center;'>
        <div style='flex: 1; padding: 1.5rem; background: rgba(255, 255, 255, 0.03); border-radius: 1.25rem; border: 1px solid rgba(255, 255, 255, 0.05);'>
            <div style='font-size: 2.5rem; font-weight: 800; color: #ff8f76;'>40%</div>
            <div style='font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.5rem;'>Global CO2 Emissions from Buildings</div>
        </div>
        <div style='flex: 1; padding: 1.5rem; background: rgba(255, 255, 255, 0.03); border-radius: 1.25rem; border: 1px solid rgba(255, 255, 255, 0.05);'>
            <div style='font-size: 2.5rem; font-weight: 800; color: #60a5fa;'>30%</div>
            <div style='font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.5rem;'>Avg. Energy Wasted in Infrastructure</div>
        </div>
        <div style='flex: 1; padding: 1.5rem; background: rgba(255, 255, 255, 0.03); border-radius: 1.25rem; border: 1px solid rgba(255, 255, 255, 0.05);'>
            <div style='font-size: 2.5rem; font-weight: 800; color: #10b981;'>$20B+</div>
            <div style='font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.5rem;'>Annual Loss due to Inefficient Cooling</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_miss, col_biz = st.columns(2)


    
    with col_miss:
        st.markdown("""
        <div class="glass-card" style="height: 100%; text-align: left; padding: 2.5rem;">
            <div style="font-size: 2.5rem; margin-bottom: 1.5rem;">🎯</div>
            <h1 style="font-size: 2.5rem; margin-bottom: 1.5rem;">Mission <span class="gradient-text">Statement</span></h1>
            <p style="color: #e2e8f0; font-size: 1.05rem; line-height: 1.8; margin-bottom: 1.5rem;">
                At SmartVolt AI, we believe that the intersection of AI and real-time IoT telemetry holds the key to a sustainable future. We integrate intelligent models with physical infrastructure to optimize energy consumption.
            </p>
            <p style="color: #e2e8f0; font-size: 1.05rem; line-height: 1.8;">
                By shifting to proactive forecasting, we slash operational costs and reduce global carbon footprints, proving that technology and responsibility must go hand in hand.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_biz:
        st.markdown("""
        <div class="glass-card" style="height: 100%; text-align: left; padding: 2.5rem;">
            <div style="font-size: 2.5rem; margin-bottom: 1.5rem;">📈</div>
            <h1 style="font-size: 2.5rem; margin-bottom: 1.5rem;">Business <span class="gradient-text">Aspect</span></h1>
            <p style="color: #e2e8f0; font-size: 1.05rem; line-height: 1.8; margin-bottom: 1.5rem;">
                SmartVolt AI drives economic value by identifying hidden inefficiencies. Our platform enables facility managers to reduce energy overhead by up to 30% through predictive peak-load management and automated waste detection.
            </p>
            <p style="color: #e2e8f0; font-size: 1.05rem; line-height: 1.8;">
                We transform environmental sustainability into a competitive financial advantage, providing a clear ROI through reduced utility bills and optimized resource allocation across entire building portfolios.
            </p>
        </div>
        """, unsafe_allow_html=True)



    # The SmartVolt Resolution
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%); padding: 3rem; border-radius: 1.5rem; border: 1px solid rgba(255, 255, 255, 0.05); text-align: center;'>
        <h2 style='margin-bottom: 1.5rem;'>The SmartVolt <span class='gradient-text'>Resolution</span></h2>
        <p style='color: #94a3b8; max-width: 800px; margin: 0 auto 2rem auto; font-size: 1.1rem;'>
            We resolve these global inefficiencies through a three-pillar technological framework:
        </p>
        <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2rem;'>
            <div>
                <h4 style='color: #60a5fa;'>1. Predictive Forecasting</h4>
                <p style='font-size: 0.9rem; color: #a3aac4;'>ML models anticipate load spikes 24 hours in advance, allowing for proactive shifting.</p>
            </div>
            <div>
                <h4 style='color: #c084fc;'>2. IoT Sensor Fusion</h4>
                <p style='font-size: 0.9rem; color: #a3aac4;'>Real-time lux and noise data detect occupancy and natural light availability instantly.</p>
            </div>
            <div>
                <h4 style='color: #10b981;'>3. Autonomous Optimization</h4>
                <p style='font-size: 0.9rem; color: #a3aac4;'>Dynamic feedback loops suggest or automate adjustments to lighting and thermal loads.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_landing_page():

    apply_style()
    render_nav("land")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Hero
    st.markdown("<h1 style='text-align: center; font-size: 4.5rem; line-height: 1.1; margin-bottom: 1.5rem;'>AI-Driven Energy <br><span class='gradient-text'>Forecasting</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.25rem; max-width: 800px; margin: 0 auto 3rem auto; line-height: 1.6;'>Connect live IoT telemetry with predictive machine learning models to monitor real-time power draw, analyze appliance-level efficiency, and cut energy costs during peak hours.</p>", unsafe_allow_html=True)
    
    col_b1, col_b2, col_b3 = st.columns([4, 2, 4])
    with col_b2:
        if st.button("Enter Live Dashboard ➡️", type="primary", use_container_width=True, key="enter_dash"):
            st.session_state.page = "dashboard"
            st.rerun()

    


def render_footer():
    st.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; background: rgba(6, 14, 32, 0.9); backdrop-filter: blur(20px); border-top: 1px solid rgba(255,255,255,0.1); padding: 1.25rem 0; z-index: 999; text-align: center; color: #94a3b8; font-family: 'Outfit', sans-serif;">
        <p style="margin: 0; font-size: 0.9rem; font-weight: 500; letter-spacing: 0.02em;">© 2026 SmartVolt AI. Built for the AntiGravity Project. All rights reserved.</p>
        <p style="margin: 0.25rem 0 0 0; font-size: 0.75rem; opacity: 0.6; font-family: 'Inter', sans-serif;">Powered by IoT Telemetry & Predictive Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Antigravity Platform", layout="wide", initial_sidebar_state="expanded")
    
    if "page" not in st.session_state:
        st.session_state.page = "landing"
        
    if st.session_state.page == "landing":
        render_landing_page()
    elif st.session_state.page == "features":
        render_features_page()
    elif st.session_state.page == "about":
        render_about_page()
    else:
        render_dashboard()
        
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    render_footer()

if __name__ == "__main__":
    main()
