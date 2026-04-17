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

    # Temperature-driven load (realistic copy for demos)
    hot_threshold = 26.0
    cold_threshold = 12.0
    if temp_used >= hot_threshold and any("air" in n or "conditioning" in n for n in app_names_lower):
        lines.append("High consumption due to temperature.")
        lines.append(
            "Cooling load is likely elevated — try a slightly higher AC setpoint and trim use during the hottest hours."
        )
    if temp_used <= cold_threshold and any("heater" in n for n in app_names_lower):
        lines.append("High consumption due to temperature.")
        lines.append(
            "Heating load is likely elevated on colder days — check insulation and lower setpoint when comfortable."
        )

    for app, _kwh in top3:
        al = app.lower()
        if "washing" in al or "dishwasher" in al or "dryer" in al:
            lines.append(f"**{app}:** run after 8 PM to reduce cost.")
        elif "oven" in al or "microwave" in al:
            lines.append(
                f"**{app}:** batch cooking or shifting use to after 8 PM can trim peak-period cost."
            )
        elif "computer" in al or "tv" in al:
            lines.append(
                f"**{app}:** enable sleep/standby and avoid idle all-day use to cut standby draw."
            )
        elif "lights" in al or "light" in al:
            lines.append(
                f"**{app}:** switch off when rooms are empty; after 8 PM, dim or zone lighting to save cost."
            )

    # De-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    if not out:
        out.append(
            "Shift flexible loads to off-peak hours and review always-on devices to lower daily cost."
        )
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
                insights.append(
                    f"**{app}** is predicted to use **{pct_vs_global:.0f}% more** than the **dataset average** "
                    f"for this appliance type (~{g:.2f} kWh/day)."
                )
            elif pct_vs_global <= -5:
                insights.append(
                    f"**{app}** is predicted to use **{abs(pct_vs_global):.0f}% less** than the **dataset average** "
                    f"for this appliance type (~{g:.2f} kWh/day)."
                )
            else:
                insights.append(
                    f"**{app}** is close to the **dataset average** for this type (~{g:.2f} kWh/day)."
                )
        else:
            insights.append(f"**{app}:** not enough history to compare to a global average.")

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


def render_dashboard():
    col_back, _ = st.columns([1, 10])
    with col_back:
        if st.button("←", key="back_btn"):
            st.session_state.page = "landing"
            st.rerun()
    
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Manrope:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    /* Global Background & Typography */
    .stApp, .stApp > header {
        background-color: #060e20 !important;
        color: #a3aac4 !important;
        font-family: 'Manrope', sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #060e20 !important;
        border-right: none !important;
    }
    
    /* Typography Overrides */
    h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 3.5rem !important; /* display-lg */
    }
    
    h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-size: 0.85rem !important;
    }

    /* Luminous Text & Secondary Elements */
    p, span, div {
        color: #a3aac4;
        font-family: 'Manrope', sans-serif;
    }

    label, .st-bb, .st-af {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.6875rem !important; /* label-sm */
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Cards & Data Tiles (The "No-Line" Rule + Tonal Layering) */
    [data-testid="stMetric"] {
        background-color: #0d1835 !important; /* surface_container_high */
        padding: 1.5rem !important;
        border-radius: 0.5rem !important; /* lg */
        border: none !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #ffffff !important;
        font-size: 2.5rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        color: #a3aac4 !important;
        margin-bottom: 0.75rem !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Status Indicators / Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ff8f76 0%, #ff785a 100%) !important;
        color: #600e00 !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 700 !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: 0 4px 12px rgba(255, 143, 118, 0.2) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(255, 143, 118, 0.3) !important;
    }

    .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #ff8f76 !important;
        border: 1px solid rgba(163, 170, 196, 0.15) !important; /* Ghost Border */
        border-radius: 0.5rem !important;
    }

    /* Inputs & Selectors */
    .stTextInput > div > div > input, 
    .stNumberInput > div > div > input, 
    .stSelectbox > div > div > div,
    .stDateInput > div > div > input {
        background-color: #101c3a !important; /* surface_container_highest */
        color: #ffffff !important;
        border: 1px solid rgba(163, 170, 196, 0.15) !important; /* Ghost border */
        border-radius: 0.375rem !important;
    }
    .stTextInput > div > div > input:focus, 
    .stNumberInput > div > div > input:focus, 
    .stSelectbox > div > div > div:focus {
        border-color: rgba(255, 143, 118, 0.5) !important;
        box-shadow: 0 0 0 1px rgba(255, 143, 118, 0.5) !important;
    }

    /* Success Alert - Glassmorphism */
    [data-testid="stAlert"] {
        background-color: rgba(105, 246, 184, 0.1) !important; /* Secondary glass */
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(105, 246, 184, 0.2) !important;
        color: #69f6b8 !important;
        border-radius: 0.5rem !important;
    }
    
    /* Error/Warning Alert */
    div.st-emotion-cache-1n76uvr, div.st-emotion-cache-1n76uvr * {
        background-color: rgba(255, 177, 72, 0.1) !important;
        color: #ffb148 !important;
        border: 1px solid rgba(255, 177, 72, 0.2) !important;
    }

    /* Dividers */
    hr {
        border-color: rgba(163, 170, 196, 0.1) !important;
        margin: 2rem 0 !important;
    }
    
    /* Hide Plotly Background */
    .js-plotly-plot .plotly .bg {
        fill: transparent !important;
    }
    .js-plotly-plot .plotly .paper-bg {
        fill: transparent !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown("<h1>Kinetic Observatory</h1>", unsafe_allow_html=True)
    with col_status:
        st.markdown("<div style='margin-top: 1rem; background: rgba(105, 246, 184, 0.1); padding: 0.5rem 1rem; border-radius: 2rem; border: 1px solid rgba(105, 246, 184, 0.2); color: #69f6b8; display: inline-block; font-family: Inter, sans-serif; font-size: 0.75rem; letter-spacing: 0.05em; font-weight: 600;'><span style='display:inline-block; width: 6px; height: 6px; background: #69f6b8; border-radius: 50%; margin-right: 8px; box-shadow: 0 0 8px #69f6b8;'></span>SYSTEM LIVE</div>", unsafe_allow_html=True)

    # --- LIVE COMMAND CENTER ---
    st.header("COMMAND CENTER")
    
    live_metrics = get_latest_metrics()
    
    if live_metrics:
        # We parse the timestamp directly since it's an ISO string
        try:
            last_sync = datetime.fromisoformat(live_metrics['timestamp'])
            time_diff = datetime.now() - last_sync
            status_color = "🟢" if time_diff.total_seconds() < 60 else "🟠"
            sync_text = last_sync.strftime('%H:%M:%S')
        except ValueError:
            status_color = "🟢"
            sync_text = live_metrics['timestamp']
            
        st.write(f"{status_color} **Live Sync Status**: Last received at {sync_text} (Phone streaming active)")
        history = get_recent_history(2)
        prev_raw = None
        if len(history) >= 2:
            prev_raw = history[-2]
            
        def get_raw_delta(live_key, prev_key):
            if prev_raw and live_key in live_metrics and prev_key in prev_raw:
                diff = live_metrics[live_key] - prev_raw[prev_key]
                if abs(diff) > 0.001:
                    return f"{diff:+.1f}"
            return "0.0"

        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Live Lighting Load", f"{live_metrics['live_lighting_kw']:.2f} kW")
        l2.metric("Live Appliance Load", f"{live_metrics['live_appliance_kw']:.2f} kW")
        l3.metric("Live Base Load", f"{live_metrics['live_base_kw']:.2f} kW")
        l4.metric("Total Live Load", f"{live_metrics['total_live_kw']:.2f} kW")
        
        # Calculate waste if live load differs significantly from expected behavior (naive approach for demo)
        st.header("LIVE INSIGHTS")
        if live_metrics['total_live_kw'] > 1.0 and live_metrics['raw_lux'] < 50:
            st.error("⚠️ **Energy Waste Alert**: High load detected but room is dark/empty. Did you leave appliances running?")
        elif live_metrics['raw_lux'] > 500 and live_metrics['live_lighting_kw'] > 0:
            st.warning("⚠️ **Energy Waste Alert**: Bright natural light detected, but lights appear to be on.")
        elif live_metrics['raw_noise'] < -50 and live_metrics['live_appliance_kw'] > 0:
            st.info("ℹ️ **Insight**: Low ambient noise but high appliance load. Ensure no silent appliances (heaters) are forgotten.")
        else:
            st.success("✅ **Consumption Status: Optimal**\n\nSystem is operating at peak efficiency. Waste detection identifies 0 leakages in current grid.")
            
        col_btn1, col_btn2 = st.columns([2, 10])
        with col_btn1:
            st.button("🔄 Refresh Live Data")
        st.divider()
    else:
        st.info("Waiting for live sensor data... Connect your Android phone to the local server.")
        st.button("🔄 Check Again")
        st.divider()
    # ---------------------------

    # Model readiness check
    needed_models = [
        os.path.join(MODELS_DIR, "appliance_day_model.joblib"),
        os.path.join(MODELS_DIR, "home_day_model.joblib"),
        os.path.join(MODELS_DIR, "building_day_model.joblib"),
    ]
    models_ready = all(os.path.exists(p) for p in needed_models)
    if not models_ready:
        st.warning("Models not trained yet. Run `python src/train.py` first.")
        return

    daily_home = load_daily_home()
    home_ids = sorted(daily_home["Home ID"].astype(str).unique().tolist())
    min_dt = pd.to_datetime(daily_home["Date"].min()).date()
    max_dt = pd.to_datetime(daily_home["Date"].max()).date()

    with st.sidebar:
        st.markdown("<h2 style='color: #ff8f76 !important; font-size: 1.5rem !important; margin-bottom: 0;'>Energy AI</h2><p style='font-family: Inter, sans-serif; font-size: 0.6875rem; color: #a3aac4; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 2rem; margin-top: 0.25rem;'><span style='display:inline-block; width: 6px; height: 6px; background: #69f6b8; border-radius: 50%; margin-right: 8px;'></span>LIVE TELEMETRY</p>", unsafe_allow_html=True)
        
        home_id = st.selectbox("RESIDENTIAL UNIT", home_ids)
        target_date = st.date_input("DAY TO PREDICT", value=max_dt, min_value=min_dt, max_value=max_dt)
        month_start = st.date_input("MONTH AGGREGATE", value=max_dt.replace(day=1), min_value=min_dt)

        tariff_per_kwh = st.slider(
            "TARIFF PER KWH",
            min_value=0.0,
            max_value=1.0,
            value=0.14,
            step=0.01,
        )

        use_expected_temp = st.toggle("Outdoor Temp", value=False)
        expected_temp = None
        if use_expected_temp:
            expected_temp = st.number_input("Expected outdoor temperature (°C)", value=28.0, step=0.5, format="%.2f")

        st.markdown("<br>", unsafe_allow_html=True)
        compute_btn = st.button("⚡ Predict Results", type="primary", use_container_width=True)

    predictor = load_predictor()

    if not compute_btn:
        st.header("PREDICTIVE ANALYSIS")
        st.markdown("<div style='text-align: center; padding: 4rem; background: #0d1835; border-radius: 0.5rem; margin-top: 1rem; box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4); border: 1px solid rgba(163, 170, 196, 0.1);'><h3 style='color: #ffffff; margin-bottom: 1rem;'>No Forecast Data Generated</h3><p>Pick inputs from the control panel and click <span style='color: #ff8f76; font-weight: 600;'>Predict Results</span> to see energy forecasting results for your selected timeline.</p></div>", unsafe_allow_html=True)
        return

    # Predict day
    appliance_preds = predictor.predict_appliance_day(home_id=str(home_id), target_date=target_date, expected_temp=expected_temp)
    day_info = predictor.predict_home_and_building_day(home_id=str(home_id), target_date=target_date, expected_temp=expected_temp)

    home_kwh_day = float(day_info["home_kwh_day"])
    building_kwh_day = float(day_info["building_kwh_day"])
    season_used = day_info["season"]
    temp_used = day_info["avg_temp"]

    # Predict monthly totals
    month_total_home_kwh = predictor.predict_home_month_kwh(home_id=str(home_id), month_date=month_start, expected_temp=expected_temp)
    month_total_building_kwh = predictor.predict_building_month_kwh(month_date=month_start, expected_temp=expected_temp)

    # Cost
    home_cost_day = home_kwh_day * float(tariff_per_kwh)
    building_cost_day = building_kwh_day * float(tariff_per_kwh)
    home_cost_month = month_total_home_kwh * float(tariff_per_kwh)
    building_cost_month = month_total_building_kwh * float(tariff_per_kwh)

    # Layout: top summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Home predicted energy (kWh/day)", f"{home_kwh_day:.2f}")
    c2.metric("Home predicted cost (currency/day)", f"{home_cost_day:.2f}")
    c3.metric("Building predicted energy (kWh/day)", f"{building_kwh_day:.2f}")

    st.write(f"Used: season = `{season_used}`, outdoor temp (avg/typical) = `{temp_used:.2f}°C`")

    # Appliance-wise chart
    if appliance_preds:
        df_app = pd.DataFrame({"Appliance Type": list(appliance_preds.keys()), "Predicted kWh": list(appliance_preds.values())})
        df_app = df_app.sort_values("Predicted kWh", ascending=False)
        fig_app = px.bar(df_app, x="Appliance Type", y="Predicted kWh", title="Appliance-wise predicted energy for the day")
        fig_app.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_app, use_container_width=True)
    else:
        st.warning("No appliances found for selected Home ID in the dataset.")

    # 2 & 3 — Optimization suggestion + efficiency insight (forecast + historical averages)
    if appliance_preds:
        daily_appliance_df = load_daily_appliance()
        avg_global = _avg_kwh_by_appliance_global(daily_appliance_df)
        avg_home = _avg_kwh_by_home_appliance(daily_appliance_df, str(home_id))
        top3 = sorted(appliance_preds.items(), key=lambda kv: kv[1], reverse=True)[:3]

        st.markdown("### 2. Optimization suggestion")
        st.caption("Actionable tips from today’s forecast and context (temperature, shiftable loads).")
        st.markdown("👉 **This is your real value** — use these to cut cost and peak demand.")
        for line in _build_optimization_suggestions(top3, temp_used):
            st.markdown(f"- {line}")

        st.markdown("### 3. Efficiency insight")
        st.caption(
            "How today’s predicted use compares to averages, plus **peak-hour concentration** inferred from **row-level `Time`** in the raw data (6–10 PM vs after 8 PM)."
        )
        for line in _build_efficiency_insights(appliance_preds, top3, avg_global, avg_home):
            st.markdown(f"- {line}")
        peak_payload = load_peak_hour_usage()
        for line in _build_peak_hour_insights(str(home_id), top3, peak_payload):
            st.markdown(f"- {line}")

    # Home vs building summary
    st.subheader("Daily totals")
    c4, c5, c6 = st.columns(3)
    c4.metric("Home total (kWh/day)", f"{home_kwh_day:.2f}")
    c5.metric("Home total cost/day", f"{home_cost_day:.2f}")
    c6.metric("Building cost/day", f"{building_cost_day:.2f}")

    # Monthly totals
    st.subheader("Monthly totals")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Home total (kWh/month)", f"{month_total_home_kwh:.2f}")
    mc2.metric("Home total cost/month", f"{home_cost_month:.2f}")
    mc3.metric("Building total cost/month", f"{building_cost_month:.2f}")

    # Optional: show actual vs predicted last 14 days for home (if within dataset)
    st.subheader("Home: last 14 days (actual vs predicted)")
    days = [pd.to_datetime(target_date) - pd.Timedelta(days=i) for i in range(13, -1, -1)]
    actual = (
        daily_home[(daily_home["Home ID"].astype(str) == str(home_id)) & (daily_home["Date"].isin(days))]
        .copy()
    )
    actual = actual.sort_values("Date")
    # Predict using typical weather (no expected_temp) to look like a forecast
    preds = []
    for d in days:
        info = predictor.predict_home_and_building_day(
            home_id=str(home_id),
            target_date=d.date(),
            expected_temp=expected_temp if use_expected_temp else None,
        )
        preds.append({"Date": d.date(), "Predicted kWh": float(info["home_kwh_day"])})
    pred_df = pd.DataFrame(preds)

    actual_df = actual[["Date", "kwh_day"]].copy()
    actual_df["Date"] = pd.to_datetime(actual_df["Date"]).dt.date
    actual_df = actual_df.rename(columns={"kwh_day": "Actual kWh"})

    plot_df = actual_df.merge(pred_df, on="Date", how="left")
    plot_df["Date"] = plot_df["Date"].astype(str)
    fig_line = px.line(plot_df, x="Date", y=["Actual kWh", "Predicted kWh"], markers=True)
    st.plotly_chart(fig_line, use_container_width=True)


def render_features_page():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
    
    .block-container {
        padding-top: 2rem !important;
    }
    
    .stApp, .stApp > header {
        background: radial-gradient(circle at center, #1e3a8a 0%, #0f172a 100%) !important;
        font-family: 'Inter', sans-serif !important;
        color: #ffffff !important;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    .feat-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
        color: #ffffff;
    }
    .feat-title span {
        background: linear-gradient(to right, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .feat-sub {
        color: #94a3b8;
        font-size: 1.1rem;
        max-width: 600px;
        margin: 0 auto 3rem auto;
        line-height: 1.5;
        text-align: center;
    }
    .feature-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 1rem;
        padding: 2rem;
        height: 100%;
    }
    .feature-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        color: #ffffff;
    }
    .feature-text {
        color: #94a3b8;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    
    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(to right, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 0.5rem !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
    # Navbar
    c1, c2, c3, c4, c5, c6 = st.columns([3, 1, 1, 1, 1, 3])
    with c1:
        st.markdown("<h3 style='margin:0; padding-top:5px;'>⚡ SmartVolt AI</h3>", unsafe_allow_html=True)
    with c2:
        if st.button("Home", use_container_width=True, key="feat_nav_home"):
            st.session_state.page = "landing"
            st.rerun()
    with c3:
        if st.button("Features", use_container_width=True, key="feat_nav_feat"):
            st.session_state.page = "features"
            st.rerun()
    with c4:
        if st.button("About Us", use_container_width=True, key="feat_nav_about"):
            st.session_state.page = "about"
            st.rerun()
    with c5:
        if st.button("Predictor", use_container_width=True, key="feat_nav_pred"):
            st.session_state.page = "dashboard"
            st.rerun()
            
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    st.markdown("<div class='feat-title'>System <span>Capabilities</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='feat-sub'>Our end-to-end pipeline leverages AI and IoT to revolutionize building management.</div>", unsafe_allow_html=True)
    
    _, center_col, _ = st.columns([1, 4, 1])
    with center_col:
        f1, f2 = st.columns(2)
        with f1:
            st.markdown('''
            <div class="feature-card">
                <div style="font-size: 2rem; margin-bottom: 1rem; color: #3b82f6;">📱</div>
                <div class="feature-title">Live Sensor Telemetry</div>
                <div class="feature-text">Connect the Sensor Logger app to stream real-time environmental data (lux, noise) to dynamically calculate live infrastructure power draw.</div>
            </div>
            ''', unsafe_allow_html=True)
        with f2:
            st.markdown('''
            <div class="feature-card">
                <div style="font-size: 2rem; margin-bottom: 1rem; color: #8b5cf6;">🤖</div>
                <div class="feature-title">Predictive Load Modeling</div>
                <div class="feature-text">Leverage historical datasets and advanced machine learning models to forecast daily and monthly energy consumption with pinpoint accuracy.</div>
            </div>
            ''', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        f3, f4 = st.columns(2)
        with f3:
            st.markdown('''
            <div class="feature-card">
                <div style="font-size: 2rem; margin-bottom: 1rem; color: #10b981;">☀️</div>
                <div class="feature-title">Daylight Harvesting</div>
                <div class="feature-text">Automatically adjust your lighting loads in real-time based on ambient sunlight availability to maximize efficiency without sacrificing comfort.</div>
            </div>
            ''', unsafe_allow_html=True)
        with f4:
            st.markdown('''
            <div class="feature-card">
                <div style="font-size: 2rem; margin-bottom: 1rem; color: #f97316;">📊</div>
                <div class="feature-title">Peak-Hour Analytics</div>
                <div class="feature-text">Gain actionable insights into appliance-level consumption patterns and identify high-cost demand spikes during the critical 6 PM - 10 PM peak window.</div>
            </div>
            ''', unsafe_allow_html=True)

def render_about_page():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
    
    .block-container {
        padding-top: 2rem !important;
    }
    
    .stApp, .stApp > header {
        background: radial-gradient(circle at center, #1e3a8a 0%, #0f172a 100%) !important;
        font-family: 'Inter', sans-serif !important;
        color: #ffffff !important;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(to right, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 0.5rem !important;
    }
    
    /* About Us Specifics */
    .mission-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        padding: 4rem 3rem;
        max-width: 800px;
        margin: 2rem auto;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .mission-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 2rem;
        color: #ffffff;
    }
    .mission-title span {
        background: linear-gradient(to right, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .mission-text {
        color: #e2e8f0;
        font-size: 1.15rem;
        line-height: 1.8;
        margin-bottom: 1.5rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
    # Navbar
    c1, c2, c3, c4, c5, c6 = st.columns([3, 1, 1, 1, 1, 3])
    with c1:
        st.markdown("<h3 style='margin:0; padding-top:5px;'>⚡ SmartVolt AI</h3>", unsafe_allow_html=True)
    with c2:
        if st.button("Home", use_container_width=True, key="abt_nav_home"):
            st.session_state.page = "landing"
            st.rerun()
    with c3:
        if st.button("Features", use_container_width=True, key="abt_nav_feat"):
            st.session_state.page = "features"
            st.rerun()
    with c4:
        if st.button("About Us", use_container_width=True, key="abt_nav_about"):
            st.session_state.page = "about"
            st.rerun()
    with c5:
        if st.button("Predictor", use_container_width=True, key="abt_nav_pred"):
            st.session_state.page = "dashboard"
            st.rerun()
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="mission-container">
        <div class="mission-title">Mission <span>Statement</span></div>
        <div class="mission-text">
            At SmartVolt AI, we believe that the intersection of advanced artificial intelligence and real-time IoT telemetry holds the key to a sustainable future. Our objective is to seamlessly integrate intelligent predictive models with physical infrastructure, empowering buildings to dynamically optimize their own energy consumption without sacrificing human comfort.
        </div>
        <div class="mission-text">
            By shifting from reactive management to proactive forecasting, we are redefining what it means to be energy-efficient. Our end-to-end platform not only slashes operational costs but fundamentally reduces global carbon footprints, proving that cutting-edge technology and environmental responsibility can—and must—go hand in hand.
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_landing_page():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
    
    .block-container {
        padding-top: 2rem !important;
    }
    
    .stApp, .stApp > header {
        background: radial-gradient(circle at center, #1e3a8a 0%, #0f172a 100%) !important;
        font-family: 'Inter', sans-serif !important;
        color: #ffffff !important;
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        text-align: center;
        color: #ffffff;
    }
    .hero-title span {
        background: linear-gradient(to right, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.25rem;
        max-width: 800px;
        margin: 0 auto 3rem auto;
        line-height: 1.6;
        text-align: center;
    }
    .feature-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 1rem;
        padding: 2rem;
        margin-bottom: 2rem;
        height: 100%;
    }
    .feature-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    .feature-text {
        color: #94a3b8;
        line-height: 1.5;
    }
    
    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(to right, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 0.5rem !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
    # Navbar
    c1, c2, c3, c4, c5, c6 = st.columns([3, 1, 1, 1, 1, 3])
    with c1:
        st.markdown("<h3 style='margin:0; padding-top:5px;'>⚡ SmartVolt AI</h3>", unsafe_allow_html=True)
    with c2:
        if st.button("Home", use_container_width=True, key="nav_home"):
            st.session_state.page = "landing"
            st.rerun()
    with c3:
        if st.button("Features", use_container_width=True, key="nav_feat"):
            st.session_state.page = "features"
            st.rerun()
    with c4:
        if st.button("About Us", use_container_width=True, key="nav_about"):
            st.session_state.page = "about"
            st.rerun()
    with c5:
        if st.button("Predictor", use_container_width=True, key="nav_pred"):
            st.session_state.page = "dashboard"
            st.rerun()
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hero
    st.markdown("<div class='hero-title'>AI-Driven Energy <span>Forecasting</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>Connect live IoT telemetry with predictive machine learning models to monitor real-time power draw, analyze appliance-level efficiency, and cut energy costs during peak hours.</div>", unsafe_allow_html=True)
    
    col_b1, col_b2, col_b3 = st.columns([4, 2, 4])
    with col_b2:
        if st.button("Enter Live Dashboard ➡️", type="primary", use_container_width=True, key="enter_dash"):
            st.session_state.page = "dashboard"
            st.rerun()
            
    st.markdown("<br>", unsafe_allow_html=True)
    


def render_footer():
    st.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: rgba(15, 23, 42, 0.95); border-top: 1px solid rgba(255,255,255,0.05); padding: 1rem 0; z-index: 999; text-align: center; color: #64748b; font-size: 0.95rem; font-family: 'Inter', sans-serif;">
        © 2026 SmartVolt AI. Built for the AntiGravity Project. All rights reserved.<br>
        <span style="font-size: 0.85rem; opacity: 0.7;">Powered by IoT Telemetry & Predictive Analytics</span>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Antigravity Platform", layout="wide")
    
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
        
    render_footer()

if __name__ == "__main__":
    main()
