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
from src.sensor_mapper import get_latest_metrics


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


def main():
    st.set_page_config(page_title="Energy Optimization Dashboard", layout="wide")
    st.title("AI-Driven Energy Consumption Optimization (Live IoT Demo)")
    st.caption("Forecasting appliance-wise + home/building energy and cost using historical data and live Android phone sensors.")

    # --- LIVE COMMAND CENTER ---
    st.header("📡 Today's Command Center (Live IoT Data)")
    
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
        
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Live Lighting Load", f"{live_metrics['live_lighting_kw']:.2f} kW", f"{live_metrics['raw_lux']:.0f} Lux", delta_color="off")
        l2.metric("Live Appliance Load", f"{live_metrics['live_appliance_kw']:.2f} kW", f"{live_metrics['raw_noise']:.0f} dBFS", delta_color="off")
        l3.metric("Live Base Load", f"{live_metrics['live_base_kw']:.2f} kW")
        l4.metric("Total Live Load", f"{live_metrics['total_live_kw']:.2f} kW")
        
        # Calculate waste if live load differs significantly from expected behavior (naive approach for demo)
        st.subheader("💡 Energy Waste & Live Insights")
        if live_metrics['total_live_kw'] > 1.0 and live_metrics['raw_lux'] < 50:
            st.error("⚠️ **Energy Waste Alert**: High load detected but room is dark/empty. Did you leave appliances running?")
        elif live_metrics['raw_lux'] > 500 and live_metrics['live_lighting_kw'] > 0:
            st.warning("⚠️ **Energy Waste Alert**: Bright natural light detected, but lights appear to be on.")
        elif live_metrics['raw_noise'] < -50 and live_metrics['live_appliance_kw'] > 0:
            st.info("ℹ️ **Insight**: Low ambient noise but high appliance load. Ensure no silent appliances (heaters) are forgotten.")
        else:
            st.success("✅ **Optimal**: Live usage matches environmental conditions. No immediate waste detected.")
            
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
        st.header("Inputs")
        home_id = st.selectbox("Home / Building unit (Home ID)", home_ids)
        target_date = st.date_input("Day to predict", value=max_dt, min_value=min_dt, max_value=max_dt)
        month_start = st.date_input("Month to predict (monthly total)", value=max_dt.replace(day=1), min_value=min_dt)

        tariff_per_kwh = st.number_input(
            "Tariff per kWh (for cost prediction)",
            min_value=0.0,
            value=7.0,
            step=0.1,
            help="Set your electricity price. Cost = kWh * tariff.",
        )

        use_expected_temp = st.checkbox("Provide expected outdoor temperature (°C)", value=False)
        expected_temp = None
        if use_expected_temp:
            expected_temp = st.number_input("Expected outdoor temperature (°C)", value=28.0, step=0.5, format="%.2f")

        compute_btn = st.button("Predict", type="primary")

    predictor = load_predictor()

    if not compute_btn:
        st.info("Pick inputs and click Predict.")
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


if __name__ == "__main__":
    main()

