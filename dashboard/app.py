import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import EnergyPredictor, PredictConfig


ARTIFACTS_DIR = "artifacts"
MODELS_DIR = "models"


@st.cache_resource
def load_predictor() -> EnergyPredictor:
    return EnergyPredictor(PredictConfig(artifacts_dir=ARTIFACTS_DIR, models_dir=MODELS_DIR))


@st.cache_data
def load_daily_home() -> pd.DataFrame:
    path = os.path.join(ARTIFACTS_DIR, "daily_home.pkl")
    return pd.read_pickle(path)


def main():
    st.set_page_config(page_title="Energy Optimization Dashboard", layout="wide")
    st.title("AI-Driven Energy Consumption Optimization (Hackathon Demo)")
    st.caption("Forecasting appliance-wise + home/building energy and cost using your provided dataset (simulated IoT).")

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

    # Optimization narrative (simple heuristic)
    if appliance_preds:
        top3 = sorted(appliance_preds.items(), key=lambda kv: kv[1], reverse=True)[:3]
        st.subheader("Optimization suggestions (based on top contributors)")
        for app, kwh in top3:
            app_lower = app.lower()
            suggestion = "Consider reducing usage or shifting to lower-demand periods when possible."
            if "air" in app_lower or "conditioning" in app_lower:
                suggestion = "Reduce cooling setpoint slightly and avoid peak hours; try 1-2°C higher target."
            elif "heater" in app_lower:
                suggestion = "Reduce heating setpoint slightly and prefer passive warming; avoid unnecessary heating hours."
            elif "lights" in app_lower:
                suggestion = "Switch to LED and turn off when not needed; aim for shorter usage windows."
            elif "washing" in app_lower or "dishwasher" in app_lower:
                suggestion = "Run during off-peak hours and enable eco modes to reduce total kWh."
            elif "oven" in app_lower or "microwave" in app_lower:
                suggestion = "Plan cooking to reduce repeated heating cycles during peak demand."
            st.write(f"- `{app}` predicted at {kwh:.2f} kWh: {suggestion}")

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

