import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessConfig:
    smart_home_csv: str
    appliance_usage_csv: Optional[str] = None
    artifacts_dir: str = "artifacts"


def _add_date_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise ValueError(f"Some rows have invalid {date_col} values.")

    df["day_of_week"] = df[date_col].dt.dayofweek  # 0=Mon
    df["month"] = df[date_col].dt.month
    df["day_of_year"] = df[date_col].dt.dayofyear

    # Cyclical encoding helps tree models slightly too.
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366.0)
    return df


def load_raw(cfg: PreprocessConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.smart_home_csv)
    required = {
        "Home ID",
        "Appliance Type",
        "Energy Consumption (kWh)",
        "Time",
        "Date",
        "Outdoor Temperature (°C)",
        "Season",
        "Household Size",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"smart_home_energy_consumption.csv missing columns: {sorted(missing)}")

    # Basic cleaning
    df["Energy Consumption (kWh)"] = pd.to_numeric(df["Energy Consumption (kWh)"], errors="coerce")
    df["Outdoor Temperature (°C)"] = pd.to_numeric(df["Outdoor Temperature (°C)"], errors="coerce")
    df["Household Size"] = pd.to_numeric(df["Household Size"], errors="coerce")
    df = df.dropna(subset=["Energy Consumption (kWh)", "Outdoor Temperature (°C)", "Household Size", "Date"])
    return df


def build_daily_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Appliance-wise daily aggregation
    daily_appliance = (
        df.groupby(["Home ID", "Appliance Type", "Date"], as_index=False)
        .agg(
            kwh_day=("Energy Consumption (kWh)", "sum"),
            avg_temp=("Outdoor Temperature (°C)", "mean"),
            season=("Season", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            household_size=("Household Size", "first"),
        )
    )
    daily_appliance = _add_date_features(daily_appliance, "Date")

    # Home-wise daily totals
    daily_home = (
        df.groupby(["Home ID", "Date"], as_index=False)
        .agg(
            kwh_day=("Energy Consumption (kWh)", "sum"),
            avg_temp=("Outdoor Temperature (°C)", "mean"),
            season=("Season", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            household_size=("Household Size", "first"),
        )
    )
    daily_home = _add_date_features(daily_home, "Date")

    # Building-wise daily totals (sum of all homes)
    daily_building = (
        df.groupby(["Date"], as_index=False)
        .agg(
            kwh_day=("Energy Consumption (kWh)", "sum"),
            avg_temp=("Outdoor Temperature (°C)", "mean"),
            season=("Season", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
        )
    )
    daily_building = _add_date_features(daily_building, "Date")

    # Typical temperature for "future" / missing dates: by day-of-year.
    typical_temp = (
        df.copy()
        .assign(Date=pd.to_datetime(df["Date"], errors="coerce"))
        .dropna(subset=["Date"])
    )
    typical_temp["day_of_year"] = typical_temp["Date"].dt.dayofyear
    typical = typical_temp.groupby("day_of_year", as_index=False).agg(typical_temp=("Outdoor Temperature (°C)", "median"))
    return daily_appliance, daily_home, daily_building, typical


def preprocess_and_save(cfg: PreprocessConfig) -> Dict[str, str]:
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    df = load_raw(cfg)
    daily_appliance, daily_home, daily_building, typical = build_daily_tables(df)

    daily_appliance_path = os.path.join(cfg.artifacts_dir, "daily_appliance.pkl")
    daily_home_path = os.path.join(cfg.artifacts_dir, "daily_home.pkl")
    daily_building_path = os.path.join(cfg.artifacts_dir, "daily_building.pkl")
    typical_path = os.path.join(cfg.artifacts_dir, "typical_temp_by_doy.pkl")

    daily_appliance.to_pickle(daily_appliance_path)
    daily_home.to_pickle(daily_home_path)
    daily_building.to_pickle(daily_building_path)
    typical.to_pickle(typical_path)

    appliances_by_home = (
        daily_appliance.groupby("Home ID")["Appliance Type"]
        .unique()
        .apply(list)
        .to_dict()
    )
    appliances_by_home_path = os.path.join(cfg.artifacts_dir, "appliances_by_home.pkl")
    pd.to_pickle(appliances_by_home, appliances_by_home_path)

    return {
        "daily_appliance": daily_appliance_path,
        "daily_home": daily_home_path,
        "daily_building": daily_building_path,
        "typical_temp": typical_path,
        "appliances_by_home": appliances_by_home_path,
    }


if __name__ == "__main__":
    # Default paths for your current workspace layout
    preprocess_and_save(
        PreprocessConfig(
            smart_home_csv="smart_home_energy_consumption.csv",
            appliance_usage_csv="appliance_usage.csv",
            artifacts_dir="artifacts",
        )
    )

