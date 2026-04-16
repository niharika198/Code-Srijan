import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load


def _season_from_month(month: int) -> str:
    # Simple hackathon-friendly mapping.
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def _date_features(d: pd.Timestamp) -> Dict[str, float]:
    month = d.month
    day_of_week = d.dayofweek
    day_of_year = d.dayofyear

    return {
        "month": float(month),
        "day_of_week": float(day_of_week),
        "day_of_year": float(day_of_year),
        "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
        "doy_sin": float(np.sin(2 * np.pi * day_of_year / 366.0)),
        "doy_cos": float(np.cos(2 * np.pi * day_of_year / 366.0)),
    }


@dataclass(frozen=True)
class PredictConfig:
    artifacts_dir: str = "artifacts"
    models_dir: str = "models"
    # If True, and the target date exists in your historical data for the given home,
    # we use the stored average temperature instead of typical temp.
    # For hackathon demos, default to False to avoid using the target day's actual weather.
    prefer_actual_weather_if_available: bool = False


class EnergyPredictor:
    def __init__(self, cfg: PredictConfig):
        self.cfg = cfg

        # Load artifacts
        self.daily_appliance = pd.read_pickle(os.path.join(cfg.artifacts_dir, "daily_appliance.pkl"))
        self.daily_home = pd.read_pickle(os.path.join(cfg.artifacts_dir, "daily_home.pkl"))

        raw_appliances_by_home = pd.read_pickle(
            os.path.join(cfg.artifacts_dir, "appliances_by_home.pkl")
        )
        self.appliances_by_home: Dict[str, List[str]] = {
            str(home_id): [str(app) for app in appliances]
            for home_id, appliances in raw_appliances_by_home.items()
        }
        self.typical_temp = pd.read_pickle(os.path.join(cfg.artifacts_dir, "typical_temp_by_doy.pkl"))

        # Load trained model pipelines (scikit-learn + one-hot encoding)
        self.appliance_model = load(os.path.join(cfg.models_dir, "appliance_day_model.joblib"))
        self.home_model = load(os.path.join(cfg.models_dir, "home_day_model.joblib"))
        self.building_model = load(os.path.join(cfg.models_dir, "building_day_model.joblib"))

        # Categorical type safety
        self.daily_appliance["Home ID"] = self.daily_appliance["Home ID"].astype(str)
        self.daily_appliance["Appliance Type"] = self.daily_appliance["Appliance Type"].astype(str)
        self.daily_appliance["season"] = self.daily_appliance["season"].astype(str)
        self.daily_home["Home ID"] = self.daily_home["Home ID"].astype(str)
        self.daily_home["season"] = self.daily_home["season"].astype(str)

        self.household_size_by_home = (
            self.daily_home.groupby("Home ID")["household_size"].first().to_dict()
        )

    def _typical_temp_for_day(self, d: pd.Timestamp) -> float:
        doy = int(d.dayofyear)
        row = self.typical_temp[self.typical_temp["day_of_year"] == doy]
        if row.empty:
            # Fallback: overall median
            return float(self.daily_home["avg_temp"].median())
        return float(row["typical_temp"].iloc[0])

    def _resolve_weather(self, home_id: str, d: pd.Timestamp, expected_temp: Optional[float]) -> float:
        if expected_temp is not None:
            return float(expected_temp)

        if self.cfg.prefer_actual_weather_if_available:
            match = self.daily_home[
                (self.daily_home["Home ID"] == str(home_id)) & (self.daily_home["Date"] == d)
            ]
            if not match.empty:
                return float(match["avg_temp"].iloc[0])

        return self._typical_temp_for_day(d)

    def predict_home_and_building_day(
        self,
        home_id: str,
        target_date: date,
        expected_temp: Optional[float] = None,
    ) -> Dict[str, float]:
        d = pd.to_datetime(target_date)
        season = _season_from_month(d.month)
        avg_temp = self._resolve_weather(home_id, d, expected_temp)

        feats = _date_features(d)
        household_size = float(self.household_size_by_home.get(str(home_id), 3.0))

        # Home prediction
        home_row = pd.DataFrame(
            [
                {
                    "Home ID": str(home_id),
                    "avg_temp": avg_temp,
                    "season": season,
                    "household_size": household_size,
                    **feats,
                }
            ]
        )
        home_pred = float(self.home_model.predict(home_row)[0])

        # Building prediction: building model does not use Home ID
        building_row = pd.DataFrame(
            [
                {
                    "avg_temp": avg_temp,
                    "season": season,
                    **feats,
                }
            ]
        )
        building_pred = float(self.building_model.predict(building_row)[0])

        return {"home_kwh_day": home_pred, "building_kwh_day": building_pred, "season": season, "avg_temp": avg_temp}

    def predict_appliance_day(
        self,
        home_id: str,
        target_date: date,
        expected_temp: Optional[float] = None,
    ) -> Dict[str, float]:
        d = pd.to_datetime(target_date)
        season = _season_from_month(d.month)
        avg_temp = self._resolve_weather(home_id, d, expected_temp)
        feats = _date_features(d)
        household_size = float(self.household_size_by_home.get(str(home_id), 3.0))

        appliances = self.appliances_by_home.get(str(home_id), [])
        if not appliances:
            return {}

        X = []
        for app in appliances:
            X.append(
                {
                    "Home ID": str(home_id),
                    "Appliance Type": str(app),
                    "avg_temp": avg_temp,
                    "season": season,
                    "household_size": household_size,
                    **feats,
                }
            )
        X_df = pd.DataFrame(X)
        preds = self.appliance_model.predict(X_df)

        appliance_pred = {str(app): float(kwh) for app, kwh in zip(appliances, preds)}

        # Calibrate so appliance sum matches home total prediction
        home_info = self.predict_home_and_building_day(home_id, target_date, expected_temp=expected_temp)
        home_total = max(0.0, float(home_info["home_kwh_day"]))
        sum_apps = max(1e-9, sum(max(0.0, v) for v in appliance_pred.values()))
        scale = home_total / sum_apps
        for k in appliance_pred:
            appliance_pred[k] = max(0.0, appliance_pred[k]) * scale

        return appliance_pred

    def predict_home_month_kwh(
        self,
        home_id: str,
        month_date: date,
        expected_temp: Optional[float] = None,
    ) -> float:
        # Sum daily home predictions for that month.
        first = pd.to_datetime(month_date).replace(day=1)
        next_month = (first + pd.offsets.MonthBegin(1)).replace(day=1)
        total = 0.0
        cur = first
        while cur < next_month:
            info = self.predict_home_and_building_day(home_id, cur.date(), expected_temp=expected_temp)
            total += max(0.0, float(info["home_kwh_day"]))
            cur += pd.Timedelta(days=1)
        return float(total)

    def predict_building_month_kwh(
        self,
        month_date: date,
        expected_temp: Optional[float] = None,
    ) -> float:
        first = pd.to_datetime(month_date).replace(day=1)
        next_month = (first + pd.offsets.MonthBegin(1)).replace(day=1)
        total = 0.0
        cur = first
        # For building model, reuse `expected_temp` if provided; otherwise use typical temp by day-of-year.
        while cur < next_month:
            avg_temp = self._typical_temp_for_day(cur) if expected_temp is None else float(expected_temp)
            season = _season_from_month(cur.month)
            feats = _date_features(cur)
            building_row = pd.DataFrame(
                [{"avg_temp": float(avg_temp), "season": season, **feats}]
            )
            total += max(0.0, float(self.building_model.predict(building_row)[0]))
            cur += pd.Timedelta(days=1)
        return float(total)

