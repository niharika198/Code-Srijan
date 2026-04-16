import os
import sys
from dataclasses import dataclass

import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import PreprocessConfig, preprocess_and_save


@dataclass(frozen=True)
class TrainConfig:
    artifacts_dir: str = "artifacts"
    models_dir: str = "models"
    smart_home_csv: str = "smart_home_energy_consumption.csv"


def _time_split_by_date(df: pd.DataFrame, date_col: str = "Date", train_ratio: float = 0.8):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    dates = sorted(df[date_col].dropna().unique())
    cutoff_idx = int(len(dates) * train_ratio)
    train_dates = set(dates[:cutoff_idx])
    is_train = df[date_col].isin(train_dates)
    return df[is_train], df[~is_train]


def _train_sklearn_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    cat_cols: list,
    model_path: str,
):
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    pipeline.fit(X_train, y_train)
    pred_val = pipeline.predict(X_val)
    mae = float(mean_absolute_error(y_val, pred_val))

    dump(pipeline, model_path)
    return mae


def main(cfg: TrainConfig) -> None:
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.artifacts_dir, exist_ok=True)

    daily_appliance_path = os.path.join(cfg.artifacts_dir, "daily_appliance.pkl")
    daily_home_path = os.path.join(cfg.artifacts_dir, "daily_home.pkl")
    daily_building_path = os.path.join(cfg.artifacts_dir, "daily_building.pkl")

    if not (
        os.path.exists(daily_appliance_path)
        and os.path.exists(daily_home_path)
        and os.path.exists(daily_building_path)
    ):
        preprocess_and_save(
            PreprocessConfig(
                smart_home_csv=cfg.smart_home_csv,
                artifacts_dir=cfg.artifacts_dir,
            )
        )

    daily_appliance = pd.read_pickle(daily_appliance_path)
    daily_home = pd.read_pickle(daily_home_path)
    daily_building = pd.read_pickle(daily_building_path)

    # Ensure categorical are strings
    for col in ["Home ID", "Appliance Type", "season"]:
        if col in daily_appliance.columns:
            daily_appliance[col] = daily_appliance[col].astype(str)
    for col in ["Home ID", "season"]:
        if col in daily_home.columns:
            daily_home[col] = daily_home[col].astype(str)
    for col in ["season"]:
        if col in daily_building.columns:
            daily_building[col] = daily_building[col].astype(str)

    date_features = ["month", "day_of_week", "day_of_year", "month_sin", "month_cos", "doy_sin", "doy_cos"]

    # 1) Appliance-wise day kWh model
    appliance_feature_cols = ["Home ID", "Appliance Type", "avg_temp", "season", "household_size"] + date_features
    appliance_target = "kwh_day"
    appliance_cat_cols = ["Home ID", "Appliance Type", "season"]
    appliance_train, appliance_val = _time_split_by_date(daily_appliance)
    appliance_model_path = os.path.join(cfg.models_dir, "appliance_day_model.joblib")
    appliance_mae = _train_sklearn_pipeline(
        train_df=appliance_train,
        val_df=appliance_val,
        feature_cols=appliance_feature_cols,
        target_col=appliance_target,
        cat_cols=appliance_cat_cols,
        model_path=appliance_model_path,
    )
    print(f"Appliance-day MAE (val): {appliance_mae:.4f}")

    # 2) Home-wise day kWh model
    home_feature_cols = ["Home ID", "avg_temp", "season", "household_size"] + date_features
    home_target = "kwh_day"
    home_cat_cols = ["Home ID", "season"]
    home_train, home_val = _time_split_by_date(daily_home)
    home_model_path = os.path.join(cfg.models_dir, "home_day_model.joblib")
    home_mae = _train_sklearn_pipeline(
        train_df=home_train,
        val_df=home_val,
        feature_cols=home_feature_cols,
        target_col=home_target,
        cat_cols=home_cat_cols,
        model_path=home_model_path,
    )
    print(f"Home-day MAE (val): {home_mae:.4f}")

    # 3) Building-wise day kWh model
    building_feature_cols = ["avg_temp", "season"] + date_features
    building_target = "kwh_day"
    building_cat_cols = ["season"]
    building_train, building_val = _time_split_by_date(daily_building)
    building_model_path = os.path.join(cfg.models_dir, "building_day_model.joblib")
    building_mae = _train_sklearn_pipeline(
        train_df=building_train,
        val_df=building_val,
        feature_cols=building_feature_cols,
        target_col=building_target,
        cat_cols=building_cat_cols,
        model_path=building_model_path,
    )
    print(f"Building-day MAE (val): {building_mae:.4f}")


if __name__ == "__main__":
    main(TrainConfig())

