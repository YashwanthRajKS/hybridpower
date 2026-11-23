
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

TRAIN_DAYS = 45
HOURS_PER_DAY = 24
HORIZON = 1

DATASETS = {
    "Synthetic": OUTPUTS / "synthetic_microgrid_hourly.csv",
    "NASA-based": OUTPUTS / "nasa_microgrid_hourly.csv"
}


def rmse(actual, predicted):
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def evaluate_dataset(dataset_name, file_path):

    print("\n" + "=" * 60)
    print(f"HOURLY FORECASTING: {dataset_name}")
    print("=" * 60)

    df = pd.read_csv(file_path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    required = [
        "time", "irradiance_wm2", "wind_ms", "temp_c",
        "pv_kw", "wind_kw", "load_kw"
    ]

    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: Missing columns: {missing_columns}"
        )

    if df[required].isna().any().any():
        raise ValueError(f"{dataset_name}: Missing values detected.")

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    X = pd.DataFrame({"time": df["time"]})

    weather_power_columns = [
        "irradiance_wm2",
        "wind_ms",
        "temp_c",
        "pv_kw",
        "wind_kw"
    ]

    for col in weather_power_columns:

        X[col] = df[col]

        # Previous 1, 2 and 4 hours
        for lag in (1, 2, 4):
            X[f"{col}_lag{lag}"] = df[col].shift(lag)

    X["irr_change"] = df["irradiance_wm2"].diff()
    X["wind_change"] = df["wind_ms"].diff()

    # Forecast the next hour
    target_time = df["time"] + pd.Timedelta(hours=HORIZON)

    hour = (
        target_time.dt.hour
        + target_time.dt.minute / 60
    )

    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # --------------------------------------------------------
    # TARGET VALUES
    # --------------------------------------------------------

    y_pv = df["pv_kw"].shift(-HORIZON)
    y_wind = df["wind_kw"].shift(-HORIZON)

    actual_renewable = (
        df["pv_kw"] + df["wind_kw"]
    ).shift(-HORIZON)

    target_load = df["load_kw"].shift(-HORIZON)

    valid = (
        X.drop(columns=["time"]).notna().all(axis=1)
        & y_pv.notna()
        & y_wind.notna()
        & actual_renewable.notna()
        & target_load.notna()
    )

    X = X.loc[valid].reset_index(drop=True)
    y_pv = y_pv.loc[valid].reset_index(drop=True)
    y_wind = y_wind.loc[valid].reset_index(drop=True)

    actual_renewable = actual_renewable.loc[valid].reset_index(drop=True)
    target_load = target_load.loc[valid].reset_index(drop=True)

    current_pv = df.loc[valid, "pv_kw"].reset_index(drop=True)
    current_wind = df.loc[valid, "wind_kw"].reset_index(drop=True)

    feature_columns = [
        col for col in X.columns if col != "time"
    ]

    # First 45 days for training, remaining 15 days for testing
    split = TRAIN_DAYS * HOURS_PER_DAY

    if len(X) <= split:
        raise ValueError(
            f"{dataset_name}: Not enough records for the "
            "45-day training and 15-day testing split."
        )

    X_train = X[feature_columns].iloc[:split]
    X_test = X[feature_columns].iloc[split:]

    results = {}
    metric_rows = []

    # --------------------------------------------------------
    # TRAIN PV AND WIND MODELS
    # --------------------------------------------------------

    for target_name, y, current_values in [
        ("PV", y_pv, current_pv),
        ("Wind", y_wind, current_wind)
    ]:

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        # Baseline: next hour equals current hour
        persistence = current_values.iloc[split:].values

        rf = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=0
        )

        gb = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            random_state=0
        )

        print(f"\nTraining {target_name} models...")

        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)

        predictions = {
            "persistence": persistence,
            "random_forest": rf.predict(X_test),
            "gradient_boosting": gb.predict(X_test)
        }

        actual = y_test.values

        results[target_name] = {
            "actual": actual,
            **predictions
        }

        for model_name, prediction in [
            ("Persistence", predictions["persistence"]),
            ("Random Forest", predictions["random_forest"]),
            ("Gradient Boosting", predictions["gradient_boosting"])
        ]:

            metric_rows.append({
                "Dataset": dataset_name,
                "Target": target_name,
                "Model": model_name,
                "MAE_kW": round(
                    mean_absolute_error(actual, prediction), 3
                ),
                "RMSE_kW": round(
                    rmse(actual, prediction), 3
                )
            })

    # --------------------------------------------------------
    # COMBINED PV + WIND FORECAST
    # --------------------------------------------------------

    combined = {}

    for model_name in [
        "persistence",
        "random_forest",
        "gradient_boosting"
    ]:

        combined[model_name] = (
            results["PV"][model_name]
            + results["Wind"][model_name]
        )

    combined_actual = (
        results["PV"]["actual"]
        + results["Wind"]["actual"]
    )

    # Best mix: RF for PV + persistence for wind
    combined["best_mix"] = (
        results["PV"]["random_forest"]
        + results["Wind"]["persistence"]
    )

    combined_models = [
        ("Persistence", "persistence"),
        ("Random Forest", "random_forest"),
        ("Gradient Boosting", "gradient_boosting"),
        ("Best Mix (RF PV + Persistence Wind)", "best_mix")
    ]

    for model_name, key in combined_models:

        prediction = combined[key]

        metric_rows.append({
            "Dataset": dataset_name,
            "Target": "PV+Wind",
            "Model": model_name,
            "MAE_kW": round(
                mean_absolute_error(combined_actual, prediction), 3
            ),
            "RMSE_kW": round(
                rmse(combined_actual, prediction), 3
            )
        })

    # --------------------------------------------------------
    # SAVE FORECAST RESULTS
    # --------------------------------------------------------

    forecast_output = pd.DataFrame({
        "time": (
            X["time"].iloc[split:]
            + pd.Timedelta(hours=HORIZON)
        ).values,

        "load_kw": target_load.iloc[split:].values,
        "actual_renewable_kw": combined_actual,

        "persistence_kw": combined["persistence"],
        "rf_kw": combined["random_forest"],
        "gb_kw": combined["gradient_boosting"],
        "mix_kw": combined["best_mix"]
    })

    output_name = (
        "synthetic_hourly_forecast_results.csv"
        if dataset_name == "Synthetic"
        else "nasa_hourly_forecast_results.csv"
    )

    forecast_output.to_csv(
        OUTPUTS / output_name,
        index=False
    )

    print(f"\nSaved: outputs/{output_name}")

    return metric_rows


# ============================================================
# RUN BOTH DATASETS
# ============================================================

all_metrics = []

for dataset_name, file_path in DATASETS.items():

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    all_metrics.extend(
        evaluate_dataset(dataset_name, file_path)
    )


# Save combined metrics
metrics_df = pd.DataFrame(all_metrics)

metrics_path = OUTPUTS / "hourly_forecast_metrics.csv"

metrics_df.to_csv(metrics_path, index=False)

print("\n" + "=" * 70)
print("HOURLY FORECASTING COMPARISON COMPLETED")
print("=" * 70)

print(metrics_df.to_string(index=False))

print(f"\nMetrics saved: {metrics_path}")
print("\nBoth datasets were evaluated separately using the same model settings.")
