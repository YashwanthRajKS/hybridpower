
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

RATED_KW = 100
MIN_LOAD_KW = 30
STARTUP_DELAY_HOURS = 1
MIN_RUN_HOURS = 1

THRESHOLD_KW = 2
SAFE_FACTOR = 0.95

F0 = 0.246
F1 = 0.08415
CO2_PER_LITRE = 2.68

DATASETS = {
    "Synthetic": {
        "file": "synthetic_hourly_forecast_results.csv",
        "forecast_column": "mix_kw"
    },
    "NASA-based": {
        "file": "nasa_hourly_forecast_results.csv",
        "forecast_column": "gb_kw"
    }
}


# ============================================================
# DISPATCH SIMULATION
# ============================================================

def simulate_dispatch(load, renewable, forecast, strategy):

    n = len(load)

    state = "off"
    pending = 0
    run_hours = 0

    diesel_output = np.zeros(n)
    unserved_power = np.zeros(n)
    generator_status = np.zeros(n)

    starts = 0
    fuel_litres = 0.0
    diesel_energy = 0.0

    for i in range(n):

        # Complete the startup process
        if state == "starting":

            pending -= 1

            if pending <= 0:
                state = "on"
                run_hours = 0

        actual_deficit = max(0, load[i] - renewable[i])

        # ----------------------------------------------------
        # GENERATOR IS OFF
        # ----------------------------------------------------

        if state == "off":

            need_to_start = False

            if strategy == "Reactive (No AI)":

                # Start only after observing the current deficit
                need_to_start = (
                    actual_deficit > THRESHOLD_KW
                )

            elif strategy in [
                "AI Forecast",
                "Perfect Forecast"
            ]:

                # Look one hour ahead
                j = i + STARTUP_DELAY_HOURS

                if j < n:

                    if strategy == "AI Forecast":

                        predicted_renewable = max(
                            0, forecast[j]
                        )

                    else:

                        # Perfect forecast is an ideal benchmark,
                        # not a deployable forecasting model.
                        predicted_renewable = renewable[j]

                    predicted_deficit = (
                        load[j]
                        - SAFE_FACTOR * predicted_renewable
                    )

                    if strategy == "Perfect Forecast":

                        predicted_deficit = (
                            load[j] - renewable[j]
                        )

                    need_to_start = (
                        predicted_deficit > THRESHOLD_KW
                    )

            if need_to_start:

                state = "starting"
                pending = STARTUP_DELAY_HOURS
                starts += 1

        # ----------------------------------------------------
        # GENERATOR IS ONLINE
        # ----------------------------------------------------

        if state == "on":

            required_power = actual_deficit

            output = min(
                RATED_KW,
                max(MIN_LOAD_KW, required_power)
            )

            diesel_output[i] = output
            generator_status[i] = 1

            remaining_deficit = max(
                0, actual_deficit - output
            )

            unserved_power[i] = remaining_deficit

            # One-hour time interval
            diesel_energy += output * 1.0
            fuel_litres += (
                F0 * RATED_KW + F1 * output
            ) * 1.0

            run_hours += 1

            # Stop only after the minimum run time
            if (
                run_hours >= MIN_RUN_HOURS
                and actual_deficit <= THRESHOLD_KW
            ):
                state = "off"
                run_hours = 0

        else:

            # No diesel output during startup or when switched off
            unserved_power[i] = actual_deficit

    total_unserved = float(np.sum(unserved_power))
    total_fuel = float(fuel_litres)
    total_diesel = float(diesel_energy)

    co2 = total_fuel * CO2_PER_LITRE

    return {
        "starts": starts,
        "unserved_kwh": total_unserved,
        "fuel_litres": total_fuel,
        "diesel_kwh": total_diesel,
        "co2_kg": co2,
        "diesel_output": diesel_output,
        "unserved_power": unserved_power,
        "generator_status": generator_status
    }


# ============================================================
# RUN BOTH DATASETS
# ============================================================

summary_rows = []

for dataset_name, config in DATASETS.items():

    print("\n" + "=" * 65)
    print(f"HOURLY DISPATCH SIMULATION: {dataset_name}")
    print("=" * 65)

    file_path = OUTPUTS / config["file"]

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    df = pd.read_csv(
        file_path,
        parse_dates=["time"]
    )

    required = [
        "time",
        "load_kw",
        "actual_renewable_kw",
        config["forecast_column"]
    ]

    if df[required].isna().any().any():
        raise ValueError(
            f"Missing values found in {dataset_name} forecast data."
        )

    load = df["load_kw"].to_numpy(dtype=float)

    renewable = df[
        "actual_renewable_kw"
    ].to_numpy(dtype=float)

    forecast = np.maximum(
        0,
        df[config["forecast_column"]].to_numpy(dtype=float)
    )

    strategies = [
        "Reactive (No AI)",
        "AI Forecast",
        "Perfect Forecast"
    ]

    detailed_results = []

    for strategy in strategies:

        result = simulate_dispatch(
            load,
            renewable,
            forecast,
            strategy
        )

        summary_rows.append({
            "Dataset": dataset_name,
            "Strategy": strategy,
            "Unserved Energy (kWh)": round(
                result["unserved_kwh"], 3
            ),
            "Diesel Fuel (L)": round(
                result["fuel_litres"], 3
            ),
            "Diesel Energy (kWh)": round(
                result["diesel_kwh"], 3
            ),
            "Diesel Starts": result["starts"],
            "CO2 Emissions (kg)": round(
                result["co2_kg"], 3
            )
        })

        detailed_results.append({
            "time": df["time"],
            "strategy": strategy,
            "load_kw": load,
            "renewable_kw": renewable,
            "forecast_kw": forecast,
            "diesel_output_kw": result["diesel_output"],
            "unserved_kw": result["unserved_power"],
            "generator_on": result["generator_status"]
        })

        print(f"\n{strategy}")
        print(
            f"Unserved energy: "
            f"{result['unserved_kwh']:.2f} kWh"
        )
        print(
            f"Fuel consumed: "
            f"{result['fuel_litres']:.2f} litres"
        )
        print(
            f"Diesel starts: {result['starts']}"
        )
        print(
            f"CO2 emissions: {result['co2_kg']:.2f} kg"
        )

    # Save hourly details for all three strategies
    detail_frames = []

    for item in detailed_results:

        temp = pd.DataFrame({
            "time": item["time"],
            "strategy": item["strategy"],
            "load_kw": item["load_kw"],
            "renewable_kw": item["renewable_kw"],
            "forecast_kw": item["forecast_kw"],
            "diesel_output_kw": item["diesel_output_kw"],
            "unserved_kw": item["unserved_kw"],
            "generator_on": item["generator_on"]
        })

        detail_frames.append(temp)

    detail_df = pd.concat(
        detail_frames,
        ignore_index=True
    )

    detail_name = (
        f"hourly_dispatch_details_"
        f"{dataset_name.lower().replace('-', '_')}.csv"
    )

    detail_df.to_csv(
        OUTPUTS / detail_name,
        index=False
    )

    print(f"\nSaved: outputs/{detail_name}")


# ============================================================
# SAVE FINAL COMPARISON
# ============================================================

summary_df = pd.DataFrame(summary_rows)

summary_path = OUTPUTS / "hourly_dispatch_summary.csv"

summary_df.to_csv(
    summary_path,
    index=False
)

print("\n" + "=" * 75)
print("HOURLY DIESEL DISPATCH COMPARISON COMPLETED")
print("=" * 75)

print(summary_df.to_string(index=False))

print(f"\nSummary saved: {summary_path}")
