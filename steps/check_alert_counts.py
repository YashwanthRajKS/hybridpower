
from pathlib import Path
import pandas as pd

OUTPUT = Path(__file__).resolve().parents[1] / "outputs"

configs = [
    (
        "Synthetic",
        "synthetic_hourly_forecast_results.csv",
        "mix_kw"
    ),
    (
        "NASA-based",
        "nasa_hourly_forecast_results.csv",
        "gb_kw"
    )
]

for name, filename, forecast_column in configs:

    df = pd.read_csv(OUTPUT / filename)

    forecast_error = (
        df[forecast_column] -
        df["actual_renewable_kw"]
    )

    actual_deficit = (
        df["load_kw"] -
        df["actual_renewable_kw"]
    )

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    for threshold in [2, 5, 10]:

        alerts = (
            (forecast_error >= threshold) &
            (actual_deficit > 0)
        )

        total = int(alerts.sum())
        training = int(alerts.iloc[:248].sum())
        testing = int(alerts.iloc[248:].sum())

        print(
            f"Threshold: {threshold} kW | "
            f"Total: {total} | "
            f"Training: {training} | "
            f"Testing: {testing}"
        )
