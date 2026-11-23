
import requests
import pandas as pd
from pathlib import Path

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Bengaluru coordinates
LATITUDE = 12.9716
LONGITUDE = 77.5946

# Historical weather period
START_DATE = "20260101"
END_DATE = "20260301"

# NASA POWER hourly API
url = "https://power.larc.nasa.gov/api/temporal/hourly/point"

params = {
    "parameters": "ALLSKY_SFC_SW_DWN,WS10M,T2M",
    "community": "RE",
    "longitude": LONGITUDE,
    "latitude": LATITUDE,
    "start": START_DATE,
    "end": END_DATE,
    "format": "JSON",
    "time-standard": "UTC"
}

print("Downloading Bengaluru weather data from NASA POWER...")

try:
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()

    result = response.json()
    weather = result["properties"]["parameter"]

    df = pd.DataFrame({
        "time_utc": pd.to_datetime(
            list(weather["ALLSKY_SFC_SW_DWN"].keys()),
            format="%Y%m%d%H",
            utc=True
        ),
        "irradiance": list(
            weather["ALLSKY_SFC_SW_DWN"].values()
        ),
        "wind_speed": list(
            weather["WS10M"].values()
        ),
        "temperature": list(
            weather["T2M"].values()
        )
    })

    # Replace NASA missing-data values
    df.replace(-999, float("nan"), inplace=True)

    # Convert UTC timestamps to Indian Standard Time
    df["time"] = df["time_utc"].dt.tz_convert(
        "Asia/Kolkata"
    )

    # Arrange columns
    df = df[
        [
            "time",
            "time_utc",
            "irradiance",
            "wind_speed",
            "temperature"
        ]
    ]

    # Save separately; do not overwrite synthetic data
    output_file = OUTPUT_DIR / "nasa_power_bengaluru_hourly.csv"
    df.to_csv(output_file, index=False)

    print("\nDOWNLOAD SUCCESSFUL")
    print("Total records:", len(df))
    print("\nFirst 5 records:")
    print(df.head().to_string(index=False))

    print("\nWeather data statistics:")
    print(df[
        ["irradiance", "wind_speed", "temperature"]
    ].describe())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nSaved file:")
    print(output_file)

except Exception as error:
    print("\nDOWNLOAD FAILED")
    print("Error:", error)
