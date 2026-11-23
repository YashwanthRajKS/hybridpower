
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SYNTHETIC_FILE = OUTPUT_DIR / "microgrid_data.csv"
NASA_FILE = OUTPUT_DIR / "nasa_power_bengaluru_hourly.csv"

# Same system ratings as the original synthetic generator
PV_RATED_KW = 100
WIND_RATED_KW = 120

# --------------------------------------------------
# CHECK INPUT FILES
# --------------------------------------------------
if not SYNTHETIC_FILE.exists():
    raise FileNotFoundError(
        f"Synthetic dataset not found: {SYNTHETIC_FILE}"
    )

if not NASA_FILE.exists():
    raise FileNotFoundError(
        f"NASA weather dataset not found: {NASA_FILE}"
    )

print("Loading synthetic and NASA datasets...")

synthetic = pd.read_csv(
    SYNTHETIC_FILE,
    parse_dates=["time"]
)

nasa = pd.read_csv(NASA_FILE)

# --------------------------------------------------
# 1. CREATE HOURLY SYNTHETIC DATASET
# --------------------------------------------------
print("\nPreparing hourly synthetic dataset...")

synthetic["time"] = pd.to_datetime(synthetic["time"])

synthetic_hourly = (
    synthetic
    .set_index("time")
    .resample("1h")
    .mean(numeric_only=True)
    .dropna()
    .reset_index()
)

synthetic_hourly["renewable_kw"] = (
    synthetic_hourly["pv_kw"] +
    synthetic_hourly["wind_kw"]
)

synthetic_hourly["deficit_kw"] = (
    synthetic_hourly["load_kw"] -
    synthetic_hourly["renewable_kw"]
).clip(lower=0)

synthetic_output = (
    OUTPUT_DIR / "synthetic_microgrid_hourly.csv"
)

synthetic_hourly.to_csv(
    synthetic_output,
    index=False
)

# --------------------------------------------------
# 2. PREPARE NASA WEATHER DATA
# --------------------------------------------------
print("Preparing NASA hourly weather dataset...")

# Convert timestamps to Indian Standard Time.
# Remove timezone information only after conversion,
# so they can align with the synthetic local timestamps.
nasa["time"] = (
    pd.to_datetime(nasa["time"], utc=True)
    .dt.tz_convert("Asia/Kolkata")
    .dt.tz_localize(None)
)

nasa = nasa.sort_values("time").reset_index(drop=True)

# Rename NASA weather variables to match the original
# synthetic dataset's column names.
nasa_microgrid = pd.DataFrame({
    "time": nasa["time"],
    "irradiance_wm2": nasa["irradiance"],
    "temp_c": nasa["temperature"],
    "wind_ms": nasa["wind_speed"]
})

# --------------------------------------------------
# 3. ESTIMATE PV POWER
# Same conversion equation as Step 1.
# --------------------------------------------------
irr = nasa_microgrid["irradiance_wm2"].to_numpy()
temp = nasa_microgrid["temp_c"].to_numpy()

t_cell = temp + 0.03 * irr

pv_kw = (
    PV_RATED_KW *
    (irr / 1000) *
    (1 - 0.004 * (t_cell - 25))
)

nasa_microgrid["pv_kw"] = np.clip(
    pv_kw,
    0,
    PV_RATED_KW
)

# --------------------------------------------------
# 4. ESTIMATE WIND POWER
# Same wind turbine power curve as Step 1.
# --------------------------------------------------
wind = nasa_microgrid["wind_ms"].to_numpy()

v_in = 3.0
v_rated = 12.0
v_out = 25.0

wind_fraction = np.where(
    wind < v_in,
    0.0,
    np.where(
        wind < v_rated,
        (wind**3 - v_in**3) /
        (v_rated**3 - v_in**3),
        np.where(wind < v_out, 1.0, 0.0)
    )
)

nasa_microgrid["wind_kw"] = (
    WIND_RATED_KW * wind_fraction
)

# --------------------------------------------------
# 5. ALIGN THE SYNTHETIC LOAD PROFILE
# NASA does not provide our village microgrid load.
# Reuse the synthetic load profile for comparison.
# --------------------------------------------------
load_series = (
    synthetic
    .set_index("time")["load_kw"]
    .sort_index()
)

target_times = pd.DatetimeIndex(
    nasa_microgrid["time"]
)

combined_index = load_series.index.union(
    target_times
).sort_values()

aligned_load = (
    load_series
    .reindex(combined_index)
    .interpolate(method="time")
    .reindex(target_times)
)

nasa_microgrid["load_kw"] = aligned_load.to_numpy()

# --------------------------------------------------
# 6. CALCULATE RENEWABLE POWER AND DEFICIT
# --------------------------------------------------
nasa_microgrid["renewable_kw"] = (
    nasa_microgrid["pv_kw"] +
    nasa_microgrid["wind_kw"]
)

nasa_microgrid["deficit_kw"] = (
    nasa_microgrid["load_kw"] -
    nasa_microgrid["renewable_kw"]
).clip(lower=0)

# Remove rows with missing values
nasa_microgrid = (
    nasa_microgrid
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index(drop=True)
)

# Arrange columns consistently
columns = [
    "time",
    "irradiance_wm2",
    "temp_c",
    "wind_ms",
    "load_kw",
    "pv_kw",
    "wind_kw",
    "renewable_kw",
    "deficit_kw"
]

nasa_microgrid = nasa_microgrid[columns]

# --------------------------------------------------
# 7. SAVE NASA-BASED MICROGRID DATASET
# --------------------------------------------------
nasa_output = (
    OUTPUT_DIR / "nasa_microgrid_hourly.csv"
)

nasa_microgrid.to_csv(
    nasa_output,
    index=False
)

# --------------------------------------------------
# 8. DISPLAY RESULTS
# --------------------------------------------------
print("\n======================================")
print("HOURLY DATA PREPARATION COMPLETED")
print("======================================")

print("\nSynthetic hourly dataset:")
print("Rows:", len(synthetic_hourly))
print("Saved:", synthetic_output)

print("\nNASA-based hourly dataset:")
print("Rows:", len(nasa_microgrid))
print("Saved:", nasa_output)

print("\nNASA dataset preview:")
print(nasa_microgrid.head().to_string(index=False))

print("\nNASA dataset statistics:")
print(
    nasa_microgrid[
        [
            "irradiance_wm2",
            "wind_ms",
            "temp_c",
            "pv_kw",
            "wind_kw",
            "load_kw",
            "renewable_kw"
        ]
    ].describe().round(2)
)

print("\nMissing values:")
print(nasa_microgrid.isna().sum())

print("\nBoth datasets are now prepared.")
