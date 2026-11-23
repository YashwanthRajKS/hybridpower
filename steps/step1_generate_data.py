# STEP 1: Generate a realistic synthetic dataset for the hybrid microgrid
# Run in Google Colab or any Python 3 (needs numpy, pandas, matplotlib).
# Output: microgrid_data.csv and data_preview.png

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)          # fixed seed so you get the same data every time

# ---------- settings you can change ----------
DAYS = 60                                # how many days of data
STEP_MIN = 15                            # minutes between samples
PV_RATED_KW = 100                        # PV size (your Simulink PV gave about 85-100 kW at 1000 W/m2)
WIND_RATED_KW = 120                      # wind turbine rated power
BASE_LOAD_KW = 50                        # average village load
# ---------------------------------------------

n = DAYS * 24 * 60 // STEP_MIN
t = pd.date_range("2026-01-01", periods=n, freq=f"{STEP_MIN}min")
hour = t.hour + t.minute / 60

# --- irradiance (W/m2): clear-sky bell shape times a cloud factor ---
clear = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) * 1000
cloud = np.ones(n)
i = 0
while i < n:
    if rng.random() < 0.004:                         # a cloud event starts
        length = rng.integers(2, 12)                 # lasts 30 min to 3 h
        depth = rng.uniform(0.25, 0.7)               # irradiance falls to 25-70 percent
        cloud[i:i + length] = depth
        i += length
    i += 1
cloud = np.convolve(cloud, np.ones(2) / 2, mode="same")      # soften the edges slightly
irr = clear * cloud * (1 + rng.normal(0, 0.02, n))
irr = np.clip(irr, 0, 1100)

# --- temperature (deg C) ---
temp = 24 + 7 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.4, n)

# --- wind speed (m/s): slowly changing random process ---
wind = np.zeros(n)
wind[0] = 6.0
for k in range(1, n):
    wind[k] = 0.97 * wind[k - 1] + 0.03 * 6.0 + rng.normal(0, 0.45)
wind = np.clip(wind + 1.0 * np.sin(2 * np.pi * (hour - 15) / 24), 0, 25)

# --- load (kW): morning and evening peaks ---
load = (BASE_LOAD_KW
        + 18 * np.exp(-((hour - 8) ** 2) / 4)
        + 30 * np.exp(-((hour - 19.5) ** 2) / 5)
        + rng.normal(0, 3, n))

# --- convert weather to power ---
t_cell = temp + 0.03 * irr
pv_kw = PV_RATED_KW * (irr / 1000) * (1 - 0.004 * (t_cell - 25))
pv_kw = np.clip(pv_kw, 0, PV_RATED_KW)

v_in, v_rated, v_out = 3.0, 12.0, 25.0                 # wind turbine curve
frac = np.where(wind < v_in, 0.0,
        np.where(wind < v_rated, (wind**3 - v_in**3) / (v_rated**3 - v_in**3),
        np.where(wind < v_out, 1.0, 0.0)))
wind_kw = WIND_RATED_KW * frac

df = pd.DataFrame({
    "time": t, "irradiance_wm2": irr.round(1), "temp_c": temp.round(2),
    "wind_ms": wind.round(2), "load_kw": load.round(2),
    "pv_kw": pv_kw.round(2), "wind_kw": wind_kw.round(2),
})
df["renewable_kw"] = (df.pv_kw + df.wind_kw).round(2)
df["deficit_kw"] = (df.load_kw - df.renewable_kw).clip(lower=0).round(2)   # what diesel must cover
df.to_csv("microgrid_data.csv", index=False)

print("Rows:", len(df))
print(df.describe().loc[["mean", "min", "max"]].round(1).T)
print("Share of time with a deficit (diesel needed): %.0f %%" % (100 * (df.deficit_kw > 0).mean()))

# --- preview plot: first 3 days ---
d = df.iloc[: 3 * 96]
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax[0].plot(d.time, d.irradiance_wm2, color="orange"); ax[0].set_ylabel("Irradiance (W/m2)")
ax[1].plot(d.time, d.wind_ms, color="teal"); ax[1].set_ylabel("Wind (m/s)")
ax[2].plot(d.time, d.load_kw, label="Load", color="black")
ax[2].plot(d.time, d.renewable_kw, label="PV + wind", color="green")
ax[2].fill_between(d.time, d.renewable_kw, d.load_kw, where=d.load_kw > d.renewable_kw,
                   color="red", alpha=0.3, label="Deficit (diesel)")
ax[2].set_ylabel("Power (kW)"); ax[2].legend()
plt.tight_layout(); plt.savefig("data_preview.png", dpi=120)
print("Saved microgrid_data.csv and data_preview.png")
