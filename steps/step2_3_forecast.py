# STEP 2 + 3: Baseline and AI forecaster (predict PV and wind power 1 hour ahead)
# Needs microgrid_data.csv from Step 1 in the same folder.
# Output: forecast_results.csv, forecast_plot.png, metrics.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

HORIZON = 4                      # 4 steps x 15 min = forecast 1 hour ahead
TRAIN_DAYS = 45                  # first 45 days train, last 15 days test (never mixed)

df = pd.read_csv("microgrid_data.csv", parse_dates=["time"])
step = len(df) // 60             # samples per day (96)

# ---------- build features (only information known at time t) ----------
X = pd.DataFrame({"time": df.time})
for col in ["irradiance_wm2", "wind_ms", "temp_c", "pv_kw", "wind_kw"]:
    X[col] = df[col]
    for lag in (1, 2, 4):                       # values 15, 30 and 60 min ago
        X[f"{col}_lag{lag}"] = df[col].shift(lag)
X["irr_change"] = df.irradiance_wm2 - df.irradiance_wm2.shift(1)
X["wind_change"] = df.wind_ms - df.wind_ms.shift(1)
target_time = df.time + pd.Timedelta(minutes=15 * HORIZON)   # the time we forecast for
hr = target_time.dt.hour + target_time.dt.minute / 60
X["hour_sin"] = np.sin(2 * np.pi * hr / 24)
X["hour_cos"] = np.cos(2 * np.pi * hr / 24)

y_pv = df.pv_kw.shift(-HORIZON)                 # what PV will be in 1 hour
y_wd = df.wind_kw.shift(-HORIZON)               # what wind will be in 1 hour

ok = X.notna().all(axis=1) & y_pv.notna() & y_wd.notna()
X, y_pv, y_wd, d = X[ok], y_pv[ok], y_wd[ok], df[ok]
feats = [c for c in X.columns if c != "time"]

split = TRAIN_DAYS * step
Xtr, Xte = X[feats].iloc[:split], X[feats].iloc[split:]
def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

rows = []
preds = {}
for name, y, now_col in [("PV", y_pv, "pv_kw"), ("Wind", y_wd, "wind_kw")]:
    ytr, yte = y.iloc[:split], y.iloc[split:]
    persist = Xte[now_col].values                       # baseline: "same as now"
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=0).fit(Xtr, ytr)
    p_rf, p_gb = rf.predict(Xte), gb.predict(Xte)
    preds[name] = {"actual": yte.values, "persistence": persist, "random_forest": p_rf, "gradient_boosting": p_gb}
    for mname, p in [("Persistence (baseline)", persist), ("Random forest", p_rf), ("Gradient boosting", p_gb)]:
        rows.append({"target": name, "model": mname,
                     "MAE_kW": round(mean_absolute_error(yte, p), 2), "RMSE_kW": round(rmse(yte, p), 2)})

# ---------- combined renewable forecast (PV + wind) ----------
tot = {k: preds["PV"][k] + preds["Wind"][k] for k in preds["PV"]}
for mname, key in [("Persistence (baseline)", "persistence"), ("Random forest", "random_forest"),
                   ("Gradient boosting", "gradient_boosting")]:
    rows.append({"target": "PV+Wind", "model": mname,
                 "MAE_kW": round(mean_absolute_error(tot["actual"], tot[key]), 2),
                 "RMSE_kW": round(rmse(tot["actual"], tot[key]), 2)})

# best mix: use the best model for each part (PV from random forest, wind from persistence)
mix = preds["PV"]["random_forest"] + preds["Wind"]["persistence"]
tot["mix"] = mix
rows.append({"target": "PV+Wind", "model": "Best mix (PV random forest + wind persistence)",
             "MAE_kW": round(mean_absolute_error(tot["actual"], mix), 2),
             "RMSE_kW": round(rmse(tot["actual"], mix), 2)})

metrics = pd.DataFrame(rows)
metrics.to_csv("metrics.csv", index=False)
print(metrics.to_string(index=False))

base = metrics[(metrics.target == "PV+Wind") & (metrics.model.str.startswith("Persistence"))].MAE_kW.iloc[0]
best = metrics[(metrics.target == "PV+Wind") & (~metrics.model.str.startswith("Persistence"))].sort_values("MAE_kW").iloc[0]
print("\nBest model for PV+Wind: %s, MAE %.2f kW vs baseline %.2f kW  -> %.0f %% lower error"
      % (best.model, best.MAE_kW, base, 100 * (base - best.MAE_kW) / base))

# ---------- save results for the next step ----------
res = pd.DataFrame({
    "time": (X.time.iloc[split:] + pd.Timedelta(minutes=15 * HORIZON)).values,
    "load_kw": df.load_kw.shift(-HORIZON)[ok].iloc[split:].values,
    "actual_renewable_kw": tot["actual"],
    "persistence_kw": tot["persistence"],
    "rf_kw": tot["random_forest"],
    "gb_kw": tot["gradient_boosting"],
    "mix_kw": tot["mix"],
})
res.to_csv("forecast_results.csv", index=False)

# ---------- plot 2 days of the test period ----------
p = res.iloc[96 * 3: 96 * 5]
plt.figure(figsize=(10, 4.5))
plt.plot(p.time, p.actual_renewable_kw, label="Actual PV+wind", color="black")
plt.plot(p.time, p.persistence_kw, label="Baseline (same as 1 h ago)", color="grey", ls="--")
plt.plot(p.time, p.mix_kw, label="AI forecast (PV model + wind persistence)", color="green")
plt.ylabel("Power (kW)"); plt.legend(); plt.title("Forecast 1 hour ahead vs actual (2 test days)")
plt.tight_layout(); plt.savefig("forecast_plot.png", dpi=120)
print("Saved forecast_results.csv, metrics.csv, forecast_plot.png")
