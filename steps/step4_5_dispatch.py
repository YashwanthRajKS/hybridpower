# STEP 4 + 5: Diesel dispatch rule and comparison (with AI versus without AI)
# Needs forecast_results.csv from Step 2/3 in the same folder.
# Output: dispatch_summary.csv, sensitivity.csv, dispatch_steps.csv, dispatch_day.png, dispatch_bars.png, sensitivity.png

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- assumptions (change them and re-run) ----------------
DT_H        = 0.25      # one step = 15 minutes = 0.25 h
RATED_KW    = 100       # diesel generator size
MIN_LOAD    = 0.30      # when running, diesel output is at least 30 % of rating
START_MAIN  = 2         # steps (2 x 15 = 30 min) from start order until diesel is online (main case)
MIN_RUN     = 4         # once on, stay on at least 4 steps (1 h) to avoid short cycling
THRESH_KW   = 2.0       # ignore tiny deficits
SAFE_AI     = 0.95      # AI rule: trust only 95 % of the forecast renewable power
F0, F1      = 0.246, 0.08415   # typical diesel fuel curve: L/h = F0*rated + F1*output
CO2_PER_L   = 2.68      # kg CO2 per litre of diesel
# ----------------------------------------------------------------------

df = pd.read_csv("forecast_results.csv", parse_dates=["time"])
load = df.load_kw.values
SAFE_ALERT = 0.80       # when the SMOTE alert fires, trust only 80 % of the forecast (start the diesel earlier)
import os
alert_smote = alert_plain = None
if os.path.exists("alert_results.csv"):
    al = pd.read_csv("alert_results.csv", parse_dates=["time"])
    m = df[["time"]].merge(al, on="time", how="left").fillna(0)
    alert_smote = np.where(m.alert_prob_smote.values >= 0.5, SAFE_ALERT, SAFE_AI)
    alert_plain = np.where(m.alert_prob_plain.values >= 0.5, SAFE_ALERT, SAFE_AI)
ren = df.actual_renewable_kw.values
n = len(df)

def sv(safe, k):
    """safety factor at step k (a single number, or one number per step)"""
    return safe[k] if hasattr(safe, "__len__") else safe

def simulate(mode, fc=None, safe=1.0, S=START_MAIN):
    """mode 'reactive': start only after a deficit is seen.
       mode 'forecast': start early when the forecast says a deficit is coming."""
    state, pending, run_steps = "off", 0, 0
    unserved = fuel = diesel_kwh = hours_on = 0.0
    starts = 0
    on_flag = np.zeros(n); out = np.zeros(n); uns = np.zeros(n)
    for i in range(n):
        deficit = max(0.0, load[i] - ren[i])
        if state == "starting" and pending == 0:
            state, run_steps = "on", 0
        # ---- decisions ----
        if state == "off":
            if mode == "reactive":
                need = deficit > THRESH_KW
            else:
                j = min(i + S, n - 1)
                need = (load[j] - sv(safe, j) * fc[j]) > THRESH_KW
            if need:
                state, pending = "starting", S
                starts += 1
                if S == 0:
                    state, run_steps = "on", 0
        elif state == "on" and run_steps >= MIN_RUN:
            if mode == "reactive":
                stop = deficit <= 0
            else:
                look = range(i, min(i + S + 2, n))
                stop = all((load[k] - sv(safe, k) * fc[k]) <= THRESH_KW for k in look)
            if stop:
                state = "off"
        # ---- what happens in this step ----
        if state == "on":
            on_flag[i] = 1
            output = min(max(deficit, MIN_LOAD * RATED_KW), RATED_KW)
            delivered = min(deficit, RATED_KW)
            uns[i] = (deficit - delivered) * DT_H
            diesel_kwh += delivered * DT_H
            fuel += (F0 * RATED_KW + F1 * output) * DT_H
            hours_on += DT_H
            out[i] = delivered
            run_steps += 1
        else:
            uns[i] = deficit * DT_H                 # nobody covers the deficit
            if state == "starting":
                fuel += F0 * RATED_KW * DT_H        # warming up burns idle fuel
                pending -= 1
    return dict(unserved_kwh=uns.sum(), run_hours=hours_on, starts=starts, fuel_l=fuel,
                diesel_kwh=diesel_kwh, on=on_flag, out=out, uns=uns)

runs = {
    "No AI: react after shortfall":  simulate("reactive"),
    "Naive forecast (same as 1 h ago)": simulate("forecast", df.persistence_kw.values, safe=SAFE_AI),
    "AI forecast":                    simulate("forecast", df.mix_kw.values, safe=SAFE_AI),
    "Perfect forecast (upper bound)": simulate("forecast", ren, safe=1.0),
}
if alert_smote is not None:
    runs["AI + alert (no balancing)"] = simulate("forecast", df.mix_kw.values, safe=alert_plain)
    runs["AI + SMOTE alert"] = simulate("forecast", df.mix_kw.values, safe=alert_smote)

rows = []
for name, r in runs.items():
    rows.append({"Strategy": name,
                 "Unserved energy (kWh)": round(r["unserved_kwh"], 1),
                 "Diesel run hours": round(r["run_hours"], 1),
                 "Starts": r["starts"],
                 "Fuel (L)": round(r["fuel_l"]),
                 "CO2 (kg)": round(r["fuel_l"] * CO2_PER_L)})
summary = pd.DataFrame(rows)
summary.to_csv("dispatch_summary.csv", index=False)
print(summary.to_string(index=False))

a, b = runs["No AI: react after shortfall"], runs["AI forecast"]

# per-step record (used by the RAG step to build the event log)
pd.DataFrame({"time": df.time, "load_kw": load, "renewable_kw": ren, "ai_forecast_kw": df.mix_kw,
              "ai_diesel_on": b["on"], "ai_diesel_kw": b["out"], "ai_unserved_kwh": b["uns"],
              "react_diesel_on": a["on"], "react_diesel_kw": a["out"], "react_unserved_kwh": a["uns"]}
             ).to_csv("dispatch_steps.csv", index=False)
print("\nMain case (start-up delay %d min). AI versus no AI over %d test days:" % (15 * START_MAIN, n // 96))
print("  Unserved energy: %.0f kWh -> %.0f kWh  (%.0f %% less)" %
      (a["unserved_kwh"], b["unserved_kwh"], 100 * (1 - b["unserved_kwh"] / a["unserved_kwh"])))
print("  Fuel: %.0f L -> %.0f L  (%+.0f %%)" % (a["fuel_l"], b["fuel_l"], 100 * (b["fuel_l"] / a["fuel_l"] - 1)))

# ---------- plot one day (the day where the AI helps most) ----------
gain = [(a["uns"][d*96:(d+1)*96].sum() - b["uns"][d*96:(d+1)*96].sum()) for d in range(n // 96)]
d0 = int(np.argmax(gain)); sl = slice(d0 * 96, (d0 + 1) * 96)
t = df.time.values[sl]
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for k, (title, r) in enumerate([("No AI: diesel starts after the shortfall", a), ("AI forecast: diesel starts before the shortfall", b)]):
    ax[k].plot(t, load[sl], color="black", label="Load")
    ax[k].plot(t, ren[sl], color="green", label="PV + wind (actual)")
    ax[k].fill_between(t, 0, r["out"][sl], color="orange", alpha=0.5, label="Diesel output")
    miss = r["uns"][sl] > 0
    ax[k].scatter(t[miss], load[sl][miss], color="red", s=22, zorder=5, label="Unserved load")
    ax[k].set_title(title); ax[k].set_ylabel("Power (kW)")
ax[0].legend(loc="upper left", ncol=2, fontsize=8)
plt.tight_layout(); plt.savefig("dispatch_day.png", dpi=120)

# ---------- bar charts ----------
short = {"No AI: react after shortfall": "No AI", "Naive forecast (same as 1 h ago)": "Naive\nforecast", "AI forecast": "AI\nforecast",
         "Perfect forecast (upper bound)": "Perfect\nforecast", "AI + alert (no balancing)": "AI + alert\n(no balancing)", "AI + SMOTE alert": "AI + SMOTE\nalert"}
names = [short[k] for k in runs]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
for k, (col, lab, colr) in enumerate([("Unserved energy (kWh)", "Unserved energy (kWh)", "#C00000"),
                                      ("Fuel (L)", "Fuel (L)", "#7F6000"),
                                      ("Starts", "Number of diesel starts", "#2E75B6")]):
    bars = ax[k].bar(names, summary[col], color=colr)
    ax[k].set_title(lab, fontsize=10)
    for bb in bars:
        ax[k].text(bb.get_x() + bb.get_width() / 2, bb.get_height(), "%g" % bb.get_height(), ha="center", va="bottom", fontsize=8)
    ax[k].tick_params(axis="x", labelsize=8)
plt.tight_layout(); plt.savefig("dispatch_bars.png", dpi=120)

# ---------- sensitivity: how does the result depend on the start-up delay? ----------
sens = []
for S in (1, 2, 3, 4):
    r0 = simulate("reactive", S=S)
    r1 = simulate("forecast", df.persistence_kw.values, SAFE_AI, S)
    r2 = simulate("forecast", df.mix_kw.values, SAFE_AI, S)
    r3 = simulate("forecast", ren, 1.0, S)
    r4 = simulate("forecast", df.mix_kw.values, alert_smote, S) if alert_smote is not None else r2
    sens.append({"Start-up delay (min)": 15 * S,
                 "No AI unserved (kWh)": round(r0["unserved_kwh"]), "Naive forecast unserved (kWh)": round(r1["unserved_kwh"]),
                 "AI unserved (kWh)": round(r2["unserved_kwh"]), "AI+SMOTE alert unserved (kWh)": round(r4["unserved_kwh"]), "Perfect unserved (kWh)": round(r3["unserved_kwh"]),
                 "No AI fuel (L)": round(r0["fuel_l"]), "AI fuel (L)": round(r2["fuel_l"]), "AI+SMOTE alert fuel (L)": round(r4["fuel_l"])})
sens = pd.DataFrame(sens)
sens.to_csv("sensitivity.csv", index=False)
print("\nSensitivity to the start-up delay:")
print(sens.to_string(index=False))
plt.figure(figsize=(7.5, 4.2))
plt.plot(sens["Start-up delay (min)"], sens["No AI unserved (kWh)"], "o-", color="#C00000", label="No AI (reactive)")
plt.plot(sens["Start-up delay (min)"], sens["Naive forecast unserved (kWh)"], "s--", color="grey", label="Naive forecast")
plt.plot(sens["Start-up delay (min)"], sens["AI unserved (kWh)"], "^-", color="green", label="AI forecast")
plt.plot(sens["Start-up delay (min)"], sens["AI+SMOTE alert unserved (kWh)"], "d-", color="#7030A0", label="AI + SMOTE alert")
plt.plot(sens["Start-up delay (min)"], sens["Perfect unserved (kWh)"], "x:", color="black", label="Perfect forecast")
plt.xlabel("Diesel start-up delay (minutes)"); plt.ylabel("Unserved energy over test period (kWh)")
plt.title("AI helps more as the diesel takes longer to start"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("sensitivity.png", dpi=120)
print("Saved dispatch_summary.csv, sensitivity.csv, dispatch_day.png, dispatch_bars.png, sensitivity.png")
