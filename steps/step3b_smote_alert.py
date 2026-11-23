# STEP 3b: "Forecast-miss alert" classifier with SMOTE
#
# WHY SMOTE HERE
#   SMOTE (Synthetic Minority Over-sampling TEchnique) is for CLASSIFICATION with a rare class.
#   Our forecaster is regression, so SMOTE does not apply to it. But we can ask a second question:
#   "Will the renewable power in 1 hour come in MUCH LOWER than the forecast?" (a forecast miss).
#   These sudden drops (clouds, wind lulls) are rare, so the classes are imbalanced:
#   class 0 = normal, class 1 = forecast-miss alert. SMOTE creates synthetic class-1 examples
#   so the classifier learns to spot them. SMOTE is applied to the TRAINING data only, never to the test data.
#
# Needs microgrid_data.csv (step 1) and forecast_results.csv (step 2/3) in the working folder.
# Output: alert_metrics.csv, alert_results.csv, alert_plot.png

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix

HORIZON, TRAIN_DAYS = 4, 45
DELTA_KW = 10          # alert = renewable power comes in at least 10 kW below the forecast (and a deficit exists)

try:                                   # use the standard library if it is installed (pip install imbalanced-learn)
    from imblearn.over_sampling import SMOTE as _IMB_SMOTE
    SMOTE_SOURCE = "imbalanced-learn"
except Exception:
    _IMB_SMOTE = None
    SMOTE_SOURCE = "built-in implementation (same algorithm)"

def smote_resample(X, y, k=5, seed=0):
    """Balance the classes by creating synthetic minority samples between a minority point and one of its k nearest minority neighbours."""
    if _IMB_SMOTE is not None:
        return _IMB_SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)
    rng = np.random.default_rng(seed)
    Xmin = X[y == 1]
    n_new = int((y == 0).sum() - (y == 1).sum())
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xmin))).fit(Xmin)
    neigh = nn.kneighbors(Xmin, return_distance=False)[:, 1:]           # drop the point itself
    base = rng.integers(0, len(Xmin), n_new)
    other = neigh[base, rng.integers(0, neigh.shape[1], n_new)]
    gap = rng.random((n_new, 1))
    Xnew = Xmin[base] + gap * (Xmin[other] - Xmin[base])                # a point on the line between the two
    return np.vstack([X, Xnew]), np.r_[y, np.ones(n_new, dtype=int)]

# ---------------- build features (same idea as step 2/3) ----------------
df = pd.read_csv("microgrid_data.csv", parse_dates=["time"])
spd = len(df) // 60
X = pd.DataFrame({"time": df.time})
for col in ["irradiance_wm2", "wind_ms", "temp_c", "pv_kw", "wind_kw"]:
    X[col] = df[col]
    for lag in (1, 2, 4):
        X[f"{col}_lag{lag}"] = df[col].shift(lag)
X["irr_change"] = df.irradiance_wm2.diff()
X["wind_change"] = df.wind_ms.diff()
tt = df.time + pd.Timedelta(minutes=15 * HORIZON)
hr = tt.dt.hour + tt.dt.minute / 60
X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
feats = [c for c in X.columns if c != "time"]

y_pv = df.pv_kw.shift(-HORIZON)
actual_ren = (df.pv_kw + df.wind_kw).shift(-HORIZON)
load_t = df.load_kw.shift(-HORIZON)
ok = X.notna().all(axis=1) & actual_ren.notna()
X, y_pv, actual_ren, load_t = X[ok], y_pv[ok], actual_ren[ok], load_t[ok]
wind_persist = df.wind_kw[ok]
split = TRAIN_DAYS * spd

# ---------------- forecast for every row (out-of-fold on the training part, so no cheating) ----------------
pv_fc = pd.Series(np.nan, index=X.index)
tr_idx, te_idx = X.index[:split], X.index[split:]
for a, b in TimeSeriesSplit(n_splits=5).split(tr_idx):
    m = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(X.loc[tr_idx[a], feats], y_pv.loc[tr_idx[a]])
    pv_fc.loc[tr_idx[b]] = m.predict(X.loc[tr_idx[b], feats])
m = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(X.loc[tr_idx, feats], y_pv.loc[tr_idx])
pv_fc.loc[te_idx] = m.predict(X.loc[te_idx, feats])
fc = pv_fc + wind_persist                                    # same "best mix" forecast as step 2/3

# ---------------- label: forecast miss ----------------
miss = fc - actual_ren
deficit_actual = load_t - actual_ren
label = ((miss >= DELTA_KW) & (deficit_actual > 0)).astype(int)

F = X[feats].copy()
F["forecast_kw"] = fc
F["load_target"] = load_t
F["forecast_deficit"] = load_t - fc
valid = fc.notna()
Ftr, ytr = F.loc[tr_idx][valid.loc[tr_idx]], label.loc[tr_idx][valid.loc[tr_idx]]
Fte, yte = F.loc[te_idx], label.loc[te_idx]
print(f"Class balance - train: {ytr.mean() * 100:.1f}% alerts ({int(ytr.sum())} of {len(ytr)}), test: {yte.mean() * 100:.1f}% alerts ({int(yte.sum())} of {len(yte)})")
print(f"SMOTE source: {SMOTE_SOURCE}")

sc = StandardScaler().fit(Ftr)
Xtr_s, Xte_s = sc.transform(Ftr), sc.transform(Fte)
ytr_a = ytr.values

def clf():
    return RandomForestClassifier(n_estimators=250, min_samples_leaf=3, n_jobs=-1, random_state=0)

variants = {}
variants["No balancing"] = clf().fit(Xtr_s, ytr_a)
variants["Class weights"] = RandomForestClassifier(n_estimators=250, min_samples_leaf=3, n_jobs=-1, random_state=0, class_weight="balanced").fit(Xtr_s, ytr_a)
Xs, ys = smote_resample(Xtr_s, ytr_a)
print(f"After SMOTE the training set has {int((ys == 0).sum())} normal and {int((ys == 1).sum())} alert samples (was {int((ytr_a == 0).sum())} / {int((ytr_a == 1).sum())})")
variants["SMOTE"] = clf().fit(Xs, ys)

rows, probs = [], {}
for name, model in variants.items():
    p = model.predict_proba(Xte_s)[:, 1]
    pred = (p >= 0.5).astype(int)
    probs[name] = p
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    rows.append({"Model": name, "Precision": round(precision_score(yte, pred, zero_division=0), 2),
                 "Recall": round(recall_score(yte, pred), 2), "F1": round(f1_score(yte, pred), 2),
                 "PR-AUC": round(average_precision_score(yte, p), 2), "Alerts caught": int(tp), "Missed": int(fn), "False alarms": int(fp)})
res = pd.DataFrame(rows)
res.to_csv("alert_metrics.csv", index=False)
print("\n" + res.to_string(index=False))

out = pd.DataFrame({"time": (X.loc[te_idx, "time"] + pd.Timedelta(minutes=15 * HORIZON)).values,
                    "alert_true": yte.values,
                    "alert_prob_plain": probs["No balancing"], "alert_prob_smote": probs["SMOTE"]})
out.to_csv("alert_results.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].bar(["Train\n(real)", "Train\n(after SMOTE)"], [ytr_a.mean() * 100, ys.mean() * 100], color=["#2E75B6", "#C55A11"])
ax[0].set_ylabel("Alert class share (%)"); ax[0].set_title("SMOTE balances the training classes")
x = np.arange(3); w = 0.25
for k, mname in enumerate(["Precision", "Recall", "F1"]):
    ax[1].bar(x + (k - 1) * w, res[mname], w, label=mname)
ax[1].set_xticks(x); ax[1].set_xticklabels(res.Model, fontsize=8); ax[1].legend(fontsize=8); ax[1].set_title("Test results (alert class)")
plt.tight_layout(); plt.savefig("alert_plot.png", dpi=120)
print("\nSaved alert_metrics.csv, alert_results.csv, alert_plot.png")
