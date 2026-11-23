from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

# ============================================================
# STEP 10B: HOURLY SMOTE ALERT CLASSIFICATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

HORIZON_HOURS = 1
TRAIN_RATIO = 0.70
RANDOM_STATE = 42


# ============================================================
# IMPORT SMOTE
# ============================================================

try:
    from imblearn.over_sampling import SMOTE

    USE_IMBLEARN = True
    print("SMOTE source: imbalanced-learn")

except ImportError:
    USE_IMBLEARN = False
    print("SMOTE source: built-in implementation")


# ============================================================
# SMOTE FUNCTION
# ============================================================

def apply_smote(X, y):

    X = np.asarray(X)
    y = np.asarray(y)

    minority_count = int((y == 1).sum())
    majority_count = int((y == 0).sum())

    if minority_count < 2:
        print("WARNING: Too few minority samples for SMOTE.")
        print("Original training data will be used.")
        return X, y

    if minority_count >= majority_count:
        print("Training data is already balanced.")
        return X, y

    k = min(5, minority_count - 1)

    if USE_IMBLEARN:

        sampler = SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=k
        )

        return sampler.fit_resample(X, y)

    # Built-in SMOTE fallback
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(RANDOM_STATE)

    minority = X[y == 1]

    n_new = majority_count - minority_count

    neighbors = NearestNeighbors(
        n_neighbors=k + 1
    ).fit(minority)

    indices = neighbors.kneighbors(
        minority,
        return_distance=False
    )[:, 1:]

    base_indices = rng.integers(
        0,
        len(minority),
        n_new
    )

    neighbor_indices = indices[
        base_indices,
        rng.integers(0, k, n_new)
    ]

    gaps = rng.random((n_new, 1))

    synthetic = minority[base_indices] + gaps * (
        minority[neighbor_indices] -
        minority[base_indices]
    )

    X_new = np.vstack([X, synthetic])

    y_new = np.concatenate([
        y,
        np.ones(n_new, dtype=int)
    ])

    return X_new, y_new


# ============================================================
# HOURLY DATASET CREATION AND MODEL EVALUATION
# ============================================================

def build_hourly_dataset(
    dataset_name,
    forecast_file,
    source_file,
    forecast_column,
    delta_kw
):

    print("\n" + "=" * 65)
    print(f"HOURLY ALERT CLASSIFICATION: {dataset_name} (Threshold >= {delta_kw} kW)")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    source = pd.read_csv(
        OUTPUT / source_file,
        parse_dates=["time"]
    )

    forecasts = pd.read_csv(
        OUTPUT / forecast_file,
        parse_dates=["time"]
    )

    source = source.sort_values("time").copy()

    forecasts = forecasts.sort_values("time").copy()

    # Forecast timestamp represents the target hour.
    # Source time is one hour before the target.
    forecasts["source_time"] = (
        forecasts["time"] -
        pd.Timedelta(hours=HORIZON_HOURS)
    )

    # --------------------------------------------------------
    # 2. CREATE SOURCE-HOUR FEATURES
    # --------------------------------------------------------

    source["irradiance_change"] = (
        source["irradiance_wm2"].diff()
    )

    source["wind_change"] = (
        source["wind_ms"].diff()
    )

    source["renewable_change"] = (
        source["renewable_kw"].diff()
    )

    source["hour_sin"] = np.sin(
        2 * np.pi * source["time"].dt.hour / 24
    )

    source["hour_cos"] = np.cos(
        2 * np.pi * source["time"].dt.hour / 24
    )

    # IMPORTANT:
    # Rename source-hour load to avoid duplicate load_kw.
    source = source.rename(
        columns={
            "time": "source_time",
            "load_kw": "source_load_kw"
        }
    )

    # --------------------------------------------------------
    # 3. DEFINE INPUT FEATURES
    # --------------------------------------------------------

    feature_columns = [
        "irradiance_wm2",
        "temp_c",
        "wind_ms",
        "source_load_kw",
        "pv_kw",
        "wind_kw",
        "renewable_kw",
        "irradiance_change",
        "wind_change",
        "renewable_change",
        "hour_sin",
        "hour_cos"
    ]

    # --------------------------------------------------------
    # 4. MERGE SOURCE AND FORECAST DATA
    # --------------------------------------------------------

    data = forecasts.merge(
        source[["source_time"] + feature_columns],
        on="source_time",
        how="inner",
        validate="one_to_one"
    )

    if len(data) == 0:
        raise ValueError(
            "No matching timestamps found between "
            "forecast results and source data."
        )

    # --------------------------------------------------------
    # 5. CREATE FORECAST AND ALERT VARIABLES
    # --------------------------------------------------------

    data["forecast_kw"] = data[forecast_column]

    # The target-hour load is treated as a known load schedule.
    data["forecast_deficit_kw"] = (
        data["load_kw"] -
        data["forecast_kw"]
    )

    # Positive value means forecast overestimated generation.
    data["forecast_miss_kw"] = (
        data["forecast_kw"] -
        data["actual_renewable_kw"]
    )

    # Actual target-hour energy deficit.
    data["actual_deficit_kw"] = (
        data["load_kw"] -
        data["actual_renewable_kw"]
    )

    # Alert condition:
    # 1. Forecast exceeds actual renewable generation by >= delta_kw.
    # 2. Actual load is greater than actual renewable generation.
    data["alert"] = (
        (data["forecast_miss_kw"] >= delta_kw) &
        (data["actual_deficit_kw"] > 0)
    ).astype(int)

    # --------------------------------------------------------
    # 6. CLEAN DATA
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    required_columns = feature_columns + [
        "forecast_kw",
        "forecast_deficit_kw",
        "actual_renewable_kw",
        "load_kw"
    ]

    data = data.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    if len(data) < 30:
        raise ValueError(
            f"Not enough aligned rows after cleaning: {len(data)}"
        )

    # --------------------------------------------------------
    # 7. PREPARE X AND Y
    # --------------------------------------------------------

    X = data[
        feature_columns + [
            "forecast_kw",
            "forecast_deficit_kw"
        ]
    ]

    y = data["alert"].astype(int)

    # --------------------------------------------------------
    # 8. CHRONOLOGICAL TRAIN-TEST SPLIT
    # --------------------------------------------------------

    split = int(len(data) * TRAIN_RATIO)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    print("Aligned rows:", len(data))
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print(
        "\nTraining normal samples:",
        int((y_train == 0).sum())
    )

    print(
        "Training alert samples:",
        int((y_train == 1).sum())
    )

    print(
        "\nTesting normal samples:",
        int((y_test == 0).sum())
    )

    print(
        "Testing alert samples:",
        int((y_test == 1).sum())
    )

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains only one class. "
            "The classifier cannot learn both normal and alert cases."
        )

    # --------------------------------------------------------
    # 9. IMPUTATION AND SCALING
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = imputer.fit_transform(X_train)

    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_imp
    )

    X_test_scaled = scaler.transform(
        X_test_imp
    )

    # --------------------------------------------------------
    # 10. RANDOM FOREST CLASSIFIER
    # --------------------------------------------------------

    def make_classifier(class_weight=None):

        return RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    models = {}

    # Model 1: No balancing
    models["No balancing"] = make_classifier().fit(
        X_train_scaled,
        y_train
    )

    # Model 2: Class weighting
    models["Class weights"] = make_classifier(
        class_weight="balanced"
    ).fit(
        X_train_scaled,
        y_train
    )

    # --------------------------------------------------------
    # 11. APPLY SMOTE TO TRAINING DATA ONLY
    # --------------------------------------------------------

    X_smote, y_smote = apply_smote(
        X_train_scaled,
        y_train.values
    )

    print("\nAfter SMOTE:")
    print("Normal samples:", int((y_smote == 0).sum()))
    print("Alert samples:", int((y_smote == 1).sum()))

    # Model 3: SMOTE
    models["SMOTE"] = make_classifier().fit(
        X_smote,
        y_smote
    )

    # --------------------------------------------------------
    # 12. EVALUATE MODELS
    # --------------------------------------------------------

    rows = []

    prediction_output = pd.DataFrame({
        "time": data.iloc[split:]["time"].values,
        "actual_alert": y_test.values,
        "actual_renewable_kw": data.iloc[split:][
            "actual_renewable_kw"
        ].values,
        "forecast_kw": data.iloc[split:][
            "forecast_kw"
        ].values
    })

    for name, model in models.items():

        probabilities = model.predict_proba(
            X_test_scaled
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        if y_test.nunique() == 2:

            pr_auc = average_precision_score(
                y_test,
                probabilities
            )

        else:
            pr_auc = np.nan

        rows.append({
            "Dataset": dataset_name,
            "Threshold_kW": delta_kw,
            "Model": name,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "PR_AUC": pr_auc,
            "True_Positive": tp,
            "False_Positive": fp,
            "False_Negative": fn,
            "True_Negative": tn
        })

        prediction_column = (
            name.replace(" ", "_").lower() +
            "_probability"
        )

        prediction_output[
            prediction_column
        ] = probabilities

    metrics = pd.DataFrame(rows)

    print("\nTEST PERFORMANCE")

    print(
        metrics.round(3).to_string(index=False)
    )

    # --------------------------------------------------------
    # 13. SAVE RESULTS
    # --------------------------------------------------------

    prefix = (
        dataset_name.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    metrics.to_csv(
        OUTPUT / f"{prefix}_hourly_alert_metrics.csv",
        index=False
    )

    prediction_output.to_csv(
        OUTPUT / f"{prefix}_hourly_alert_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # 14. PLOT CLASS BALANCE
    # --------------------------------------------------------

    before = [
        int((y_train == 0).sum()),
        int((y_train == 1).sum())
    ]

    after = [
        int((y_smote == 0).sum()),
        int((y_smote == 1).sum())
    ]

    x = np.arange(2)

    width = 0.35

    plt.figure(figsize=(7, 4))

    plt.bar(
        x - width / 2,
        before,
        width,
        label="Original training data"
    )

    plt.bar(
        x + width / 2,
        after,
        width,
        label="After SMOTE"
    )

    plt.xticks(
        x,
        ["Normal", "Alert"]
    )

    plt.ylabel("Number of samples")

    plt.title(
        f"{dataset_name}: Training class balance ({delta_kw} kW threshold)"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT / f"{prefix}_hourly_smote_balance.png",
        dpi=150
    )

    plt.show()

    # --------------------------------------------------------
    # 15. PLOT MODEL PERFORMANCE
    # --------------------------------------------------------

    metrics_plot = metrics.set_index("Model")[
        ["Precision", "Recall", "F1"]
    ]

    metrics_plot.plot(
        kind="bar",
        figsize=(8, 4)
    )

    plt.title(
        f"{dataset_name}: Alert classification ({delta_kw} kW threshold)"
    )

    plt.ylabel("Score")

    plt.ylim(0, 1)

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        OUTPUT / f"{prefix}_hourly_alert_performance.png",
        dpi=150
    )

    plt.show()

    return metrics


# ============================================================
# 16. RUN SYNTHETIC DATASET (Severe Alert: 10 kW)
# ============================================================

synthetic_metrics = build_hourly_dataset(
    dataset_name="Synthetic",
    source_file="synthetic_microgrid_hourly.csv",
    forecast_file="synthetic_hourly_forecast_results.csv",
    forecast_column="mix_kw",
    delta_kw=10
)


# ============================================================
# 17. RUN NASA-BASED DATASET (Moderate Alert: 2 kW)
# ============================================================

nasa_metrics = build_hourly_dataset(
    dataset_name="NASA-based",
    source_file="nasa_microgrid_hourly.csv",
    forecast_file="nasa_hourly_forecast_results.csv",
    forecast_column="gb_kw",
    delta_kw=2
)


# ============================================================
# 18. COMBINE BOTH DATASET RESULTS
# ============================================================

combined = pd.concat(
    [
        synthetic_metrics,
        nasa_metrics
    ],
    ignore_index=True
)

combined.to_csv(
    OUTPUT / "hourly_smote_alert_comparison.csv",
    index=False
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("HOURLY SMOTE ALERT EXPERIMENT COMPLETED")
print("=" * 65)

print(
    combined.round(3).to_string(index=False)
)

print("\nSaved output files:")

output_files = [
    "synthetic_hourly_alert_metrics.csv",
    "nasa_based_hourly_alert_metrics.csv",
    "synthetic_hourly_alert_predictions.csv",
    "nasa_based_hourly_alert_predictions.csv",
    "synthetic_hourly_smote_balance.png",
    "nasa_based_hourly_smote_balance.png",
    "synthetic_hourly_alert_performance.png",
    "nasa_based_hourly_alert_performance.png",
    "hourly_smote_alert_comparison.csv"
]

for filename in output_files:

    filepath = OUTPUT / filename

    if filepath.exists():
        print("  ", filename)

print("\nSTEP 10B FINISHED")