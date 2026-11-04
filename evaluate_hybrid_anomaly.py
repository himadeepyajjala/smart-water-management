
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_FILE = "dataset/water_consumption_dataset.csv"
MODEL_FILE = "model/anomaly_detection_models.pkl"

OUTPUT_PREDICTIONS = "outputs/hybrid_anomaly_predictions_2025.csv"
OUTPUT_METRICS = "outputs/hybrid_anomaly_metrics_2025.csv"

os.makedirs("outputs", exist_ok=True)

FEATURES = [
    "occupancy",
    "temperature_c",
    "rainfall_mm",
    "consumption_liters",
    "previous_day_consumption_liters",
    "rolling_7day_avg_liters"
]

# ==========================================
# LOAD DATASET AND MODELS
# ==========================================

print("\n===== HYBRID ANOMALY MODEL EVALUATION =====")

df = pd.read_csv(DATASET_FILE)
df["date"] = pd.to_datetime(df["date"])

saved_data = joblib.load(MODEL_FILE)

building_models = saved_data["models"]

df = df.dropna(
    subset=FEATURES + ["synthetic_anomaly_flag"]
).copy()

# Select 2025 records only
test = df[df["date"].dt.year == 2025].copy()

if test.empty:
    raise ValueError("No eligible 2025 testing records found.")

# ==========================================
# HYBRID ANOMALY DETECTION
# ==========================================

predictions = pd.Series(
    0,
    index=test.index,
    dtype=int
)

isolation_predictions = pd.Series(
    0,
    index=test.index,
    dtype=int
)

threshold_predictions = pd.Series(
    0,
    index=test.index,
    dtype=int
)

for building_id, group in test.groupby("building_id"):

    if building_id not in building_models:
        print(f"Skipping {building_id}: model not found")
        continue

    info = building_models[building_id]

    scaler = info["scaler"]
    model = info["model"]
    threshold = info["consumption_threshold"]

    X_test = scaler.transform(group[FEATURES])

    # Isolation Forest prediction
    isolation_result = (
        model.predict(X_test) == -1
    ).astype(int)

    # Historical consumption threshold
    threshold_result = (
        group["consumption_liters"].values > threshold
    ).astype(int)

    # Hybrid OR decision
    hybrid_result = (
        (isolation_result == 1) |
        (threshold_result == 1)
    ).astype(int)

    isolation_predictions.loc[group.index] = isolation_result
    threshold_predictions.loc[group.index] = threshold_result
    predictions.loc[group.index] = hybrid_result

# ==========================================
# EVALUATION
# ==========================================

actual = test["synthetic_anomaly_flag"].astype(int)

precision = precision_score(
    actual, predictions, zero_division=0
)

recall = recall_score(
    actual, predictions, zero_division=0
)

f1 = f1_score(
    actual, predictions, zero_division=0
)

matrix = confusion_matrix(
    actual, predictions, labels=[0, 1]
)

print("\n===== HYBRID MODEL RESULTS =====")
print("Testing year: 2025")
print("Total testing records:", len(test))
print("Actual anomalies:", int(actual.sum()))

print(
    "Isolation Forest detections:",
    int(isolation_predictions.sum())
)

print(
    "Threshold detections:",
    int(threshold_predictions.sum())
)

print("Hybrid detections:", int(predictions.sum()))

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nConfusion Matrix:")
print(matrix)

print("\nClassification Report:")
print(
    classification_report(
        actual,
        predictions,
        zero_division=0
    )
)

# ==========================================
# SAVE PREDICTIONS
# ==========================================

output = test[
    [
        "date",
        "building_id",
        "building_type",
        "occupancy",
        "consumption_liters",
        "synthetic_anomaly_flag"
    ]
].copy()

output["isolation_forest_prediction"] = (
    isolation_predictions
)

output["threshold_prediction"] = (
    threshold_predictions
)

output["predicted_anomaly"] = predictions

output.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)

# ==========================================
# SAVE METRICS
# ==========================================

metrics = pd.DataFrame([{
    "testing_year": 2025,
    "total_records": len(test),
    "actual_anomalies": int(actual.sum()),
    "isolation_forest_detections": int(
        isolation_predictions.sum()
    ),
    "threshold_detections": int(
        threshold_predictions.sum()
    ),
    "hybrid_detections": int(predictions.sum()),
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}])

metrics.to_csv(
    OUTPUT_METRICS,
    index=False
)

print("\n===== FILES SAVED =====")
print(OUTPUT_PREDICTIONS)
print(OUTPUT_METRICS)

print("\nHybrid anomaly evaluation completed successfully!")
