
import os
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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

os.makedirs("outputs", exist_ok=True)
os.makedirs("model", exist_ok=True)

CONTAMINATION_VALUES = [0.02, 0.03, 0.04, 0.05, 0.06]

FEATURES = [
    "occupancy",
    "temperature_c",
    "rainfall_mm",
    "consumption_liters",
    "previous_day_consumption_liters",
    "rolling_7day_avg_liters"
]

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATASET_FILE)
df["date"] = pd.to_datetime(df["date"])

df = df.dropna(
    subset=FEATURES + ["synthetic_anomaly_flag"]
).copy()

df["synthetic_anomaly_flag"] = (
    df["synthetic_anomaly_flag"].astype(int)
)

# ==========================================
# CHRONOLOGICAL SPLIT
# ==========================================

# January-September 2024: training
train_2024 = df[
    (df["date"] >= "2024-01-01") &
    (df["date"] < "2024-10-01") &
    (df["synthetic_anomaly_flag"] == 0)
].copy()

# October-December 2024: validation
validation_2024 = df[
    (df["date"] >= "2024-10-01") &
    (df["date"] < "2025-01-01")
].copy()

# All normal records from 2024: final training
final_train = df[
    (df["date"].dt.year == 2024) &
    (df["synthetic_anomaly_flag"] == 0)
].copy()

# All records from 2025: final testing
final_test = df[
    df["date"].dt.year == 2025
].copy()


# ==========================================
# TRAIN AND DETECT FUNCTION
# ==========================================

def detect_anomalies(training_data, testing_data, contamination):

    predictions = pd.Series(
        0,
        index=testing_data.index,
        dtype=int
    )

    for building_id, train_group in training_data.groupby(
        "building_id"
    ):

        test_group = testing_data[
            testing_data["building_id"] == building_id
        ]

        if test_group.empty or len(train_group) < 10:
            continue

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            train_group[FEATURES]
        )

        X_test = scaler.transform(
            test_group[FEATURES]
        )

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42
        )

        model.fit(X_train)

        building_predictions = (
            model.predict(X_test) == -1
        ).astype(int)

        predictions.loc[test_group.index] = (
            building_predictions
        )

    return predictions


# ==========================================
# PART 1: VALIDATION
# ==========================================

validation_results = []

print("\n===== 2024 VALIDATION RESULTS =====")
print("Training: Normal records from January-September 2024")
print("Validation: October-December 2024")

actual_validation = (
    validation_2024["synthetic_anomaly_flag"].astype(int)
)

for contamination in CONTAMINATION_VALUES:

    predicted = detect_anomalies(
        train_2024,
        validation_2024,
        contamination
    )

    precision = precision_score(
        actual_validation,
        predicted,
        zero_division=0
    )

    recall = recall_score(
        actual_validation,
        predicted,
        zero_division=0
    )

    f1 = f1_score(
        actual_validation,
        predicted,
        zero_division=0
    )

    validation_results.append({
        "contamination": contamination,
        "detected_anomalies": int(predicted.sum()),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print(f"\nContamination: {contamination}")
    print(f"Detected anomalies: {predicted.sum()}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")


validation_df = pd.DataFrame(validation_results)

validation_df.to_csv(
    "outputs/anomaly_validation_2024.csv",
    index=False
)

# Select best contamination using validation F1
best_index = validation_df["f1_score"].idxmax()

best_contamination = float(
    validation_df.loc[best_index, "contamination"]
)

print("\n===== SELECTED CONTAMINATION =====")
print(f"Best contamination: {best_contamination}")


# ==========================================
# PART 2: FINAL 2025 EVALUATION
# ==========================================

print("\n===== FINAL 2025 TEST EVALUATION =====")
print("Training: Normal records from 2024")
print("Testing: All eligible records from 2025")

final_predictions = detect_anomalies(
    final_train,
    final_test,
    best_contamination
)

actual = final_test[
    "synthetic_anomaly_flag"
].astype(int)

precision = precision_score(
    actual,
    final_predictions,
    zero_division=0
)

recall = recall_score(
    actual,
    final_predictions,
    zero_division=0
)

f1 = f1_score(
    actual,
    final_predictions,
    zero_division=0
)

matrix = confusion_matrix(
    actual,
    final_predictions,
    labels=[0, 1]
)

print(f"\nSelected contamination: {best_contamination}")
print(f"Actual anomalies: {actual.sum()}")
print(f"Detected anomalies: {final_predictions.sum()}")

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 score: {f1:.4f}")

print("\nConfusion Matrix:")
print(matrix)

print("\nClassification Report:")
print(
    classification_report(
        actual,
        final_predictions,
        zero_division=0
    )
)


# ==========================================
# PART 3: SAVE FINAL PREDICTIONS
# ==========================================

final_output = final_test[
    [
        "date",
        "building_id",
        "building_type",
        "occupancy",
        "consumption_liters",
        "synthetic_anomaly_flag"
    ]
].copy()

final_output["predicted_anomaly"] = (
    final_predictions
)

final_output.to_csv(
    "outputs/final_anomaly_predictions_2025.csv",
    index=False
)


# ==========================================
# PART 4: SAVE FINAL METRICS
# ==========================================

final_metrics = pd.DataFrame([{
    "selected_contamination": best_contamination,
    "actual_anomalies": int(actual.sum()),
    "detected_anomalies": int(final_predictions.sum()),
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}])

final_metrics.to_csv(
    "outputs/final_anomaly_metrics_2025.csv",
    index=False
)


# ==========================================
# PART 5: TRAIN AND SAVE FINAL MODELS
# ==========================================

building_models = {}

print("\n===== SAVING FINAL BUILDING MODELS =====")

for building_id, group in final_train.groupby("building_id"):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        group[FEATURES]
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=best_contamination,
        random_state=42
    )

    model.fit(X_train)

    building_models[building_id] = {
        "scaler": scaler,
        "model": model,
        "building_type": group["building_type"].iloc[0],
        "training_records": len(group)
    }

    print(
        f"{building_id}: {len(group)} normal records trained"
    )


saved_data = {
    "models": building_models,
    "features": FEATURES,
    "contamination": best_contamination,
    "training_year": 2024
}

joblib.dump(saved_data, MODEL_FILE)

print("\n===== PROCESS COMPLETED =====")
print(f"Selected contamination: {best_contamination}")
print(f"Buildings trained: {len(building_models)}")
print(f"Final model saved: {MODEL_FILE}")
print("Results saved in the outputs folder.")
