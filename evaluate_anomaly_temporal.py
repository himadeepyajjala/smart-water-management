
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

df = pd.read_csv("dataset/water_consumption_dataset.csv")

df["date"] = pd.to_datetime(df["date"])

features = [
    "occupancy",
    "temperature_c",
    "rainfall_mm",
    "consumption_liters",
    "previous_day_consumption_liters",
    "rolling_7day_avg_liters"
]

df = df.dropna(subset=features).copy()

train = df[df["date"].dt.year == 2024].copy()
test = df[df["date"].dt.year == 2025].copy()

all_predictions = []

for building_id, train_group in train.groupby("building_id"):

    test_group = test[
        test["building_id"] == building_id
    ].copy()

    if test_group.empty:
        continue

    X_train = train_group[features]
    X_test = test_group[features]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42
    )

    model.fit(X_train_scaled)

    predictions = (
        model.predict(X_test_scaled) == -1
    ).astype(int)

    test_group["predicted_anomaly"] = predictions

    all_predictions.append(test_group)

results = pd.concat(all_predictions, ignore_index=True)

actual = results["synthetic_anomaly_flag"]
predicted = results["predicted_anomaly"]

print("\n===== TEMPORAL ANOMALY EVALUATION =====")

print("Training period: 2024")
print("Testing period: 2025")
print("Test records:", len(results))
print("Actual anomalies:", int(actual.sum()))
print("Detected anomalies:", int(predicted.sum()))

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predicted))

print("\nClassification Report:")
print(classification_report(
    actual,
    predicted,
    zero_division=0
))

print("\nPrecision:", precision_score(
    actual, predicted, zero_division=0
))

print("Recall:", recall_score(
    actual, predicted, zero_division=0
))

print("F1 Score:", f1_score(
    actual, predicted, zero_division=0
))

results.to_csv(
    "outputs/temporal_anomaly_results.csv",
    index=False
)

print("\nResults saved successfully!")
