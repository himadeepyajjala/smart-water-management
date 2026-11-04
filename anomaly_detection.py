
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv("dataset/water_consumption_dataset.csv")

# Features used for anomaly detection
features = [
    "occupancy",
    "temperature_c",
    "rainfall_mm",
    "consumption_liters",
    "previous_day_consumption_liters",
    "rolling_7day_avg_liters"
]

# Remove rows with missing feature values
data = df.dropna(subset=features).copy()

X = data[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.031,
    random_state=42
)

model.fit(X_scaled)

# Predict anomalies
data["predicted_anomaly"] = (
    model.predict(X_scaled) == -1
).astype(int)

# Compare with existing synthetic labels
actual = data["synthetic_anomaly_flag"]
predicted = data["predicted_anomaly"]

print("\n===== ISOLATION FOREST RESULTS =====")

print("Total records analysed:", len(data))
print("Existing synthetic anomalies:", actual.sum())
print("Isolation Forest anomalies:", predicted.sum())

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(actual, predicted))

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(actual, predicted, zero_division=0))

print("\n===== EVALUATION METRICS =====")
print("Precision:", precision_score(actual, predicted))
print("Recall:", recall_score(actual, predicted))
print("F1 Score:", f1_score(actual, predicted))

# Save results
os.makedirs("outputs", exist_ok=True)

data.to_csv(
    "outputs/isolation_forest_results.csv",
    index=False
)

# Plot detected anomalies
plt.figure(figsize=(12, 6))

normal = data[data["predicted_anomaly"] == 0]
anomalies = data[data["predicted_anomaly"] == 1]

plt.scatter(
    normal.index,
    normal["consumption_liters"],
    label="Normal",
    alpha=0.5
)

plt.scatter(
    anomalies.index,
    anomalies["consumption_liters"],
    label="Detected Anomaly",
    alpha=0.8
)

plt.title("Isolation Forest - Water Consumption Anomaly Detection")
plt.xlabel("Record Index")
plt.ylabel("Water Consumption (Litres)")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/isolation_forest_anomalies.png"
)

plt.show()

print("\nResults saved successfully!")
