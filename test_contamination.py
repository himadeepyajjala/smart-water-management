
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("dataset/water_consumption_dataset.csv")

features = [
    "occupancy",
    "temperature_c",
    "rainfall_mm",
    "consumption_liters",
    "previous_day_consumption_liters",
    "rolling_7day_avg_liters"
]

data = df.dropna(subset=features).copy()

X = data[features]
y = data["synthetic_anomaly_flag"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

contamination_values = [0.02, 0.03, 0.04, 0.05, 0.06]

print("\n===== CONTAMINATION EXPERIMENT =====")

for value in contamination_values:

    model = IsolationForest(
        n_estimators=200,
        contamination=value,
        random_state=42
    )

    predictions = (model.fit_predict(X_scaled) == -1).astype(int)

    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)

    print(f"\nContamination: {value}")
    print("Detected anomalies:", predictions.sum())
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

print("\nExperiment completed.")
