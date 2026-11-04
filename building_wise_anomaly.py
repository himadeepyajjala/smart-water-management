
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

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

results = []

print("\n===== BUILDING-WISE ISOLATION FOREST =====")

for building_id, group in data.groupby("building_id"):

    X = group[features]
    y = group["synthetic_anomaly_flag"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42
    )

    predictions = (
        model.fit_predict(X_scaled) == -1
    ).astype(int)

    precision = precision_score(
        y, predictions, zero_division=0
    )

    recall = recall_score(
        y, predictions, zero_division=0
    )

    f1 = f1_score(
        y, predictions, zero_division=0
    )

    results.append({
        "building_id": building_id,
        "records": len(group),
        "actual_anomalies": int(y.sum()),
        "detected_anomalies": int(predictions.sum()),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    })

result_df = pd.DataFrame(results)

print(result_df.to_string(index=False))

print("\nAverage building-wise F1:",
      round(result_df["f1_score"].mean(), 4))

result_df.to_csv(
    "outputs/building_wise_anomaly_results.csv",
    index=False
)

print("\nResults saved successfully!")
