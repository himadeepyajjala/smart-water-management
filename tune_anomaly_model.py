
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
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

# Temporal split
train = df[df["date"].dt.year == 2024]
test = df[df["date"].dt.year == 2025].copy()

contamination_values = [0.02, 0.03, 0.04, 0.05, 0.06]

results = []

for contamination in contamination_values:

    test_predictions = pd.Series(
        0, index=test.index, dtype=int
    )

    # Separate model for each building
    for building_id, train_group in train.groupby("building_id"):

        mask = test["building_id"] == building_id
        test_group = test.loc[mask]

        if test_group.empty:
            continue

        scaler = StandardScaler()

        X_train = scaler.fit_transform(train_group[features])
        X_test = scaler.transform(test_group[features])

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42
        )

        model.fit(X_train)

        predictions = (
            model.predict(X_test) == -1
        ).astype(int)

        test_predictions.loc[test_group.index] = predictions

    actual = test["synthetic_anomaly_flag"]
    predicted = test_predictions.loc[test.index]

    precision = precision_score(
        actual, predicted, zero_division=0
    )

    recall = recall_score(
        actual, predicted, zero_division=0
    )

    f1 = f1_score(
        actual, predicted, zero_division=0
    )

    results.append({
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

# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/anomaly_tuning_results.csv",
    index=False
)

print("\n===== FINAL COMPARISON =====")
print(results_df.to_string(index=False))

print("\nResults saved to outputs/anomaly_tuning_results.csv")
