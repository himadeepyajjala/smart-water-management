import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

os.makedirs("outputs", exist_ok=True)

# Load final predictions
df = pd.read_csv("outputs/final_anomaly_predictions_2025.csv")

results = []

print("\n===== FINAL BUILDING-WISE ANOMALY EVALUATION =====")

for building_id, group in df.groupby("building_id"):

    actual = group["synthetic_anomaly_flag"].astype(int)
    predicted = group["predicted_anomaly"].astype(int)

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
        "building_id": building_id,
        "records": len(group),
        "actual_anomalies": int(actual.sum()),
        "detected_anomalies": int(predicted.sum()),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print(
        f"{building_id}: "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1={f1:.4f}"
    )

results_df = pd.DataFrame(results)

print("\n===== FINAL SUMMARY =====")
print(results_df.to_string(index=False))

print(
    "\nAverage building-wise F1:",
    round(results_df["f1_score"].mean(), 4)
)

results_df.to_csv(
    "outputs/final_building_wise_metrics.csv",
    index=False
)

print("\nResults saved successfully!")
