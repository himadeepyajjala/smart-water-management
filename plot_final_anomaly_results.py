import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

os.makedirs("outputs", exist_ok=True)

# Load building-wise results
df = pd.read_csv(
    "outputs/final_building_wise_metrics.csv"
)

# 1. Building-wise F1 score
plt.figure(figsize=(10, 5))

plt.bar(df["building_id"], df["f1_score"])

plt.title("Building-wise Anomaly Detection F1 Score")
plt.xlabel("Building ID")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    "outputs/building_wise_f1_score.png",
    dpi=300
)
plt.show()

# 2. Actual vs detected anomalies
plt.figure(figsize=(10, 5))

x = range(len(df))
width = 0.35

plt.bar(
    [i - width/2 for i in x],
    df["actual_anomalies"],
    width,
    label="Actual Anomalies"
)

plt.bar(
    [i + width/2 for i in x],
    df["detected_anomalies"],
    width,
    label="Detected Anomalies"
)

plt.xticks(x, df["building_id"])
plt.title("Actual vs Detected Anomalies by Building")
plt.xlabel("Building ID")
plt.ylabel("Number of Anomalies")
plt.legend()

plt.tight_layout()
plt.savefig(
    "outputs/actual_vs_detected_anomalies.png",
    dpi=300
)
plt.show()

# 3. Confusion matrix
predictions = pd.read_csv(
    "outputs/final_anomaly_predictions_2025.csv"
)

actual = predictions["synthetic_anomaly_flag"]
predicted = predictions["predicted_anomaly"]

cm = confusion_matrix(actual, predicted)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Anomaly"],
    yticklabels=["Normal", "Anomaly"]
)

plt.title("Anomaly Detection Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()
plt.savefig(
    "outputs/anomaly_confusion_matrix.png",
    dpi=300
)
plt.show()

print("All graphs saved successfully in outputs folder.")
