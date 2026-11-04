
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("outputs/hybrid_anomaly_predictions_2025.csv")

# 1. Confusion matrix
actual = df["synthetic_anomaly_flag"]
predicted = df["predicted_anomaly"]

cm = confusion_matrix(actual, predicted, labels=[0, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Hybrid Anomaly Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/hybrid_confusion_matrix.png", dpi=150)
plt.show()

# 2. Actual vs detected anomalies
actual_counts = df.groupby("building_id")[
    "synthetic_anomaly_flag"
].sum()

detected_counts = df.groupby("building_id")[
    "predicted_anomaly"
].sum()

comparison = pd.DataFrame({
    "Actual anomalies": actual_counts,
    "Detected anomalies": detected_counts
})

comparison.plot(kind="bar", figsize=(11, 5))

plt.title("Actual vs Detected Anomalies by Building - Hybrid Model")
plt.xlabel("Building ID")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("outputs/hybrid_actual_vs_detected.png", dpi=150)
plt.show()

# 3. Detection method comparison
method_counts = [
    int(df["isolation_forest_prediction"].sum()),
    int(df["threshold_prediction"].sum()),
    int(df["predicted_anomaly"].sum())
]

plt.figure(figsize=(9, 5))
plt.bar(
    ["Isolation Forest", "Threshold", "Hybrid"],
    method_counts
)

plt.title("Anomaly Detections by Method - 2025")
plt.ylabel("Number of Detected Records")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("outputs/hybrid_method_comparison.png", dpi=150)
plt.show()

print("Hybrid anomaly graphs generated successfully!")
