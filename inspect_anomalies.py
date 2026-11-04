
import pandas as pd

df = pd.read_csv("dataset/water_consumption_dataset.csv")

anomalies = df[df["synthetic_anomaly_flag"] == 1]

print("\n===== ANOMALY ANALYSIS =====")

print("Total anomaly records:", len(anomalies))

print("\nAnomalies by building:")
print(anomalies.groupby("building_id").size())

print("\nAnomaly consumption statistics:")
print(anomalies["consumption_liters"].describe())

print("\nSample anomaly records:")
print(
    anomalies[
        [
            "date",
            "building_id",
            "building_type",
            "occupancy",
            "consumption_liters",
            "synthetic_anomaly_flag"
        ]
    ].head(20).to_string(index=False)
)

print("\nAnalysis completed.")
