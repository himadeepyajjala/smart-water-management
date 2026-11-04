
import pandas as pd

df = pd.read_csv("dataset/water_consumption_dataset.csv")

print("\n===== NORMAL VS ANOMALOUS RECORDS =====")

comparison = df.groupby("synthetic_anomaly_flag").agg(
    record_count=("consumption_liters", "count"),
    average_consumption=("consumption_liters", "mean"),
    minimum_consumption=("consumption_liters", "min"),
    maximum_consumption=("consumption_liters", "max"),
    average_occupancy=("occupancy", "mean")
).round(2)

print(comparison)

print("\n===== COMPARISON BY BUILDING TYPE =====")

building_comparison = df.groupby(
    ["building_type", "synthetic_anomaly_flag"]
).agg(
    records=("consumption_liters", "count"),
    average_consumption=("consumption_liters", "mean"),
    average_occupancy=("occupancy", "mean")
).round(2)

print(building_comparison.to_string())

print("\nComparison completed.")
