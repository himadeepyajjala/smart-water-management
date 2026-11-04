
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/water_consumption_dataset.csv")

# Select B02 records
b02 = df[df["building_id"] == "B02"].copy()

# Separate normal and synthetic anomaly records
normal = b02[b02["synthetic_anomaly_flag"] == 0]
anomalies = b02[b02["synthetic_anomaly_flag"] == 1]

print("\n===== B02 CONSUMPTION ANALYSIS =====")

print("\nNormal consumption statistics:")
print(normal["consumption_liters"].describe())

print("\nSynthetic anomaly consumption statistics:")
print(anomalies["consumption_liters"].describe())

print("\nHighest 10 normal consumption values:")
print(
    normal["consumption_liters"]
    .nlargest(10)
    .to_string(index=False)
)

print("\nHighest 10 synthetic anomaly values:")
print(
    anomalies["consumption_liters"]
    .nlargest(10)
    .to_string(index=False)
)
