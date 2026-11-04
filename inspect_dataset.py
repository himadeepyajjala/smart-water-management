
import pandas as pd

df = pd.read_csv("dataset/water_consumption_dataset.csv")

print("\n===== DATASET INFORMATION =====")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\n===== BUILDING TYPES =====")
print(df.groupby("building_type")["building_id"].nunique())

print("\n===== SAMPLE RECORDS =====")
print(df.head(10).to_string(index=False))

print("\n===== CONSUMPTION BY BUILDING TYPE =====")
print(
    df.groupby("building_type")["consumption_liters"]
    .agg(["count", "mean", "min", "max"])
    .round(2)
)

print("\n===== HOLIDAY AND WEEKEND DISTRIBUTION =====")
for col in ["is_weekend", "is_long_holiday", "is_academic_holiday"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

print("\n===== ANOMALY FLAG DISTRIBUTION =====")
if "synthetic_anomaly_flag" in df.columns:
    print(df["synthetic_anomaly_flag"].value_counts(dropna=False))

print("\nInspection completed.")
