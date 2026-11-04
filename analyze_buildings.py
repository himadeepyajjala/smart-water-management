
import os
import pandas as pd

# Load original dataset
df = pd.read_csv("dataset/water_consumption_dataset.csv")

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Display available buildings
print("\n========== AVAILABLE BUILDINGS ==========")
print(sorted(df["building_id"].unique()))

# 1. Occupancy and consumption statistics
summary = df.groupby(
    ["building_id", "building_type"]
).agg(
    total_records=("occupancy", "count"),
    min_occupancy=("occupancy", "min"),
    avg_occupancy=("occupancy", "mean"),
    max_occupancy=("occupancy", "max"),
    min_consumption=("consumption_liters", "min"),
    avg_consumption=("consumption_liters", "mean"),
    max_consumption=("consumption_liters", "max")
).round(2)

print("\n========== ALL BUILDINGS ANALYSIS ==========")
print(summary.to_string())

# Save statistics
summary.to_csv("outputs/all_buildings_analysis.csv")

# 2. Occupancy distribution
bins = [-1, 10, 25, 50, 100, 150, 200, 300, 400, 500, 600]

labels = [
    "0-10", "11-25", "26-50", "51-100",
    "101-150", "151-200", "201-300",
    "301-400", "401-500", "501-600"
]

df["occupancy_range"] = pd.cut(
    df["occupancy"],
    bins=bins,
    labels=labels
)

distribution = pd.crosstab(
    df["building_id"],
    df["occupancy_range"]
)

print("\n========== OCCUPANCY DISTRIBUTION ==========")
print(distribution.to_string())

# Save occupancy distribution
distribution.to_csv(
    "outputs/all_buildings_occupancy_distribution.csv"
)

print("\nAnalysis completed successfully!")
print("Files saved in outputs folder.")
