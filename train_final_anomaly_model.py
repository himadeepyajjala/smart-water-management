
import os
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_FILE = "dataset/water_consumption_dataset.csv"
OUTPUT_FILE = "model/anomaly_detection_models.pkl"

CONTAMINATION = 0.03
TRAINING_YEAR = 2024
THRESHOLD_PERCENTILE = 99

FEATURES = [
    "occupancy",
    "temperature_c",
    "rainfall_mm",
    "consumption_liters",
    "previous_day_consumption_liters",
    "rolling_7day_avg_liters"
]

os.makedirs("model", exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATASET_FILE)
df["date"] = pd.to_datetime(df["date"])

required_columns = FEATURES + [
    "building_id",
    "building_type",
    "synthetic_anomaly_flag"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=FEATURES).copy()

# ==========================================
# SELECT NORMAL RECORDS FROM 2024
# ==========================================

train = df[
    (df["date"].dt.year == TRAINING_YEAR) &
    (df["synthetic_anomaly_flag"] == 0)
].copy()

if train.empty:
    raise ValueError("No normal training records found.")

# ==========================================
# TRAIN BUILDING-WISE MODELS
# ==========================================

building_models = {}

print("\n===== HYBRID ANOMALY MODEL TRAINING =====")
print("Training year:", TRAINING_YEAR)
print("Training records: Normal records only")
print("Isolation Forest contamination:", CONTAMINATION)
print("Consumption threshold percentile:", THRESHOLD_PERCENTILE)

for building_id, group in train.groupby("building_id"):

    if len(group) < 10:
        print(f"Skipping {building_id}: insufficient records")
        continue

    scaler = StandardScaler()

    X_train = scaler.fit_transform(group[FEATURES])

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42
    )

    model.fit(X_train)

    # Calculate the building-specific threshold
    consumption_threshold = group["consumption_liters"].quantile(
        THRESHOLD_PERCENTILE / 100
    )

    building_models[building_id] = {
        "scaler": scaler,
        "model": model,
        "building_type": group["building_type"].iloc[0],
        "training_records": len(group),
        "consumption_threshold": float(consumption_threshold)
    }

    print(f"\nBuilding: {building_id}")
    print(f"Building type: {group['building_type'].iloc[0]}")
    print(f"Training records: {len(group)}")
    print(
        f"99th percentile consumption threshold: "
        f"{consumption_threshold:.2f} litres"
    )

# ==========================================
# SAVE MODELS
# ==========================================

saved_data = {
    "models": building_models,
    "features": FEATURES,
    "contamination": CONTAMINATION,
    "training_year": TRAINING_YEAR,
    "threshold_percentile": THRESHOLD_PERCENTILE
}

joblib.dump(saved_data, OUTPUT_FILE)

print("\n===== TRAINING COMPLETED =====")
print("Buildings trained:", len(building_models))
print("Contamination:", CONTAMINATION)
print("Threshold percentile:", THRESHOLD_PERCENTILE)
print("Saved model:", OUTPUT_FILE)
