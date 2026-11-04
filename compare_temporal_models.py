
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

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

train = df[df["date"].dt.year == 2024]
test = df[df["date"].dt.year == 2025].copy()

# -------------------------------
# GLOBAL ISOLATION FOREST
# -------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(train[features])
X_test = scaler.transform(test[features])

global_model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

global_model.fit(X_train)

test["global_prediction"] = (
    global_model.predict(X_test) == -1
).astype(int)

# -------------------------------
# BUILDING-WISE ISOLATION FOREST
# -------------------------------

test["building_prediction"] = 0

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
        contamination=0.02,
        random_state=42
    )

    model.fit(X_train)

    test.loc[mask, "building_prediction"] = (
        model.predict(X_test) == -1
    ).astype(int)

# -------------------------------
# EVALUATION
# -------------------------------

actual = test["synthetic_anomaly_flag"]

print("\n===== TEMPORAL MODEL COMPARISON =====")

for name, column in [
    ("Global Isolation Forest", "global_prediction"),
    ("Building-wise Isolation Forest", "building_prediction")
]:

    prediction = test[column]

    print(f"\n{name}")

    print("Detected anomalies:", int(prediction.sum()))

    print("Precision:", round(
        precision_score(actual, prediction, zero_division=0), 4
    ))

    print("Recall:", round(
        recall_score(actual, prediction, zero_division=0), 4
    ))

    print("F1 Score:", round(
        f1_score(actual, prediction, zero_division=0), 4
    ))

test.to_csv(
    "outputs/temporal_model_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")
