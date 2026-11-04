
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv("dataset/water_consumption_dataset.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# 2. Sort observations chronologically
df = df.sort_values(["date", "building_id"]).reset_index(drop=True)

# 3. Select features available for prediction
# Exclude synthetic_anomaly_flag because it is a generated label.
# Exclude rolling_7day_avg_liters until we verify how it was calculated.
numeric_features = [
    "occupancy",
    "temperature_c",
    "rainfall_mm",
    "previous_day_consumption_liters",
    "is_weekend",
    "is_long_holiday",
    "is_academic_holiday"
]

categorical_features = [
    "building_id",
    "building_type"
]

target = "consumption_liters"

# 4. Remove rows missing required values
required = numeric_features + categorical_features + [target, "date"]
data = df.dropna(subset=required).copy()

# Convert Boolean features to numeric values
for col in ["is_weekend", "is_long_holiday", "is_academic_holiday"]:
    data[col] = data[col].astype(int)

# 5. Split by DATE, not randomly by row
unique_dates = np.sort(data["date"].unique())
cutoff = unique_dates[int(len(unique_dates) * 0.8)]

train = data[data["date"] < cutoff].copy()
test = data[data["date"] >= cutoff].copy()

features = numeric_features + categorical_features
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training ends:", train["date"].max())
print("Testing starts:", test["date"].min())

# 6. Prepare numerical and categorical features
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"),
     categorical_features),
    ("numeric", "passthrough", numeric_features)
])

# 7. Create two models
baseline = DummyRegressor(strategy="median")

random_forest = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    ))
])

# 8. Train and evaluate
models = {
    "Baseline": baseline,
    "Random Forest": random_forest
}

results = {}

for name, model in models.items():
    if name == "Baseline":
        model.fit(np.zeros((len(X_train), 1)), y_train)
        predictions = model.predict(np.zeros((len(X_test), 1)))
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "predictions": predictions
    }

    print(f"\n{name}")
    print(f"MAE:  {mae:.2f} litres")
    print(f"RMSE: {rmse:.2f} litres")
    print(f"R2:   {r2:.4f}")

# 9. Plot actual versus predicted values
rf_predictions = results["Random Forest"]["predictions"]

plt.figure(figsize=(10, 5))
plt.plot(y_test.to_numpy()[:200], label="Actual")
plt.plot(rf_predictions[:200], label="Random Forest prediction")
plt.title("Actual vs Predicted Water Consumption")
plt.xlabel("Test observations (chronological order)")
plt.ylabel("Consumption (Litres)")
plt.legend()
plt.tight_layout()
plt.savefig("dataset/prediction_graph.png", dpi=150)
plt.show()

# 10. Save the trained model
joblib.dump(random_forest, "model/water_consumption_model.pkl")

print("\nModel saved successfully!")
print("File: model/water_consumption_model.pkl")
