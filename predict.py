
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("model/water_consumption_model.pkl")

# Load the dataset
df = pd.read_csv("dataset/water_consumption_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

# Select one record with complete input features
features = [
    "occupancy",
    "temperature_c",
    "rainfall_mm",
    "previous_day_consumption_liters",
    "is_weekend",
    "is_long_holiday",
    "is_academic_holiday",
    "building_id",
    "building_type"
]

sample = df.dropna(subset=features).iloc[[100]]

# Generate a prediction
prediction = model.predict(sample[features])[0]

print("Building ID:", sample["building_id"].iloc[0])
print("Building type:", sample["building_type"].iloc[0])
print("Occupancy:", sample["occupancy"].iloc[0])
print("Actual consumption:",
      sample["consumption_liters"].iloc[0], "litres")
print("Predicted consumption:",
      round(prediction, 2), "litres")
