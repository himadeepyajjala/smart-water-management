
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. LOAD DATASET AND MODEL
# --------------------------------------------------

df = pd.read_csv("dataset/water_consumption_dataset.csv")
df["date"] = pd.to_datetime(df["date"])
model = joblib.load("model/water_consumption_model.pkl")

latest_date = df["date"].max()

forecast_dates = pd.date_range(
    start=latest_date + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

print("\n===== SMART WATER CONSUMPTION PREDICTION =====")
print("Latest dataset date:", latest_date.date())
print("Forecast period:", forecast_dates[0].date(),
      "to", forecast_dates[-1].date())

# --------------------------------------------------
# 2. SELECT MODE
# --------------------------------------------------

print("\n1. Predict for a specific date")
print("2. Generate a 30-day forecast")

mode = input("Enter choice (1 or 2): ").strip()

if mode not in ("1", "2"):
    raise ValueError("Enter 1 or 2.")

# --------------------------------------------------
# 3. SELECT BUILDING
# --------------------------------------------------

buildings = (
    df[["building_id", "building_type"]]
    .drop_duplicates()
    .sort_values("building_id")
)

print("\nAvailable buildings:")
print(buildings.to_string(index=False))

building_id = input(
    "\nEnter building ID (example: B01): "
).strip().upper()

building_data = df[
    df["building_id"] == building_id
].sort_values("date")

if building_data.empty:
    raise ValueError("Building ID not found.")

last_record = building_data.iloc[-1]
building_type = last_record["building_type"]

# --------------------------------------------------
# 4. INPUT VALIDATION HELPERS
# --------------------------------------------------


def read_occupancy(prompt):
    value = int(input(prompt))
    if value < 0:
        raise ValueError("Occupancy cannot be negative.")
    return value


def read_weather():
    temperature = float(
        input("Expected temperature (C): ")
    )
    rainfall = float(
        input("Expected rainfall (mm): ")
    )

    if not -50 <= temperature <= 60:
        raise ValueError("Temperature must be between -50 and 60 C.")

    if not 0 <= rainfall <= 1000:
        raise ValueError("Rainfall must be between 0 and 1000 mm.")

    return temperature, rainfall


def read_holiday(prompt):
    value = int(input(prompt))
    if value not in (0, 1):
        raise ValueError("Enter 0 for no or 1 for yes.")
    return value


# --------------------------------------------------
# 5. SELECT DATE(S) AND DEFAULT INPUTS
# --------------------------------------------------

if mode == "1":
    print("\nAvailable dates:",
          forecast_dates[0].date(), "to",
          forecast_dates[-1].date())

    date_text = input(
        "Enter prediction date (YYYY-MM-DD): "
    ).strip()

    selected_date = pd.to_datetime(
        date_text, format="%Y-%m-%d", errors="raise"
    )

    if selected_date not in forecast_dates:
        raise ValueError("Date must be within the 30-day period.")

    prediction_dates = [selected_date]

    # Inputs apply only to the selected date
    defaults = {
        "occupancy": read_occupancy("Expected occupancy: ")
    }

    defaults["temperature_c"], defaults["rainfall_mm"] = (
        read_weather()
    )

    defaults["is_long_holiday"] = read_holiday(
        "Is it a long holiday? (0=no, 1=yes): "
    )
    defaults["is_academic_holiday"] = read_holiday(
        "Is it an academic holiday? (0=no, 1=yes): "
    )

    overrides = {}

else:
    prediction_dates = list(forecast_dates)

    print("\nEnter default values for the 30-day forecast.")
    print("You can override selected dates afterward.")

    defaults = {
        "occupancy": read_occupancy(
            "Default daily occupancy: "
        )
    }

    defaults["temperature_c"], defaults["rainfall_mm"] = (
        read_weather()
    )

    defaults["is_long_holiday"] = 0
    defaults["is_academic_holiday"] = 0

    print("\nDefault holiday flags are set to 0 (no).")
    print("Holiday dates can be added as exceptions.")

    # Store date-specific overrides
    overrides = {}

    count = int(input(
        "\nHow many dates need custom inputs? (0-30): "
    ))

    if not 0 <= count <= 30:
        raise ValueError("Enter a number between 0 and 30.")

    valid_dates = {d.strftime("%Y-%m-%d") for d in forecast_dates}

    for _ in range(count):
        date_text = input(
            "\nException date (YYYY-MM-DD): "
        ).strip()

        date_value = pd.to_datetime(
            date_text, format="%Y-%m-%d", errors="raise"
        )

        date_key = date_value.strftime("%Y-%m-%d")

        if date_key not in valid_dates:
            raise ValueError("Exception date is outside the forecast period.")

        if date_key in overrides:
            raise ValueError("This date has already been entered.")

        print("Leave a field blank to keep its default value.")

        changes = {}

        value = input(
            f"Occupancy [{defaults['occupancy']}]: "
        ).strip()
        if value:
            changes["occupancy"] = int(value)
            if changes["occupancy"] < 0:
                raise ValueError("Occupancy cannot be negative.")

        value = input(
            f"Temperature C [{defaults['temperature_c']}]: "
        ).strip()
        if value:
            changes["temperature_c"] = float(value)
            if not -50 <= changes["temperature_c"] <= 60:
                raise ValueError("Invalid temperature.")

        value = input(
            f"Rainfall mm [{defaults['rainfall_mm']}]: "
        ).strip()
        if value:
            changes["rainfall_mm"] = float(value)
            if not 0 <= changes["rainfall_mm"] <= 1000:
                raise ValueError("Invalid rainfall.")

        value = input("Long holiday (0/1) [0]: ").strip()
        if value:
            changes["is_long_holiday"] = int(value)
            if changes["is_long_holiday"] not in (0, 1):
                raise ValueError("Enter 0 or 1.")

        value = input("Academic holiday (0/1) [0]: ").strip()
        if value:
            changes["is_academic_holiday"] = int(value)
            if changes["is_academic_holiday"] not in (0, 1):
                raise ValueError("Enter 0 or 1.")

        overrides[date_key] = changes

# --------------------------------------------------
# 6. GENERATE PREDICTIONS
# --------------------------------------------------

previous_consumption = float(last_record["consumption_liters"])
results = []

for prediction_date in prediction_dates:
    date_key = prediction_date.strftime("%Y-%m-%d")

    # Begin with defaults, then apply any custom inputs
    daily = defaults.copy()
    daily.update(overrides.get(date_key, {}))

    # Calculate weekend automatically
    is_weekend = int(prediction_date.dayofweek >= 5)

    input_data = pd.DataFrame([{
        "occupancy": daily["occupancy"],
        "temperature_c": daily["temperature_c"],
        "rainfall_mm": daily["rainfall_mm"],
        "previous_day_consumption_liters": previous_consumption,
        "is_weekend": is_weekend,
        "is_long_holiday": daily["is_long_holiday"],
        "is_academic_holiday": daily["is_academic_holiday"],
        "building_id": building_id,
        "building_type": building_type
    }])

    prediction = max(0.0, float(model.predict(input_data)[0]))

    results.append({
        "date": prediction_date.date(),
        "building_id": building_id,
        "building_type": building_type,
        "occupancy": daily["occupancy"],
        "temperature_c": daily["temperature_c"],
        "rainfall_mm": daily["rainfall_mm"],
        "is_weekend": is_weekend,
        "is_long_holiday": daily["is_long_holiday"],
        "is_academic_holiday": daily["is_academic_holiday"],
        "predicted_consumption_liters": round(prediction, 2)
    })

    # Feed the predicted consumption into the next day
    previous_consumption = prediction

forecast_df = pd.DataFrame(results)

# --------------------------------------------------
# 7. DISPLAY AND SAVE RESULTS
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

if mode == "1":
    output_file = "outputs/specific_date_prediction.csv"
    graph_file = "outputs/specific_date_prediction.png"
else:
    output_file = f"outputs/{building_id}_30_day_forecast.csv"
    graph_file = f"outputs/{building_id}_30_day_forecast.png"

print("\n===== PREDICTION RESULTS =====")

print(forecast_df[
    ["date", "occupancy", "temperature_c", "rainfall_mm",
     "is_weekend", "is_long_holiday",
     "is_academic_holiday", "predicted_consumption_liters"]
].to_string(index=False))

if mode == "2":
    print(
        "\nTotal predicted consumption:",
        f"{forecast_df['predicted_consumption_liters'].sum():,.2f}",
        "litres"
    )
    print(
        "Average daily consumption:",
        f"{forecast_df['predicted_consumption_liters'].mean():,.2f}",
        "litres/day"
    )

forecast_df.to_csv(output_file, index=False)

plt.figure(figsize=(12, 5))
plt.plot(
    pd.to_datetime(forecast_df["date"]),
    forecast_df["predicted_consumption_liters"],
    marker="o"
)
plt.title(f"Water Consumption Prediction - {building_id}")
plt.xlabel("Date")
plt.ylabel("Predicted consumption (litres/day)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(graph_file, dpi=150)
plt.show()

print("\nCSV saved to:", output_file)
print("Graph saved to:", graph_file)
print("Prediction completed successfully.")
