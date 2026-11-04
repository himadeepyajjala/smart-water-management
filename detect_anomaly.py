
import joblib
import pandas as pd

MODEL_FILE = "model/anomaly_detection_models.pkl"


def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))

            if value < 0:
                print("Please enter a non-negative value.")
                continue

            return value

        except ValueError:
            print("Invalid input. Enter a numeric value.")


def main():

    # Load trained models
    try:
        saved_data = joblib.load(MODEL_FILE)

    except FileNotFoundError:
        print("Model file not found. Train the model first.")
        return

    building_models = saved_data["models"]
    features = saved_data["features"]

    print("\n===== WATER CONSUMPTION ANOMALY DETECTION =====")

    # Display available buildings
    print("\nAvailable buildings:")

    for building_id, info in building_models.items():
        print(f"{building_id} - {info['building_type']}")

    # Get building ID
    building_id = input("\nEnter building ID: ").strip().upper()

    if building_id not in building_models:
        print("Invalid building ID.")
        return

    # Collect inputs
    occupancy = get_float_input("Enter occupancy: ")
    temperature = get_float_input("Enter temperature (C): ")
    rainfall = get_float_input("Enter rainfall (mm): ")

    consumption = get_float_input(
        "Enter actual water consumption (litres): "
    )

    previous_consumption = get_float_input(
        "Enter previous day's consumption (litres): "
    )

    rolling_average = get_float_input(
        "Enter 7-day rolling average (litres): "
    )

    # Prepare input
    input_data = pd.DataFrame([{
        "occupancy": occupancy,
        "temperature_c": temperature,
        "rainfall_mm": rainfall,
        "consumption_liters": consumption,
        "previous_day_consumption_liters": previous_consumption,
        "rolling_7day_avg_liters": rolling_average
    }])[features]

    # Get building-specific model
    building_info = building_models[building_id]

    scaler = building_info["scaler"]
    model = building_info["model"]

    # Isolation Forest detection
    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]
    anomaly_score = model.decision_function(scaled_input)[0]

    isolation_anomaly = prediction == -1

    # Historical consumption threshold
    threshold = building_info["consumption_threshold"]

    high_consumption = consumption > threshold

    # Hybrid decision
    anomaly_detected = isolation_anomaly or high_consumption

    # Display result
    print("\n===== ANOMALY DETECTION RESULT =====")
    print(f"Building: {building_id}")
    print(f"Building type: {building_info['building_type']}")

    print(f"\nActual consumption: {consumption:.2f} litres")
    print(f"Historical threshold: {threshold:.2f} litres")
    print(f"Isolation Forest score: {anomaly_score:.4f}")

    print("\nDetection details:")
    print(
        "Isolation Forest:",
        "ANOMALY" if isolation_anomaly else "NORMAL"
    )

    print(
        "Consumption threshold:",
        "EXCEEDED" if high_consumption else "NOT EXCEEDED"
    )

    if anomaly_detected:
        print("\nResult: ANOMALY DETECTED")
        print("Warning: Water consumption is unusually high or unusual.")
        print("Recommended action: Inspect the building's water usage.")

    else:
        print("\nResult: NORMAL CONSUMPTION")
        print("Consumption is within the configured detection limits.")

    print("\nNote:")
    print("An anomaly indicates unusual consumption, not proof of a leak.")
    print("The historical threshold is based on normal 2024 training data.")
    print("The threshold is a screening rule, not a calibrated probability.")


if __name__ == "__main__":
    main()
