import matplotlib.pyplot as plt
import os
import calendar
import joblib
import pandas as pd

# Prevent Tkinter GUI errors: graphs are saved directly to PNG files.
import matplotlib
matplotlib.use("Agg")

DATASET_FILE = "dataset/water_consumption_dataset.csv"
MODEL_FILE = "model/water_consumption_model.pkl"
OUTPUT_DIR = "outputs"
YEAR = 2026
HOLIDAY_MONTHS = {5, 6, 7, 12}


def ask_int(prompt, low=None, high=None):
    while True:
        try:
            value = int(input(prompt).strip())
            if low is not None and value < low:
                raise ValueError
            if high is not None and value > high:
                raise ValueError
            return value
        except ValueError:
            print("Enter a valid whole number.")


def ask_float(prompt, low, high):
    while True:
        try:
            value = float(input(prompt).strip())
            if low <= value <= high:
                return value
        except ValueError:
            pass
        print(f"Enter a number from {low} to {high}.")


def ask_yes_no(prompt):
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Enter yes or no.")


def ask_date(prompt):
    while True:
        try:
            date = pd.to_datetime(input(prompt).strip(), format="%Y-%m-%d")
            if date.year != YEAR:
                print(f"Enter a date in {YEAR}.")
                continue
            return date
        except ValueError:
            print("Use YYYY-MM-DD format.")


def load_resources():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Prediction model not found: {MODEL_FILE}")

    df = pd.read_csv(DATASET_FILE)
    required = {
        "date", "building_id", "building_type", "consumption_liters",
        "occupancy", "temperature_c", "rainfall_mm",
        "previous_day_consumption_liters", "is_weekend",
        "is_long_holiday", "is_academic_holiday"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    model = joblib.load(MODEL_FILE)
    buildings = (df[["building_id", "building_type"]]
                 .drop_duplicates().sort_values("building_id")
                 .reset_index(drop=True))
    return df, model, buildings


def choose_building(buildings):
    print("\nAvailable buildings:")
    print(buildings.to_string(index=False))
    bid = input("Enter building ID (e.g., B01): ").strip().upper()
    match = buildings[buildings["building_id"].astype(str).str.upper() == bid]
    if match.empty:
        print("Building ID not found.")
        return None
    row = match.iloc[0]
    return str(row["building_id"]), str(row["building_type"])


def predict_one(model, bid, btype, date, occupancy, temperature, rainfall,
                long_holiday, academic_holiday, previous):
    features = pd.DataFrame([{
        "occupancy": occupancy,
        "temperature_c": temperature,
        "rainfall_mm": rainfall,
        "previous_day_consumption_liters": previous,
        "is_weekend": int(date.dayofweek >= 5),
        "is_long_holiday": int(long_holiday),
        "is_academic_holiday": int(academic_holiday),
        "building_id": bid,
        "building_type": btype
    }])
    return round(max(0.0, float(model.predict(features)[0])), 2)


def previous_consumption_for_building(df, bid):
    history = df[df["building_id"].astype(str) == bid].sort_values("date")
    if history.empty:
        return 0.0
    return float(history.iloc[-1]["consumption_liters"])


def save_daily_result(row, filename, title):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, filename)
    pd.DataFrame([row]).to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 4))
    plt.bar([row["date"]], [row["predicted_consumption_liters"]])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Predicted consumption (litres/day)")
    plt.tight_layout()
    png_path = os.path.splitext(csv_path)[0] + ".png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    return csv_path, png_path


def specific_day_mode(df, model, buildings):
    print(f"\n--- SPECIFIC-DATE PREDICTION ({YEAR}) ---")
    date = ask_date(f"Enter prediction date (YYYY-MM-DD, within {YEAR}): ")
    selected = choose_building(buildings)
    if selected is None:
        return
    bid, btype = selected

    occupancy = ask_int("Expected occupancy: ", 0)
    temperature = ask_float("Temperature (C): ", -50, 60)
    rainfall = ask_float("Rainfall (mm): ", 0, 1000)
    long_holiday = ask_yes_no("Is it a long holiday? (yes/no): ")
    academic_holiday = ask_yes_no("Is it an academic holiday? (yes/no): ")
    previous = previous_consumption_for_building(df, bid)

    prediction = predict_one(
        model, bid, btype, date, occupancy, temperature, rainfall,
        long_holiday, academic_holiday, previous
    )
    row = {
        "date": date.strftime("%Y-%m-%d"),
        "building_id": bid,
        "building_type": btype,
        "occupancy": occupancy,
        "temperature_c": temperature,
        "rainfall_mm": rainfall,
        "is_weekend": int(date.dayofweek >= 5),
        "is_long_holiday": int(long_holiday),
        "is_academic_holiday": int(academic_holiday),
        "previous_day_consumption_liters": round(previous, 2),
        "predicted_consumption_liters": prediction
    }

    print("\n========== PREDICTION RESULT ==========")
    print(f"Date: {row['date']}")
    print(f"Building: {bid} ({btype})")
    print(f"Weekend: {'Yes' if row['is_weekend'] else 'No'}")
    print(f"Predicted water consumption: {prediction:,.2f} litres")
    csv_path, png_path = save_daily_result(
        row, f"specific_day_{YEAR}_prediction_{bid}.csv",
        f"Predicted Water Consumption - {bid} - {row['date']}"
    )
    print(f"Saved CSV: {csv_path}")
    print(f"Saved graph: {png_path}")
    print("\nNote: the dataset ends in 2025. This 2026 result is a simulated forecast.")
    print("The previous-day input uses the latest historical reading because")
    print("an actual reading immediately before the 2026 date is unavailable.")


def monthly_mode(df, model, buildings):
    print(f"\n--- MONTHLY PREDICTION ({YEAR}) ---")
    month = ask_int("Enter month number (1=January ... 12=December): ", 1, 12)
    selected = choose_building(buildings)
    if selected is None:
        return
    bid, btype = selected

    default_academic = month in HOLIDAY_MONTHS
    print(f"Project default for {calendar.month_name[month]}: "
          f"{'Academic holiday' if default_academic else 'Regular academic month'}")
    academic_holiday = ask_yes_no(
        f"Use the default academic-holiday setting ({'yes' if default_academic else 'no'}) "
        "for the whole month? (yes/no): "
    )
    if not academic_holiday:
        academic_holiday = not default_academic

    long_holiday = ask_yes_no(
        "Treat the whole month as a long holiday? (yes/no): ")
    occupancy = ask_int("Default daily occupancy: ", 0)
    temperature = ask_float("Default temperature (C): ", -50, 60)
    rainfall = ask_float("Default daily rainfall (mm): ", 0, 1000)

    overrides = {}
    if ask_yes_no("Add date-specific overrides within this month? (yes/no): "):
        count = ask_int("Number of dates to override: ",
                        1, calendar.monthrange(YEAR, month)[1])
        for _ in range(count):
            date = ask_date("Override date (YYYY-MM-DD): ")
            if date.month != month:
                print("That date is not in the selected month; skipped.")
                continue
            key = date.strftime("%Y-%m-%d")
            item = {}
            if ask_yes_no("Override occupancy? (yes/no): "):
                item["occupancy"] = ask_int("Occupancy: ", 0)
            if ask_yes_no("Override temperature? (yes/no): "):
                item["temperature_c"] = ask_float("Temperature (C): ", -50, 60)
            if ask_yes_no("Override rainfall? (yes/no): "):
                item["rainfall_mm"] = ask_float("Rainfall (mm): ", 0, 1000)
            if ask_yes_no("Override long-holiday flag? (yes/no): "):
                item["is_long_holiday"] = int(
                    ask_yes_no("Is it a long holiday? (yes/no): "))
            if ask_yes_no("Override academic-holiday flag? (yes/no): "):
                item["is_academic_holiday"] = int(
                    ask_yes_no("Is it an academic holiday? (yes/no): "))
            overrides[key] = item

    days = pd.date_range(
        f"{YEAR}-{month:02d}-01",
        periods=calendar.monthrange(YEAR, month)[1], freq="D"
    )
    previous = previous_consumption_for_building(df, bid)
    rows = []

    for date in days:
        key = date.strftime("%Y-%m-%d")
        override = overrides.get(key, {})
        occ = override.get("occupancy", occupancy)
        temp = override.get("temperature_c", temperature)
        rain = override.get("rainfall_mm", rainfall)
        day_long = override.get("is_long_holiday", int(long_holiday))
        day_academic = override.get(
            "is_academic_holiday", int(academic_holiday))

        prediction = predict_one(
            model, bid, btype, date, occ, temp, rain,
            day_long, day_academic, previous
        )
        rows.append({
            "date": key, "building_id": bid, "building_type": btype,
            "occupancy": occ, "temperature_c": temp, "rainfall_mm": rain,
            "is_weekend": int(date.dayofweek >= 5),
            "is_long_holiday": day_long,
            "is_academic_holiday": day_academic,
            "previous_day_consumption_liters": round(previous, 2),
            "predicted_consumption_liters": prediction
        })
        previous = prediction

    forecast = pd.DataFrame(rows)
    total = round(float(forecast["predicted_consumption_liters"].sum()), 2)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    month_name = calendar.month_name[month].lower()

    daily_path = os.path.join(
        OUTPUT_DIR, f"{YEAR}_{month_name}_daily_predictions_{bid}.csv")
    forecast.to_csv(daily_path, index=False)

    summary_path = os.path.join(
        OUTPUT_DIR, f"{YEAR}_{month_name}_monthly_summary_{bid}.csv")
    pd.DataFrame([{
        "year": YEAR, "month_number": month,
        "month": calendar.month_name[month], "building_id": bid,
        "building_type": btype, "days_in_month": len(days),
        "monthly_predicted_consumption_liters": total
    }]).to_csv(summary_path, index=False)

    graph_path = os.path.join(
        OUTPUT_DIR, f"{YEAR}_{month_name}_daily_predictions_{bid}.png")
    plt.figure(figsize=(12, 5))
    plt.plot(forecast["date"], forecast["predicted_consumption_liters"],
             marker="o", markersize=3)
    plt.title(
        f"Daily Predicted Water Consumption - {calendar.month_name[month]} {YEAR} - {bid}")
    plt.xlabel("Date")
    plt.ylabel("Predicted consumption (litres/day)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graph_path, dpi=150)
    plt.close()

    print("\n========== MONTHLY PREDICTION RESULT ==========")
    print(f"Building: {bid} ({btype})")
    print(f"Month: {calendar.month_name[month]} {YEAR}")
    print(f"Days forecast: {len(days)}")
    print(f"Predicted monthly consumption: {total:,.2f} litres")
    print(f"Average predicted daily consumption: "
          f"{forecast['predicted_consumption_liters'].mean():,.2f} litres/day")
    print(f"Saved daily predictions: {daily_path}")
    print(f"Saved monthly summary: {summary_path}")
    print(f"Saved graph: {graph_path}")
    print("\nNote: these are simulated forecasts, not actual measured 2026 consumption.")


def main():
    print("=" * 62)
    print(" SMART WATER CONSUMPTION PREDICTION & MANAGEMENT")
    print(f" {YEAR} DATE AND MONTHLY PREDICTION")
    print("=" * 62)
    print("1. Predict water consumption for a specific date in 2026")
    print("2. Predict monthly water consumption for a selected month in 2026")
    choice = input("Enter your choice (1 or 2): ").strip()
    if choice not in ("1", "2"):
        print("Invalid choice. Run again and enter 1 or 2.")
        return

    df, model, buildings = load_resources()
    print(f"\nLatest dataset date: {df['date'].max().date()}")
    print("Note: the dataset ends in 2025; all 2026 values are forecasts.")
    if choice == "1":
        specific_day_mode(df, model, buildings)
    else:
        monthly_mode(df, model, buildings)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrediction cancelled by user.")
    except Exception as error:
        print(f"\nERROR: {error}")
