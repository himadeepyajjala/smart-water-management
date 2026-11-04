import os
import calendar
import pandas as pd
import joblib
import matplotlib.pyplot as plt

DATASET_FILE = "dataset/water_consumption_dataset.csv"
MODEL_FILE = "model/water_consumption_model.pkl"
OUTPUT_DIR = "outputs"
DATES = pd.date_range("2026-01-01", "2026-12-31", freq="D")


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
            limits = ""
            if low is not None:
                limits += f" minimum {low}"
            if high is not None:
                limits += f", maximum {high}"
            print("Enter a valid whole number." + limits)


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
            return pd.to_datetime(input(prompt).strip(), format="%Y-%m-%d")
        except ValueError:
            print("Use YYYY-MM-DD format.")


def main():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}. Run from the project folder.")
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}. Train your model first.")

    df = pd.read_csv(DATASET_FILE)
    df["date"] = pd.to_datetime(df["date"])
    model = joblib.load(MODEL_FILE)

    print("=" * 60)
    print(" SMART WATER CONSUMPTION PREDICTION & MANAGEMENT")
    print(" FULL YEAR 2026 FORECAST")
    print("=" * 60)
    print("Latest dataset date:", df["date"].max().date())
    print("Forecast dates:", DATES[0].date(), "to", DATES[-1].date())
    print("Days to predict:", len(DATES))
    print("Note: These are simulated forecasts, not actual 2026 readings.")

    buildings = (
        df[["building_id", "building_type"]]
        .drop_duplicates()
        .sort_values("building_id")
        .reset_index(drop=True)
    )
    print("\nAvailable buildings:")
    print(buildings.to_string(index=False))
    print("\n1. Forecast one building")
    print("2. Forecast all buildings")
    mode = input("Enter choice (1 or 2): ").strip()
    if mode not in ("1", "2"):
        print("Invalid choice. Run again and enter 1 or 2.")
        return

    if mode == "1":
        bid = input("Enter building ID (e.g. B01): ").strip().upper()
        selected = buildings[buildings["building_id"].astype(
            str).str.upper() == bid]
        if selected.empty:
            print("Building ID not found.")
            return
    else:
        selected = buildings.copy()

    temperature = ask_float("Default temperature (C): ", -50, 60)
    rainfall = ask_float("Default daily rainfall (mm): ", 0, 1000)

    # Occupancy is entered separately for each building.
    occupancies = {}
    print("\nEnter expected daily occupancy for each selected building.")
    for _, row in selected.iterrows():
        bid = str(row["building_id"])
        occupancies[bid] = ask_int(
            f"Occupancy for {bid} ({row['building_type']}): ", 0
        )

    # Exceptions can override weather, holiday flags, and occupancy by building.
    exceptions = {}
    if ask_yes_no("\nAdd any exception dates? (yes/no): "):
        count = ask_int("Number of exception dates (0-365): ", 0, 365)
        valid_dates = set(DATES)
        selected_ids = set(selected["building_id"].astype(str))
        for _ in range(count):
            print("\nException date:")
            date = ask_date("Date (YYYY-MM-DD): ")
            if date not in valid_dates:
                print("Date is outside 2026; skipped.")
                continue
            key = date.strftime("%Y-%m-%d")
            if key in exceptions:
                print("Date already entered; skipped.")
                continue

            item = {"occupancy": {}, "temperature_c": temperature,
                    "rainfall_mm": rainfall, "is_long_holiday": 0,
                    "is_academic_holiday": 0}

            if ask_yes_no("Is this a long holiday? (yes/no): "):
                item["is_long_holiday"] = 1
            if ask_yes_no("Is this an academic holiday? (yes/no): "):
                item["is_academic_holiday"] = 1
            if ask_yes_no("Override temperature for this date? (yes/no): "):
                item["temperature_c"] = ask_float("Temperature (C): ", -50, 60)
            if ask_yes_no("Override rainfall for this date? (yes/no): "):
                item["rainfall_mm"] = ask_float("Rainfall (mm): ", 0, 1000)

            if ask_yes_no("Override occupancy for any building? (yes/no): "):
                while True:
                    override_id = input(
                        f"Building ID {sorted(selected_ids)} (Enter to finish): "
                    ).strip().upper()
                    if not override_id:
                        break
                    if override_id not in selected_ids:
                        print("That building is not in this forecast.")
                        continue
                    item["occupancy"][override_id] = ask_int(
                        f"Occupancy for {override_id}: ", 0
                    )
            exceptions[key] = item

    # Predict sequentially for each building. Each day's prediction becomes
    # the previous-day consumption input for the following day.
    rows = []
    for _, building in selected.iterrows():
        bid = str(building["building_id"])
        btype = str(building["building_type"])
        history = df[df["building_id"].astype(str) == bid].sort_values("date")
        if history.empty:
            print(f"Skipping {bid}: no historical records.")
            continue
        previous = float(history.iloc[-1]["consumption_liters"])
        print(f"\nForecasting {bid} ({btype})...")
        for date in DATES:
            key = date.strftime("%Y-%m-%d")
            ex = exceptions.get(key, {})
            occ = ex.get("occupancy", {}).get(bid, occupancies[bid])
            temp = ex.get("temperature_c", temperature)
            rain = ex.get("rainfall_mm", rainfall)
            long_holiday = ex.get("is_long_holiday", 0)
            academic_holiday = ex.get("is_academic_holiday", 0)
            weekend = int(date.dayofweek >= 5)

            features = pd.DataFrame([{
                "occupancy": occ,
                "temperature_c": temp,
                "rainfall_mm": rain,
                "previous_day_consumption_liters": previous,
                "is_weekend": weekend,
                "is_long_holiday": long_holiday,
                "is_academic_holiday": academic_holiday,
                "building_id": bid,
                "building_type": btype
            }])
            predicted = max(0.0, float(model.predict(features)[0]))
            rows.append({
                "date": date,
                "building_id": bid,
                "building_type": btype,
                "occupancy": occ,
                "temperature_c": temp,
                "rainfall_mm": rain,
                "is_weekend": weekend,
                "is_long_holiday": long_holiday,
                "is_academic_holiday": academic_holiday,
                "predicted_consumption_liters": round(predicted, 2)
            })
            previous = predicted

    forecast = pd.DataFrame(rows)
    if forecast.empty:
        print("No forecasts were generated.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    forecast["month_number"] = forecast["date"].dt.month
    forecast["month"] = forecast["date"].dt.strftime("%B")

    daily_path = os.path.join(OUTPUT_DIR, "2026_daily_forecast.csv")
    forecast.to_csv(daily_path, index=False)

    monthly = (
        forecast.groupby(
            ["month_number", "month", "building_id", "building_type"],
            as_index=False
        )["predicted_consumption_liters"].sum()
        .rename(columns={
            "predicted_consumption_liters": "monthly_predicted_consumption_liters"
        })
        .sort_values(["month_number", "building_id"])
    )
    monthly["monthly_predicted_consumption_liters"] = monthly[
        "monthly_predicted_consumption_liters"
    ].round(2)
    monthly_path = os.path.join(OUTPUT_DIR, "2026_monthly_forecast.csv")
    monthly.to_csv(monthly_path, index=False)

    campus_daily = (
        forecast.groupby("date", as_index=False)[
            "predicted_consumption_liters"]
        .sum()
        .rename(columns={
            "predicted_consumption_liters": "campus_predicted_consumption_liters"
        })
        .sort_values("date")
    )
    campus_daily_path = os.path.join(
        OUTPUT_DIR, "2026_campus_daily_totals.csv")
    campus_daily.to_csv(campus_daily_path, index=False)

    campus_monthly = (
        campus_daily.assign(
            month_number=campus_daily["date"].dt.month,
            month=campus_daily["date"].dt.strftime("%B")
        )
        .groupby(["month_number", "month"], as_index=False)[
            "campus_predicted_consumption_liters"
        ].sum()
        .sort_values("month_number")
    )
    campus_monthly["campus_predicted_consumption_liters"] = campus_monthly[
        "campus_predicted_consumption_liters"
    ].round(2)
    campus_monthly_path = os.path.join(
        OUTPUT_DIR, "2026_campus_monthly_totals.csv")
    campus_monthly.to_csv(campus_monthly_path, index=False)

    annual = (
        forecast.groupby(["building_id", "building_type"], as_index=False)[
            "predicted_consumption_liters"
        ].sum()
        .rename(columns={
            "predicted_consumption_liters": "annual_predicted_consumption_liters"
        })
    )
    annual["annual_predicted_consumption_liters"] = annual[
        "annual_predicted_consumption_liters"
    ].round(2)
    annual_path = os.path.join(OUTPUT_DIR, "2026_annual_summary.csv")
    annual.to_csv(annual_path, index=False)

    # Daily graph, with a line for each building.
    plt.figure(figsize=(14, 6))
    for bid, group in forecast.groupby("building_id"):
        plt.plot(group["date"], group["predicted_consumption_liters"],
                 label=bid, linewidth=1)
    plt.title("Predicted Daily Water Consumption - 2026")
    plt.xlabel("Date")
    plt.ylabel("Predicted consumption (litres/day)")
    plt.grid(True, alpha=0.3)
    if forecast["building_id"].nunique() <= 15:
        plt.legend(title="Building", ncol=2)
    plt.tight_layout()
    daily_graph = os.path.join(OUTPUT_DIR, "2026_daily_forecast.png")
    plt.savefig(daily_graph, dpi=150)
    plt.close()

    # Monthly totals grouped by building.
    pivot = monthly.pivot(
        index="month_number", columns="building_id",
        values="monthly_predicted_consumption_liters"
    ).sort_index()
    ax = pivot.plot(kind="bar", figsize=(14, 6))
    ax.set_title("Monthly Predicted Consumption by Building - 2026")
    ax.set_xlabel("Month")
    ax.set_ylabel("Predicted consumption (litres/month)")
    ax.set_xticklabels([calendar.month_abbr[i]
                       for i in pivot.index], rotation=0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    monthly_graph = os.path.join(OUTPUT_DIR, "2026_monthly_forecast.png")
    plt.savefig(monthly_graph, dpi=150)
    plt.close()

    # Combined campus monthly graph.
    plt.figure(figsize=(12, 5))
    plt.bar(campus_monthly["month"],
            campus_monthly["campus_predicted_consumption_liters"])
    plt.title("Combined Campus Monthly Water Consumption - 2026")
    plt.xlabel("Month")
    plt.ylabel("Predicted consumption (litres/month)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    campus_graph = os.path.join(OUTPUT_DIR, "2026_campus_monthly_totals.png")
    plt.savefig(campus_graph, dpi=150)
    plt.close()

    print("\n========== MONTHLY CAMPUS TOTALS ==========")
    print(campus_monthly.to_string(index=False))
    print("\n========== ANNUAL TOTALS BY BUILDING ==========")
    print(annual.to_string(index=False))
    print(
        "\nCombined predicted campus consumption in 2026: "
        f"{campus_monthly['campus_predicted_consumption_liters'].sum():,.2f} litres"
    )
    print("\nSaved files:")
    for path in [daily_path, monthly_path, campus_daily_path, campus_monthly_path,
                 annual_path, daily_graph, monthly_graph, campus_graph]:
        print(" -", path)
    print("\nForecast completed successfully.")


if __name__ == "__main__":
    main()
