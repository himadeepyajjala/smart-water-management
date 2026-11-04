
import os
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. DATASET ANALYSIS
# ==========================================

def dataset_analysis():

    print("\n===== DATASET ANALYSIS =====")

    dataset_file = "dataset/water_consumption_dataset.csv"

    if not os.path.exists(dataset_file):
        print("Dataset file not found:", dataset_file)
        return

    df = pd.read_csv(dataset_file)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print("\nDataset shape:", df.shape)
    print("Unique buildings:", df["building_id"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nWater consumption statistics:")
    print(df["consumption_liters"].describe())

    # Daily consumption across all buildings
    daily_total = (
        df.groupby("date")["consumption_liters"]
        .sum()
        .sort_index()
    )

    plt.figure(figsize=(12, 5))
    plt.plot(daily_total.index, daily_total.values)

    plt.title("Daily Water Consumption Across All Buildings")
    plt.xlabel("Date")
    plt.ylabel("Total Consumption (Litres)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig("dataset/daily_consumption.png", dpi=150)
    plt.close()

    # Average consumption by building type
    type_usage = (
        df.groupby("building_type")["consumption_liters"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nAverage consumption by building type:")
    print(type_usage.round(2))

    type_usage.plot(kind="bar", figsize=(10, 5))

    plt.title("Average Daily Consumption by Building Type")
    plt.xlabel("Building Type")
    plt.ylabel("Average Consumption (Litres)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig("dataset/building_type_comparison.png", dpi=150)
    plt.close()

    print("\nDataset analysis completed successfully!")
    print("Graphs saved in the dataset folder.")


# ==========================================
# 2. WATER CONSUMPTION PREDICTION
# ==========================================

def water_prediction():

    print("\n===== WATER CONSUMPTION PREDICTION =====")

    try:
        import yearly_predict_updated

        yearly_predict_updated.main()

    except Exception as error:
        print("Prediction module error:", error)


# ==========================================
# 3. ANOMALY DETECTION
# ==========================================

def anomaly_detection():

    print("\n===== WATER CONSUMPTION ANOMALY DETECTION =====")

    try:
        import detect_anomaly

        detect_anomaly.main()

    except Exception as error:
        print("Anomaly detection module error:", error)


# ==========================================
# 4. MAIN APPLICATION MENU
# ==========================================

def main():

    while True:

        print("\n")
        print("=" * 55)
        print(" SMART WATER CONSUMPTION PREDICTION & MANAGEMENT")
        print("=" * 55)

        print("1. Dataset Analysis")
        print("2. Daily / Monthly Water Consumption Prediction")
        print("3. Water Consumption Anomaly Detection")
        print("4. Exit")

        print("=" * 55)

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":

            dataset_analysis()

        elif choice == "2":

            water_prediction()

        elif choice == "3":

            anomaly_detection()

        elif choice == "4":

            print("\nThank you for using Smart Water Management System!")
            break

        else:

            print("\nInvalid choice. Please enter 1, 2, 3 or 4.")


if __name__ == "__main__":
    main()
