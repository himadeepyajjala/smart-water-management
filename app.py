import os

import pickle

import calendar

import joblib

import streamlit as st

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import date


# --------------------------------------------------

# PAGE CONFIGURATION

# --------------------------------------------------


st.set_page_config(

    page_title="Smart Water Management",

    page_icon="💧",

    layout="wide",

    initial_sidebar_state="expanded"

)


# --------------------------------------------------

# FILE PATHS

# --------------------------------------------------


DATASET_FILE = "dataset/water_consumption_dataset.csv"

MODEL_FILE = "model/water_consumption_model.pkl"

OUTPUT_DIR = "outputs"


# --------------------------------------------------

# CUSTOM CSS

# --------------------------------------------------


st.markdown("""

<style>

.main {

          background-color: #0e1117;

}

[data-testid="stSidebar"] {

          background-color: #172536;

}

.metric-card {

          background-color: #192c40;

          padding: 18px;

          border-radius: 12px;

}

h1, h2, h3 {

          color: #64c8ff;

}

</style>

""", unsafe_allow_html=True)


# --------------------------------------------------

# LOAD DATASET

# --------------------------------------------------


@st.cache_data
def load_data():

    if not os.path.exists(DATASET_FILE):

        raise FileNotFoundError(

            f"Dataset not found: {DATASET_FILE}"

        )

    data = pd.read_csv(DATASET_FILE)

    data["date"] = pd.to_datetime(data["date"])

    return data


try:

    df = load_data()

except Exception as error:

    st.error(f"Dataset loading failed: {error}")

    st.stop()


# --------------------------------------------------

# LOAD RANDOM FOREST MODEL

# --------------------------------------------------


@st.cache_resource
def load_prediction_model():

    if not os.path.exists(MODEL_FILE):

        return None

    try:

        return joblib.load(MODEL_FILE)

    except Exception:

        try:

            with open(MODEL_FILE, "rb") as file:

                return pickle.load(file)

        except Exception as error:

            st.error(f"Prediction model loading failed: {error}")

            return None


prediction_model = load_prediction_model()


# --------------------------------------------------

# BUILDING INFORMATION

# --------------------------------------------------


building_info = (

    df[["building_id", "building_type"]]

    .drop_duplicates()

    .sort_values("building_id")

)


building_ids = building_info["building_id"].tolist()


building_type_map = dict(

    zip(

        building_info["building_id"],

        building_info["building_type"]

    )

)


# --------------------------------------------------

# SIDEBAR NAVIGATION

# --------------------------------------------------


# --------------------------------------------------

# INTERACTIVE SIDEBAR

# --------------------------------------------------


st.markdown("""

<style>

[data-testid="stSidebar"] {

          background: linear-gradient(180deg, #14283d 0%, #0d1b2a 100%);

}



[data-testid="stSidebar"] h1,

[data-testid="stSidebar"] h2,

[data-testid="stSidebar"] h3 {

          color: #ffffff;

}



[data-testid="stSidebar"] p {

          color: #d5e3f0;

}



[data-testid="stSidebar"] [role="radiogroup"] {

          gap: 12px;

}



[data-testid="stSidebar"] [role="radiogroup"] label {

          background: rgba(255,255,255,0.06);

          border: 1px solid rgba(255,255,255,0.12);

          border-radius: 10px;

          padding: 10px 12px;

          transition: all 0.2s ease;

}



[data-testid="stSidebar"] [role="radiogroup"] label:hover {

          background: rgba(59, 130, 246, 0.25);

          border-color: #60a5fa;

}



[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {

          background: #2563eb;

          border-color: #60a5fa;

          color: white;

          font-weight: 600;

}



.sidebar-footer {

          color: #9fb3c8;

          font-size: 12px;

          border-top: 1px solid #38516a;

          padding-top: 15px;

          margin-top: 25px;

}

</style>

""", unsafe_allow_html=True)


with st.sidebar:

    st.title("💧 Smart Water")

    st.subheader("Management System")

    st.caption("Campus Water Intelligence Platform")

    st.divider()

    st.markdown("### 🧭 NAVIGATION")

    pages = {

        "🏠 Dashboard": "Dashboard",

        "📊 Dataset Analysis": "Dataset Analysis",

        "🔮 Water Prediction": "Water Prediction",

        "🚨 Anomaly Detection": "Anomaly Detection"

    }

    selected_page = st.radio(

        "Select a page",

        options=list(pages.keys()),

        label_visibility="collapsed"

    )

    page = pages[selected_page]

    st.divider()

    st.markdown("### 📌 PROJECT OVERVIEW")

    st.metric("Total Records", f"{len(df):,}")

    st.metric("Total Buildings", df["building_id"].nunique())

    st.metric(

        "Average Daily Building Consumption",

        f"{df['consumption_liters'].mean():,.0f} L"

    )

    anomaly_count = int(df["synthetic_anomaly_flag"].sum())

    st.metric("Synthetic Anomalies", anomaly_count)

    st.divider()

    st.markdown("### ⚙️ SYSTEM INFORMATION")

    st.caption("🤖 Prediction Model: Random Forest")

    st.caption("🔍 Anomaly Model: Isolation Forest")

    st.caption("📁 Dataset: Synthetic Campus Data")

    st.markdown("""

          <div class="sidebar-footer">

                    💧 Smart Water Consumption<br>

                    Prediction & Management<br><br>

                    AI-Based Campus Project

          </div>

          """, unsafe_allow_html=True)


# --------------------------------------------------

# DASHBOARD

# --------------------------------------------------


if page == "Dashboard":

    st.title("💧 Smart Water Consumption Prediction & Management")

    st.write(

        "AI-based campus water consumption prediction "

        "and anomaly detection system."

    )

    total_records = len(df)

    total_buildings = df["building_id"].nunique()

    average_consumption = df["consumption_liters"].mean()

    total_anomalies = int(df["synthetic_anomaly_flag"].sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{total_records:,}")

    col2.metric("Buildings", total_buildings)

    col3.metric(

        "Average Consumption",

        f"{average_consumption:,.0f} L"

    )

    col4.metric("Synthetic Anomalies", total_anomalies)

    st.markdown("---")

    st.subheader("Daily Campus Water Consumption")

    daily_usage = (

        df.groupby("date")["consumption_liters"]

        .sum()

        .sort_index()

    )

    st.line_chart(daily_usage)

    st.subheader("Average Consumption by Building")

    building_usage = (

        df.groupby("building_id")["consumption_liters"]

        .mean()

        .sort_values(ascending=False)

    )

    st.bar_chart(building_usage)

    st.subheader("Consumption by Building Type")

    type_usage = (

        df.groupby("building_type")["consumption_liters"]

        .mean()

        .sort_values(ascending=False)

    )

    st.bar_chart(type_usage)

    st.caption(

        "Note: Dashboard statistics are calculated from the "

        "synthetic dataset."

    )


# --------------------------------------------------

# DATASET ANALYSIS

# --------------------------------------------------


elif page == "Dataset Analysis":

    st.title("📊 Dataset Analysis")

    st.write(

        "Explore campus water consumption records and "

        "understand the dataset."

    )

    col1, col2 = st.columns(2)

    with col1:

        selected_building = st.selectbox(

            "Select Building",

            ["All Buildings"] + building_ids

        )

    with col2:

        selected_type = st.selectbox(

            "Select Building Type",

            ["All Types"] + sorted(df["building_type"].unique())

        )

    filtered_df = df.copy()

    if selected_building != "All Buildings":

        filtered_df = filtered_df[

            filtered_df["building_id"] == selected_building

        ]

    if selected_type != "All Types":

        filtered_df = filtered_df[

            filtered_df["building_type"] == selected_type

        ]

    st.subheader("Dataset Records")

    records_per_building = st.selectbox(
        "Records to display per building",
        [5, 10, 20, 50],
        index=1
    )

    if selected_building == "All Buildings":
        preview_df = (
            filtered_df.sort_values(["building_id", "date"])
            .groupby("building_id", group_keys=False)
            .head(records_per_building)
        )
    else:
        preview_df = filtered_df.sort_values("date").head(
            records_per_building
        )

    st.dataframe(
        preview_df,
        use_container_width=True,
        height=420
    )

    st.caption(
        f"Showing {len(preview_df):,} sample records from "
        f"{filtered_df['building_id'].nunique()} matching building(s). "
        f"Total matching records: {len(filtered_df):,}"
    )

    with st.expander("View all matching records"):
        st.dataframe(
            filtered_df.sort_values(["building_id", "date"]),
            use_container_width=True,
            height=450
        )

    st.subheader("Consumption Statistics")

    st.dataframe(

        filtered_df["consumption_liters"].describe().to_frame(),

        use_container_width=True

    )

    st.subheader("Consumption by Building")

    usage = (

        filtered_df.groupby("building_id")["consumption_liters"]

        .mean()

        .sort_values(ascending=False)

    )

    st.bar_chart(usage)

    st.subheader("Anomaly Distribution")

    anomaly_counts = (

        filtered_df["synthetic_anomaly_flag"]

        .value_counts()

        .rename(index={0: "Normal", 1: "Anomaly"})

    )

    st.bar_chart(anomaly_counts)

    st.download_button(

        "Download Filtered Dataset",

        filtered_df.to_csv(index=False),

        file_name="filtered_water_dataset.csv",

        mime="text/csv"

    )


# --------------------------------------------------

# WATER PREDICTION

# --------------------------------------------------


elif page == "Water Prediction":

    st.title("🔮 Water Consumption Prediction")

    if prediction_model is None:
        st.error(
            "The Random Forest model could not be loaded. "
            "Check model/water_consumption_model.pkl."
        )
        st.stop()

    st.write(
        "Choose daily prediction for one date or monthly prediction "
        "for a complete month in 2026."
    )

    prediction_mode = st.radio(
        "Prediction Type",
        ["Daily Prediction", "Monthly Prediction"],
        horizontal=True,
        key="prediction_mode"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------
    # DAILY PREDICTION
    # --------------------------------------------

    if prediction_mode == "Daily Prediction":

        st.subheader("Daily Water Consumption Prediction")

        with st.form("daily_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:
                selected_date = st.date_input(
                    "Prediction Date",
                    value=date(2026, 1, 1),
                    min_value=date(2026, 1, 1),
                    max_value=date(2026, 12, 31)
                )

                building_id = st.selectbox(
                    "Building",
                    building_ids,
                    key="daily_building"
                )

                occupancy = st.number_input(
                    "Expected Occupancy",
                    min_value=0,
                    max_value=10000,
                    value=100,
                    key="daily_occupancy"
                )

                temperature = st.number_input(
                    "Temperature (°C)",
                    min_value=-10.0,
                    max_value=60.0,
                    value=30.0,
                    key="daily_temperature"
                )

            with col2:
                rainfall = st.number_input(
                    "Rainfall (mm)",
                    min_value=0.0,
                    max_value=500.0,
                    value=0.0,
                    key="daily_rainfall"
                )

                long_holiday = st.selectbox(
                    "Long Holiday",
                    [0, 1],
                    format_func=lambda x: "Yes" if x else "No",
                    key="daily_long_holiday"
                )

                academic_holiday = st.selectbox(
                    "Academic Holiday",
                    [0, 1],
                    format_func=lambda x: "Yes" if x else "No",
                    key="daily_academic_holiday"
                )

            submit_daily = st.form_submit_button(
                "Predict Daily Water Consumption",
                type="primary",
                use_container_width=True
            )

        if submit_daily:

            building_type = building_type_map[building_id]
            historical = (
                df[df["building_id"] == building_id]
                .sort_values("date")
            )
            previous_consumption = float(
                historical["consumption_liters"].iloc[-1]
            )

            input_data = pd.DataFrame([{
                "occupancy": occupancy,
                "temperature_c": temperature,
                "rainfall_mm": rainfall,
                "previous_day_consumption_liters": previous_consumption,
                "is_weekend": int(selected_date.weekday() >= 5),
                "is_long_holiday": long_holiday,
                "is_academic_holiday": academic_holiday,
                "building_id": building_id,
                "building_type": building_type
            }])

            try:
                predicted_value = float(
                    prediction_model.predict(input_data)[0]
                )

                st.success("Daily prediction completed successfully!")

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Predicted Consumption",
                    f"{predicted_value:,.2f} L"
                )
                col2.metric("Building", building_id)
                col3.metric("Building Type", building_type)

                st.info(
                    f"Prediction date: {selected_date.strftime('%d %B %Y')}"
                )

                result = input_data.copy()
                result["prediction_date"] = selected_date
                result["predicted_consumption_liters"] = predicted_value

                st.download_button(
                    "Download Daily Prediction",
                    result.to_csv(index=False),
                    file_name=f"daily_prediction_{building_id}_{selected_date}.csv",
                    mime="text/csv"
                )

                st.caption(
                    "This is a simulated forecast. Previous-day consumption "
                    "uses the latest available historical reading in the "
                    "dataset, not an actual reading immediately before the "
                    "prediction date."
                )

            except Exception as error:
                st.error(f"Daily prediction failed: {error}")

    # --------------------------------------------
    # MONTHLY PREDICTION
    # --------------------------------------------

    else:

        st.subheader("Monthly Water Consumption Prediction")
        st.write(
            "The system predicts consumption for every day of the selected "
            "month and adds the daily predictions to estimate monthly usage."
        )

        with st.form("monthly_prediction_form"):

            col1, col2 = st.columns(2)

            with col1:
                selected_month = st.selectbox(
                    "Select Month (2026)",
                    options=list(range(1, 13)),
                    format_func=lambda month: calendar.month_name[month],
                    key="monthly_month"
                )

                monthly_building = st.selectbox(
                    "Select Building",
                    building_ids,
                    key="monthly_building"
                )

                monthly_occupancy = st.number_input(
                    "Default Daily Occupancy",
                    min_value=0,
                    max_value=10000,
                    value=100,
                    key="monthly_occupancy"
                )

                monthly_temperature = st.number_input(
                    "Default Temperature (°C)",
                    min_value=-10.0,
                    max_value=60.0,
                    value=30.0,
                    key="monthly_temperature"
                )

            with col2:
                monthly_rainfall = st.number_input(
                    "Default Daily Rainfall (mm)",
                    min_value=0.0,
                    max_value=500.0,
                    value=0.0,
                    key="monthly_rainfall"
                )

                default_academic_holiday = selected_month in [5, 6, 7, 12]
                monthly_academic_holiday = st.selectbox(
                    "Academic Holiday for the Month",
                    [0, 1],
                    index=1 if default_academic_holiday else 0,
                    format_func=lambda x: "Yes" if x else "No",
                    key="monthly_academic_holiday"
                )

                monthly_long_holiday = st.selectbox(
                    "Treat the Whole Month as a Long Holiday?",
                    [0, 1],
                    format_func=lambda x: "Yes" if x else "No",
                    key="monthly_long_holiday"
                )

            submit_monthly = st.form_submit_button(
                "Predict Monthly Water Consumption",
                type="primary",
                use_container_width=True
            )

        if submit_monthly:

            try:
                building_type = building_type_map[monthly_building]
                number_of_days = calendar.monthrange(2026, selected_month)[1]

                historical = (
                    df[df["building_id"] == monthly_building]
                    .sort_values("date")
                )
                previous_consumption = float(
                    historical["consumption_liters"].iloc[-1]
                )

                daily_results = []

                progress = st.progress(
                    0, text="Generating daily predictions...")

                for day in range(1, number_of_days + 1):

                    prediction_date = date(2026, selected_month, day)

                    input_data = pd.DataFrame([{
                        "occupancy": monthly_occupancy,
                        "temperature_c": monthly_temperature,
                        "rainfall_mm": monthly_rainfall,
                        "previous_day_consumption_liters": previous_consumption,
                        "is_weekend": int(prediction_date.weekday() >= 5),
                        "is_long_holiday": monthly_long_holiday,
                        "is_academic_holiday": monthly_academic_holiday,
                        "building_id": monthly_building,
                        "building_type": building_type
                    }])

                    predicted_value = float(
                        prediction_model.predict(input_data)[0]
                    )

                    daily_results.append({
                        "date": prediction_date,
                        "building_id": monthly_building,
                        "building_type": building_type,
                        "occupancy": monthly_occupancy,
                        "temperature_c": monthly_temperature,
                        "rainfall_mm": monthly_rainfall,
                        "is_weekend": int(prediction_date.weekday() >= 5),
                        "is_long_holiday": monthly_long_holiday,
                        "is_academic_holiday": monthly_academic_holiday,
                        "previous_day_consumption_liters": previous_consumption,
                        "predicted_consumption_liters": predicted_value
                    })

                    # Feed today's prediction into the next day's prediction.
                    previous_consumption = predicted_value
                    progress.progress(
                        day / number_of_days,
                        text=f"Predicting day {day} of {number_of_days}..."
                    )

                daily_df = pd.DataFrame(daily_results)
                monthly_total = float(
                    daily_df["predicted_consumption_liters"].sum()
                )
                average_daily = float(
                    daily_df["predicted_consumption_liters"].mean()
                )

                month_name = calendar.month_name[selected_month]
                daily_filename = (
                    f"2026_{month_name.lower()}_daily_predictions_"
                    f"{monthly_building}.csv"
                )
                summary_filename = (
                    f"2026_{month_name.lower()}_monthly_summary_"
                    f"{monthly_building}.csv"
                )
                graph_filename = (
                    f"2026_{month_name.lower()}_daily_predictions_"
                    f"{monthly_building}.png"
                )

                daily_path = os.path.join(OUTPUT_DIR, daily_filename)
                summary_path = os.path.join(OUTPUT_DIR, summary_filename)
                graph_path = os.path.join(OUTPUT_DIR, graph_filename)

                daily_df.to_csv(daily_path, index=False)

                summary_df = pd.DataFrame([{
                    "building_id": monthly_building,
                    "building_type": building_type,
                    "year": 2026,
                    "month": month_name,
                    "number_of_days": number_of_days,
                    "predicted_monthly_consumption_liters": monthly_total,
                    "average_daily_consumption_liters": average_daily,
                    "academic_holiday": monthly_academic_holiday,
                    "long_holiday": monthly_long_holiday
                }])
                summary_df.to_csv(summary_path, index=False)

                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(
                    daily_df["date"],
                    daily_df["predicted_consumption_liters"],
                    marker="o"
                )
                ax.set_title(
                    f"Daily Predicted Water Consumption - "
                    f"{monthly_building} - {month_name} 2026"
                )
                ax.set_xlabel("Date")
                ax.set_ylabel("Predicted Consumption (Litres)")
                ax.grid(True, alpha=0.3)
                fig.autofmt_xdate()
                fig.tight_layout()
                fig.savefig(graph_path, dpi=150)
                plt.close(fig)

                st.success("Monthly prediction completed successfully!")

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Predicted Monthly Consumption",
                    f"{monthly_total:,.2f} L"
                )
                col2.metric(
                    "Average Daily Consumption",
                    f"{average_daily:,.2f} L/day"
                )
                col3.metric("Number of Days", number_of_days)

                st.info(
                    f"Building: {monthly_building} ({building_type}) | "
                    f"Month: {month_name} 2026"
                )

                st.subheader("Daily Prediction Graph")
                st.image(
                    graph_path,
                    caption="Daily Predicted Water Consumption",
                    use_container_width=True
                )

                st.subheader("Daily Prediction Records")
                st.dataframe(daily_df, use_container_width=True)

                download_col1, download_col2 = st.columns(2)
                with download_col1:
                    st.download_button(
                        "Download Daily Predictions CSV",
                        daily_df.to_csv(index=False),
                        file_name=daily_filename,
                        mime="text/csv",
                        use_container_width=True
                    )
                with download_col2:
                    st.download_button(
                        "Download Monthly Summary CSV",
                        summary_df.to_csv(index=False),
                        file_name=summary_filename,
                        mime="text/csv",
                        use_container_width=True
                    )

                st.caption(
                    "This is a simulated forecast, not actual measured 2026 "
                    "consumption. The first day's previous consumption uses "
                    "the latest available historical reading. Each later "
                    "day uses the previous day's predicted consumption. "
                    "The academic-holiday default for May, June, July and "
                    "December follows the project's current assumption; "
                    "change it if your institution's calendar differs."
                )

            except Exception as error:
                st.error(f"Monthly prediction failed: {error}")


# --------------------------------------------------


# ANOMALY DETECTION

# --------------------------------------------------


elif page == "Anomaly Detection":

    st.title("🚨 Water Consumption Anomaly Detection")

    st.write(

        "View the anomaly detection results produced by "

        "the trained anomaly detection models."

    )

    prediction_file = os.path.join(

        OUTPUT_DIR,

        "final_anomaly_predictions_2025.csv"

    )

    metrics_file = os.path.join(

        OUTPUT_DIR,

        "final_anomaly_metrics_2025.csv"

    )

    if os.path.exists(metrics_file):

        st.subheader("Model Evaluation Metrics")

        metrics_df = pd.read_csv(metrics_file)

        st.dataframe(metrics_df, use_container_width=True)

    else:

        st.warning(

            "Anomaly metrics file was not found in the outputs folder."

        )

    if os.path.exists(prediction_file):

        anomaly_df = pd.read_csv(prediction_file)

        st.subheader("Anomaly Detection Results")

        st.dataframe(

            anomaly_df.head(100),

            use_container_width=True

        )

        st.download_button(

            "Download Anomaly Results",

            anomaly_df.to_csv(index=False),

            file_name="anomaly_detection_results.csv",

            mime="text/csv"

        )

    else:

        st.warning(

            "The final anomaly predictions file was not found."

        )

    st.subheader("Anomaly Detection Graphs")

    graph_files = [

        ("Confusion Matrix", "anomaly_confusion_matrix.png"),

        ("Building-wise F1 Score", "building_wise_f1_score.png"),

        ("Actual vs Detected Anomalies", "actual_vs_detected_anomalies.png"),

        ("Hybrid Confusion Matrix", "hybrid_confusion_matrix.png"),

        ("Hybrid Method Comparison", "hybrid_method_comparison.png"),

        ("Hybrid Actual vs Detected", "hybrid_actual_vs_detected.png")

    ]

    available_graphs = [

        (title, os.path.join(OUTPUT_DIR, filename))

        for title, filename in graph_files

        if os.path.exists(os.path.join(OUTPUT_DIR, filename))

    ]

    if available_graphs:

        for index in range(0, len(available_graphs), 2):

            columns = st.columns(2)

            for column, (title, filepath) in zip(

                columns,

                available_graphs[index:index + 2]

            ):

                with column:

                    st.markdown(f"**{title}**")

                    st.image(filepath, use_container_width=True)

    else:

        st.info(

            "No anomaly graphs were found in the outputs folder."

        )

    st.caption(

        "The anomaly results are evaluated against synthetic anomaly "

        "labels. They do not establish real-world leak detection accuracy."

    )

    # --------------------------------------------
    # LIVE / INTERACTIVE ANOMALY DETECTION
    # --------------------------------------------

    import joblib
    import pandas as pd

    st.markdown("---")
    st.header("🔍 Live Water Consumption Anomaly Detection")

    st.write(
        "Enter the current water consumption details "
        "to check whether the reading is normal or anomalous."
    )

    MODEL_FILE = os.path.join(
        "model", "anomaly_detection_models.pkl"
    )

    try:
        saved_data = joblib.load(MODEL_FILE)

        building_models = saved_data["models"]
        features = saved_data["features"]

        with st.form("live_anomaly_detection_form"):

            st.subheader("Building Information")

            building_id = st.selectbox(
                "Select Building",
                options=list(building_models.keys())
            )

            building_info = building_models[building_id]

            st.info(
                f"Building Type: {building_info['building_type']}"
            )

            st.subheader("Water Consumption Details")

            col1, col2 = st.columns(2)

            with col1:
                occupancy = st.number_input(
                    "Occupancy",
                    min_value=0,
                    value=100,
                    step=1
                )

                temperature = st.number_input(
                    "Temperature (°C)",
                    value=30.0,
                    step=0.5
                )

                rainfall = st.number_input(
                    "Rainfall (mm)",
                    min_value=0.0,
                    value=0.0,
                    step=0.5
                )

            with col2:
                consumption = st.number_input(
                    "Current Consumption (Litres)",
                    min_value=0.0,
                    value=50000.0,
                    step=100.0
                )

                previous_consumption = st.number_input(
                    "Previous Day Consumption (Litres)",
                    min_value=0.0,
                    value=50000.0,
                    step=100.0
                )

                rolling_average = st.number_input(
                    "7-Day Rolling Average (Litres)",
                    min_value=0.0,
                    value=50000.0,
                    step=100.0
                )

            submitted = st.form_submit_button(
                "Detect Water Consumption Anomaly",
                use_container_width=True
            )

        if submitted:

            input_data = pd.DataFrame([{
                "occupancy": occupancy,
                "temperature_c": temperature,
                "rainfall_mm": rainfall,
                "consumption_liters": consumption,
                "previous_day_consumption_liters":
                    previous_consumption,
                "rolling_7day_avg_liters": rolling_average
            }])[features]

            scaler = building_info["scaler"]
            model = building_info["model"]

            scaled_input = scaler.transform(input_data)

            prediction = model.predict(scaled_input)[0]

            anomaly_score = model.decision_function(
                scaled_input
            )[0]

            isolation_anomaly = prediction == -1

            threshold = building_info["consumption_threshold"]

            high_consumption = consumption > threshold

            anomaly_detected = (
                isolation_anomaly or high_consumption
            )

            st.markdown("---")
            st.subheader("Detection Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Building", building_id)

            col2.metric(
                "Current Consumption",
                f"{consumption:,.2f} L"
            )

            col3.metric(
                "Historical Threshold",
                f"{threshold:,.2f} L"
            )

            st.write(
                f"**Isolation Forest Score:** "
                f"{anomaly_score:.4f}"
            )

            st.write("**Detection Details**")

            st.write(
                "Isolation Forest: "
                + ("ANOMALY" if isolation_anomaly else "NORMAL")
            )

            st.write(
                "Consumption Threshold: "
                + (
                    "EXCEEDED"
                    if high_consumption
                    else "NOT EXCEEDED"
                )
            )

            if anomaly_detected:

                st.error(
                    "🚨 ANOMALY DETECTED!"
                )

                st.warning(
                    "Water consumption is unusually high "
                    "or follows an unusual pattern."
                )

                st.info(
                    "Recommended Action: Inspect the building's "
                    "water usage and check for possible leaks."
                )

            else:

                st.success(
                    "✅ NORMAL WATER CONSUMPTION"
                )

                st.info(
                    "Consumption is within the configured "
                    "detection limits."
                )

            st.caption(
                "An anomaly indicates unusual consumption, "
                "not proof of a water leak. The historical "
                "threshold is a screening rule, not a "
                "calibrated probability."
            )

    except FileNotFoundError:

        st.error(
            "Anomaly detection model not found. "
            "Please train the anomaly detection models first."
        )

    except Exception as e:

        st.error(f"Anomaly detection error: {e}")


# --------------------------------------------------

# FOOTER

# --------------------------------------------------


st.markdown("---")


st.caption(

    "Smart Water Consumption Prediction & Management | "

    "AI-Based Campus Water Management System"

)
