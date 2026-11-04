# Smart Water Consumption Prediction & Management

## 1. Project Overview

Smart Water Consumption Prediction & Management is a machine learning-based project designed to predict water consumption across different campus buildings and identify unusual water usage patterns.

The system uses historical water consumption data, building information, occupancy, and weather conditions to estimate future water demand and detect potential anomalies.

The project provides an interactive dashboard developed using Streamlit.

## 2. Project Objectives

- Predict daily water consumption for individual campus buildings.
- Estimate monthly water consumption.
- Analyze water usage across different buildings and building types.
- Detect unusual water consumption patterns.
- Visualize consumption trends through interactive charts.
- Support better water resource planning and management.

## 3. Key Features

### Dashboard
- Displays overall campus water consumption statistics.
- Shows daily water consumption trends.
- Compares average consumption across buildings.
- Displays consumption by building type.

### Dataset Analysis
- Explore the water consumption dataset.
- Filter records by building and building type.
- View statistical summaries.
- Visualize consumption patterns.
- Download filtered dataset records.

### Water Consumption Prediction
- Predict daily water consumption.
- Estimate monthly water consumption.
- Display daily prediction charts.
- Download prediction results as CSV files.

### Anomaly Detection
- Identify unusual water consumption patterns.
- Use Isolation Forest for anomaly detection.
- Apply a consumption threshold to identify unusually high usage.
- Display anomaly detection results and model evaluation metrics.

## 4. Technologies Used

- **Programming Language:** Python
- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Storage:** Joblib
- **Data Visualization:** Matplotlib and Streamlit charts

## 5. Machine Learning Models

### Water Consumption Prediction

A Random Forest Regression model is used to predict water consumption based on:

- Occupancy
- Temperature
- Rainfall
- Previous day's water consumption
- Weekend information
- Holiday information
- Building ID
- Building type

### Anomaly Detection

Isolation Forest is used to identify unusual water consumption patterns.

A hybrid detection approach combines the Isolation Forest result with a high-consumption threshold to identify potential anomalies.

## 6. Dataset Description

The project uses a synthetically generated campus water consumption dataset.

- **Dataset period:** January 2024 – December 2025
- **Number of records:** 8,772
- **Number of buildings:** 12
- **Number of columns:** 13
- **Synthetic anomaly records:** 271

The dataset contains information about building IDs, building types, occupancy, weather conditions, daily water consumption, historical consumption features, and anomaly labels.
smart_water_consumption_dataset.csv` is a **synthetic** dataset. It is not real meter data. I generated it with `generate_water_dataset.py`, using per-person water-use figures from published campus reports as the starting point.

**Sources used**
- Crescent Institute, [Water Consumption Details](https://crescent.education/wp-content/uploads/2021/11/6.4.2-b-Crescent-Water-Consumption-Details.pdf)
- Redeemer's University, [Water Management Report 2023](<https://sdg.run.edu.ng/assets/img/Redeemer's-University-Water-Management-Report-2023.pdf>)
- IIT Bombay, [campus water report](https://www.insightiitb.org/wp-content/uploads/2015/10/18.1-final-draft.pdf)
- CIT Chennai, [SDG 6 report](https://www.citchennai.edu.in/wp-content/uploads/2024/11/sdg/SDG_6_17.pdf)
- Iversen, J. O. (2010). *Analysis of Potable Water Use at Yale University.* Yale University. Used as a reference for water use by building type.

The building list, occupancy patterns, weather, holiday calendar (May, June, July and December), noise and anomalies are my own assumptions.
**Note:** The dataset is synthetic and does not represent actual measured campus water consumption. Therefore, model results demonstrate performance on the generated dataset and require validation using real campus data before real-world deployment.

## 7. Project Folder Structure

```text
Smart-Water-Management/
│
├── app.py
├── README.md
├── requirements.txt
│
├── dataset/
│   └── water_consumption_dataset.csv
│
├── model/
│   ├── water_consumption_model.pkl
│   └── anomaly_detection_models.pkl
│
├── outputs/
│   ├── final_anomaly_metrics_2025.csv
│   ├── anomaly_confusion_matrix.png
│   └── building-wise_f1.png
│
└── .venv/
```

*Additional Python scripts and output files may also be present in the project.*

## 8. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### Step 2: Open the project folder

```bash
cd Smart-Water-Management
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### Step 5: Install the required libraries

```bash
pip install -r requirements.txt
```

### Step 6: Run the application

```bash
python -m streamlit run app.py
```

The application will open in your web browser at:

```text
http://localhost:8501
```

## 9. Expected Outcome

The application provides an interactive platform for:

- Understanding campus water consumption patterns.
- Predicting future water requirements.
- Comparing water usage between buildings.
- Identifying unusual consumption patterns.
- Supporting data-driven water management decisions.

## 10. Future Enhancements

- Integration with real-time water flow sensors.
- Use of actual campus water consumption data.
- Improved anomaly detection accuracy.
- Automated leak alerts.
- Real-time monitoring of water consumption.
- Advanced forecasting models.

## 11. Conclusion

The Smart Water Consumption Prediction & Management project demonstrates how machine learning and data visualization can be used to analyze, predict, and monitor water consumption across campus buildings. It aims to support efficient water resource management and help identify unusual consumption patterns.
