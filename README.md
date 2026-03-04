# Solar Water Heater Efficiency Prediction using ANN

This project predicts the **efficiency of a solar water heater system** using an **Artificial Neural Network (ANN) regression model**.  
The objective is to analyze how environmental and operational parameters affect system performance and build a model that can estimate efficiency under different conditions.

The dataset used in this project simulates **raw sensor data from a solar thermal system**, including missing values and measurement errors to reflect real-world scenarios.

---

# Project Overview

Solar water heaters convert sunlight into thermal energy to heat water. The efficiency of a solar collector depends on several factors such as:

* Solar radiation
* Ambient temperature
* Inlet water temperature
* Water flow rate
* Collector area
* Wind speed
* Tilt angle of the collector

These parameters are used as **input features** to predict the **efficiency of the solar water heater**.

---

# Dataset Description

The dataset contains **9000 rows** representing simulated sensor readings from a solar water heating system.

## Features

| Feature | Description | Unit |
|------|------|------|
| Solar_Radiation_Wm2 | Solar energy received per unit area | W/m² |
| Ambient_Temp_C | Surrounding air temperature | °C |
| Inlet_Water_Temp_C | Temperature of water entering the collector | °C |
| Flow_Rate_Lmin | Water flow rate through the collector | L/min |
| Collector_Area_m2 | Surface area of the solar collector | m² |
| Wind_Speed_ms | Wind speed affecting heat loss | m/s |
| Tilt_Angle_deg | Angle of the collector relative to ground | degrees |
| Efficiency_percent | Thermal efficiency of the system | % |

## Dataset Characteristics

The dataset intentionally includes:

* Missing values (simulating sensor faults)
* Outliers and measurement errors
* Realistic efficiency range (~25–70%)

This allows demonstration of **data preprocessing and cleaning techniques** before training the ANN model.

---

# Project Workflow

1. Data Collection / Simulation  
2. Data Cleaning  
3. Handling Missing Values  
4. Outlier Detection  
5. Feature Scaling  
6. Train-Test Split  
7. ANN Regression Model Training  
8. Model Evaluation  

---

# ANN Model Structure

The neural network used in this project includes:

* **Input Layer:** 7 neurons (system parameters)
* **Hidden Layers:** Dense layers with ReLU activation
* **Output Layer:** 1 neuron (Efficiency prediction)

### Loss Function
* Mean Squared Error (MSE)

### Evaluation Metrics

* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

# Technologies Used

* Python  
* NumPy  
* Pandas  
* Scikit-learn  
* TensorFlow / Keras  
* Matplotlib  

---

# Example Use Case

The trained ANN model can estimate solar water heater efficiency under different environmental conditions. This can help in:

* Optimizing solar collector design
* Improving energy efficiency
* Predicting system performance
* Supporting renewable energy planning

---

# Future Improvements

* Use real-world IoT sensor datasets  
* Implement advanced deep learning models  
* Add real-time prediction system  
* Deploy the model as a web application  

---

# Project Type

This project was developed as part of a **Project Based Learning (PBL)** activity for **Artificial Intelligence / Machine Learning coursework**.
