**Flight Delay Prediction Model**

Overview
This project involves predicting flight delays using historical flight data and weather data. The model uses machine learning techniques to predict whether a flight will be delayed based on features like weather conditions, flight times, and airport information. Two models are built: Random Forest and XGBoost. Additionally, uncertainty in the predictions is evaluated through Monte Carlo simulations.

**Data Sources**

**The following datasets are used for training and evaluation:**

Flight Arrivals Data (Detailed_Statistics_Arrivals.csv): Contains historical data on flight arrivals.
Flight Departures Data (Detailed_Statistics_Departures.csv): Contains historical data on flight departures.
Weather Data for Arrivals (weather_data_arrivals.csv): Weather data related to arrival flights.
Weather Data for Departures (weather_data_depature.csv): Weather data related to departure flights.

****Steps Overview:**

**Data Cleaning & Preprocessing:****

**Cleaned column names.**
Converted date columns to proper datetime format.
Merged flight data with weather data based on the date.
Handled missing values by forward filling them.

**Feature Engineering:**

Created new features like temp_deviation (difference between temperature and precipitation) and high_wind (indicator if wind speed exceeds 15).
Encoded categorical features like carrier code, origin airport, and destination airport using Label Encoding.
Extracted day of the week and month from the date column for additional features.

**Target Variable:**

Created a binary target variable is_delayed, where 1 indicates the flight was delayed and 0 indicates it was not.

**Model Building:**

Split the dataset into training and testing sets (80% for training, 20% for testing).
Trained two machine learning models: Random Forest and XGBoost.

**Model Evaluation:**

Evaluated model performance using metrics such as the classification report, ROC curve, and calibration curve.
Plotted ROC and calibration curves to evaluate model performance.
Uncertainty Evaluation:

Applied Monte Carlo simulations to evaluate the uncertainty of predictions by introducing random noise to weather features and observing variations in predicted probabilitie

**model saving:**

Saved the trained models (RandomForest and XGBoost) using joblib.
Saved the uncertainty results (mean probabilities and standard deviations) in a CSV file.
Files

random_forest_model.pkl: The trained Random Forest model.
xgboost_model.pkl: The trained XGBoost model.
uncertainty_results.csv: A CSV file containing the uncertainty evaluation results from the Monte Carlo simulations.

**Requirements**

To run this project, you need the following libraries installed:

**pandas
numpy
matplotlib
seaborn
sklearn
xgboost
joblib**

**You can install these dependencies using pip:
**
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib

**How to Run the Code**

Ensure the datasets (Detailed_Statistics_Arrivals.csv, Detailed_Statistics_Departures.csv, weather_data_arrivals.csv, and weather_data_depature.csv) are available in the specified file paths in the code. If using a different directory, update the file paths accordingly.

**Run the script in your preferred Python environment.**

The models will be trained and evaluated, and the results will be saved in the respective files (random_forest_model.pkl, xgboost_model.pkl, and uncertainty_results.csv).

The ROC curve and calibration curve for Random Forest will be plotted to evaluate its performance.

**Conclusion**

This project demonstrates how to predict flight delays using machine learning techniques. The models' performance is evaluated, and the uncertainty in predictions is quantified using Monte Carlo simulations. The results can be used to improve decision-making for airlines and passengers.
