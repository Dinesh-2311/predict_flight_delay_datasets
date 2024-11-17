**Flight Delays Prediction Model**

**Overview:**

The following project predicts flight delays using historic data of flights and weather. The model will use machine learning techniques to make predictions if a given flight will be late depending on the weather conditions, flight schedule, and the airport where the flight took off from originally. It builds two models: Random Forest and XGBoost. Additionally, it also evaluates the uncertainty of the predictions using Monte Carlo simulations.

**Data Sources:**

The datasets used for training and evaluation are as follows:

Flight Arrivals Data: Detailed_Statistics_Arrivals.csv - This includes the historical data related to flight arrivals. Flight Departures Data: Detailed_Statistics_Departures.csv - This consists of historical data about flight departures. Weather Data for Arrivals: weather_data_arrivals.csv - Weather data for arrival flights. Weather Data for Departures: weather_data_depature.csv - Weather data for departure flights.

**Overview of Steps**:

**Data Cleaning & Preprocessing:**

Cleaned column names. Converted date columns into the right datetime format. Merged flight data with weather data by date. Handled missing values by doing a forward fill.

**Feature Engineering:**

The new features created are temp_deviation, which is the difference between temperature and precipitation, and high_wind, which is an indicator if wind speed surpasses 15. The categorical features such as carrier code, origin airport, and destination airport are encoded using Label Encoding. The date column is used to extract additional features day of the week and month.

**Target Variable**:

A binary target variable is created, is_delayed, where 1- flight delayed and 0- it was not.

**Model Building**:

Split the dataset into training and testing sets, 80% for training and 20% for testing. The work involved training two machine learning models: Random Forest and XGBoost.

**Model Evaluation:**

The performance of the model was then evaluated based on the classification report, ROC curve, and calibration curve. Plotted ROC and calibration curves were used to evaluate the model performance. Uncertainty Evaluation:

Applied Monte Carlo simulations to quantify the uncertainty of the predictions by adding random noise to the weather features and measuring changes in the predicted probabilities.

**Saving the Model:**

The models are saved through joblib for both RandomForest and XGBoost. The uncertainty results-mean probabilities and their standard deviations-are saved into a CSV file. Files

random_forest_model.pkl: Trained Random Forest model. xgboost_model.pkl: Trained XGBoost model. uncertainty_results.csv: A CSV file including the result of uncertainty evaluation via Monte Carlo simulations.

**Dependencies**:

**this project requires the following libraries to be installed:
****

**pandas numpy matplotlib seaborn sklearn xgboost joblib**

****You can install these dependencies using pip: ** pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib**

**Running Code**

Make sure the datasets Detailed_Statistics_Arrivals.csv, Detailed_Statistics_Departures.csv, weather_data_arrivals.csv, and weather_data_depature.csv can be found in the defined pathways in the code. If another directory is used, the pathway must be updated in the code.

**Run the script in your Python environment of choice.**

This will train and evaluate the models with the results being saved in respective files named random_forest_model.pkl, xgboost_model.pkl, and uncertainty_results.csv.

The performance of the Random Forest is evaluated by plotting the ROC curve and calibration curve.

**Conclusion**:

The project explains the prediction of flight delays using techniques from machine learning. It evaluates the performance of the models, and the uncertainty in predictions will be quantified using Monte Carlo simulations. Such results have several applications in decision-making for airlines and passengers.
