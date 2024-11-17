import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the datasets
arrivals_data = pd.read_csv(r"C:\Users\dines\OneDrive\Desktop\data\Detailed_Statistics_Arrivals.csv")
departures_data = pd.read_csv(r"C:\Users\dines\OneDrive\Desktop\data\Detailed_Statistics_Departures.csv")
weather_arrivals_data = pd.read_csv(r"C:\Users\dines\OneDrive\Desktop\data\weather_data_arrivals.csv")
weather_departures_data = pd.read_csv(r"C:\Users\dines\OneDrive\Desktop\data\weather_data_depature.csv")

# Step 2: Clean and preprocess the flight data
# Standardize column names
arrivals_data.columns = [col.strip().replace(' ', '_').lower() for col in arrivals_data.columns]
departures_data.columns = [col.strip().replace(' ', '_').lower() for col in departures_data.columns]

# Parse the date column into datetime
arrivals_data['date'] = pd.to_datetime(arrivals_data['date_(dd/mm/yyyy)'], format='%d-%m-%Y', errors='coerce')
departures_data['date'] = pd.to_datetime(departures_data['date_(dd/mm/yyyy)'], format='%d-%m-%Y', errors='coerce')

# Step 3: Preprocess the weather data
weather_key_features = [
    'STATION',
    'DATE',
    'DailyAverageDryBulbTemperature',  # Temperature
    'DailyPrecipitation',              # Precipitation
    'HourlyVisibility',                # Visibility
    'DailySustainedWindSpeed',         # Wind Speed
    'HourlyWindDirection'              # Wind Direction
]

weather_arrivals_data = weather_arrivals_data[weather_key_features]
weather_departures_data = weather_departures_data[weather_key_features]

# Parse the date column in weather data
weather_arrivals_data['DATE'] = pd.to_datetime(weather_arrivals_data['DATE'], errors='coerce')
weather_departures_data['DATE'] = pd.to_datetime(weather_departures_data['DATE'], errors='coerce')

# Step 4: Merge flight data with weather data
arrivals_data_merged = arrivals_data.merge(weather_arrivals_data, how='left', left_on='date', right_on='DATE')
departures_data_merged = departures_data.merge(weather_departures_data, how='left', left_on='date', right_on='DATE')

# Combine arrivals and departures data into a unified dataset
flight_data = pd.concat([arrivals_data_merged, departures_data_merged], ignore_index=True)

# Step 5: Handle missing values
flight_data.ffill(inplace=True)  # Using forward fill to handle missing values
flight_data = flight_data.infer_objects()  # Ensures object columns are inferred correctly

# Step 6: Create target variable (delay indicator)
flight_data['is_delayed'] = (flight_data['actual_arrival_time'] > flight_data['scheduled_arrival_time']).astype(int)

# Step 7: Feature engineering
# Add derived weather features
flight_data['temp_deviation'] = abs(flight_data['DailyAverageDryBulbTemperature'] - flight_data['DailyPrecipitation'])
flight_data['high_wind'] = (flight_data['DailySustainedWindSpeed'] > 15).astype(int)

# Convert categorical columns to numeric using Label Encoding
categorical_columns = ['carrier_code', 'origin_airport', 'destination_airport']
for col in categorical_columns:
    le = LabelEncoder()
    flight_data[col] = le.fit_transform(flight_data[col])

# Convert date-related features to useful numeric features
flight_data['day_of_week'] = flight_data['date'].dt.dayofweek  # Day of the week (0 = Monday, 6 = Sunday)
flight_data['month'] = flight_data['date'].dt.month  # Month (1 = January, 12 = December)

# Step 8: Remove non-numeric columns (such as 'date' and 'DATE') that can't be used in the model
X = flight_data.drop(['is_delayed', 'actual_arrival_time', 'scheduled_arrival_time', 'date', 'DATE',
                      'date_(dd/mm/yyyy)', 'tail_number', 'scheduled_departure_time', 'actual_departure_time'], axis=1)

# Explicitly check if there are any non-numeric columns left in X
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_columns) > 0:
    print("Non-numeric columns found:", non_numeric_columns)
    # Drop the non-numeric columns to avoid issues during model fitting
    X = X.drop(non_numeric_columns, axis=1)

# Target variable
y = flight_data['is_delayed']

# Step 9: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Build models
# Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# XGBoost model
xgb_model = xgb.XGBClassifier(eval_metric='logloss')  # Removed 'use_label_encoder'
xgb_model.fit(X_train, y_train)

# Step 11: Evaluate models
# Random Forest predictions and probabilistic predictions
rf_predictions = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Classification report for Random Forest
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

# ROC Curve for Random Forest
fpr, tpr, _ = roc_curve(y_test, rf_probs)
plt.plot(fpr, tpr, label='Random Forest (AUC = {:.2f})'.format(roc_auc_score(y_test, rf_probs)))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve')
plt.show()

# Calibration Curve for Random Forest
prob_true, prob_pred = calibration_curve(y_test, rf_probs, n_bins=10)
plt.plot(prob_pred, prob_true, label='Random Forest')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.title('Calibration Curve')
plt.show()

# Step 12: Incorporate uncertainty
# Monte Carlo simulations
simulated_uncertainty = []
for _ in range(100):  # Simulate 100 variations
    perturbed_X = X_test.copy()
    perturbed_X['DailyPrecipitation'] += np.random.normal(0, 0.1, size=X_test.shape[0])  # Add random noise
    simulated_uncertainty.append(rf_model.predict_proba(perturbed_X)[:, 1])

# Aggregate Monte Carlo results
simulated_uncertainty = np.array(simulated_uncertainty)
uncertainty_mean = simulated_uncertainty.mean(axis=0)
uncertainty_std = simulated_uncertainty.std(axis=0)

# Step 13: Save models and results
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(xgb_model, 'xgboost_model.pkl')

# Save Monte Carlo uncertainty results
uncertainty_results = pd.DataFrame({'mean_prob': uncertainty_mean, 'std_dev': uncertainty_std})
uncertainty_results.to_csv('uncertainty_results.csv', index=False)
