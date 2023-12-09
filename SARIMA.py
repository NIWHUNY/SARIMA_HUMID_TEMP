
# BELUM DI MODIFIKASI, MOHON PAKAI SARIMA_MULTI_CORE\
#made by NIWH

import os
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to load and preprocess data from a folder
def load_and_preprocess_data(folder_path):
    all_data = pd.DataFrame()
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Explicitly specify column names
                data = pd.read_csv(file_path, header=None, names=['DATETIME', 'TEMPERATURE', 'HUMIDITY'])
                # Convert DATETIME column to datetime with specified format and handle errors
                data['DATETIME'] = pd.to_datetime(data['DATETIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                all_data = pd.concat([all_data, data], ignore_index=True)
            except pd.errors.ParserError as e:
                print(f"ParserError in file {file_path}: {e}")
                continue
    return all_data
# Load data from CSV files in the three different folders
folder_paths = ["SEPTEMBER", "OKTOBER"]

all_data = pd.DataFrame()

for folder_path in folder_paths:
    folder_data = load_and_preprocess_data(folder_path)
    all_data = pd.concat([all_data, folder_data], ignore_index=True)

# Verify if 'DATETIME', 'TEMPERATURE', and 'HUMIDITY' columns exist
if 'DATETIME' not in all_data.columns or 'TEMPERATURE' not in all_data.columns or 'HUMIDITY' not in all_data.columns:
    print("Error: 'DATETIME', 'TEMPERATURE', or 'HUMIDITY' columns are missing in the combined data.")
    exit()

# You may want to sort the data by datetime
all_data = all_data.sort_values(by='DATETIME')

# Feature engineering: extracting relevant features from datetime
all_data['hour'] = all_data['DATETIME'].dt.hour
all_data['minute'] = all_data['DATETIME'].dt.minute
all_data['second'] = all_data['DATETIME'].dt.second

# Select features and target variables
features = all_data[['hour', 'minute', 'second']]
target_temperature = all_data['TEMPERATURE']
target_humidity = all_data['HUMIDITY']

# Split data into training and testing sets
X_train, X_test, y_train_temp, y_test_temp, y_train_hum, y_test_hum = train_test_split(
    features, target_temperature, target_humidity, test_size=0.2, random_state=42
)

# Convert target variables to numeric type
y_train_temp = pd.to_numeric(y_train_temp, errors='coerce')
y_train_hum = pd.to_numeric(y_train_hum, errors='coerce')

# Train SARIMA model for temperature prediction
order_temp = (1, 1, 1)  # Example order - adjust based on your data
model_temp = sm.tsa.statespace.SARIMAX(y_train_temp, order=order_temp, seasonal_order=(1, 1, 1, 24))
results_temp = model_temp.fit()

# Train SARIMA model for humidity prediction
order_hum = (1, 1, 1)  # Example order - adjust based on your data
model_hum = sm.tsa.statespace.SARIMAX(y_train_hum, order=order_hum, seasonal_order=(1, 1, 1, 24))
results_hum = model_hum.fit()
# Make predictions
predictions_temp = results_temp.get_forecast(steps=len(X_test))
predictions_hum = results_hum.get_forecast(steps=len(X_test))

# Evaluate the model
mse_temp = mean_squared_error(y_test_temp, predictions_temp.predicted_mean)
mse_hum = mean_squared_error(y_test_hum, predictions_hum.predicted_mean)

print(f"Mean Squared Error (Temperature): {mse_temp}")
print(f"Mean Squared Error (Humidity): {mse_hum}")

# Now, you can use the trained models (results_temp and results_hum) to predict temperature and humidity for a given date and time.
# For example:
new_data = pd.DataFrame({'hour': [12], 'minute': [0], 'second': [0]})
predicted_temp = results_temp.get_forecast(steps=1, exog=new_data).predicted_mean
predicted_hum = results_hum.get_forecast(steps=1, exog=new_data).predicted_mean

print(f"Predicted Temperature: {predicted_temp.values[0]}")
print(f"Predicted Humidity: {predicted_hum.values[0]}")
