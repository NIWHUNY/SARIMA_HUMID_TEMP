
#SARIMA MULTI CORE ADALAH VERSI SARIMA.PY yang menggunakan pararel processing yang mempercepat proses training model
# dengan menggunakan semua core yang ada di CPU, namun juga menggunakan memory yang lebih besar(tergantung data yang di training)
# biasanya penggunaan ram akan penuh sebelum penggunaan CPU mencapai 100%., dan ada kemungkinan gagal jika memory tidak cukup
# made by NIWH
import os
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# Function untuk memasukan data dan prosesing data agar untuk dibaca pandas
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

# lokasi folder data
folder_paths = ["SEPTEMBER", "OKTOBER", "NOVEMBER"]

all_data = pd.DataFrame()

for folder_path in folder_paths:
    folder_data = load_and_preprocess_data(folder_path)
    all_data = pd.concat([all_data, folder_data], ignore_index=True)

# verifikasi 'DATETIME', 'TEMPERATURE', dan 'HUMIDITY' columns
if 'DATETIME' not in all_data.columns or 'TEMPERATURE' not in all_data.columns or 'HUMIDITY' not in all_data.columns:
    print("Error: 'DATETIME', 'TEMPERATURE', or 'HUMIDITY' columns are missing in the combined data.")
    exit()


all_data = all_data.sort_values(by='DATETIME')

# mengambil data waktu untuk fitur prediksi
all_data['hour'] = all_data['DATETIME'].dt.hour
all_data['minute'] = all_data['DATETIME'].dt.minute
all_data['second'] = all_data['DATETIME'].dt.second

# memilih fitur prediksi dan variable yang akan di prediksikan
features = all_data[['hour', 'minute', 'second']]
target_temperature = all_data['TEMPERATURE']
target_humidity = all_data['HUMIDITY']

# modifikasi data agar menjadi training dan testing sets
X_train, X_test, y_train_temp, y_test_temp, y_train_hum, y_test_hum = train_test_split(
    features, target_temperature, target_humidity, test_size=0.2, random_state=42
)

# mengubah variable menjadi numeric
y_train_temp = pd.to_numeric(y_train_temp, errors='coerce')
y_train_hum = pd.to_numeric(y_train_hum, errors='coerce')

# function untuk training model SARIMA dan adjustment sarima
def train_sarima_model(y_train, order):
    model = sm.tsa.statespace.SARIMAX(y_train, order=order, seasonal_order=(1, 1, 1, 24))
    results = model.fit()
    return results

# melatih sarima untuk temperatur dan kelembapan
order_temp = (1, 1, 1)  # Example order - adjust based on your data
order_hum = (1, 1, 1)   # Example order - adjust based on your data

results_temp, results_hum = Parallel(n_jobs=-1)(
    delayed(train_sarima_model)(y_train, order)
    for y_train, order in [(y_train_temp, order_temp), (y_train_hum, order_hum)]
)

# membuat prediksi
predictions_temp = results_temp.get_forecast(steps=len(X_test))
predictions_hum = results_hum.get_forecast(steps=len(X_test))

# evaluasi model menggunakan MSE (di replace dengan MAE dibawah)
#mse_temp = mean_squared_error(y_test_temp, predictions_temp.predicted_mean)
#mse_hum = mean_squared_error(y_test_hum, predictions_hum.predicted_mean)

#print(f"Mean Squared Error (Temperature): {mse_temp}")
#print(f"Mean Squared Error (Humidity): {mse_hum}")

# kode dibawah hanya untuk prediksi satu waktu saja
#new_data = pd.DataFrame({'hour': [12], 'minute': [0], 'second': [0]})
#predicted_temp = results_temp.get_forecast(steps=1, exog=new_data).predicted_mean
#predicted_hum = results_hum.get_forecast(steps=1, exog=new_data).predicted_mean

#print(f"Predicted Temperature: {predicted_temp.values[0]}")
#print(f"Predicted Humidity: {predicted_hum.values[0]}")

# additional path adalah sebagai input data berupa csv pembanding data asli yang akan di bandingkan dengan prediksi
additional_data_path = "26_november_2023.csv"
additional_data = pd.read_csv(additional_data_path, header=None, names=['DATETIME', 'TEMPERATURE', 'HUMIDITY'])
additional_data['DATETIME'] = pd.to_datetime(additional_data['DATETIME'], format='%Y-%m-%d %H:%M:%S')

# memastikan fitur prediski sama
additional_data['hour'] = additional_data['DATETIME'].dt.hour
additional_data['minute'] = additional_data['DATETIME'].dt.minute
additional_data['second'] = additional_data['DATETIME'].dt.second

# membuat prediksi untuk temperature dan kelembapan untuk perhari
additional_data_features = additional_data[['hour', 'minute', 'second']]
predicted_temp_additional = results_temp.get_forecast(steps=len(additional_data_features), exog=additional_data_features).predicted_mean
predicted_hum_additional = results_hum.get_forecast(steps=len(additional_data_features), exog=additional_data_features).predicted_mean

output_df = pd.DataFrame({
    'DATETIME': additional_data['DATETIME'],
    'Actual_Temperature': additional_data['TEMPERATURE'],
    'Predicted_Temperature': predicted_temp_additional,
    'Actual_Humidity': additional_data['HUMIDITY'],
    'Predicted_Humidity': predicted_hum_additional
})

output_csv_path = 'predicted_output.csv'
output_df.to_csv(output_csv_path, index=False)

# Print the path to the saved CSV file
print(f"Predicted output saved to: {output_csv_path}")

# menghitung Mean Absolute Error (MAE)
mae_temp = mean_absolute_error(additional_data['TEMPERATURE'], predicted_temp_additional)
mae_hum = mean_absolute_error(additional_data['HUMIDITY'], predicted_hum_additional)

# Print the MAE
print(f"Mean Absolute Error (Temperature): {mae_temp}")
print(f"Mean Absolute Error (Humidity): {mae_hum}")

# Plot perbandingan antara nilai asli dan nilai prediksi the comparison between the actual and predicted values
plt.figure(figsize=(12, 9))

plt.subplot(3, 1, 1)
plt.plot(additional_data['DATETIME'], additional_data['TEMPERATURE'], label='Temperature asli')
plt.plot(additional_data['DATETIME'], predicted_temp_additional, label='Prediksi Temperature')
plt.title('Perbandingan prediksi temperature')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(additional_data['DATETIME'], additional_data['HUMIDITY'], label='Humidity Asli')
plt.plot(additional_data['DATETIME'], predicted_hum_additional, label='Prediksi Humidity')
plt.title('perbandingan prediksi kelembapan')
plt.legend()

# Plot MAE
plt.subplot(3, 1, 3)
plt.bar(['Temperature', 'Humidity'], [mae_temp, mae_hum], color=['blue', 'orange'])
plt.title('Mean Absolute Error (MAE)')
plt.ylabel('MAE')
plt.show()
