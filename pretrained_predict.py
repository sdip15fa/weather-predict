import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the data
input_file = 'data/processed_data.csv'
df = pd.read_csv(input_file)

# Rename the columns for better readability
df.columns = ['DateTime', 'Year', 'Month', 'Date', 'Time', 'Minute', 'Temperature', 'Previous Day Average',
              'Two Days Before Average', 'Three Days Before average', 'Last 7 Days Average',
              'Previous Day Wind Speed', 'Previous Day Rainfall']

# Convert the 'Date' and 'Time' columns to integers
df['Date'] = df['Date'].astype(int)
df['Time'] = df['Time'].astype(int)

# Fill leading zeros for the 'Time' column
df['Time'] = df['Time'].apply(lambda x: str(x).zfill(4))

# Combine the 'Date' and 'Time' columns into a single 'DateTime' column
df['DateTime'] = pd.to_datetime(df['DateTime'].astype(int).astype(str) + df['Time'].astype(int).astype(str) +
                                df['Minute'].astype(str), format='%Y%m%d%H%M')

# Filter the data for June 2023
start_date = pd.Timestamp('2023-06-01')
end_date = pd.Timestamp('2023-06-30')
june_data = df[(df['DateTime'] >= start_date) & (df['DateTime'] <= end_date)]

# Prepare the data for prediction
time_steps = 60
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_temperature = scaler.fit_transform(june_data['Temperature'].values.reshape(-1, 1))
data = []
for i in range(len(scaled_temperature) - time_steps):
    data.append(scaled_temperature[i:i + time_steps])
data = np.array(data)

# Load the trained LSTM model
model = tf.keras.models.load_model("lstm.keras")

# Make predictions
predictions = model.predict(data)

# Rescale the predictions back to the original range
scaled_predictions = predictions.reshape(-1, 1)
predicted_temperature = scaler.inverse_transform(scaled_predictions)

# Extract the actual temperatures for June 2023
actual_temperature = june_data['Temperature'].values[time_steps:]

# Calculate MSE and MAE
mse = mean_squared_error(actual_temperature, predicted_temperature)
mae = mean_absolute_error(actual_temperature, predicted_temperature)

# Convert the predictions to a DataFrame
prediction_df = pd.DataFrame({'DateTime': june_data['DateTime'].values[time_steps:], 'Actual': actual_temperature.flatten(),
                              'Predicted': predicted_temperature.flatten()})

# Print MSE and MAE
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# Plot the actual vs. predicted temperatures
plt.figure(figsize=(12, 6))
plt.plot(prediction_df['DateTime'], prediction_df['Actual'], label='Actual')
plt.plot(prediction_df['DateTime'], prediction_df['Predicted'], label='Predicted')
plt.xlabel('DateTime')
plt.ylabel('Temperature')
plt.title('Actual vs. Predicted Temperatures for June 2023')
plt.legend()
plt.xticks(rotation=45)
plt.show()
