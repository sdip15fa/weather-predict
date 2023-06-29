from datetime import timedelta
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the data
input_file = 'data/processed_data.csv'
df = pd.read_csv(input_file, parse_dates=['DateTime'], index_col='DateTime')
df = df.resample('60T').mean()
df = df[(df['Temperature'] >= df['Temperature'].quantile(0.1)) & (df['Temperature'] <= df['Temperature'].quantile(0.9))]
df = df.reset_index()

# Convert the 'Date' and 'Time' columns to integers
df['Date'] = df['Date'].astype(int)
df['Time'] = df['Time'].astype(int)

# Fill leading zeros for the 'Time' column
df['Time'] = df['Time'].apply(lambda x: str(x).zfill(4))

# Remove rows with a specific value (e.g., 32767) in 'Temperature' column
df = df[df['Temperature'] != 32767]

# Prepare the data for LSTM
time_steps = 60  # Number of time steps for the LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))  # Scale the data to [0, 1]

# Scale the temperature values
scaled_temperature = scaler.fit_transform(df['Temperature'].values.reshape(-1, 1))

# Generate predictions for June 30, 2023
date_30th_june = pd.date_range('2023-06-29 00:00:00', '2023-06-29 23:00:00', freq='H')

# Load the pre-trained model
model = tf.keras.models.load_model("lstm.keras")

# Predict temperatures for each hour on June 30, 2023
predicted_temperatures = []
for i in range(len(date_30th_june)):
    try:
        end_index = df[df["DateTime"] == date_30th_june[i] - timedelta(days=1)].index[0]
        input_data = scaled_temperature[end_index-time_steps:end_index]
        input_data = input_data.reshape(1, time_steps, 1)
        prediction = model.predict(input_data)
        predicted_temperature = scaler.inverse_transform(prediction)[0][0]
        predicted_temperatures.append(predicted_temperature)
        scaled_temperature = np.append(scaled_temperature, prediction, axis=0)
    except IndexError as e:
        print(e)

# Print the predicted temperatures per hour for June 30, 2023
for i, dt in enumerate(date_30th_june):
    print(f'{dt}: {predicted_temperatures[i]:.2f}°C')
