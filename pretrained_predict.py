import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the data
input_file = 'data/processed_data.csv'
df = pd.read_csv(input_file)

# Rename the columns for better readability
df.columns = ['DateTime', 'Year', 'Month', 'Date', 'Time', 'Minute', 'Temperature', 'Previous Day Average', 'Two Days Before Average', 'Three Days Before average', 'Last 7 Days Average', 'Previous Day Wind Speed', 'Previous Day Rainfall']

# Convert the 'Date' and 'Time' columns to integers
df['Date'] = df['Date'].astype(int)
df['Time'] = df['Time'].astype(int)

# Fill leading zeros for the 'Time' column
df['Time'] = df['Time'].apply(lambda x: str(x).zfill(4))

# Combine the 'Date' and 'Time' columns into a single 'DateTime' column
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + df['Time'], format='%Y%m%d%H%M')

# Filter the data for June 2023
june_2023_df = df.loc[(df['DateTime'] >= '2023-06-01') & (df['DateTime'] < '2023-07-01')]

# Normalize the temperature values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
june_2023_df['Temperature'] = scaler.fit_transform(june_2023_df['Temperature'].values.reshape(-1, 1))

# Prepare the input features and target variable
input_features = june_2023_df[['Previous Day Average', 'Two Days Before Average', 'Three Days Before average', 'Last 7 Days Average', 'Previous Day Wind Speed', 'Previous Day Rainfall']]
target_variable = june_2023_df['Temperature']

# Convert input features and target variable to numpy arrays
input_features = input_features.to_numpy()
target_variable = target_variable.to_numpy()

# Reshape the input features to 3D shape (samples, timesteps, features)
input_features = input_features.reshape(input_features.shape[0], 1, input_features.shape[1])

# Load the pre-trained LSTM model
model = tf.keras.models.load_model('lstm.keras')

# Make predictions for the June 2023 data
predicted_temperature = model.predict(input_features)

# Rescale the predicted temperature values to the original scale
predicted_temperature = scaler.inverse_transform(predicted_temperature)

# Visualize the actual vs. predicted temperatures for June 2023
plt.figure(figsize=(12, 6))
plt.plot(june_2023_df['DateTime'], target_variable, label='Actual')
plt.plot(june_2023_df['DateTime'], predicted_temperature, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Actual vs. Predicted Temperatures (June 2023)')
plt.legend()
plt.show()
