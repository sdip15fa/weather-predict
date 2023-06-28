import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the data
input_file = 'data/processed_data.csv'
df = pd.read_csv(input_file)

# Rename the columns for better readability
df.columns = ['DateTime', 'Year', 'Month', 'Date', 'Time', 'Minute', 'Temperature', 'Previous Day Average', 'Two Days Before Average',
              'Three Days Before average', 'Last 7 Days Average', 'Previous Day Wind Speed', 'Previous Day Rainfall']

# Convert the 'Date' and 'Time' columns to integers
df['Date'] = df['Date'].astype(int)
df['Time'] = df['Time'].astype(int)

# Fill leading zeros for the 'Time' column
df['Time'] = df['Time'].apply(lambda x: str(x).zfill(4))

# Combine the 'Date' and 'Time' columns into a single 'DateTime' column
df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + df['Time'], format='%Y%m%d%H%M')

# Remove rows with a specific value (e.g., 32767) in 'Temperature' column
df = df[df['Temperature'] != 32767]

# Calculate the mean and standard deviation of Y
threshold = 5
mean_Y = np.mean(df['Temperature'])
std_Y = np.std(df['Temperature'])

# Define the range of acceptable Y values
lower_bound = mean_Y - threshold * std_Y
upper_bound = mean_Y + threshold * std_Y

# Filter out rows with Y values outside the acceptable range
df = df[(df['Temperature'] >= lower_bound) & (df['Temperature'] <= upper_bound)]

# Prepare the data for LSTM
time_steps = 60  # Number of time steps for the LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))  # Scale the data to [0, 1]

# Scale the temperature values
scaled_temperature = scaler.fit_transform(df['Temperature'].values.reshape(-1, 1))

# Create sequences of input data and corresponding target values
data = []
target = []
for i in range(len(scaled_temperature) - time_steps):
    data.append(scaled_temperature[i:i+time_steps])
    target.append(scaled_temperature[i+time_steps])

data = np.array(data)
target = np.array(target)

# Split the dataset into training and validation sets (80% for training, 20% for validation)
train_size = int(0.8 * len(data))
train_data, train_target = data[:train_size], target[:train_size]
val_data, val_target = data[train_size:], target[train_size:]

# Build the LSTM model architecture
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(time_steps, 1)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the LSTM model
batch_size = 64
epochs = 100
history = model.fit(train_data, train_target, batch_size=batch_size, epochs=epochs, validation_data=(val_data, val_target))

# Make predictions using the trained LSTM model
predictions = model.predict(val_data)

# Rescale the predictions back to the original range
scaled_predictions = predictions.reshape(-1, 1)
predicted_temperature = scaler.inverse_transform(scaled_predictions)

# Visualize the actual vs. predicted temperatures for the validation set
plt.figure(figsize=(12, 6))
plt.plot(range(len(val_target)), val_target, label='Actual')
plt.plot(range(len(val_target)), predicted_temperature, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Actual vs. Predicted Temperatures (Validation Set)')
plt.legend()
plt.show()
