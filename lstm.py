#!/usr/bin/env python
# coding: utf-8

# In[121]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Load the data

# In[122]:


input_file = 'data/processed_data.csv'
df = pd.read_csv('data/processed_data.csv',
                 parse_dates=['DateTime'], index_col='DateTime')
df = df.resample('60T').mean()
df = df[(df['Temperature'] >= df['Temperature'].quantile(0.1)) &
        (df['Temperature'] <= df['Temperature'].quantile(0.9))]
# df["DateTime"] = pd.to_datetime(df.index)
df = df.reset_index()


# Rename the columns for better readability

# In[123]:


# df.columns = ['DateTime', 'Year', 'Month', 'Date', 'Time', 'Minute', 'Temperature', 'Previous Day Average', 'Two Days Before Average',
# 'Three Days Before average', 'Last 7 Days Average', 'Previous Day Wind Speed', 'Previous Day Rainfall']


# Convert the 'Date' and 'Time' columns to integers

# In[124]:


df['Date'] = df['Date'].astype(int)
df['Time'] = df['Time'].astype(int)


# Fill leading zeros for the 'Time' column

# In[125]:


df['Time'] = df['Time'].apply(lambda x: str(x).zfill(4))


# Combine the 'Date' and 'Time' columns into a single 'DateTime' column

# In[126]:


# df['DateTime'] = pd.to_datetime(df["DateTime"], format="%Y-%m-%d %H:%M:%S")


# In[ ]:


# Remove rows with a specific value (e.g., 32767) in 'Temperature' column

# In[127]:


df = df[df['Temperature'] != 32767]


# Calculate the mean and standard deviation of Y

# In[128]:


threshold = 5
mean_Y = np.mean(df['Temperature'])
std_Y = np.std(df['Temperature'])


# Define the range of acceptable Y values

# In[129]:


lower_bound = mean_Y - threshold * std_Y
upper_bound = mean_Y + threshold * std_Y


# Filter out rows with Y values outside the acceptable range

# In[130]:


df = df[(df['Temperature'] >= lower_bound) &
        (df['Temperature'] <= upper_bound)]


# Prepare the data for LSTM

# In[131]:


time_steps = 60  # Number of time steps for the LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))  # Scale the data to [0, 1]


# Create sequences of input data and corresponding target values

# In[133]:


# Filter out the outliers and invalid values for the new features
# Replace the specific invalid values (e.g., 32767) with np.nan
df['Wind Speed'] = df['Wind Speed'].replace(32767, np.nan)
df['Rainfall'] = df['Rainfall'].replace(32767, np.nan)
df['Wind Direction'] = df['Wind Direction'].replace(32767, np.nan)
df = df[0 <= df['Wind Direction'] <= 360]

# Remove rows with missing values
df = df.dropna()

# Scale the temperature, wind speed, rainfall, and wind direction values
features = df[['Temperature', 'Wind Speed',
               'Rainfall', 'Wind Direction']].values
scaled_features = scaler.fit_transform(features)

# Create sequences of input data and corresponding target values
data = []
target = []
for i in range(24, len(scaled_features) - time_steps):
    data.append(scaled_features[i-24:i+time_steps-24])
    # Only the temperature is the target
    target.append(scaled_features[i+time_steps, 0])

data = np.array(data)
target = np.array(target)


# In[135]:


df["DateTime"]


# Define exclude date (year, month, and day)

# In[136]:


exclude_year = 2022
exclude_month = 6
exclude_day = 1


# Split the dataset into training and validation sets

# In[137]:


exclude_date = pd.to_datetime(f"{exclude_year}-{exclude_month}-{exclude_day}")
exclude_index = df[df['DateTime'] >= exclude_date].index[0]
print(exclude_index)
train_data, train_target = data[:exclude_index], target[:exclude_index]
val_data, val_target = data[exclude_index:], target[exclude_index:]


# Build the LSTM model architecture

# In[138]:


model = tf.keras.Sequential([
    tf.keras.layers.LSTM(256, return_sequences=True,
                         input_shape=(time_steps, 4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(256),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])


# Compile the model

# In[139]:


optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_absolute_error', optimizer=optimizer)


# Define early stopping and learning rate scheduler

# In[140]:


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)


# Train the LSTM model

# In[141]:


batch_size = 64
epochs = 100
history = model.fit(train_data, train_target, batch_size=batch_size, epochs=epochs, validation_data=(val_data, val_target),
                    callbacks=[early_stopping, lr_scheduler])


# Make predictions using the trained LSTM model

# In[142]:


predictions = model.predict(val_data)


# Rescale the predictions back to the original range

# In[143]:


scaled_predictions = predictions.reshape(-1, 1)
predicted_temperature = scaler.inverse_transform(scaled_predictions)


# Calculate MSE and MAE

# In[144]:


mse = mean_squared_error(scaler.inverse_transform(
    val_target), predicted_temperature)
mae = mean_absolute_error(scaler.inverse_transform(
    val_target), predicted_temperature)


# In[145]:


print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)


# Visualize the actual vs. predicted temperatures for the validation set

# In[146]:


plt.figure(figsize=(12, 6))
plt.plot(range(len(val_target)), scaler.inverse_transform(
    val_target), label='Actual')
plt.plot(range(len(val_target)), predicted_temperature, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Actual vs. Predicted Temperatures (Validation Set)')
plt.legend()
plt.show()


# Visualize the actual vs. predicted temperatures for the training set

# In[147]:


plt.figure(figsize=(12, 6))
plt.plot(range(len(train_target)), train_target, label='Actual')
plt.plot(range(len(train_target)), model.predict(
    train_data), label='Predicted')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Actual vs. Predicted Temperatures (Training Set)')
plt.legend()
plt.show()


# Save the trained model

# In[148]:


model.save("lstm.keras")
