import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the data
input_file = 'data/TaiMoShan.csv'

# Read the CSV file
df = pd.read_csv(input_file)

# Rename the columns for better readability
df.columns = ['Date', 'Time', 'Temperature', 'Relative Humidity', 'Wind Speed', 'Wind Direction', 'Rainfall', 'Wind Direction 2']

# Convert the 'Date' and 'Time' columns to integers
df['Date'] = df['Date'].astype(int)
df['Time'] = df['Time'].astype(int)

# Fill leading zeros for the 'Time' column
df['Time'] = df['Time'].apply(lambda x: str(x).zfill(4))

# Combine the 'Date' and 'Time' columns into a single 'DateTime' column
# df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + df['Time'], format='%Y%m%d%H%M')

# Remove rows with a specific value (e.g., 32767) in 'Temperature' column
df = df[df['Temperature'] != 32767]

# Divide the 'Temperature' column by 10 to convert it to degrees Celsius
df['Temperature'] = df['Temperature'] / 10

# Prepare the data for Prophet
prophet_df = df[['DateTime', 'Temperature']]
prophet_df.columns = ['ds', 'y']

# Split the dataset into training and validation sets (80% for training, 20% for validation)
train_df = prophet_df.loc[prophet_df['ds'] < '2023-06-01']
validation_df = prophet_df.loc[(prophet_df['ds'] >= '2023-06-01') & (prophet_df['ds'] < '2023-07-01')]

validation_df.reset_index(drop=True, inplace=True)

# Initialize and fit the Prophet model
model = Prophet()
model.fit(train_df)

# Make predictions for the validation dataset
validation_predictions = model.predict(validation_df[['ds']])

# Calculate the mean squared error (MSE) for the validation set
mse = np.mean((validation_df['y'] - validation_predictions['yhat']) ** 2)
print("Mean Squared Error (MSE) for the validation set:", mse)

# Generate future date times for prediction
future_dates = pd.date_range(start='2023-06-01', periods=30000, freq='1min')  # Adjust the start date and number of periods as needed

# Create a dataframe with the future dates
future_df = pd.DataFrame({'ds': future_dates})

# Use the trained model to make predictions
predictions = model.predict(future_df)

# Print the predicted temperatures for the future dates
print(predictions[['ds', 'yhat']].tail(10))  # Adjust the number of rows to display as needed

# Print the actual and predicted temperatures for the future dates
actual_values = df.loc[df['DateTime'].isin(future_dates)]
predicted_values = predictions.loc[predictions['ds'].isin(actual_values['DateTime'])]['yhat'].values
comparison_df = pd.DataFrame({'DateTime': actual_values['DateTime'].values, 'Actual': actual_values["Temperature"].values, 'Predicted': predicted_values})
print(comparison_df)

# Calculate accuracy (optional, depending on the desired accuracy metric)
accuracy = np.mean(np.abs(actual_values['Temperature'].values - predicted_values) / actual_values['Temperature'].values)
print("Accuracy:", (1 - accuracy) * 100, "%")

# Visualize the actual vs. predicted temperatures for the validation set
plt.figure(figsize=(12, 6))
plt.plot(validation_df['ds'], validation_df['y'], label='Actual')
plt.plot(validation_predictions['ds'], validation_predictions['yhat'], label='Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Actual vs. Predicted Temperatures (Validation Set)')
plt.legend()
plt.grid(True)
plt.show()
