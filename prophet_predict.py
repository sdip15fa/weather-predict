#!/usr/bin/env python
# coding: utf-8

# In[518]:


import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.offline as py
import matplotlib.pyplot as plt
py.init_notebook_mode()
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')


# Load the data

# In[519]:


input_file = 'data/processed_data.csv'


# Read the CSV file

# In[520]:


df = pd.read_csv(input_file)


# In[521]:


df.head()


# In[522]:


df.info()


# Rename the columns for better readability

# In[523]:


df.columns = ['DateTime', 'Year', 'Month', 'Date', 'Time', 'Minute', 'Temperature', 'Previous Day Average', 'Two Days Before Average', 'Three Days Before average', 'Last 7 Days Average', 'Previous Day Wind Speed', 'Previous Day Rainfall']


# Convert the 'Date' and 'Time' columns to integers

# In[524]:


df['Date'] = df['Date'].astype(int)
df['Time'] = df['Time'].astype(int)


# Fill leading zeros for the 'Time' column

# In[525]:


df['Time'] = df['Time'].apply(lambda x: str(x).zfill(4))


# Combine the 'Date' and 'Time' columns into a single 'DateTime' column<br>
# df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + df['Time'], format='%Y%m%d%H%M')

# Remove rows with a specific value (e.g., 32767) in 'Temperature' column

# In[526]:


df = df[df['Temperature'] != 32767]


# Calculate the mean and standard deviation of Y

# In[527]:


threshold = 5
mean_Y = np.mean(df['Temperature'])
std_Y = np.std(df['Temperature'])


# Define the range of acceptable Y values

# In[528]:


lower_bound = mean_Y - threshold * std_Y
upper_bound = mean_Y + threshold * std_Y


# Filter out rows with Y values outside the acceptable range

# In[529]:


df = df[(df['Temperature'] >= lower_bound) & (df['Temperature'] <= upper_bound)]


# Divide the 'Temperature' column by 10 to convert it to degrees Celsius

# In[530]:


# df['Temperature'] = df['Temperature'] / 10


# Prepare the data for Prophet

# In[531]:


df['DateTime'] = pd.to_datetime(df['DateTime'].astype(int).astype(str) + df['Time'].astype(int).astype(str) + df['Minute'].astype(str), format='%Y%m%d%H%M')
prophet_df = df.copy()
prophet_df.rename(columns={'DateTime': 'ds', 'Temperature': 'y'}, inplace=True)
prophet_df.dropna(inplace=True)
prophet_df = prophet_df[prophet_df['Month'] == 7]


# In[532]:


prophet_df.head()


# In[533]:


prophet_df.info()


# In[534]:


ax = prophet_df[["ds", "y"]].set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('Temperature')
ax.set_xlabel('Date')

plt.show()


# Split the dataset into training and validation sets (80% for training, 20% for validation)

# In[535]:


train_df = prophet_df # .loc[prophet_df['ds'] < '2022-07-01']
validation_df = prophet_df.loc[(prophet_df['ds'] >= '2022-07-01') & (prophet_df['ds'] < '2022-08-01')]


# In[536]:


validation_df.reset_index(drop=True, inplace=True)


# Initialize and fit the Prophet model

# In[537]:


model = Prophet(
    interval_width=0.95,  # Increase the interval width to capture more uncertainty
    n_changepoints=250,  # Increase the number of changepoints for more flexibility
    yearly_seasonality=True,  # Keep yearly seasonality
    weekly_seasonality=False,  # Add weekly seasonality
    daily_seasonality=True,  # Add daily seasonality
    changepoint_prior_scale=20,  # Reduce the changepoint prior scale for smoother trends
    seasonality_mode='additive',  # Use multiplicative seasonality for better handling of varying scales
    # changepoint_range=0.8,  # Restrict changepoints to the first 80% of the data
    seasonality_prior_scale=10,
    # growth='logistic'
)
model.add_regressor('Previous Day Average')
model.add_regressor('Two Days Before Average')
model.add_regressor('Three Days Before average')
model.add_regressor('Last 7 Days Average')
model.add_regressor('Previous Day Wind Speed')
model.add_regressor('Previous Day Rainfall')
model.add_regressor('Time')
model.add_seasonality(name='daily', period=1, fourier_order=10)
model.add_seasonality(name='hourly', period=1/24, fourier_order=10)
model.fit(train_df)


# Make predictions for the validation dataset

# In[ ]:


validation_predictions = model.predict(validation_df)


# Calculate the mean squared error (MSE) for the validation set

# In[ ]:


mse = np.mean((validation_df['y'] - validation_predictions['yhat']) ** 2)
print("Mean Squared Error (MSE) for the validation set:", mse)
mas = np.mean(abs(validation_df['y'] - validation_predictions['yhat']))
print("Mean Absolute Error (MAE) for the validation set:", mas)
# accuracy = 


# Generate future date times for prediction

# In[ ]:


future_dates = pd.date_range(start='2022-07-01', periods=30000, freq='1min')  # Adjust the start date and number of periods as needed


# Create a dataframe with the future dates

# In[ ]:


future_df =  pd.DataFrame({'ds': future_dates})


# Use the trained model to make predictions

# In[ ]:


predictions = model.predict(validation_df)
predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()


# In[ ]:


all_predictions = model.predict(prophet_df.dropna(inplace=True))


# In[ ]:


model.plot(all_predictions, uncertainty=True)


# Print the predicted temperatures for the future dates

# In[ ]:


print(predictions[['ds', 'yhat']].tail(10))  # Adjust the number of rows to display as needed


# Print the actual and predicted temperatures for the future dates

# In[ ]:


# actual_values = df.loc[df['DateTime'].isin(future_dates)]
# predicted_values = predictions.loc[predictions['ds'].isin(actual_values['DateTime'])]['yhat'].values
# comparison_df = pd.DataFrame({'DateTime': actual_values['DateTime'].values, 'Actual': actual_values["Temperature"].values, 'Predicted': predicted_values})
# print(comparison_df)


# Calculate accuracy (optional, depending on the desired accuracy metric)

# In[ ]:


# accuracy = np.mean(np.abs(actual_values['Temperature'].values - predicted_values) / actual_values['Temperature'].values)
# print("Accuracy:", (1 - accuracy) * 100, "%")


# Visualize the actual vs. predicted temperatures for the validation set

# In[ ]:


plt.figure(figsize=(12, 6))
plt.plot(validation_df['ds'], validation_df['y'], label='Actual')
plt.plot(validation_predictions['ds'], validation_predictions['yhat'], label='Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Actual vs. Predicted Temperatures (Validation Set)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:


plt.figure(figsize=(12, 6))
# condition = '2021-12-31' < prophet_df['ds'] < '2023-01-01'
plt.plot(prophet_df[(prophet_df['ds'] < '2023-01-01') & (prophet_df['ds'] > '2021-12-31')]['ds'], prophet_df[(prophet_df['ds'] < '2023-01-01') & (prophet_df['ds'] > '2021-12-31')]['y'], label='Actual')
plt.plot(all_predictions[(all_predictions['ds'] < '2023-01-01') & (all_predictions['ds'] > '2021-12-31')]['ds'], all_predictions[(all_predictions['ds'] < '2023-01-01') & (all_predictions['ds'] > '2021-12-31')]['yhat'], label='Predicted')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Actual vs. Predicted Temperatures (Validation Set)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


import pickle

# Assuming you have trained the Prophet model and stored it in a variable called 'model'
with open('prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)

