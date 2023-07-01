import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

data_df = pd.read_csv("data/processed_data.csv", parse_dates=["DateTime"], index_col="DateTime")
predict_df = pd.read_csv("predict.csv", parse_dates=["DateTime"], index_col="DateTime")

data_df = data_df.resample("60T").mean()
predict_df = predict_df.resample("60T").mean()

dates = predict_df.index

actual_data = data_df[data_df.index.isin(dates)]

# Extract the actual and predicted temperature values
actual_temperatures = actual_data["Temperature"]
predicted_temperatures = predict_df["Temperature"]

mse = mean_squared_error(actual_temperatures, predicted_temperatures[predicted_temperatures.index.isin(actual_temperatures.index)])
mae = mean_absolute_error(actual_temperatures, predicted_temperatures[predicted_temperatures.index.isin(actual_temperatures.index)])

print("mean squared error:", mse)
print("mean absolute error:", mae)

# Plot the graph
plt.figure(figsize=(12, 6))
plt.plot(actual_temperatures.index, actual_temperatures, label="Actual Temperature")
plt.plot(predicted_temperatures.index, predicted_temperatures, label="Predicted Temperature")
plt.xlabel("Date and Time")
plt.ylabel("Temperature")
plt.title("Actual vs Predicted Temperatures")
plt.legend()
plt.show()