import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

# Load the processed CSV file
input_file = './data/processed_data.csv'

df = pd.read_csv(input_file)

# Remove rows with NaN values
df.dropna(inplace=True)

# Define the threshold for removing extreme Y values
threshold = 3.5  # Adjust this value based on your dataset and requirements

# Calculate the mean and standard deviation of Y
mean_Y = np.mean(df['Temperature'])
std_Y = np.std(df['Temperature'])

# Define the range of acceptable Y values
lower_bound = mean_Y - threshold * std_Y
upper_bound = mean_Y + threshold * std_Y

# Filter out rows with Y values outside the acceptable range
df = df[(df['Temperature'] >= lower_bound) & (df['Temperature'] <= upper_bound)]

# Extract the columns
X = df[['Month', 'Date', 'Time', 'Previous Day Average', 'Two Days Before Average', 'Three Days Before Average', 'Last 7 Days Average']].values
Y = df['Temperature'].values

# Normalize X values
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the model
model = GradientBoostingRegressor()

# Train the model
model.fit(X_train, Y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, predictions)

print("Mean Squared Error (MSE):", mse)

# Print some example predictions
for i in range(10):
    print("Predicted:", predictions[i])
    print("Actual:", Y_test[i])
    print()

# Save the trained model
import joblib
joblib.dump(model, "./model.pkl")
