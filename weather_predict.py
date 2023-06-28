import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Load the processed CSV file
input_file = './data/processed_data.csv'

df = pd.read_csv(input_file)

# Remove rows with NaN values
df.dropna(inplace=True)

# df['DateTime'] = pd.to_datetime(df['DateTime'], format="%Y%m%d%H%M")

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

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', input_shape=(7,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1) # Output layer with 1 unit for air temperature
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# Compile the model
model.compile(optimizer=optimizer, loss="mean_squared_error")

# Define early stopping and learning rate scheduler
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

# Train the model
history = model.fit(X_train, Y_train, epochs=500, batch_size=64, validation_data=(X_test, Y_test), callbacks=[early_stopping, lr_scheduler])

# Evaluate the model
loss = model.evaluate(X_test, Y_test)
print("Test loss:", loss)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy and mean squared error (MSE)
accuracy = 100 - np.mean(np.abs((Y_test - predictions) / Y_test)) * 100
mse = mean_squared_error(Y_test, predictions)

print("Accuracy:", accuracy)
print("Mean Squared Error (MSE):", mse)

# Plot error (predicted - actual)
error = predictions.flatten() - Y_test

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, error)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual Air Temperature')
plt.ylabel('Error (Predicted - Actual)')
plt.title('Error Plot')
plt.show()

# Plot training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Print some example predictions
for i in range(10):
    print("Predicted:", predictions[i])
    print("Actual:", Y_test[i])
    print()

# Save the TensorFlow model
model.save("./model.keras")
