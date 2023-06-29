import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

# Load data and preprocess
data = pd.read_csv('data/processed_data.csv', parse_dates=['DateTime'], index_col='DateTime')
data = data.resample('60T').mean()
data = data[(data['Temperature'] >= data['Temperature'].quantile(0.01)) & (data['Temperature'] <= data['Temperature'].quantile(0.99))]

# Prepare dataset
def create_dataset(series, look_back=1):
    X, Y = [], []
    for i in range(len(series)-look_back-1):
        X.append(series[i:(i+look_back), 0])
        Y.append(series[i + look_back, 0])
    return np.array(X), np.array(Y)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(data['Temperature'].values.reshape(-1, 1))

train_size = int(len(dataset) * 0.8)
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

look_back = 24
X_train, Y_train = create_dataset(train, look_back)
X_test, Y_test = create_dataset(test, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Create and train LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
lr_scheduler = LearningRateScheduler(lambda epoch: 0.001 * 0.9 ** epoch)

history = model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2, validation_split=0.1, callbacks=[early_stopping, lr_scheduler])

# Predictions and metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])[0]
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])[0]

print('Train Mean Absolute Error:', mean_absolute_error(Y_train, train_predict[:,0]))
print('Train Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_train, train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test, test_predict[:,0]))
print('Test Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test, test_predict[:,0])))

# Plot training and test predictions
plt.figure(figsize=(15, 8))
plt.plot(data['Temperature'], label='True Temperature')
plt.plot(pd.Series(np.concatenate([train_predict[:, 0], test_predict[:, 0]]), index=data.index[look_back+1:]), label='Predicted Temperature')
plt.xlabel("DateTime")
plt.ylabel("Temperature")
plt.legend()
plt.show()

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Predict next 24 hours
last_24_hours = dataset[-look_back:]
last_24_hours = np.reshape(last_24_hours, (1, 1, look_back))
next_24_hours = []

for _ in range(24):
    prediction = model.predict(last_24_hours)
    next_24_hours.append(prediction[0][0])
    last_24_hours = np.append(last_24_hours[:, :, 1:], prediction)
    last_24_hours = np.reshape(last_24_hours, (1, 1, look_back))

next_24_hours = scaler.inverse_transform(np.array(next_24_hours).reshape(-1, 1))

print("Predicted temperatures for next 24 hours:")
print(next_24_hours)