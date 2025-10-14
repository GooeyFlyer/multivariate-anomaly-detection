# code from https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.optimizers import Adam


def create_dataset(data, time_step=1):
    """Prepares dataset for timeseries forecasting"""
    print("Creating dataset")
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


def gru():
    print("Reading csv")
    df = pd.read_csv("data/temperature_data.csv", parse_dates=['Date'], index_col='Date')
    print(df.head())

    # scale data to ensure all features have equal weight and avoid any bias
    print("Scaling data")
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df.values)

    time_step = 100
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # building GRU model
    print("Building model")
    neurons = 50
    model = Sequential()
    model.add(GRU(units=neurons, return_sequences=True, input_shape=(X.shape[1], 1)))  # GRU returns entire sequence
    model.add(GRU(units=neurons))
    model.add(Dense(units=1))  # output layer that predicts a single value
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')  # Adam() - an adaptive optimizer

    # training the model. epochs is no. of iterations over dataset. batch_size is no. of samples per batch
    print("Training model")
    model.fit(X, y, epochs=10, batch_size=32)

    # predictions. uses last 100 temps in dataset. reshapes dataset as GRU expects 3D data.
    # samples = 1 for 1 prediction. time_steps = 100 and features = 1 because we are predicting only the temperature
    print("Predicting data")
    input_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
    predicted_values = model.predict(input_sequence)

    # inverse transforming converts scaled predictions back to normal
    print("inverse transformation")
    predicted_values = scaler.inverse_transform(predicted_values)

    print(f"The predicted temp for the next day is {predicted_values[0][0]:.2f}°C")


if __name__ == '__main__':
    gru()
