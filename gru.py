# code from https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/
# code from https://www.geeksforgeeks.ord/deep-learning/multivariate-time-series-forecasting-with-grus/
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

from data.get_data import preprocess_data


def create_dataset(data, time_step=1):
    """Prepares dataset for timeseries forecasting"""
    print("Creating dataset")
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


def gru():
    """Single prediction from temperature_data"""
    print("Reading csv")
    df = pd.read_csv("data/temperature_data.csv", parse_dates=['Date'], index_col='Date')
    print(df.head())

    # scale data to ensure all features have equal weight and avoid any bias
    print("Scaling data")
    scaler = MinMaxScaler(feature_range=(0, 1))
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

    # training the model. epochs = no. of iterations over dataset. batch_size is no. of samples per batch
    print("Training model")
    model.fit(X, y, epochs=3, batch_size=32)

    # predictions. uses last 100 temps in dataset. reshapes dataset as GRU expects 3D data.
    # samples = 1 for 1 prediction. time_steps = 100 and features = 1 because we are predicting only the temperature
    print("Predicting data")
    input_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)  # 3D reshape. 1 block, time_steps rows, 1 column
    predicted_values = model.predict(input_sequence)

    # inverse transforming converts scaled predictions back to normal
    print("inverse transformation")
    predicted_values = scaler.inverse_transform(predicted_values)

    print(f"The predicted temp for the next day is {predicted_values[0][0]:.2f}°C")

    print(predicted_values)


def singleStepSampler(df: pd.DataFrame, window: int) -> (np.ndarray, np.ndarray):
    """Prepares the data for single-step time-series forecasting"""
    xRes = []  # store input features
    yRes = []  # store target values

    # create sequences of input features and corresponding target values based on window size
    # input features constructed as a sequence of windowed data points,
    # where each data point is a list containing values from each column of the dataframe
    for i in range(0, len(df) - window):
        res = []
        for j in range(0, window):
            r = []
            for col in df.columns:
                r.append(df[col][i + j])
            res.append(r)
        xRes.append(res)
        # filter output columns here if needed
        yRes.append(df.iloc[i + window].values)  # filter columns here
    return np.array(xRes), np.array(yRes)


def gru_plot_predictions(file_path: str):
    """splits file_path data into train and test data.
    plots gru predictions for each column in .csv file"""
    data = preprocess_data(pd.read_csv(file_path, sep=";"))

    # missing values imputed with np.nan
    imputer = SimpleImputer(missing_values=np.nan)
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    data = data.reset_index(drop=True)

    # feature scaling - ensure they all fall within 0 to 1
    scalar = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scalar.fit_transform(data.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=list(data.columns))

    # use for target scaling of specific columns
    # target_scalar = MinMaxScaler(feature_range=(0, 1))

    df_scaled = df_scaled.astype(float)

    # apply singleStepSample with window size of 20
    (xVal, yVal) = singleStepSampler(df_scaled, window=20)

    # a constant split with a value of 0.85 is defined, specifying proportion of data to be used for training
    # xVal and yVal are split into training and testing sets according to the split ratio.
    # training set has 0.85% of data, testing has 0.15%
    SPLIT = 0.85

    x_train = xVal[:int(SPLIT * len(xVal))]
    y_train = yVal[:int(SPLIT * len(yVal))]
    x_test = xVal[int(SPLIT * len(xVal)):]
    y_test = yVal[int(SPLIT * len(yVal)):]

    # initialise Sequential model, a linear stack of layers
    multivariate_gru = tf.keras.Sequential()

    # GRU layer with 200 units
    # takes input sequences with shape defined by number of features in training data (x_train)
    multivariate_gru.add(tf.keras.layers.GRU(200, input_shape=(x_train.shape[1], x_train.shape[2])))

    # dropout layer to prevent overfitting
    multivariate_gru.add(tf.keras.layers.Dropout(0.5))

    # output layer for predicted variables
    # units equal to number of columns in y_train
    multivariate_gru.add(tf.keras.layers.Dense(y_train.shape[1], activation="linear"))

    multivariate_gru.compile(loss="MeanSquaredError",  # loss function
                             metrics=["MAE", "MSE"],  # metrics for further evaluation
                             optimizer=tf.keras.optimizers.Adam())  # for training
    multivariate_gru.summary()

    # train model
    history = multivariate_gru.fit(x_train, y_train, epochs=50)

    # predict values and plot
    predicted_values = multivariate_gru.predict(x_test)

    d = {}
    for x in range(predicted_values.shape[1]):
        d[f"predicted column {x}"] = predicted_values[:, x]
        d[f"actual column {x}"] = y_test[:, x]

    d = pd.DataFrame(d)

    for x in range(predicted_values.shape[1]):
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(d[[f"actual column {x}", f"predicted column {x}"]],
                 label=[f"actual column {x}", f"predicted column {x}"])
        plt.xlabel("Timestamps")
        plt.ylabel("Values")
        ax.legend()

        plt.show()


if __name__ == '__main__':
    file_path = sys.argv[1]
    gru_plot_predictions(file_path)
