import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD,Adadelta,Adam,RMSprop 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import optimizers
import itertools
import numpy as np
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    # Target timestep (t=lag)
    cols.append(data.shift(-lag))
    names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def get_sales():
    # Read data
    df = pd.read_csv(
        '../data/sales_top50groups.csv'
        # 'data/product_salestop50.csv'
    )
    
    df.start_date = pd.to_datetime(df.start_date, format='%m/%d/%Y').dt.date
    df = df.rename(columns={'start_date': 'date'})
    df = df.sort_values(by=['object_id', 'natural_week'], ascending=[True, True])

    df.set_index('date', inplace=True)

    return df

sales = get_sales()


window = 29
lag = 4

series = pd.DataFrame()

for obj in sales.object_id.unique():

    obj_series = series_to_supervised(
        sales[sales.object_id == obj][['quantity']], window=window, lag=lag)
    obj_series['object_id'] = obj
    series = series.append(obj_series)

# series.head()

# Label
labels_col = 'quantity(t+%d)' % lag

epochs = 50
batch = 256
lr = 0.003
adam = optimizers.Adam(lr)

for obj in series.object_id.unique():
    print(obj)

    obj_series = series[series.object_id == obj]
    labels = obj_series[labels_col]
    obj_series = obj_series.drop(labels_col, axis=1)
    obj_series = obj_series.drop('object_id', axis=1)
    x_train, x_valid, y_train, y_valid = train_test_split(obj_series, labels.values, test_size=0.1, random_state=0)

    x_train = x_train.astype(np.float32)
    x_valid = x_valid.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_valid = y_valid.astype(np.float32)

    # model_mlp = Sequential()
    # model_mlp.add(Dense(100, activation='relu', input_dim=x_train.shape[1]))
    # model_mlp.add(Dense(1))
    # model_mlp.compile(loss='mse', optimizer=adam)
    # model_mlp.summary()
    # # breakpoint()
    # mlp_history = model_mlp.fit(x_train.values, y_train, validation_data=(x_valid.values, y_valid), epochs=epochs, verbose=0)
    
    # mlp_train_pred = model_mlp.predict(x_train.values)
    # mlp_valid_pred = model_mlp.predict(x_valid.values)
    # print('Train rmse:', np.sqrt(mean_squared_error(y_train, mlp_train_pred)))
    # print('Validation rmse:', np.sqrt(mean_squared_error(y_valid, mlp_valid_pred)))

    X_train_series = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))
    X_valid_series = x_valid.values.reshape((x_valid.shape[0], x_valid.shape[1], 1))

    model_lstm = Sequential()
    model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mse', optimizer=adam)
    model_lstm.summary()
    # breakpoint()
    lstm_history = model_lstm.fit(X_train_series, y_train, validation_data=(X_valid_series, y_valid), epochs=epochs, verbose=2)
    
    lstm_train_pred = model_lstm.predict(X_train_series)
    lstm_valid_pred = model_lstm.predict(X_valid_series)
    print('Train rmse:', np.sqrt(mean_squared_error(y_train, lstm_train_pred)))
    print('Validation rmse:', np.sqrt(mean_squared_error(y_valid, lstm_valid_pred)))
    # breakpoint()
