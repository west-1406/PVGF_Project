from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation
from tensorflow.keras.layers import LSTM

import joblib

# XGBoost API接口
def XGBoostModel(input_data):
    model = joblib.load("./checkpoints/XGBoost.pkl")
    return model.predict(input_data)


# demo测试样例
def LSTMnet(n_steps_in, out, features):
    lstm_net = Sequential([
        LSTM(1024, return_sequences=True, input_shape=(n_steps_in, features)),
        Dropout(0.2),
        LSTM(512, return_sequences=False),
        Dropout(0.2),
        Dense(1024),
        Activation("linear"),
        Dropout(0.2),
        Dense(512),
        Activation("linear"),
        Dropout(0.2),
        Dense(out),
    ])
    return lstm_net
