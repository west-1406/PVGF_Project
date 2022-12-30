from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM

# demo测试样例
def LSTMnet(n_steps_in, out, features):
    lstm_net = Sequential([
        LSTM(96, return_sequences=True, input_shape=(n_steps_in, features)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32),
        Dropout(0.2),
        Dense(out),
    ])
    return lstm_net
