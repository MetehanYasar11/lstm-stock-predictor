import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def create_lstm_model(input_shape):
    """
    LSTM modeli oluşturur.
    """
    model = Sequential()

    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_lstm_model(X_train, y_train, epochs=50, batch_size=64):
    """
    LSTM modelini eğitir ve döndürür.
    """
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model
