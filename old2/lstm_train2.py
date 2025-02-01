import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

def train_lstm_model(X_train, y_train, epochs=50, batch_size=64):
    """
    LSTM modelini eğitir ve döndürür.
    - L2 regularization (0.001)
    - Dropout (0.1)
    """

    model = Sequential([
        LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.1),
        
        LSTM(units=60, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001)),
        Dropout(0.1),
        
        LSTM(units=80, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001)),
        Dropout(0.1),
        
        LSTM(units=120, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.1),
        
        Dense(units=1)  # Çıkış katmanı
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    print("🎯 Model Eğitimi Başlıyor...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    return model
