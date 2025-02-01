import numpy as np
import pandas as pd
import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import joblib
from indicators import get_technical_indicators

# Veri Çekme (yfinance kullanarak)
data = yf.download('^NSEBANK', start='2019-01-31', end='2021-04-19')
data = data[['Close']]
data.columns = ['Price']

# Teknik İndikatörleri Hesaplama
data = get_technical_indicators(data)
data.dropna(inplace=True)

# Eğitim ve Test Setlerine Ayırma
train_data = data[data.index < '2020-06-01'].copy()
test_data = data[data.index >= '2020-06-01'].copy()

# MinMax Ölçekleme
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
joblib.dump(scaler, 'scaler.pkl')

# LSTM İçin Veri Hazırlama
X_train, y_train = [], []
sequence_length = 60

for i in range(sequence_length, train_scaled.shape[0]):
    X_train.append(train_scaled[i-sequence_length:i])
    y_train.append(train_scaled[i, 0])  # Fiyat hedef değişkeni

X_train, y_train = np.array(X_train), np.array(y_train)

# LSTM Modeli Oluşturma
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(60, activation='relu', return_sequences=True),
    Dropout(0.3),
    LSTM(80, activation='relu', return_sequences=True),
    Dropout(0.4),
    LSTM(120, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli Eğitme
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)

# Modeli Kaydetme
model.save('lstm_stock_model.h5')
