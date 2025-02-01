import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from indicators import get_technical_indicators

# Model ve Ölçekleyiciyi Yükleme
model = tf.keras.models.load_model('lstm_stock_model.h5')
scaler = joblib.load('scaler.pkl')

# Veriyi Çekme (yfinance kullanarak)
data = yf.download('^NSEBANK', start='2019-01-31', end='2021-04-19')
data = data[['Close']]
data.columns = ['Price']
data = get_technical_indicators(data)
data.dropna(inplace=True)

# Veriyi Ölçekleme
data_scaled = scaler.transform(data)

# LSTM İçin Test Verisi Hazırlama
X_test, y_test = [], []
sequence_length = 60

for i in range(sequence_length, data_scaled.shape[0]):
    X_test.append(data_scaled[i-sequence_length:i])
    y_test.append(data_scaled[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Model ile Tahmin Yapma
y_pred = model.predict(X_test)

# Ölçekleme Geri Alma
y_pred = y_pred * (1 / scaler.scale_[0])
y_test = y_test * (1 / scaler.scale_[0])

# Sonuçları Görselleştirme
plt.figure(figsize=(14, 7))
plt.plot(y_test, color='red', label='Gerçek Fiyat')
plt.plot(y_pred, color='blue', label='Tahmin Edilen Fiyat')
plt.title('Hisse Senedi Kapanış Fiyatı Tahmini')
plt.xlabel('Zaman')
plt.ylabel('Fiyat')
plt.legend()
plt.show()
