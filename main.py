import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from indicators import get_technical_indicators

# Modeli Yükleme
model = tf.keras.models.load_model('lstm_stock_model.h5')

# Test Verisini Çekme ve İşleme
data = yf.download('^NSEBANK', start='2007-10-16', end='2021-04-19')

# **MultiIndex'i düzelterek veri sütunlarını al**
data.columns = [col[0] for col in data.columns]  
print(f"Düzeltilmiş sütunlar: {data.columns}")  

# **Sadece kapanış fiyatını al ve 'Price' olarak adlandır**
data = data[['Close']]
data.columns = ['Price']

# Teknik Göstergeleri Ekle
data = get_technical_indicators(data)
data.dropna(inplace=True)

# Train ve Test Setlerini Ayır 
train_data = data[data.index < '2019-01-31'].copy()
test_data = data[data.index >= '2019-01-31'].copy()

# **ÇÖZÜM: Train verisinin SON 60 gününü alarak scaler oluştur**
past_60 = train_data.tail(600)  
dt = pd.concat([past_60, test_data], ignore_index=True)  # **HATA DÜZELTİLDİ!**

# **Scaler'ı sadece bu veri setiyle oluştur **
scaler_test = MinMaxScaler()
inputs = scaler_test.fit_transform(dt)

# LSTM İçin Test Verisi Hazırlama 
X_test, y_test = [], []
sequence_length = 60

for i in range(sequence_length, inputs.shape[0]):
    X_test.append(inputs[i-sequence_length:i])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Model ile Tahmin Yapma
y_pred = model.predict(X_test)

# **Ölçekleme Geri Alma (Sadece Tahmin ve Gerçek Kapanış Fiyatları)**
y_pred = scaler_test.inverse_transform(np.hstack([y_pred, np.zeros((y_pred.shape[0], inputs.shape[1]-1))]))[:, 0]
y_test = scaler_test.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], inputs.shape[1]-1))]))[:, 0]

# **Zaman Kaymasını Düzeltme**
prediction_range = range(len(y_test) - len(y_pred), len(y_test))  # **Tahminleri doğru yere hizala**

# Sonuçları Görselleştirme
plt.figure(figsize=(14, 7))
plt.plot(y_test, color='red', label='Gerçek Fiyat')
plt.plot(prediction_range, y_pred, color='blue', label='Tahmin Edilen Fiyat')  # 🔥 **Tahminler kaydırıldı**
plt.title('Hisse Senedi Kapanış Fiyatı Tahmini')
plt.xlabel('Zaman')
plt.ylabel('Fiyat')
plt.legend()
plt.show()
