import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def train_lstm_model(X_train, y_train, epochs=50, batch_size=64, save_path="lstm_model.h5"):
    """
    LSTM modelini eğitir ve kaydeder.
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
    
    # Eğitilen modeli kaydet
    model.save(save_path)
    print(f"✅ Model kaydedildi: {save_path}")

    return model

def prepare_test_data(data_testing, scaler):
    """
    Test verisini hazırlar.
    """
    past_60_days = data_testing.iloc[:60]
    test_data = past_60_days.append(data_testing, ignore_index=True)
    inputs = scaler.transform(test_data)

    X_test, y_test = [], []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i])
        y_test.append(inputs[i, 0])  # Sadece 'Price' sütunu tahmin edilecek

    return np.array(X_test), np.array(y_test)

def predict_and_visualize(model, X_test, y_test, scaler):
    """
    Modeli kullanarak tahmin yapar ve sonuçları görselleştirir.
    """
    y_pred = model.predict(X_test)
    
    # 🛠 Tahminleri eski ölçeğe döndür!
    y_pred = scaler.inverse_transform(np.concatenate((y_pred, np.zeros((y_pred.shape[0], 11))), axis=1))[:, 0] # Sadece Price sütunu
    y_test = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 11))), axis=1))[:, 0] # Gerçek değerleri de geri döndür

    # Min ve Max değerleri al
    price_min = scaler.data_min_[0]
    price_max = scaler.data_max_[0]

    # Model tahminlerini geri ölçekle
    y_pred = y_pred * (price_max - price_min) + price_min

    # Gerçek test değerlerini de geri ölçekle
    y_test = y_test * (price_max - price_min) + price_min

    # 📊 Grafik Çizimi
    plt.figure(figsize=(14, 5))
    plt.plot(y_test, label="Gerçek Değer", color="red")
    plt.plot(y_pred, label="Tahmin", color="blue")
    plt.xlabel("Zaman")
    plt.ylabel("Fiyat")
    plt.legend()
    plt.title("Gerçek vs. Tahmin Edilen Fiyat")
    plt.show()
    
    return y_pred
