import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import date
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from lstm_train3 import train_lstm_model, predict_and_visualize
from indicators import get_technical_indicators
import yfinance as yf

def download_data(ticker, start_date, end_date):
    """
    Yahoo Finance verisini `yfinance` ile çeker ve 'Adj Close' veya 'Close' sütununu kullanır.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)

        print(f"📊 Çekilen veri sütunları: {data.columns.tolist()}")

        if "Adj Close" in data.columns:
            data = data[['Adj Close']]
        elif "Close" in data.columns:
            data = data[['Close']]
        else:
            raise KeyError(f"Hata: 'Adj Close' veya 'Close' sütunu bulunamadı. Mevcut sütunlar: {data.columns.tolist()}")

        data.columns = ['Price']
        print(f"✅ Veri başarıyla çekildi! {data.shape[0]} gün kayıt bulundu.")
        return data

    except Exception as e:
        raise RuntimeError(f"Veri çekme hatası: {e}")

def preprocess_data(data):
    """
    Teknik göstergeleri hesaplar, eksik verileri temizler ve veriyi ölçekler.
    """
    data = get_technical_indicators(data)

    data_training = data[data.index < '2019-01-31'].copy()
    data_testing = data[data.index >= '2019-01-31'].copy()

    scaler = MinMaxScaler()
    data_training_scaled = scaler.fit_transform(data_training)

    X_train, y_train = [], []
    for i in range(60, len(data_training_scaled)):
        X_train.append(data_training_scaled[i-60: i])
        y_train.append(data_training_scaled[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    return X_train, y_train, scaler, data_testing

def main():
    """
    1. Veriyi indir
    2. Teknik göstergeleri hesapla
    3. LSTM modelini eğit veya yükle
    4. Test seti ile tahmin yap ve görselleştir
    """
    ticker = "^NSEBANK"
    start_date = datetime.datetime(2000, 1, 2)
    # end_date = date.today()
    # set end date to Apr 19, 2021
    end_date = datetime.datetime(2021, 4, 19)

    print("📥 Veri indiriliyor...")
    data = download_data(ticker, start_date, end_date)

    print("🔍 Veri ön işleniyor...")
    X_train, y_train, scaler, data_testing = preprocess_data(data)

    # Modeli yükle veya eğit
    model_path = "lstm_model.h5"
    try:
        model = load_model(model_path)
        print(f"✅ Önceden eğitilmiş model yüklendi: {model_path}")
    except:
        print("🎯 LSTM Modeli Eğitiliyor...")
        model = train_lstm_model(X_train, y_train, save_path=model_path)

    print("📊 Model Test Verisi ile Çalıştırılıyor...")
    past_60 = data_testing.head(60)
    full_test_data = pd.concat([past_60, data_testing], ignore_index=True)

    test_scaled = scaler.transform(full_test_data)

    X_test, y_test = [], []
    for i in range(60, len(test_scaled)):
        X_test.append(test_scaled[i-60: i])
        y_test.append(test_scaled[i, 0])

    X_test, y_test = np.array(X_test), np.array(y_test)

    # Modeli kullanarak tahmin yap ve görselleştir
    predict_and_visualize(model, X_test, y_test, scaler)

if __name__ == "__main__":
    main()
