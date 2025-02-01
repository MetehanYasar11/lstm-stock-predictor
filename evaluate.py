import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import yfinance as yf
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from indicators import get_technical_indicators

# ✅ Test Verisini Çekme
data = yf.download('^NSEBANK', start='2007-10-16', end='2021-04-19')
data = data[['Close']]
data.columns = ['Price']

# ✅ Teknik Göstergeleri Ekle
data = get_technical_indicators(data)
data.dropna(inplace=True)

# ✅ Train ve Test Setlerini Ayır
train_data = data[data.index < '2019-01-31'].copy()
test_data = data[data.index >= '2019-01-31'].copy()

# ✅ Train Verisinin Son 60 Gününü Kullanarak Ölçekleme Yap
past_60 = train_data.tail(60)  
dt = pd.concat([past_60, test_data], ignore_index=True)  
scaler_test = MinMaxScaler()
inputs = scaler_test.fit_transform(dt)

# ✅ LSTM İçin Test Verisi Hazırlama
X_test, y_test = [], []
sequence_length = 60

for i in range(sequence_length, inputs.shape[0]):
    X_test.append(inputs[i-sequence_length:i])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# ✅ MODELS KLASÖRÜNDEKİ TÜM .h5 MODELLERİ BUL
model_dir = "models/"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".h5")]

# ✅ HER MODELİ EVALUATE ET
results = []

for model_file in model_files:
    print(f"Evaluating model: {model_file}")
    
    # 🔥 Modeli Yükle
    model_path = os.path.join(model_dir, model_file)
    model = tf.keras.models.load_model(model_path)

    # ✅ Model ile Tahmin Yap
    y_pred = model.predict(X_test)

    # ✅ Ölçekleme Geri Alma
    y_pred = scaler_test.inverse_transform(np.hstack([y_pred, np.zeros((y_pred.shape[0], inputs.shape[1]-1))]))[:, 0]
    y_test_real = scaler_test.inverse_transform(np.hstack([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], inputs.shape[1]-1))]))[:, 0]

    # ✅ RMSE ve MAPE Hesaplama
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
    mape = mean_absolute_percentage_error(y_test_real, y_pred)

    print(f"Model: {model_file} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

    # ✅ Sonuçları Listeye Ekle
    results.append({"Model": model_file, "RMSE": rmse, "MAPE": mape})

# ✅ Sonuçları CSV Olarak Kaydet
df_results = pd.DataFrame(results)
df_results.to_csv("model_evaluation_results.csv", index=False)
print("\n🎯 Model değerlendirme tamamlandı! Sonuçlar 'model_evaluation_results.csv' olarak kaydedildi.")
