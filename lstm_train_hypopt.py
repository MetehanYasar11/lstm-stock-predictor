import numpy as np
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import joblib
from indicators import get_technical_indicators
import keras_tuner as kt  # Hiperparametre optimizasyonu için Keras Tuner kullanıyoruz

# Veri Çekme (yfinance kullanarak)
data = yf.download('^NSEBANK', start='2007-10-16', end='2021-04-19')
data = data[['Close']]
data.columns = ['Price']

# Teknik İndikatörleri Hesaplama
data = get_technical_indicators(data)
data.dropna(inplace=True)

# Eğitim ve Test Setlerini Ayırma
train_data = data[data.index < '2019-01-31'].copy()
test_data = data[data.index >= '2019-01-31'].copy()

# MinMax Ölçekleme
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
joblib.dump(scaler, 'scaler.pkl')

# LSTM İçin Eğitim Verisi Hazırlama
X_train, y_train = [], []
sequence_length = 60

for i in range(sequence_length, train_scaled.shape[0]):
    X_train.append(train_scaled[i-sequence_length:i])
    y_train.append(train_scaled[i, 0])  # Hedef değişken (Price)

X_train, y_train = np.array(X_train), np.array(y_train)

# Modeli Optimize Etmek İçin Fonksiyon
def build_model(hp):
    model = Sequential()
    
    # İlk LSTM katmanı
    model.add(LSTM(
        units=hp.Int('units_1', min_value=32, max_value=128, step=32),
        activation='relu',
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

    # Orta LSTM katmanları
    for i in range(hp.Int('num_layers', 1, 3)):  # 1 ila 3 ekstra LSTM katmanı arasında değiştirilebilir
        model.add(LSTM(
            units=hp.Int(f'units_{i+2}', min_value=32, max_value=128, step=32),
            activation='relu',
            return_sequences=True
        ))
        model.add(Dropout(hp.Float(f'dropout_{i+2}', min_value=0.1, max_value=0.5, step=0.1)))

    # Son LSTM katmanı
    model.add(LSTM(
        units=hp.Int('units_final', min_value=64, max_value=256, step=64),
        activation='relu'
    ))
    model.add(Dropout(hp.Float('dropout_final', min_value=0.1, max_value=0.5, step=0.1)))

    # Çıkış katmanı
    model.add(Dense(1))

    # Modeli derleme
    model.compile(
        optimizer=hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd']),
        loss='mean_squared_error'
    )
    
    return model

# Hiperparametre Tuner Tanımlama
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,  # 10 farklı model yapılandırmasını deneyecek
    executions_per_trial=1,
    directory='tuner_results',
    project_name='lstm_stock_hyperopt'
)

# Hiperparametre optimizasyonunu başlatma
tuner.search(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=1)

# En iyi modeli seçme
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# En iyi modeli eğitme
best_model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)

# Modeli "models" klasörüne kaydetme (_hypopt ekleyerek)
if not os.path.exists('models'):
    os.makedirs('models')

best_model.save('models/lstm_stock_model_hypopt.h5')
