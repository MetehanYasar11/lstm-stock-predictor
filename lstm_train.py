import numpy as np
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Conv1D, Flatten
import joblib
import tensorflow as tf
from indicators import get_technical_indicators

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

# **📌 Model Seçim Menüsü**
def select_model():
    print("\n🚀 Hangi modeli eğitmek istiyorsunuz?")
    print("1️⃣ LSTM")
    print("2️⃣ BiLSTM + CNN (Önerilen)")
    print("3️⃣ Transformer (Deneysel)")
    
    choice = input("\nSeçiminizi yapın (1/2/3): ")

    if choice == "1":
        return create_lstm_model()
    elif choice == "2":
        return create_bilstm_cnn_model()
    elif choice == "3":
        return create_transformer_model()
    else:
        print("⚠️ Geçersiz seçim! Varsayılan olarak LSTM modeli seçildi.")
        return create_lstm_model()

# **📌 LSTM Modeli**
def create_lstm_model():
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
    return model

# **📌 BiLSTM + CNN Modeli (Önerilen)**
def create_bilstm_cnn_model():
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Bidirectional(LSTM(60, return_sequences=True, activation='relu')),
        Dropout(0.3),
        Bidirectional(LSTM(80, return_sequences=True, activation='relu')),
        Dropout(0.4),
        Flatten(),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# **📌 Transformer Modeli (Deneysel)**
def create_transformer_model():
    inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=64)(inputs, inputs)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = Flatten()(x)
    outputs = Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# **📌 Modeli Eğitme**
model = select_model()
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)

# **📌 Modeli Kaydetme**
if not os.path.exists('models'):
    os.makedirs('models')

model_type = model.name if hasattr(model, 'name') else "custom_model"
model_path = f"models/{model_type}.h5"
model.save(model_path)

print(f"✅ Model kaydedildi: {model_path}")
