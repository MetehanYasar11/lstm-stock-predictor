import numpy as np

def get_technical_indicators(dataset):
    """
    Teknik göstergeleri hesaplar:
    - Hareketli Ortalamalar (7 & 21 günlük)
    - MACD
    - Bollinger Bandları
    - Üstel Hareketli Ortalama (EMA)
    - Momentum
    """
    dataset['ma7'] = dataset['Price'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Price'].rolling(window=21).mean()
    
    # MACD Hesaplama
    dataset['26ema'] = dataset['Price'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Price'].ewm(span=12).mean()
    dataset['MACD'] = dataset['12ema'] - dataset['26ema']

    # Bollinger Bantları
    dataset['20sd'] = dataset['Price'].rolling(window=21).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)
    
    # EMA Hesaplama
    dataset['ema'] = dataset['Price'].ewm(com=0.5).mean()
    
    # Momentum Hesaplama
    dataset['momentum'] = dataset['Price'] - 1
    dataset['log_momentum'] = np.log(dataset['momentum'])
    
    return dataset.dropna()  # NaN değerleri temizle
