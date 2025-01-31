import numpy as np

def get_technical_indicators(dataset):
    dataset['ma7'] = dataset['Price'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Price'].rolling(window=21).mean()
    dataset['26ema'] = dataset['Price'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Price'].ewm(span=12).mean()
    dataset['MACD'] = dataset['12ema'] - dataset['26ema']
    dataset['20sd'] = dataset['Price'].rolling(window=21).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)
    dataset['ema'] = dataset['Price'].ewm(com=0.5).mean()
    dataset['momentum'] = dataset['Price'].diff()
    dataset['log_momentum'] = np.log(dataset['Price']) - np.log(dataset['Price'].shift(1))
    return dataset
