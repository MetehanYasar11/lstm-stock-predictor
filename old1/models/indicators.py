import talib
import pandas as pd

class Indicators:
    @staticmethod
    def add_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI to the dataframe."""
        # Pandas sütununu numpy array'e dönüştür ve 1D array'e sıkıştır
        close_prices = data['Close'].to_numpy().squeeze()
        print(f"Close sütununun boyutları: {close_prices.shape}")  # Hata kontrolü için
        data['RSI'] = talib.RSI(close_prices, timeperiod=period)
        return data

    @staticmethod
    def add_macd(data: pd.DataFrame) -> pd.DataFrame:
        """Add MACD to the dataframe."""
        # MACD hesaplama
        close_prices = data['Close'].to_numpy().squeeze()
        macd, macdsignal, macdhist = talib.MACD(close_prices)
        data['MACD'] = macd
        data['Signal'] = macdsignal
        return data
