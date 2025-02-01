import yfinance as yf
import pandas as pd

class DataLoader:
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance."""
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return data

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values and normalize the data."""
        data = data.fillna(method='ffill')  # Eksik verileri bir önceki değerle doldur
        data = data.dropna()  # Hala eksik olanları kaldır
        return data

