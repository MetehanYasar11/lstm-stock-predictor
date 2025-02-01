import os
from models.data_loader import DataLoader
from models.indicators import Indicators
from models.roberta_model import CustomSentimentModel
import pandas as pd

def main():
    # Veri Yükleme
    loader = DataLoader(ticker="AAPL", start_date="2020-01-01", end_date="2023-01-01")
    data = loader.fetch_data()
    data = loader.preprocess_data(data)
    print(f"Veri Yüklendi: {data.head()}")

    # Teknik Göstergeler Ekleniyor
    data = Indicators.add_rsi(data)
    data = Indicators.add_macd(data)
    print(f"Teknik Göstergeler Eklendi: {data[['RSI', 'MACD', 'Signal']].head()}")

    # Model Eğitimi ve Tahmin
    model_path = "./sentiment_model"
    
    if not os.path.exists(model_path):
        print("Model Eğitimi Başlıyor...")
        # Model eğitimi için script çalıştırılabilir.
        os.system("python train_roberta.py")
        print("Model Eğitimi Tamamlandı!")
    else:
        print("Eğitilmiş Model Yüklendi.")
    
    # Modeli Yükleme ve Test Etme
    model = CustomSentimentModel(model_path)
    example_text = "Apple stock is expected to rise after strong earnings."
    predicted_class, probabilities = model.predict(example_text)

    # Tahmin Sonucu
    prediction_label = 'Positive' if predicted_class == 1 else 'Negative'
    print(f"Tahmin Sonucu: {prediction_label}")
    print(f"Olasılıklar: Pozitif={probabilities[0][1]:.2f}, Negatif={probabilities[0][0]:.2f}")

if __name__ == "__main__":
    main()
