# LSTM Stock Price Predictor

## 📌 Proje Açıklaması
Bu proje, **LSTM (Long Short-Term Memory) tabanlı bir derin öğrenme modeli kullanarak borsa fiyat tahmini yapmayı** amaçlamaktadır. Model, geçmiş fiyat verileri ve teknik göstergeleri kullanarak hisse senedi fiyatlarının gelecekteki hareketlerini tahmin etmeye çalışır.

---

## 🚀 Özellikler
- 📈 **LSTM Modeli:** Uzun vadeli bağımlılıkları öğrenerek borsa tahmini yapar.
- 📊 **Teknik İndikatörler:** Model, **MACD, RSI, Bollinger Bands** gibi teknik göstergeleri kullanır.
- 📅 **Geçmiş Fiyat Verileri:** **Yahoo Finance API** kullanılarak borsa verileri otomatik olarak çekilir.
- 🔄 **Veri Ön İşleme:** Eksik veriler temizlenir, ölçeklendirme uygulanır ve uygun formatta modele beslenir.
- 🎯 **Performans Değerlendirmesi:** RMSE ve MAPE gibi hata metrikleri kullanılarak model performansı ölçülür.

---

## 📂 Proje Dosya Yapısı

```
📂 lstm-stock-predictor/
│-- indicators.py       # Teknik indikatörleri hesaplayan yardımcı dosya
│-- lstm_train.py       # Modeli eğiten Python scripti
│-- main.py             # Eğitilmiş modeli kullanarak tahmin yapan script
│-- lstm_stock_model.h5 # Eğitilmiş model dosyası
│-- scaler.pkl          # MinMaxScaler modeli (ölçekleme için)
│-- README.txt          # Bu dosya 😃
```

---

## 📌 Kullanım Talimatları

### 🔹 1. Ortamı Hazırlayın
Aşağıdaki komut ile gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

### 🔹 2. Modeli Eğitme
Eğer modeli sıfırdan eğitmek istiyorsanız aşağıdaki komutu çalıştırın:
```bash
python lstm_train.py
```
Bu işlem, **lstm_stock_model.h5** adında eğitilmiş bir model oluşturacaktır.

### 🔹 3. Modeli Kullanarak Tahmin Yapma
Önceden eğitilmiş model ile tahmin yapmak için:
```bash
python main.py
```
Bu komut, test verileri üzerinde tahmin yaparak **gerçek fiyatlarla karşılaştırmalı bir grafik** çizecektir.

---

## 📌 Modelin Çalışma Mantığı
1️⃣ **Veri Çekme:** Yahoo Finance API kullanılarak belirlenen tarih aralığındaki borsa verileri çekilir.
2️⃣ **Özellik Mühendisliği:** **RSI, MACD, Bollinger Bantları gibi teknik göstergeler hesaplanır.**
3️⃣ **Veri Ön İşleme:** Eksik veriler temizlenir, MinMaxScaler ile ölçekleme yapılır.
4️⃣ **LSTM Modeli Eğitimi:** Model geçmiş 60 günlük veriyi kullanarak **gelecek kapanış fiyatını tahmin edecek şekilde eğitilir.**
5️⃣ **Test ve Değerlendirme:** Modelin performansı RMSE ve MAPE gibi metriklerle ölçülür.
6️⃣ **Tahmin Sonuçları Görselleştirme:** Gerçek fiyatlar ve model tahminleri grafik üzerinde gösterilir.

---

## 📌 Geliştirme Planları
✅ **Kısa Vadeli Hedefler:**
- Modelin doğruluğunu artırmak için hiperparametre optimizasyonu yapmak.
- Yeni teknik göstergeler ekleyerek modeli iyileştirmek.

🚀 **Uzun Vadeli Hedefler:**
- Modelin piyasa haberleriyle entegre çalışmasını sağlamak.
- Derin öğrenme tekniklerini geliştirerek tahmin hassasiyetini artırmak.

---

## 📌 Katkıda Bulunma
Projeye katkıda bulunmak istiyorsanız:
1️⃣ **Fork** yapın 🍴
2️⃣ **Yeni bir branch oluşturun** 🚀
3️⃣ **Değişiklikleri yapın ve commit edin** 📌
4️⃣ **Pull Request gönderin!** 🔥

---

## 📌 Lisans
Bu proje **MIT Lisansı** ile lisanslanmıştır. Açık kaynak olarak kullanabilirsiniz. ⭐

---

🚀 **Bu proje finansal tahmin modelleri geliştirmek isteyen herkes için harika bir başlangıçtır!** 😊

