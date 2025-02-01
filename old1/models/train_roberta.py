from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

class FinanceSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 1. Veri Yükleme ve Hazırlık
df = pd.read_csv("data/sentiment_data.csv")  # Haber metni ve etiketler CSV dosyası
df['label'] = df['Label'].map({'Positive': 1, 'Negative': 0})  # Metin etiketlerini sayıya dönüştür

texts = df['Text'].tolist()
labels = df['label'].tolist()

# Veriyi eğitim ve test setlerine ayır
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 2. Tokenizer ve Dataset
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)

train_dataset = FinanceSentimentDataset(train_texts, train_labels, tokenizer, max_len=128)
val_dataset = FinanceSentimentDataset(val_texts, val_labels, tokenizer, max_len=128)

# 3. Model ve Eğitim Ayarları
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 4. Modeli Eğit
trainer.train()

# 5. Modeli Kaydet
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")
