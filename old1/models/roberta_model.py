from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

class CustomSentimentModel:
    def __init__(self, model_path: str):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)

    def predict(self, text: str):
        tokens = self.tokenizer(text, max_length=128, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**tokens)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        return predicted_class, probabilities
