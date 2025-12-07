import os
from abc import ABC, abstractmethod
from typing import Literal

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from scripts.fine_tuned_classification_model import train


class RoleClassifier(ABC):
    name: str

    @abstractmethod
    def classify(self, question: str) -> Literal["user", "assistant"]:
        pass


class UserRoleClassifier(RoleClassifier):
    name = "user_role_classifier"

    def classify(self, question: str) -> Literal["user", "assistant"]:
        return "user"


class FineTuningClassifier(RoleClassifier):
    name = "fine_tuning_classifier"

    def __init__(self, model_path: str):
        if not os.path.isdir(model_path):
            train()

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)

    def classify(self, question: str) -> Literal["user", "assistant"]:
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        predictions = outputs.logits.argmax(axis=1).item()
        return "user" if predictions == 0 else "assistant"
