from src.datasets.dataset import LongMemEvalDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def train():
    dataset = LongMemEvalDataset("oracle", "longmemeval")

    question = []
    role = []
    question_added = False

    for instance in dataset[:]:
        question_added = False
        for session in instance.sessions:
            if question_added:
                break
            for message in session.messages:
                if message.get("has_answer", False):
                    question.append(instance.question)
                    role.append(0 if message["role"] == "user" else 1)
                    question_added = True
                    break

    df = pd.DataFrame({"sentence": question, "label": role})
    df.reset_index(inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(df[["index", "sentence"]], df["label"], test_size=0.1,
                                                        random_state=42)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    train_encodings = tokenizer(
        X_train["sentence"].tolist(),
        truncation=True,
        padding="max_length"
    )
    test_encodings = tokenizer(
        X_test["sentence"].tolist(),
        truncation=True,
        padding="max_length"
    )

    train_dataset = TextDataset(train_encodings, y_train.to_list())
    test_dataset = TextDataset(test_encodings, y_test.to_list())

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./training/results",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./training/logs",
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    trainer.train()
    model.save_pretrained("./models/fine_tuned_classification_model")

    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = test_dataset.labels
    print(classification_report(true_labels, predicted_labels))


if __name__ == "__main__":
    train()
