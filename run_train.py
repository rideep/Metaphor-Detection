# %%
# !pip install transformers[torch] evaluate sentencepiece

# %%
import pandas as pd
import numpy as np
from argparse import ArgumentParser

# %%
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)
import evaluate

f1_metric = evaluate.load("f1")
acc_metric = evaluate.load("accuracy")
pre_metric = evaluate.load("precision")
rec_metric = evaluate.load("recall")


# %%
class MetaphorDataset(Dataset):
    def __init__(self, x, y, model_checkpoint="microsoft/deberta-v3-base") -> None:
        super().__init__()
        self.x = x.reset_index()
        self.y = y
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x.iloc[idx]
        y = self.y[idx]
        tokenized_data = self.tokenizer(
            x["metaphorID"],
            x["text"],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        tokenized_data["input_ids"] = tokenized_data["input_ids"].squeeze(0)
        tokenized_data["attention_mask"] = tokenized_data["attention_mask"].squeeze(0)
        tokenized_data["label"] = torch.tensor(y)
        return tokenized_data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--training_data_csv", default="train.csv")

    args = parser.parse_args()
    df = pd.read_csv(f"{args.training_data_csv}")

    map_dct = {
        0: "road",
        1: "candle",
        2: "light",
        3: "spice",
        4: "ride",
        5: "train",
        6: "boat",
    }

    df["metaphorID"] = df["metaphorID"].apply(lambda x: map_dct[x])

    # %%
    x = df[["metaphorID", "text"]].astype(str)
    y = df["label_boolean"].astype(int).tolist()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, stratify=y, test_size=0.2, random_state=7
    )

    # %%
    model_checkpoint = "microsoft/deberta-v3-base"
    train_dataset = MetaphorDataset(x_train, y_train, model_checkpoint)
    test_dataset = MetaphorDataset(x_test, y_test, model_checkpoint)

    # %%
    def compute_metric(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        f1, acc, prec, recall = (
            f1_metric.compute(predictions=predictions, references=labels),
        )
        acc_metric.compute(
            predictions=predictions, references=labels
        ), pre_metric.compute(predictions=predictions, references=labels),
        rec_metric.compute(predictions=predictions, references=labels)
        return f1 | acc | prec | recall

    # %%
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # %%
    # tokenizer.decode(next(iter(train_dataset))["input_ids"].squeeze(0))

    # %%
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2
    )

    # %%
    model.config.num_labels

    # %%
    args = TrainingArguments(
        output_dir="metaphor",
        evaluation_strategy="steps",
        logging_steps=50,
        eval_steps=50,
        num_train_epochs=10,
        save_steps=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_total_limit=1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        gradient_accumulation_steps=2,
    )

    # %%
    earlycallback = EarlyStoppingCallback(early_stopping_patience=3)

    # %%
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metric,
        callbacks=[earlycallback],
    )

    # %%
    trainer.train()
