import pandas as pd
import numpy as np
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import evaluate
from tqdm import tqdm

f1_metric = evaluate.load("f1")
acc_metric = evaluate.load("accuracy")
pre_metric = evaluate.load("precision")
rec_metric = evaluate.load("recall")


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
        tokenized_data["label"] = y
        tokenized_data["idx"] = idx
        return tokenized_data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--testing_data_csv", default="./test.csv")
    parser.add_argument("--model_path", default="./metaphor/checpoint-150")

    args = parser.parse_args()
    print(args)
    df = pd.read_csv(f"{args.testing_data_csv}")

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
    df["preds"] = 0

    # %%
    x = df[["metaphorID", "text"]].astype(str)
    y = df["label_boolean"].astype(int).tolist()

    # %%
    model_checkpoint = "microsoft/deberta-v3-base"
    test_dataset = MetaphorDataset(x, y, model_checkpoint)

    # %%
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # %%
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.eval()

    for dt in tqdm(test_dataset, total=len(test_dataset)):
        with torch.inference_mode():
            ref = dt.pop("label")
            idx = dt.pop("idx")

            for k, v in dt.items():
                dt[k] = v.to(device)
            op = model(**dt)
            logits = op[0]
            pred = np.argmax(logits.detach().cpu().numpy(), -1)
            df.at[idx, "pred"] = pred
            f1_metric.add(references=ref, predictions=pred)
            acc_metric.add(references=ref, predictions=pred)
            pre_metric.add(references=ref, predictions=pred)
            rec_metric.add(references=ref, predictions=pred)

    df["pred"] = df["pred"].astype(bool)
    df.to_csv("prediction.csv", index=False)

    f1_metric = f1_metric.compute()
    acc_metric = acc_metric.compute()
    pre_metric = pre_metric.compute()
    rec_metric = rec_metric.compute()

    print(
        f1_metric,
        acc_metric,
        pre_metric,
        rec_metric,
    )
