"""BHAID loading, splitting and batching."""
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

LABEL2ID = {"human": 0, "ai": 1}


def load_bhaid(dataset_name: str, data_file: str, local_file: str = None) -> pd.DataFrame:
    """Return a DataFrame with columns Text, Label, Lang."""
    if local_file:
        df = pd.read_parquet(local_file)
    else:
        ds = load_dataset(dataset_name, data_files=data_file)
        df = ds["train"].to_pandas()

    df = df[["Text", "Label", "Lang"]].dropna().copy()
    df["Text"] = df["Text"].astype(str)
    df["Label"] = df["Label"].astype(str).str.strip().str.lower()
    df["Lang"] = df["Lang"].astype(str).str.strip()
    df = df[df["Label"].isin(LABEL2ID)]
    df = df[df["Text"].str.strip().str.len() >= 5]
    df["y"] = df["Label"].map(LABEL2ID)
    return df.reset_index(drop=True)


def make_splits(df: pd.DataFrame, seed=42, val_size=0.1, test_size=0.2,
                train_langs=None, test_langs=None):
    """70/10/20 split, stratified by (Lang, Label).

    train_langs / test_langs (lists of language codes) enable cross-lingual
    runs: train and validation use train_langs, test uses test_langs.
    """
    strat = df["Lang"] + "_" + df["Label"]
    rest, test = train_test_split(df, test_size=test_size, random_state=seed,
                                  stratify=strat)
    strat_rest = rest["Lang"] + "_" + rest["Label"]
    train, val = train_test_split(rest, test_size=val_size / (1 - test_size),
                                  random_state=seed, stratify=strat_rest)

    def keep(d, langs):
        if not langs:
            return d
        langs = {l.lower() for l in langs}
        return d[d["Lang"].str.lower().isin(langs)]

    train, val = keep(train, train_langs), keep(val, train_langs)
    test = keep(test, test_langs or train_langs)
    return {k: v.reset_index(drop=True) for k, v in
            {"train": train, "validation": val, "test": test}.items()}


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.texts = df["Text"].tolist()
        self.labels = df["y"].tolist()
        self.langs = df["Lang"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


def make_collate(tokenizer, max_len):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), truncation=True, max_length=max_len,
                        padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    return collate
