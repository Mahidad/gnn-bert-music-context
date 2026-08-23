"""
Task 1 data utilities.

Loads MusicCaps (captions + aspect tags only -- no audio download needed),
builds a fixed multi-label tag vocabulary from the most common aspect
tags, and exposes a PyTorch Dataset that tokenizes captions for BERT.
"""

import ast
import random
from collections import Counter

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def load_musiccaps_dataframe():
    """
    Downloads MusicCaps from Hugging Face and returns it as a pandas
    DataFrame. Only `caption` and `aspect_list` are used for Task 1, so
    this is a small, fast download -- the actual audio clips are NOT
    fetched here (Task 1 doesn't need them; Tasks 3/4 will).
    """
    ds = load_dataset("google/MusicCaps", split="train")
    return ds.to_pandas()


def parse_aspect_list(raw):
    """
    aspect_list is stored as a string that *looks* like a Python list,
    e.g. "['pop', 'mellow piano melody', 'female vocal']".
    ast.literal_eval turns that string back into a real Python list
    without the security risk of eval().
    """
    if isinstance(raw, list):
        return raw
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []


def build_tag_vocab(df, top_k=50):
    """
    Counts every aspect tag across the dataset and keeps the `top_k`
    most frequent ones as our fixed label vocabulary.

    Why top-k instead of all tags? aspect_list has a long tail of
    highly specific phrases that appear once or twice. Training a
    classifier head for a tag with 1-2 positive examples gives it
    almost nothing to learn from and makes the F1 numbers noisy.
    Keeping the most common ~50 tags means every label has enough
    examples for the model to actually learn a pattern.
    """
    counter = Counter()
    for raw in df["aspect_list"]:
        counter.update(parse_aspect_list(raw))
    top_tags = [tag for tag, _ in counter.most_common(top_k)]
    return {tag: i for i, tag in enumerate(top_tags)}


def make_multi_hot(aspects, tag_to_idx):
    vec = torch.zeros(len(tag_to_idx), dtype=torch.float32)
    for tag in aspects:
        if tag in tag_to_idx:
            vec[tag_to_idx[tag]] = 1.0
    return vec


def split_dataframe(df, seed=42, val_frac=0.1, test_frac=0.1):
    """
    MusicCaps doesn't ship an official train/val/test split for this
    tagging setup, so we make a reproducible random split ourselves.
    Fixing the seed means every run (yours, and anyone re-running your
    code) gets the identical split -- important for fairly comparing
    this BERT-only baseline against the GNN-only and fusion models
    you'll train in later tasks.
    """
    indices = list(range(len(df)))
    random.Random(seed).shuffle(indices)

    n_val = int(len(indices) * val_frac)
    n_test = int(len(indices) * test_frac)

    val_idx = indices[:n_val]
    test_idx = indices[n_val:n_val + n_test]
    train_idx = indices[n_val + n_test:]

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


class MusicCapsTagDataset(Dataset):
    """
    Wraps a MusicCaps dataframe slice so PyTorch's DataLoader can batch
    it: each item is (tokenized caption, multi-hot tag vector).
    """

    def __init__(self, df, tokenizer, tag_to_idx, max_length=128):
        self.captions = df["caption"].tolist()
        self.aspects = [parse_aspect_list(a) for a in df["aspect_list"]]
        self.tokenizer = tokenizer
        self.tag_to_idx = tag_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.captions[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": make_multi_hot(self.aspects[idx], self.tag_to_idx),
        }
