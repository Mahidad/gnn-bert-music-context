"""
Shared split + normalisation helpers for Task 2.

Kept in one place so preprocess_task2.py and rebuild_graphs.py cannot
drift apart -- if they built splits differently, every model comparison
in the report would be invalid.
"""

import json
import random
from pathlib import Path

import numpy as np

SPLITS = Path("data/splits")


def stratified_split(labels, seed=42, val_frac=0.1, test_frac=0.2):
    """
    Stratified = each genre keeps the same proportion in train/val/test.
    With only 100 tracks per genre an unstratified split can easily put
    12 blues tracks in test and 4 of something else, which moves accuracy
    by several points on its own.
    """
    by_label = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(int(lab), []).append(i)

    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label in sorted(by_label):
        idxs = by_label[label][:]
        rng.shuffle(idxs)
        n_val = int(len(idxs) * val_frac)
        n_test = int(len(idxs) * test_frac)
        val_idx += idxs[:n_val]
        test_idx += idxs[n_val:n_val + n_test]
        train_idx += idxs[n_val + n_test:]

    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


def save_splits(train_idx, val_idx, test_idx):
    SPLITS.mkdir(parents=True, exist_ok=True)
    with open(SPLITS / "task2_splits.json", "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)


def load_splits():
    path = SPLITS / "task2_splits.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def train_feature_stats(records, train_idx):
    """
    Dataset-wide mean/std per feature dimension, computed from TRAINING
    tracks only. Including val/test tracks here would leak information
    about the evaluation data into the model's inputs -- exactly what the
    rubric's "no leakage" criterion is checking for.
    """
    stacked = np.concatenate([records[i]["features"] for i in train_idx], axis=0)
    return (stacked.mean(axis=0, keepdims=True).astype(np.float32),
            stacked.std(axis=0, keepdims=True).astype(np.float32))
