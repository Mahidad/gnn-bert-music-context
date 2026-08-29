"""
Baseline B2: CNN on mel-spectrograms (no graph, no text).

This is the comparison that gives your Task 2 result meaning. It uses the
SAME split indices as the GNN so the two numbers are directly comparable
-- if the two models saw different test sets, the comparison would be
worthless.

Prerequisite: run `python preprocess_task2.py` first (with
build_cnn_cache: true in config.yaml).

Run with:
    python train_cnn_baseline.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from src.gnn_model import MelCNNBaseline

PROCESSED = Path("data/processed")
SPLITS = Path("data/splits")
RESULTS = Path("results")


def run_epoch(model, loader, criterion, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss, preds, targets = 0.0, [], []

    with torch.set_grad_enabled(train_mode):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds.append(logits.argmax(dim=1).cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
    }, preds, targets


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)["task2"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache = np.load(PROCESSED / "gtzan_mels.npz")
    mels, labels = cache["mels"], cache["labels"]

    with open(SPLITS / "task2_splits.json") as f:
        splits = json.load(f)

    # Normalise using TRAIN statistics only. Computing mean/std over the
    # whole dataset would leak information about the test set into
    # training -- a subtle but real form of data leakage that graders
    # specifically look for under "correct splits, no leakage".
    train_mels = mels[splits["train"]]
    mean, std = train_mels.mean(), train_mels.std() + 1e-8

    def make_loader(indices, shuffle=False):
        x = torch.from_numpy((mels[indices] - mean) / std).unsqueeze(1)
        y = torch.from_numpy(labels[indices]).long()
        return DataLoader(TensorDataset(x, y),
                          batch_size=cfg["cnn_batch_size"], shuffle=shuffle)

    train_loader = make_loader(splits["train"], shuffle=True)
    val_loader = make_loader(splits["val"])
    test_loader = make_loader(splits["test"])

    num_classes = int(labels.max()) + 1
    model = MelCNNBaseline(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["cnn_learning_rate"])
    criterion = nn.CrossEntropyLoss()

    RESULTS.mkdir(exist_ok=True)
    best_val_f1, history = -1.0, []

    for epoch in range(1, cfg["cnn_epochs"] + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics, _, _ = run_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch:3d}/{cfg['cnn_epochs']} | "
              f"train acc {train_metrics['accuracy']:.3f} | "
              f"val acc {val_metrics['accuracy']:.3f} "
              f"macroF1 {val_metrics['macro_f1']:.3f}")

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), RESULTS / "task2_best_cnn.pt")

    model.load_state_dict(torch.load(RESULTS / "task2_best_cnn.pt"))
    test_metrics, preds, targets = run_epoch(model, test_loader, criterion, device)

    print(f"\nCNN BASELINE TEST -> accuracy {test_metrics['accuracy']:.4f} | "
          f"macro-F1 {test_metrics['macro_f1']:.4f}")

    # Save a confusion matrix too, so the report can compare WHICH genres
    # each model gets right. Two models with similar overall accuracy but
    # different error patterns is the strongest evidence that fusing them
    # (Task 3) should help -- if they failed on the same tracks, combining
    # them would add nothing.
    genres = None
    graph_bundle = PROCESSED / "gtzan_graphs.pt"
    if graph_bundle.exists():
        genres = torch.load(graph_bundle, weights_only=False)["genres"]

    with open(RESULTS / "task2_cnn_metrics.json", "w") as f:
        json.dump({"history": history, "test": test_metrics,
                   "genres": genres,
                   "confusion_matrix": confusion_matrix(targets, preds).tolist()},
                  f, indent=2)

    if genres:
        cm = confusion_matrix(targets, preds)
        print("\nPer-genre test recall (CNN):")
        for i, genre in enumerate(genres):
            total = cm[i].sum()
            if total:
                print(f"  {genre:12s} {cm[i][i] / total:.2f}")


if __name__ == "__main__":
    main()
