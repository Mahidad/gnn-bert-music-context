"""
Task 2: train the GraphSAGE genre classifier on GTZAN segment graphs.

Prerequisite: run `python preprocess_task2.py` first.

Run with:
    python train_task2.py
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch_geometric.loader import DataLoader

from src.gnn_model import GraphSAGEClassifier

PROCESSED = Path("data/processed")
SPLITS = Path("data/splits")
RESULTS = Path("results")


def drop_edges(edge_index, p, device):
    """
    Randomly remove a fraction of edges each training step.

    This is dropout for graph structure. With only ~700 training graphs the
    GNN otherwise memorises each track's exact wiring; hiding a random
    slice of edges every step forces it to rely on patterns that survive
    perturbation instead. Applied at training time only.
    """
    if p <= 0:
        return edge_index
    keep = torch.rand(edge_index.shape[1], device=device) > p
    if keep.sum() == 0:
        return edge_index
    return edge_index[:, keep]


def run_epoch(model, loader, criterion, device, optimizer=None, edge_drop=0.0):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss, preds, targets = 0.0, [], []

    with torch.set_grad_enabled(train_mode):
        for batch in loader:
            batch = batch.to(device)
            edge_index = drop_edges(batch.edge_index, edge_drop, device) if train_mode \
                else batch.edge_index
            logits = model(batch.x, edge_index, batch.batch)
            loss = criterion(logits, batch.y)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            preds.append(logits.argmax(dim=1).cpu())
            targets.append(batch.y.cpu())

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

    bundle = torch.load(PROCESSED / "gtzan_graphs.pt", weights_only=False)
    graphs, genres = bundle["graphs"], bundle["genres"]
    with open(SPLITS / "task2_splits.json") as f:
        splits = json.load(f)

    train_set = [graphs[i] for i in splits["train"]]
    val_set = [graphs[i] for i in splits["val"]]
    test_set = [graphs[i] for i in splits["test"]]
    print(f"Graphs -> train {len(train_set)}, val {len(val_set)}, test {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"])

    in_dim = graphs[0].x.shape[1]
    model = GraphSAGEClassifier(
        in_dim=in_dim,
        hidden_dim=cfg["hidden_dim"],
        num_classes=len(genres),
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()   # single-label: one genre per track

    RESULTS.mkdir(exist_ok=True)
    history = []
    best_val_loss, best_epoch, epochs_without_improvement = float("inf"), 0, 0
    patience = cfg.get("early_stopping_patience", 25)
    edge_drop = cfg.get("edge_dropout", 0.0)

    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, device,
                                        optimizer, edge_drop=edge_drop)
        val_metrics, _, _ = run_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:3d}/{cfg['epochs']} | "
            f"train_loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.3f} | "
            f"val_loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.3f} "
            f"macroF1 {val_metrics['macro_f1']:.3f}"
        )

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        # Select on validation LOSS, not macro-F1.
        # On a 99-track validation set, F1 is noisy enough that its peak
        # often lands on a badly overfit epoch that guessed luckily -- the
        # previous run picked epoch 97, where val loss was near its worst.
        # Loss reflects calibration as well as correctness, so it is the
        # more stable selection signal at this dataset size.
        if val_metrics["loss"] < best_val_loss:
            best_val_loss, best_epoch = val_metrics["loss"], epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), RESULTS / "task2_best_gnn.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping: no val-loss improvement for {patience} "
                      f"epochs. Best was epoch {best_epoch} (val_loss {best_val_loss:.4f}).")
                break

    print(f"\nRestoring best checkpoint from epoch {best_epoch} "
          f"(val_loss {best_val_loss:.4f})")

    model.load_state_dict(torch.load(RESULTS / "task2_best_gnn.pt"))
    test_metrics, preds, targets = run_epoch(model, test_loader, criterion, device)

    print(f"\nTEST -> accuracy {test_metrics['accuracy']:.4f} | "
          f"macro-F1 {test_metrics['macro_f1']:.4f}")

    cm = confusion_matrix(targets, preds).tolist()
    with open(RESULTS / "task2_metrics.json", "w") as f:
        json.dump({"history": history, "test": test_metrics, "genres": genres,
                   "confusion_matrix": cm, "best_epoch": best_epoch,
                   "best_val_loss": best_val_loss}, f, indent=2)

    print("\nPer-genre test accuracy:")
    for i, genre in enumerate(genres):
        row_total = sum(cm[i])
        if row_total:
            print(f"  {genre:12s} {cm[i][i] / row_total:.2f}")

    print(f"\nSaved metrics + confusion matrix to {RESULTS / 'task2_metrics.json'}")


if __name__ == "__main__":
    main()
