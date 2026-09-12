"""
Task 3: train the GNN-BERT fusion model.

Run one variant at a time:
    python train_task3.py --variant cross_attention
    python train_task3.py --variant cross_attention --caption-mode unmasked
    python train_task3.py --variant bert_only
    python train_task3.py --variant gnn_only
    python train_task3.py --variant concat

Or run the whole ablation grid in one go:
    python run_task3_ablation.py

Results land in results/task3/<variant>_<caption_mode>.json, so every
combination is kept separately and the ablation table can be rebuilt
without retraining.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, average_precision_score
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer

from src.fusion_model import GNNBertFusion, VARIANTS
from src.io_utils import safe_save
from src.musiccaps_data import load_metadata, attach_text

PROCESSED = Path("data/processed")
SPLITS = Path("data/splits")
RESULTS = Path("results/task3")


def collect_predictions(model, loader, criterion, device):
    """Run the model over a split and return (loss, probabilities, labels)."""
    model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y)
            total_loss += loss.item() * batch.num_graphs
            all_probs.append(torch.sigmoid(logits.float()).cpu())
            all_labels.append(batch.y.cpu())

    return (total_loss / len(loader.dataset),
            torch.cat(all_probs).numpy(),
            torch.cat(all_labels).numpy())


def score(probs, labels, loss, thresholds=0.5):
    preds = (probs >= thresholds).astype(float)

    # AUC-PR is computed only over tags that actually occur in this split.
    # A tag with zero positives has an undefined precision-recall curve, and
    # including it would silently drag the mean toward zero.
    present = labels.sum(axis=0) > 0
    auc_pr = float(average_precision_score(
        labels[:, present], probs[:, present], average="macro")) if present.any() else 0.0

    return {
        "loss": loss,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "auc_pr": auc_pr,
        "tags_evaluated": int(present.sum()),
    }


def evaluate(model, loader, criterion, device, thresholds=0.5):
    loss, probs, labels = collect_predictions(model, loader, criterion, device)
    return score(probs, labels, loss, thresholds)


def tune_thresholds(probs, labels, grid=None):
    """
    Pick a decision threshold per tag on the VALIDATION set.

    A fixed 0.5 cutoff for every tag is arbitrary: a tag present in 40% of
    clips and one present in 2% should not share a decision boundary. The
    model's job is to RANK the right tags highest (which AUC-PR measures);
    the threshold merely decides where to cut that ranked list, and cutting
    every tag in the same place discards performance already earned.

    Thresholds are chosen on validation only and then applied unchanged to
    test -- tuning them on test would be leakage.
    """
    if grid is None:
        grid = np.arange(0.05, 0.95, 0.025)

    num_tags = labels.shape[1]
    thresholds = np.full(num_tags, 0.5, dtype=np.float32)

    for k in range(num_tags):
        if labels[:, k].sum() == 0:      # tag never occurs: leave at default
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(labels[:, k], (probs[:, k] >= t).astype(float),
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[k] = best_t

    return thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="cross_attention")
    parser.add_argument("--caption-mode", choices=["masked", "unmasked"], default=None)
    parser.add_argument("--aggressive-mask", action="store_true",
                        help="also mask individual words from multi-word tags")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override config; lower = less GPU power draw")
    parser.add_argument("--max-length", type=int, default=None,
                        help="override caption token length; 64 roughly halves BERT compute")
    parser.add_argument("--freeze-bert", action="store_true",
                        help="train only the GNN/fusion/head. Cuts memory and power "
                             "substantially, and trains several times faster.")
    parser.add_argument("--amp", action="store_true",
                        help="mixed precision: less time under load, less heat")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any saved checkpoint and start fresh")
    parser.add_argument("--select-by", choices=["auc_pr", "macro_f1", "val_loss"],
                        default="auc_pr",
                        help="validation metric used to pick the best epoch. "
                             "auc_pr is threshold-free and the right default for "
                             "multi-label: BCE loss is dominated by the many absent "
                             "tags and penalises the model for growing confident, "
                             "while macro-F1 at a fixed 0.5 cutoff rewards exactly "
                             "that drift.")
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)["task3"]

    caption_mode = args.caption_mode or cfg["caption_mode"]
    epochs = args.epochs or cfg["epochs"]
    batch_size = args.batch_size or cfg["batch_size"]
    max_length = args.max_length or cfg["max_length"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | variant: {args.variant} | captions: {caption_mode}")

    bundle = torch.load(PROCESSED / "musiccaps_graphs.pt", weights_only=False)
    graphs, splits = bundle["graphs"], bundle["splits"]

    with open(SPLITS / "task3_tag_vocab.json") as f:
        tag_vocab = json.load(f)

    records = load_metadata()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    graphs = attach_text(graphs, records, tokenizer, tag_vocab,
                         caption_mode=caption_mode,
                         max_length=max_length,
                         aggressive=args.aggressive_mask)
    print(f"{len(graphs)} paired graph+caption examples, {len(tag_vocab)} tags.")

    train_set = [graphs[i] for i in splits["train"] if i < len(graphs)]
    val_set = [graphs[i] for i in splits["val"] if i < len(graphs)]
    test_set = [graphs[i] for i in splits["test"] if i < len(graphs)]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = GNNBertFusion(
        node_dim=graphs[0].x.shape[1],
        num_tags=len(tag_vocab),
        variant=args.variant,
        bert_name=cfg["model_name"],
        gnn_hidden=cfg["gnn_hidden"],
        gnn_layers=cfg["gnn_layers"],
        gnn_dropout=cfg["gnn_dropout"],
        fusion_dim=cfg["fusion_dim"],
        attn_heads=cfg["attn_heads"],
        dropout=cfg["dropout"],
    ).to(device)

    if args.freeze_bert and hasattr(model, "bert"):
        for param in model.bert.parameters():
            param.requires_grad = False
        print("BERT frozen: only the GNN, fusion block and head will train.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params/1e6:.1f}M | z dim: {model.z_dim}")

    optimizer = torch.optim.AdamW(
        model.param_groups(cfg["bert_lr"], cfg["head_lr"], cfg["weight_decay"]))
    criterion = nn.BCEWithLogitsLoss()

    RESULTS.mkdir(parents=True, exist_ok=True)
    ckpt = RESULTS / f"{args.variant}_{caption_mode}.pt"
    resume_path = RESULTS / f"{args.variant}_{caption_mode}_resume.pt"

    history, best_score, best_epoch, stale = [], -float("inf"), 0, 0
    patience = cfg["early_stopping_patience"]
    start_epoch = 1

    def selection_score(metrics):
        # negated so that "higher is better" holds for every option
        return -metrics["loss"] if args.select_by == "val_loss" else metrics[args.select_by]

    # Resume from the last completed epoch if a checkpoint is present.
    # Training this model pins the GPU for minutes at a time, and on a
    # laptop that can end in a thermal or power cutoff. Saving after every
    # epoch means a crash costs one epoch, not the whole run.
    if resume_path.exists() and not args.no_resume:
        state = torch.load(resume_path, weights_only=False, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        best_score = state["best_score"]
        best_epoch = state["best_epoch"]
        stale = state["stale"]
        start_epoch = state["epoch"] + 1
        print(f"Resuming from epoch {start_epoch} "
              f"(best so far: epoch {best_epoch}, {args.select_by} {best_score:.4f})")

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    if args.amp:
        print("Mixed precision enabled.")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss = criterion(model(batch), batch.y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * batch.num_graphs

        train_loss = running / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} | "
              f"val_loss {val_metrics['loss']:.4f} "
              f"macroF1 {val_metrics['macro_f1']:.4f} "
              f"AUC-PR {val_metrics['auc_pr']:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})

        current = selection_score(val_metrics)
        if current > best_score:
            best_score, best_epoch, stale = current, epoch, 0
            safe_save(model.state_dict(), ckpt)
        else:
            stale += 1

        safe_save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history, "best_score": best_score,
                    "best_epoch": best_epoch, "stale": stale, "epoch": epoch},
                   resume_path)

        if stale >= patience:
            print(f"\nEarly stopping at epoch {epoch}; best was {best_epoch}.")
            break

    model.load_state_dict(torch.load(ckpt, weights_only=True))

    # Thresholds are fitted on VALIDATION and then applied unchanged to
    # test. Fitting them on test would be leakage.
    val_loss_f, val_probs, val_labels = collect_predictions(
        model, val_loader, criterion, device)
    thresholds = tune_thresholds(val_probs, val_labels)

    test_loss_f, test_probs, test_labels = collect_predictions(
        model, test_loader, criterion, device)

    test_metrics = score(test_probs, test_labels, test_loss_f, 0.5)
    test_tuned = score(test_probs, test_labels, test_loss_f, thresholds)

    print(f"\nTEST [{args.variant} / {caption_mode}]")
    print(f"  fixed 0.5 threshold : macro-F1 {test_metrics['macro_f1']:.4f} | "
          f"micro-F1 {test_metrics['micro_f1']:.4f} | "
          f"AUC-PR {test_metrics['auc_pr']:.4f}")
    print(f"  tuned per-tag       : macro-F1 {test_tuned['macro_f1']:.4f} | "
          f"micro-F1 {test_tuned['micro_f1']:.4f}")
    print(f"  (AUC-PR is threshold-free, so it is identical for both rows)")

    out = RESULTS / f"{args.variant}_{caption_mode}.json"
    with open(out, "w") as f:
        json.dump({"variant": args.variant, "caption_mode": caption_mode,
                   "aggressive_mask": args.aggressive_mask,
                   "params": n_params, "best_epoch": best_epoch,
                   "select_by": args.select_by,
                   "history": history, "test": test_metrics,
                   "test_tuned": test_tuned,
                   "thresholds": thresholds.tolist(),
                   "batch_size": batch_size, "max_length": max_length,
                   "frozen_bert": args.freeze_bert, "amp": args.amp}, f, indent=2)
    print(f"Saved {out}")

    # Training finished cleanly -- the resume file is no longer needed.
    if resume_path.exists():
        resume_path.unlink()


if __name__ == "__main__":
    main()
