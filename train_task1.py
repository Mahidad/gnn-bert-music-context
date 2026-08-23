"""
Entry point for Task 1: trains and evaluates the BERT tag classifier on
MusicCaps captions -> top-50 aspect tags.

Run with:
    python train_task1.py
(edit config.yaml first if you want to change batch size, epochs, etc.)
"""

import json
import os

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data_utils import (
    load_musiccaps_dataframe,
    build_tag_vocab,
    split_dataframe,
    MusicCapsTagDataset,
)
from src.bert_baseline import BertTagClassifier, train_epoch, evaluate


def print_example_predictions(model, test_ds, test_df, tag_to_idx, device, n=5):
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    model.eval()
    print("\nExample predictions (test set):")
    with torch.no_grad():
        for i in range(min(n, len(test_ds))):
            item = test_ds[i]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).squeeze(0)

            pred_tags = [idx_to_tag[j] for j, p in enumerate(probs) if p >= 0.5]
            true_tags = [idx_to_tag[j] for j, v in enumerate(item["labels"]) if v == 1.0]

            caption_preview = test_df.iloc[i]["caption"][:150]
            print(f"\nCaption: {caption_preview}...")
            print(f"  True tags:      {true_tags}")
            print(f"  Predicted tags: {pred_tags}")


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)["task1"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading MusicCaps...")
    df = load_musiccaps_dataframe()
    print(f"Loaded {len(df)} caption/tag examples.")

    tag_to_idx = build_tag_vocab(df, top_k=cfg["num_tags"])
    os.makedirs("data/splits", exist_ok=True)
    with open("data/splits/tag_vocab.json", "w") as f:
        json.dump(tag_to_idx, f, indent=2)
    print(f"Built tag vocabulary with {len(tag_to_idx)} tags.")

    train_df, val_df, test_df = split_dataframe(df, seed=cfg["seed"])
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    train_ds = MusicCapsTagDataset(train_df, tokenizer, tag_to_idx, cfg["max_length"])
    val_ds = MusicCapsTagDataset(val_df, tokenizer, tag_to_idx, cfg["max_length"])
    test_ds = MusicCapsTagDataset(test_df, tokenizer, tag_to_idx, cfg["max_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"])

    model = BertTagClassifier(
        model_name=cfg["model_name"],
        num_tags=len(tag_to_idx),
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()

    history = []
    best_val_f1 = -1.0
    os.makedirs("results", exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{cfg['epochs']} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"val_macro_f1: {val_metrics['macro_f1']:.4f} | "
            f"val_micro_f1: {val_metrics['micro_f1']:.4f}"
        )

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), "results/task1_best_model.pt")

    # Final evaluation on the held-out test set, using the best checkpoint
    # (not just whatever the model looked like after the last epoch).
    model.load_state_dict(torch.load("results/task1_best_model.pt"))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\nTest set -> macro_f1: {test_metrics['macro_f1']:.4f} | micro_f1: {test_metrics['micro_f1']:.4f}")

    with open("results/task1_metrics.json", "w") as f:
        json.dump({"history": history, "test": test_metrics}, f, indent=2)

    print_example_predictions(model, test_ds, test_df, tag_to_idx, device)

    print("\nSaved best model to results/task1_best_model.pt")
    print("Saved metrics + F1-vs-epoch history to results/task1_metrics.json")


if __name__ == "__main__":
    main()
