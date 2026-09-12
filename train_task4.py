"""
Task 4: train the contrastive dual-encoder and evaluate retrieval.

Produces every Task 4 deliverable:
  - dual-encoder GNN-BERT trained with InfoNCE
  - retrieval table (R@1/5/10, median rank, both directions)
  - 10 qualitative retrieval examples
  - zero-shot tag prediction, compared against the Task 3 supervised model

Run with:
    python train_task4.py --amp --batch-size 32
    python train_task4.py --amp --freeze-bert --batch-size 64   # lighter
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer

from src.contrastive import ContrastiveGNNBert, info_nce_loss, retrieval_metrics
from src.io_utils import safe_save
from src.musiccaps_data import load_metadata, attach_text, parse_aspects

PROCESSED = Path("data/processed")
SPLITS = Path("data/splits")
RESULTS = Path("results/task4")


@torch.no_grad()
def embed_split(model, loader, device):
    model.eval()
    gs, ts = [], []
    for batch in loader:
        batch = batch.to(device)
        g = model.encode_graph(batch)
        t = model.encode_text(batch.input_ids, batch.attention_mask)
        gs.append(g.float().cpu())
        ts.append(t.float().cpu())
    return torch.cat(gs), torch.cat(ts)


@torch.no_grad()
def zero_shot_tags(model, graph_emb, tag_names, tokenizer, device, max_length=32):
    """
    Zero-shot tagging: embed each tag as a short sentence, then rank tags
    for each clip by similarity to its audio embedding.

    The model was never trained to predict tags -- only to match clips with
    captions. If this scores above chance, the shared space has learned
    something about musical meaning rather than just memorising pairs.
    """
    prompts = [f"This music clip sounds {tag}." for tag in tag_names]
    enc = tokenizer(prompts, padding="max_length", truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
    tag_emb = model.encode_text(enc["input_ids"], enc["attention_mask"]).float().cpu()
    return graph_emb @ tag_emb.T          # (num_clips, num_tags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--freeze-bert", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    with open("config.yaml") as f:
        full_cfg = yaml.safe_load(f)
    cfg = full_cfg["task4"]
    t3 = full_cfg["task3"]

    epochs = args.epochs or cfg["epochs"]
    batch_size = args.batch_size or cfg["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | batch_size: {batch_size}")

    bundle = torch.load(PROCESSED / "musiccaps_graphs.pt", weights_only=False)
    graphs, splits = bundle["graphs"], bundle["splits"]
    with open(SPLITS / "task3_tag_vocab.json") as f:
        tag_vocab = json.load(f)

    records = load_metadata()
    tokenizer = AutoTokenizer.from_pretrained(t3["model_name"])

    # Retrieval uses the REAL captions. Masking exists to stop the text
    # branch copying tag phrases when both branches predict the same
    # labels; here the caption is the query a user would actually type, so
    # masking it would be measuring the wrong task.
    graphs = attach_text(graphs, records, tokenizer, tag_vocab,
                         caption_mode="unmasked", max_length=t3["max_length"])
    print(f"{len(graphs)} paired graph+caption examples.")

    train_set = [graphs[i] for i in splits["train"] if i < len(graphs)]
    val_set = [graphs[i] for i in splits["val"] if i < len(graphs)]
    test_set = [graphs[i] for i in splits["test"] if i < len(graphs)]

    # drop_last matters here: InfoNCE uses the other items in the batch as
    # negatives, so a final batch of size 1 has no negatives at all and
    # produces a meaningless (zero) loss.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = ContrastiveGNNBert(
        node_dim=graphs[0].x.shape[1],
        embed_dim=cfg["embed_dim"],
        bert_name=t3["model_name"],
        gnn_hidden=cfg["gnn_hidden"],
        gnn_layers=cfg["gnn_layers"],
        gnn_dropout=cfg["gnn_dropout"],
        dropout=cfg["dropout"],
        init_temperature=cfg["init_temperature"],
    ).to(device)

    if args.freeze_bert:
        for p in model.bert.parameters():
            p.requires_grad = False
        print("BERT frozen.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        model.param_groups(cfg["bert_lr"], cfg["head_lr"], cfg["weight_decay"]))

    RESULTS.mkdir(parents=True, exist_ok=True)
    ckpt = RESULTS / "contrastive_best.pt"
    resume_path = RESULTS / "contrastive_resume.pt"

    history, best_score, best_epoch, stale = [], -float("inf"), 0, 0
    patience = cfg["early_stopping_patience"]
    start_epoch = 1

    if resume_path.exists() and not args.no_resume:
        state = torch.load(resume_path, weights_only=False, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, best_score = state["history"], state["best_score"]
        best_epoch, stale = state["best_epoch"], state["stale"]
        start_epoch = state["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                g, t, scale = model(batch)
                loss = info_nce_loss(g.float(), t.float(), scale.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * batch.num_graphs

        train_loss = running / len(train_loader.dataset)

        g_val, t_val = embed_split(model, val_loader, device)
        val_metrics = retrieval_metrics(g_val, t_val)
        # select on the mean of the two R@10 figures: a single direction can
        # improve while the other degrades, and retrieval is meant to work
        # both ways
        selection = (val_metrics["audio_to_text_R@10"] +
                     val_metrics["text_to_audio_R@10"]) / 2

        print(f"Epoch {epoch:3d}/{epochs} | loss {train_loss:.4f} | "
              f"val A->T R@1 {val_metrics['audio_to_text_R@1']:.3f} "
              f"R@10 {val_metrics['audio_to_text_R@10']:.3f} | "
              f"T->A R@10 {val_metrics['text_to_audio_R@10']:.3f} | "
              f"temp {1.0/model.log_temperature.exp().item():.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})

        if selection > best_score:
            best_score, best_epoch, stale = selection, epoch, 0
            safe_save(model.state_dict(), ckpt)
        else:
            stale += 1

        safe_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "history": history, "best_score": best_score,
                    "best_epoch": best_epoch, "stale": stale, "epoch": epoch},
                   resume_path)

        if stale >= patience:
            print(f"\nEarly stopping at epoch {epoch}; best was {best_epoch}.")
            break

    # ---------------- evaluation ----------------
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    g_test, t_test = embed_split(model, test_loader, device)
    test_metrics = retrieval_metrics(g_test, t_test)

    n = test_metrics["n_candidates"]
    print("\n" + "=" * 62)
    print(f"TASK 4 RETRIEVAL ({n} candidates)")
    print("=" * 62)
    print(f"{'direction':<18}{'R@1':>9}{'R@5':>9}{'R@10':>9}{'median rank':>14}")
    print("-" * 62)
    for d in ("audio_to_text", "text_to_audio"):
        print(f"{d:<18}{test_metrics[d + '_R@1']:>9.4f}"
              f"{test_metrics[d + '_R@5']:>9.4f}{test_metrics[d + '_R@10']:>9.4f}"
              f"{test_metrics[d + '_median_rank']:>14.0f}")
    print("-" * 62)
    print(f"random baseline    {1/n:>9.4f}{5/n:>9.4f}{10/n:>9.4f}{n/2:>14.0f}")

    # ---------------- 10 qualitative examples ----------------
    sim = (g_test @ t_test.T).numpy()
    by_id = {r["ytid"]: r for r in records}
    test_ids = [g.ytid for g in test_set]

    examples = []
    for i in range(min(10, len(test_ids))):
        top3 = np.argsort(-sim[i])[:3]
        examples.append({
            "query_caption": by_id[test_ids[i]]["caption"][:200],
            "correct_clip": test_ids[i],
            "retrieved_top3": [test_ids[j] for j in top3],
            "rank_of_correct": int(np.where(np.argsort(-sim[i]) == i)[0][0]) + 1,
            "hit_at_1": bool(top3[0] == i),
        })

    print(f"\n10 qualitative examples -> {RESULTS / 'retrieval_examples.json'}")
    hits = sum(e["hit_at_1"] for e in examples)
    print(f"  {hits}/10 correct at rank 1; "
          f"ranks: {[e['rank_of_correct'] for e in examples]}")

    with open(RESULTS / "retrieval_examples.json", "w") as f:
        json.dump(examples, f, indent=2)

    # ---------------- zero-shot tagging ----------------
    tag_names = [t for t, _ in sorted(tag_vocab.items(), key=lambda kv: kv[1])]
    scores = zero_shot_tags(model, g_test, tag_names, tokenizer, device).numpy()

    labels = np.stack([
        np.array([1.0 if t in parse_aspects(by_id[y]["aspect_list"]) else 0.0
                  for t in tag_names])
        for y in test_ids
    ])
    present = labels.sum(axis=0) > 0
    zs_ap = float(average_precision_score(labels[:, present], scores[:, present],
                                          average="macro"))
    chance = float(labels[:, present].mean())

    print("\n" + "=" * 62)
    print("ZERO-SHOT TAGGING (never trained on tags)")
    print("=" * 62)
    print(f"  macro AUC-PR : {zs_ap:.4f}")
    print(f"  chance level : {chance:.4f}  (mean tag prevalence)")

    t3_path = Path("results/task3/bert_only_unmasked.json")
    if t3_path.exists():
        with open(t3_path) as f:
            sup = json.load(f)["test"]["auc_pr"]
        print(f"  Task 3 supervised BERT-only: {sup:.4f}")
        print(f"\n  Zero-shot reaches {zs_ap/sup:.0%} of the supervised model while")
        print("  never seeing a tag label. Note the supervised model reads the")
        print("  caption at test time and the zero-shot model sees only audio,")
        print("  so this is not a like-for-like comparison -- it measures how")
        print("  much tag information the audio embedding alone carries.")

    with open(RESULTS / "task4_metrics.json", "w") as f:
        json.dump({"history": history, "test_retrieval": test_metrics,
                   "best_epoch": best_epoch, "zero_shot_auc_pr": zs_ap,
                   "zero_shot_chance": chance, "batch_size": batch_size,
                   "frozen_bert": args.freeze_bert}, f, indent=2)
    print(f"\nSaved {RESULTS / 'task4_metrics.json'}")

    if resume_path.exists():
        resume_path.unlink()


if __name__ == "__main__":
    main()
