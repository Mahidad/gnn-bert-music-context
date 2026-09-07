"""
Task 3 analysis: the remaining deliverables, plus the control that tells
you whether the fusion model uses its audio branch at all.

Produces four things:

1. GRAPH RELIANCE TEST. Re-runs the trained fusion model on test with the
   graph vectors randomly permuted across the batch, so every caption is
   paired with the wrong track's audio. If the score barely moves, the
   model was ignoring the graph -- which turns "fusion did not help" from
   a guess into a measurement.

2. PER-TAG COMPARISON. Average precision per tag for BERT-only vs the
   fusion model. A flat overall result can still hide tags where audio
   genuinely helps (instrumentation, audio quality) and tags where it
   cannot (lyrical themes). That breakdown is the interesting finding.

3. t-SNE of the fused representation z, coloured by tag.

4. CASE STUDIES pairing the caption tokens the model attended to against
   the clip's estimated chord sequence.

Run with:
    python analyze_task3.py                       # uses masked condition
    python analyze_task3.py --caption-mode unmasked
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer

from src.fusion_model import GNNBertFusion
from src.musiccaps_data import load_metadata, attach_text

PROCESSED = Path("data/processed")
SPLITS = Path("data/splits")
RESULTS = Path("results/task3")


def build_model(variant, cfg, node_dim, num_tags, device):
    return GNNBertFusion(
        node_dim=node_dim, num_tags=num_tags, variant=variant,
        bert_name=cfg["model_name"], gnn_hidden=cfg["gnn_hidden"],
        gnn_layers=cfg["gnn_layers"], gnn_dropout=cfg["gnn_dropout"],
        fusion_dim=cfg["fusion_dim"], attn_heads=cfg["attn_heads"],
        dropout=cfg["dropout"],
    ).to(device)


@torch.no_grad()
def run_test(model, loader, device, shuffle_graph=False, collect_z=False):
    model.eval()
    probs, labels, zs = [], [], []
    for batch in loader:
        batch = batch.to(device)
        if collect_z:
            z = model.encode(batch, shuffle_graph=shuffle_graph)
            logits = model.head(z)
            zs.append(z.float().cpu())
        else:
            logits = model(batch, shuffle_graph=shuffle_graph)
        probs.append(torch.sigmoid(logits.float()).cpu())
        labels.append(batch.y.cpu())
    out = (torch.cat(probs).numpy(), torch.cat(labels).numpy())
    return out + ((torch.cat(zs).numpy(),) if collect_z else ())


def macro_ap(probs, labels):
    present = labels.sum(axis=0) > 0
    return float(average_precision_score(labels[:, present], probs[:, present],
                                         average="macro"))


def per_tag_ap(probs, labels):
    aps = np.full(labels.shape[1], np.nan)
    for k in range(labels.shape[1]):
        if labels[:, k].sum() > 0:
            aps[k] = average_precision_score(labels[:, k], probs[:, k])
    return aps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-mode", choices=["masked", "unmasked"],
                        default="masked")
    parser.add_argument("--fusion-variant", default="cross_attention")
    parser.add_argument("--repeats", type=int, default=5,
                        help="repetitions of the shuffle test (it is random)")
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)["task3"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = args.caption_mode

    bundle = torch.load(PROCESSED / "musiccaps_graphs.pt", weights_only=False)
    graphs, splits = bundle["graphs"], bundle["splits"]
    with open(SPLITS / "task3_tag_vocab.json") as f:
        tag_vocab = json.load(f)
    idx_to_tag = {v: k for k, v in tag_vocab.items()}

    records = load_metadata()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    graphs = attach_text(graphs, records, tokenizer, tag_vocab,
                         caption_mode=mode, max_length=cfg["max_length"])

    test_set = [graphs[i] for i in splits["test"] if i < len(graphs)]
    test_loader = DataLoader(test_set, batch_size=cfg["batch_size"])
    node_dim, num_tags = graphs[0].x.shape[1], len(tag_vocab)
    print(f"Test set: {len(test_set)} clips | captions: {mode}\n")

    # ---------------- 1. graph reliance test ----------------
    fusion_ckpt = RESULTS / f"{args.fusion_variant}_{mode}.pt"
    if not fusion_ckpt.exists():
        raise FileNotFoundError(f"{fusion_ckpt} not found. Train it first.")

    fusion = build_model(args.fusion_variant, cfg, node_dim, num_tags, device)
    fusion.load_state_dict(torch.load(fusion_ckpt, weights_only=True))

    probs_f, labels, z_f = run_test(fusion, test_loader, device, collect_z=True)
    ap_intact = macro_ap(probs_f, labels)

    shuffled = []
    for _ in range(args.repeats):
        p_s, _ = run_test(fusion, test_loader, device, shuffle_graph=True)
        shuffled.append(macro_ap(p_s, labels))

    print("=" * 66)
    print("1. GRAPH RELIANCE TEST")
    print("=" * 66)
    print(f"  AUC-PR with correct audio : {ap_intact:.4f}")
    print(f"  AUC-PR with shuffled audio: {np.mean(shuffled):.4f} "
          f"(+/- {np.std(shuffled):.4f} over {args.repeats} runs)")
    drop = ap_intact - np.mean(shuffled)
    print(f"  drop when audio is wrong  : {drop:+.4f}")
    if abs(drop) < 0.01:
        print("\n  The model is essentially IGNORING the audio branch: pairing")
        print("  captions with the wrong track's audio changes nothing. This is")
        print("  direct evidence that the fusion result is a data/signal problem,")
        print("  not a tuning problem -- more epochs or a bigger GNN will not fix")
        print("  a branch the model has learned to route around.")
    else:
        print("\n  The model IS using the audio branch: corrupting it costs real")
        print("  performance. Fusion failing to beat BERT-only despite this means")
        print("  the audio signal is genuine but weaker than the text signal.")

    # ---------------- 2. per-tag comparison ----------------
    bert_ckpt = RESULTS / f"bert_only_{mode}.pt"
    if bert_ckpt.exists():
        bert = build_model("bert_only", cfg, node_dim, num_tags, device)
        bert.load_state_dict(torch.load(bert_ckpt, weights_only=True))
        probs_b, _ = run_test(bert, test_loader, device)

        ap_b, ap_f = per_tag_ap(probs_b, labels), per_tag_ap(probs_f, labels)
        prevalence = labels.mean(axis=0)
        diff = ap_f - ap_b

        order = np.argsort(-np.nan_to_num(diff, nan=-9))
        print("\n" + "=" * 66)
        print("2. PER-TAG AVERAGE PRECISION: fusion minus BERT-only")
        print("=" * 66)
        print(f"{'tag':<28}{'prev':>7}{'BERT':>8}{'fusion':>8}{'diff':>8}")
        print("-" * 66)
        print("  tags where AUDIO HELPS MOST:")
        for k in order[:8]:
            if np.isnan(diff[k]):
                continue
            print(f"  {idx_to_tag[k][:26]:<26}{prevalence[k]:>7.3f}"
                  f"{ap_b[k]:>8.3f}{ap_f[k]:>8.3f}{diff[k]:>+8.3f}")
        print("\n  tags where AUDIO HURTS MOST:")
        for k in order[-8:]:
            if np.isnan(diff[k]):
                continue
            print(f"  {idx_to_tag[k][:26]:<26}{prevalence[k]:>7.3f}"
                  f"{ap_b[k]:>8.3f}{ap_f[k]:>8.3f}{diff[k]:>+8.3f}")

        helped = int(np.nansum(diff > 0.01))
        hurt = int(np.nansum(diff < -0.01))
        print(f"\n  tags improved by >0.01 AP: {helped} / {num_tags}")
        print(f"  tags degraded by >0.01 AP: {hurt} / {num_tags}")

        with open(RESULTS / f"per_tag_comparison_{mode}.json", "w") as f:
            json.dump({idx_to_tag[k]: {"prevalence": float(prevalence[k]),
                                       "ap_bert": None if np.isnan(ap_b[k]) else float(ap_b[k]),
                                       "ap_fusion": None if np.isnan(ap_f[k]) else float(ap_f[k])}
                       for k in range(num_tags)}, f, indent=2)

    # ---------------- 3. t-SNE of z ----------------
    print("\n" + "=" * 66)
    print("3. t-SNE OF THE FUSED REPRESENTATION")
    print("=" * 66)
    n = min(len(z_f), 1500)
    z_sub, lab_sub = z_f[:n], labels[:n]
    emb = TSNE(n_components=2, perplexity=30, init="pca",
               learning_rate="auto", random_state=42).fit_transform(z_sub)

    # colour by the most frequent tags, since a 50-way legend is unreadable
    top_tags = np.argsort(-labels.sum(axis=0))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, k in zip(axes.ravel(), top_tags):
        has = lab_sub[:, k] > 0
        ax.scatter(emb[~has, 0], emb[~has, 1], s=5, c="lightgrey", label="absent")
        ax.scatter(emb[has, 0], emb[has, 1], s=6, c="crimson", label="present")
        ax.set_title(f"{idx_to_tag[k]}  (n={int(has.sum())})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    axes.ravel()[0].legend(fontsize=8, markerscale=2)
    fig.suptitle(f"t-SNE of fused representation z ({args.fusion_variant}, {mode})")
    plt.tight_layout()
    plt.savefig(RESULTS / f"tsne_z_{mode}.png", dpi=150)
    plt.close()
    print(f"  Saved {RESULTS / f'tsne_z_{mode}.png'}")
    print("  Read it as: if red points cluster, z separates that tag. If they")
    print("  are scattered through the grey, the representation does not.")

    # ---------------- 4. attention analysis + case studies ----------------
    if args.fusion_variant == "cross_attention":
        print("\n" + "=" * 66)
        print("4. ATTENTION ANALYSIS")
        print("=" * 66)

        fusion.eval()
        entropies, lengths = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                _, w = fusion(batch, return_attention=True)
                w = w.float().cpu().numpy()
                m = batch.attention_mask.cpu().numpy()
                for row, mask_row in zip(w, m):
                    valid = row[mask_row.astype(bool)]
                    valid = valid / (valid.sum() + 1e-12)
                    ent = -np.sum(valid * np.log(valid + 1e-12))
                    # normalise by log(n) so different caption lengths compare:
                    # 1.0 = attention spread evenly, 0.0 = all on one token
                    entropies.append(ent / np.log(len(valid) + 1e-12))
                    lengths.append(int(len(valid)))

        mean_ent = float(np.mean(entropies))
        mean_len = float(np.mean(lengths))
        print(f"  mean normalised attention entropy: {mean_ent:.4f}")
        print(f"  mean caption length (tokens)     : {mean_len:.1f}")
        print("  (1.0 = attention spread evenly over all tokens, 0.0 = focused")
        print("   entirely on one token)")
        print("\n  Compare this figure between the masked and unmasked runs. If")
        print("  masking raises entropy, it means that with the answer phrases")
        print("  removed the model has nothing specific left to attend to and")
        print("  spreads its attention over function words -- the extraction")
        print("  shortcut made visible in the attention itself.")

        print("\n  CAVEAT: masking shortens captions (a multi-word phrase collapses")
        print("  to a single [MASK] token), and entropy normalised by log(n) is not")
        print("  perfectly length-invariant. Report the mean length alongside the")
        print("  entropy so the comparison can be judged rather than taken on faith.")

        with open(RESULTS / f"attention_entropy_{mode}.json", "w") as f:
            json.dump({"mean_normalised_entropy": mean_ent,
                       "mean_caption_length": mean_len,
                       "n_clips": len(entropies)}, f, indent=2)

        print("\n" + "=" * 66)
        print("   CASE STUDIES: attended tokens vs chord sequence")
        print("=" * 66)

        # Pick clips that actually have several true tags. Taking the first
        # three in the split can land on a clip whose aspects fall outside
        # the top-50 vocabulary, giving an empty ground truth and a case
        # study that demonstrates nothing.
        tag_counts = labels.sum(axis=1)
        candidates = np.where(tag_counts >= 3)[0]
        if len(candidates) < 3:
            candidates = np.argsort(-tag_counts)[:3]
        chosen = candidates[:3]
        print(f"  selected clips with >=3 ground-truth tags "
              f"({len(candidates)} of {len(test_set)} qualify)\n")

        # Function words carry little meaning; showing them crowds out the
        # content words. Both lists are printed so nothing is hidden.
        FUNCTION = {"the","a","an","and","or","of","in","on","with","is","are",
                    "it","its","this","that","to","for","as","at","by","be",
                    "was","were","has","have","you","there","also","being",
                    "very","which","while",",",".","'","-","##s","##al","##ed"}

        studies = []
        subset = DataLoader([test_set[i] for i in chosen], batch_size=1)
        with torch.no_grad():
            for i, batch in zip(chosen, subset):
                batch = batch.to(device)
                logits, weights = fusion(batch, return_attention=True)
                probs = torch.sigmoid(logits.float()).squeeze(0).cpu().numpy()

                ids = batch.input_ids.squeeze(0).cpu().tolist()
                mask_row = batch.attention_mask.squeeze(0).cpu().numpy()
                w = weights.squeeze(0).float().cpu().numpy()
                tokens = tokenizer.convert_ids_to_tokens(ids)

                valid = [(tok, float(wt)) for tok, wt, m in zip(tokens, w, mask_row)
                         if m and tok not in ("[CLS]", "[SEP]", "[PAD]")]
                valid.sort(key=lambda x: -x[1])
                content = [(t, wt) for t, wt in valid if t.lower() not in FUNCTION]

                graph = test_set[i]
                chords = getattr(graph, "chords", [])
                true = [idx_to_tag[k] for k in np.where(labels[i] > 0)[0]]
                pred = [idx_to_tag[k] for k in np.argsort(-probs)[:5]]
                hits = [t for t in pred if t in true]

                # most repeated chord = the clip's harmonic centre
                chord_mode = max(set(chords), key=chords.count) if chords else "n/a"

                print(f"--- case: {graph.ytid} ---")
                print(f"  true tags        : {true}")
                print(f"  top-5 predicted  : {pred}")
                print(f"  correct in top-5 : {len(hits)}/{len(true)}  {hits}")
                print(f"  top content words: {[t for t, _ in content[:6]]}")
                print(f"  top tokens (raw) : {[t for t, _ in valid[:6]]}")
                print(f"  chords           : {' '.join(chords[:12])}")
                print(f"  dominant chord   : {chord_mode} "
                      f"({chords.count(chord_mode)}/{len(chords)} segments)\n")

                studies.append({"ytid": graph.ytid, "true_tags": true,
                                "top5_predicted": pred, "correct_in_top5": hits,
                                "top_content_words": content[:12],
                                "top_tokens_raw": valid[:12],
                                "chords": chords, "dominant_chord": chord_mode})

        with open(RESULTS / f"case_studies_{mode}.json", "w") as f:
            json.dump(studies, f, indent=2)
        print(f"  Saved {RESULTS / f'case_studies_{mode}.json'}")


if __name__ == "__main__":
    main()
