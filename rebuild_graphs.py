"""
Rebuild the graph cache at a different similarity_tau in seconds.

Audio feature extraction is the slow part (~20 min for GTZAN); graph
construction is fast. This reuses the cached raw features and redoes only
the graph construction, so a tau ablation costs seconds per setting.

It reuses the existing split file, so every tau setting is evaluated on
the identical test set, and recomputes the normalisation statistics from
the training split only.

Run with:
    python rebuild_graphs.py --tau 0.6
    python rebuild_graphs.py --tau 0.6 --max-pairs 40
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.graph_builder import build_segment_graph, graph_summary
from src.split_utils import (stratified_split, save_splits, load_splits,
                             train_feature_stats)

PROCESSED = Path("data/processed")
SAMPLES = PROCESSED / "graph_samples"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, required=True,
                        help="cosine similarity threshold for similarity edges")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="cap on similarity PAIRS per graph (default: from config)")
    args = parser.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)["task2"]

    max_pairs = args.max_pairs if args.max_pairs is not None else cfg["max_similarity_edges"]

    path = PROCESSED / "gtzan_features.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Re-run `python preprocess_task2.py` once -- the updated version\n"
            "caches raw features so graphs can be rebuilt without re-decoding audio."
        )

    bundle = torch.load(path, weights_only=False)
    records, genres = bundle["features"], bundle["genres"]
    print(f"Rebuilding {len(records)} graphs at tau={args.tau}, max_pairs={max_pairs}")

    splits = load_splits()
    if splits is None:
        labels = [r["label"] for r in records]
        tr, va, te = stratified_split(labels, seed=cfg["seed"],
                                      val_frac=cfg["val_frac"], test_frac=cfg["test_frac"])
        save_splits(tr, va, te)
        splits = {"train": tr, "val": va, "test": te}
        print("Wrote new split file.")
    else:
        print("Kept existing split file (same test set across tau settings).")

    g_mean, g_std = train_feature_stats(records, splits["train"])

    graphs = []
    for rec in records:
        graphs.append(build_segment_graph(
            rec["features"],
            label=rec["label"],
            tau=args.tau,
            max_similarity_edges=max_pairs,
            chords=rec["chords"],
            track_id=rec["track_id"],
            global_mean=g_mean,
            global_std=g_std,
        ))

    torch.save({"graphs": graphs, "genres": genres,
                "genre_to_idx": bundle["genre_to_idx"]},
               PROCESSED / "gtzan_graphs.pt")

    SAMPLES.mkdir(parents=True, exist_ok=True)
    for old in SAMPLES.glob("*.pt"):
        old.unlink()
    for i, g in enumerate(graphs[:20]):
        torch.save(g, SAMPLES / f"graph_{i:02d}_{g.track_id}.pt")

    stats = [graph_summary(g) for g in graphs]
    sims = np.array([s["similarity_edges"] for s in stats])
    print(f"\nnode feature dim: {graphs[0].x.shape[1]} (absolute + relative)")
    print(f"Similarity edges per graph (both directions):")
    print(f"  mean {sims.mean():.1f}  median {np.median(sims):.1f}  "
          f"min {sims.min()}  max {sims.max()}  std {sims.std():.1f}")

    if sims.std() < 4:
        print("\n  WARNING: edge counts barely vary across tracks. Graph topology")
        print("  is nearly identical everywhere, so the GNN gets no structural")
        print("  signal. Try a different tau (run tune_tau.py).")
    else:
        print("\n  Topology varies across tracks -- good.")

    print(f"\nSaved to {PROCESSED / 'gtzan_graphs.pt'}. Next: python train_task2.py")


if __name__ == "__main__":
    main()
