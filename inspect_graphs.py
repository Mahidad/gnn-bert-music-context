"""
Inspect the saved graph .pt files.

Two uses:
1. Verify the 20 example graphs required by the submission checklist
   actually contain what you claim they do.
2. Produce graph statistics for your report's dataset section.

Run with:
    python inspect_graphs.py                # summarise the 20 sample graphs
    python inspect_graphs.py --all          # summarise the full cached set
    python inspect_graphs.py --show 3       # dump details of 3 graphs
"""

import argparse
from pathlib import Path

import numpy as np
import torch

PROCESSED = Path("data/processed")
SAMPLES = PROCESSED / "graph_samples"


def load_sample_graphs():
    files = sorted(SAMPLES.glob("*.pt"))
    if not files:
        raise FileNotFoundError(
            f"No .pt files in {SAMPLES}. Run `python preprocess_task2.py` first."
        )
    return [(f.name, torch.load(f, weights_only=False)) for f in files]


def load_all_graphs():
    path = PROCESSED / "gtzan_graphs.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python preprocess_task2.py` first."
        )
    bundle = torch.load(path, weights_only=False)
    graphs, genres = bundle["graphs"], bundle["genres"]
    return [(g.track_id, g) for g in graphs], genres


def describe(name, g, genres=None):
    label = int(g.y.item())
    genre = genres[label] if genres else label
    temporal = int(getattr(g, "num_temporal_edges", 0))
    similarity = int(getattr(g, "num_similarity_edges", 0))

    print(f"\n{name}")
    print(f"  genre          : {genre} (label {label})")
    print(f"  nodes          : {g.num_nodes}  (each = one 3s segment)")
    print(f"  node feat dim  : {g.x.shape[1]}")
    print(f"  edges total    : {g.edge_index.shape[1]}  (stored both directions)")
    print(f"    temporal     : {temporal}")
    print(f"    similarity   : {similarity}")
    if hasattr(g, "chords") and g.chords:
        # Show the chord sequence -- this is the metadata that powers the
        # Task 3 interpretability analysis.
        seq = " ".join(g.chords[:12])
        print(f"  chord sketch   : {seq}{' ...' if len(g.chords) > 12 else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="summarise the full cached graph set, not just samples")
    parser.add_argument("--show", type=int, default=2,
                        help="how many individual graphs to print in detail")
    args = parser.parse_args()

    genres = None
    if args.all:
        items, genres = load_all_graphs()
        print(f"Loaded {len(items)} graphs from the full cache.")
    else:
        items = load_sample_graphs()
        print(f"Loaded {len(items)} example graphs from {SAMPLES}/")

    for name, g in items[:args.show]:
        describe(name, g, genres)

    # --- aggregate statistics, ready to paste into the report ---------
    nodes = np.array([g.num_nodes for _, g in items])
    edges = np.array([g.edge_index.shape[1] for _, g in items])
    sims = np.array([int(getattr(g, "num_similarity_edges", 0)) for _, g in items])
    temps = np.array([int(getattr(g, "num_temporal_edges", 0)) for _, g in items])

    print("\n" + "=" * 55)
    print("AGGREGATE GRAPH STATISTICS (for your report)")
    print("=" * 55)
    print(f"  graphs                 : {len(items)}")
    print(f"  nodes      mean/min/max: {nodes.mean():.1f} / {nodes.min()} / {nodes.max()}")
    print(f"  edges      mean/min/max: {edges.mean():.1f} / {edges.min()} / {edges.max()}")
    print(f"  temporal   mean        : {temps.mean():.1f}")
    print(f"  similarity mean/min/max: {sims.mean():.1f} / {sims.min()} / {sims.max()}")

    # Diagnostics: is tau sensible?
    frac_saturated = float((sims >= 120).mean())
    frac_empty = float((sims == 0).mean())
    print(f"\n  graphs hitting the edge cap : {frac_saturated:.1%}")
    print(f"  graphs with NO similarity edges: {frac_empty:.1%}")

    if frac_saturated > 0.5:
        print("\n  -> Most graphs are hitting the cap. similarity_tau is too LOW:")
        print("     almost every segment pair counts as 'similar', so the")
        print("     repetition signal is washed out. Try tau 0.90.")
    elif frac_empty > 0.3:
        print("\n  -> Many graphs have no similarity edges at all. tau is too HIGH:")
        print("     the graph is reduced to a plain chain and the GNN loses the")
        print("     repetition structure that motivates using a graph. Try tau 0.80.")
    else:
        print("\n  -> Similarity edge density looks reasonable for this tau.")


if __name__ == "__main__":
    main()
