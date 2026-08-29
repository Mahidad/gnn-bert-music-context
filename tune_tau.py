"""
Choose similarity_tau from your actual data instead of guessing.

Loads the cached raw features, standardises them exactly as the graph
builder does, and reports the real distribution of segment-to-segment
cosine similarities -- plus how many edges each candidate tau would
produce.

Prerequisite: run `python preprocess_task2.py` (which now caches raw
features to data/processed/gtzan_features.pt).

Run with:
    python tune_tau.py
"""

import numpy as np
import torch
from pathlib import Path

PROCESSED = Path("data/processed")

CANDIDATE_TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]


def standardise(features):
    mu = features.mean(axis=0, keepdims=True)
    sigma = features.std(axis=0, keepdims=True) + 1e-8
    return (features - mu) / sigma


def offdiag_similarities(features):
    """Cosine similarity between all non-adjacent segment pairs."""
    x = standardise(features)
    normed = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = normed @ normed.T
    n = sim.shape[0]
    vals = []
    for i in range(n):
        for j in range(i + 2, n):     # skip i+1: those are temporal edges
            vals.append(sim[i, j])
    return np.array(vals)


def main():
    path = PROCESSED / "gtzan_features.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Re-run `python preprocess_task2.py` once -- the updated version\n"
            "caches raw features so tau can be tuned without re-extracting audio."
        )

    bundle = torch.load(path, weights_only=False)
    records = bundle["features"]
    genres = bundle["genres"]
    print(f"Loaded cached features for {len(records)} tracks.\n")

    all_sims = []
    per_track_counts = {tau: [] for tau in CANDIDATE_TAUS}

    for rec in records:
        sims = offdiag_similarities(rec["features"])
        all_sims.append(sims)
        for tau in CANDIDATE_TAUS:
            per_track_counts[tau].append(int((sims > tau).sum()))

    flat = np.concatenate(all_sims)

    print("=" * 62)
    print("SEGMENT-PAIR COSINE SIMILARITY DISTRIBUTION (standardised)")
    print("=" * 62)
    for p in [1, 5, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:3d}th percentile : {np.percentile(flat, p):+.3f}")
    print(f"  mean             : {flat.mean():+.3f}")

    print("\n" + "=" * 62)
    print("SIMILARITY PAIRS PER GRAPH AT EACH CANDIDATE TAU")
    print("=" * 62)
    print(f"{'tau':>6}{'mean':>9}{'median':>9}{'min':>7}{'max':>7}"
          f"{'% empty':>10}{'verdict':>22}")
    print("-" * 62)

    best_tau, best_score = None, -1
    for tau in CANDIDATE_TAUS:
        counts = np.array(per_track_counts[tau])
        pct_empty = float((counts == 0).mean())
        spread = counts.std()

        # A good tau gives graphs that DIFFER from each other: some tracks
        # repetitive (many edges), some not (few). Zero variation means the
        # topology carries no information about the track.
        if pct_empty > 0.4:
            verdict = "too sparse"
        elif counts.mean() > 60:
            verdict = "too dense"
        elif spread < 2:
            verdict = "no variation"
        else:
            verdict = "usable"
            score = spread
            if score > best_score:
                best_score, best_tau = score, tau

        print(f"{tau:>6.2f}{counts.mean():>9.1f}{np.median(counts):>9.1f}"
              f"{counts.min():>7d}{counts.max():>7d}{pct_empty:>9.1%}{verdict:>22}")

    print("\n" + "=" * 62)
    if best_tau is not None:
        print(f"RECOMMENDED tau = {best_tau}")
        print("This gives the widest spread of edge counts across tracks, which")
        print("means graph topology actually varies by track -- the whole point")
        print("of modelling a song as a graph.")
        print(f"\nNext: python rebuild_graphs.py --tau {best_tau}")
        print("      python train_task2.py")
    else:
        print("No candidate tau looks clearly usable. Try widening the range")
        print("in CANDIDATE_TAUS, or reconsider the node feature set.")

    # per-genre check: do genres differ in repetitiveness?
    if best_tau is not None:
        print("\n" + "=" * 62)
        print(f"MEAN SIMILARITY PAIRS PER GENRE (at tau={best_tau})")
        print("=" * 62)
        by_genre = {}
        for rec, sims in zip(records, all_sims):
            by_genre.setdefault(rec["label"], []).append(int((sims > best_tau).sum()))
        for label in sorted(by_genre):
            vals = np.array(by_genre[label])
            print(f"  {genres[label]:<12} mean {vals.mean():6.1f}   std {vals.std():5.1f}")
        print("\nIf these means differ noticeably between genres, graph structure")
        print("is carrying genre signal -- that difference is worth a sentence")
        print("in your report.")


if __name__ == "__main__":
    main()
