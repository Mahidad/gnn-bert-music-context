"""
Task 2 graph construction.

Turns a (num_segments, feat_dim) feature matrix into a PyTorch Geometric
graph. Two edge types, following the brief's segment-graph spec:

1. TEMPORAL edges  : segment i <-> segment i+1.
   Encodes "this moment is followed by that moment" -- the song's
   timeline. Without these the graph would forget what order anything
   happened in.

2. SIMILARITY edges : segment i <-> segment j when cosine(f_i, f_j) > tau.
   Encodes "these two moments sound alike" -- which is how repetition
   (chorus returning, riff looping) becomes visible to the model. This
   is the structural information a plain CNN cannot easily see.

Both edge types are stored undirected (each edge added in both
directions) because "A precedes B" and "B follows A" are equally useful
during message passing.
"""

import numpy as np
import torch
from torch_geometric.data import Data


def cosine_similarity_matrix(features):
    normed = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    return normed @ normed.T


def build_segment_graph(features, label, tau=0.85, max_similarity_edges=None,
                        chords=None, track_id=None,
                        global_mean=None, global_std=None):
    """
    Args:
        features : (N, D) numpy array of per-segment features
        label    : integer genre label for the whole track
        global_mean, global_std : dataset-wide feature statistics computed
                   from the TRAINING SPLIT ONLY (passing val/test statistics
                   here would leak information). When supplied, node features
                   become the concatenation of the absolute and relative
                   views, doubling the feature dimension.
        max_similarity_edges : cap on the number of similarity PAIRS.
                   Each pair is stored as 2 directed edges, so a cap of 60
                   yields up to 120 entries in edge_index.
        tau      : cosine-similarity threshold for adding a similarity edge.
                   Higher tau -> fewer, more confident "these repeat" edges.
                   0.85 is a reasonable default; tune it and report the
                   effect (good ablation material for your write-up).
    Returns:
        torch_geometric.data.Data
    """
    num_nodes = features.shape[0]

    # --- two complementary views of the same features ------------------
    # WITHIN-TRACK (relative): z-scored against the track's own mean/std.
    # Answers "how does this segment differ from the rest of THIS song?"
    # Good for finding repeats; used for the similarity edges.
    mu = features.mean(axis=0, keepdims=True)
    sigma = features.std(axis=0, keepdims=True) + 1e-8
    x_relative = ((features - mu) / sigma).astype(np.float32)

    # ACROSS-TRACK (absolute): z-scored against dataset-wide (train-split)
    # statistics. Answers "is this segment bright/loud/noisy compared to
    # music in general?" -- which is where most of the genre signal lives
    # (metal is bright and dense; classical is not).
    #
    # Using ONLY the relative view silently deletes that signal: every
    # track gets centred on its own mean, so absolute timbre is erased and
    # the GNN is left guessing from within-track contrasts alone. That
    # alone can cost 20+ accuracy points against a CNN, which reads
    # absolute values straight off the spectrogram.
    if global_mean is not None and global_std is not None:
        x_absolute = ((features - global_mean) / (global_std + 1e-8)).astype(np.float32)
        x = np.concatenate([x_absolute, x_relative], axis=1).astype(np.float32)
    else:
        x = x_relative

    src, dst, edge_type = [], [], []

    # --- temporal edges ------------------------------------------------
    for i in range(num_nodes - 1):
        src += [i, i + 1]
        dst += [i + 1, i]
        edge_type += [0, 0]

    # --- similarity edges ----------------------------------------------
    # IMPORTANT: similarity is computed on the STANDARDISED features `x`,
    # not the raw ones. The raw feature vector mixes scales that differ by
    # three orders of magnitude (chroma ~0-1, MFCC ~-400..100, spectral
    # rolloff ~4000). Cosine similarity measures the ANGLE between
    # vectors, so those few huge dimensions dominate the direction of
    # every vector and all pairs come out at ~0.99 -- which makes every
    # graph in the dataset structurally identical and destroys the
    # repetition signal the graph exists to capture.
    # Standardising first puts every dimension on equal footing, so the
    # similarity actually reflects "do these two segments sound alike".
    sim = cosine_similarity_matrix(x_relative)
    candidates = []
    for i in range(num_nodes):
        for j in range(i + 2, num_nodes):      # skip i+1: already temporal
            if sim[i, j] > tau:
                candidates.append((sim[i, j], i, j))

    # Cap the number of similarity edges so a very repetitive track can't
    # produce a near-complete graph (which would make message passing
    # meaningless -- if everyone talks to everyone, nobody has structure).
    candidates.sort(reverse=True)
    if max_similarity_edges is not None:
        candidates = candidates[:max_similarity_edges]

    for _, i, j in candidates:
        src += [i, j]
        dst += [j, i]
        edge_type += [1, 1]

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        edge_type=torch.tensor(edge_type, dtype=torch.long),
        y=torch.tensor([label], dtype=torch.long),
    )
    data.num_temporal_edges = int(sum(1 for t in edge_type if t == 0))
    data.num_similarity_edges = int(sum(1 for t in edge_type if t == 1))
    if chords is not None:
        data.chords = chords          # metadata for Task 3 interpretability
    if track_id is not None:
        data.track_id = track_id
    return data


def graph_summary(data):
    """Small helper for sanity-checking / reporting graph statistics."""
    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "temporal_edges": int(getattr(data, "num_temporal_edges", 0)),
        "similarity_edges": int(getattr(data, "num_similarity_edges", 0)),
        "label": int(data.y.item()),
    }
