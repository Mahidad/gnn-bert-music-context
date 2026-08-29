"""
Task 2 preprocessing: GTZAN audio -> per-track segment graphs.

Run this ONCE. It reads every .wav under data/raw/gtzan/genres_original/,
extracts segment features, builds a graph per track, and caches
everything to data/processed/.

Order matters here: features are extracted first, then the split is made,
then dataset-wide normalisation statistics are computed FROM THE TRAINING
SPLIT ONLY, and only then are graphs built. Computing those statistics
over the whole dataset would leak test information into training.

Expected input layout (the standard GTZAN release):
    data/raw/gtzan/genres_original/blues/blues.00000.wav
    ...

Run with:
    python preprocess_task2.py
"""

import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.audio_features import extract_segment_features, extract_mel_spectrogram
from src.graph_builder import build_segment_graph, graph_summary
from src.split_utils import stratified_split, save_splits, train_feature_stats

GTZAN_ROOT = Path("data/raw/gtzan/genres_original")
PROCESSED = Path("data/processed")
SAMPLES = PROCESSED / "graph_samples"


def find_audio_files():
    if not GTZAN_ROOT.exists():
        raise FileNotFoundError(
            f"Could not find {GTZAN_ROOT}.\n"
            "Download GTZAN and extract it so that genre folders live at\n"
            "  data/raw/gtzan/genres_original/<genre>/<genre>.00000.wav\n"
            "See the README section 'Getting GTZAN' for download options."
        )
    genres = sorted([d.name for d in GTZAN_ROOT.iterdir() if d.is_dir()])
    files = []
    for genre in genres:
        for wav in sorted((GTZAN_ROOT / genre).glob("*.wav")):
            files.append((wav, genre))
    return files, genres


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)["task2"]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)

    files, genres = find_audio_files()
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    print(f"Found {len(files)} audio files across {len(genres)} genres: {genres}")

    # ---------- PASS 1: extract features (the slow part) ---------------
    records, mels, skipped = [], [], []

    for wav_path, genre in tqdm(files, desc="Extracting features"):
        features, chords = extract_segment_features(str(wav_path))
        if features is None:
            # GTZAN ships at least one corrupted file (jazz.00054.wav).
            # Skipping and logging it is the honest fix -- report the count.
            skipped.append(str(wav_path))
            continue

        records.append({
            "features": features,
            "chords": chords,
            "label": genre_to_idx[genre],
            "track_id": wav_path.stem,
        })

        if cfg["build_cnn_cache"]:
            mel = extract_mel_spectrogram(str(wav_path))
            if mel is not None:
                mels.append((mel, genre_to_idx[genre]))

    print(f"\nExtracted features for {len(records)} tracks. "
          f"Skipped {len(skipped)} unreadable files.")
    if skipped:
        print("Skipped:", skipped)

    # ---------- PASS 2: split, then train-only statistics ---------------
    labels = [r["label"] for r in records]
    train_idx, val_idx, test_idx = stratified_split(
        labels, seed=cfg["seed"], val_frac=cfg["val_frac"], test_frac=cfg["test_frac"]
    )
    save_splits(train_idx, val_idx, test_idx)
    print(f"Split -> train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

    g_mean, g_std = train_feature_stats(records, train_idx)

    # ---------- PASS 3: build graphs -----------------------------------
    graphs = []
    for rec in records:
        graphs.append(build_segment_graph(
            rec["features"],
            label=rec["label"],
            tau=cfg["similarity_tau"],
            max_similarity_edges=cfg["max_similarity_edges"],
            chords=rec["chords"],
            track_id=rec["track_id"],
            global_mean=g_mean,
            global_std=g_std,
        ))

    torch.save({"graphs": graphs, "genres": genres, "genre_to_idx": genre_to_idx},
               PROCESSED / "gtzan_graphs.pt")
    torch.save({"features": records, "genres": genres, "genre_to_idx": genre_to_idx,
                "global_mean": g_mean, "global_std": g_std},
               PROCESSED / "gtzan_features.pt")

    if mels:
        np.savez_compressed(
            PROCESSED / "gtzan_mels.npz",
            mels=np.stack([m for m, _ in mels]),
            labels=np.array([l for _, l in mels]),
        )
        print(f"Saved {len(mels)} mel-spectrograms for the CNN baseline.")

    # 20 example graphs -- a required submission deliverable
    for old in SAMPLES.glob("*.pt"):
        old.unlink()
    for i, g in enumerate(graphs[:20]):
        torch.save(g, SAMPLES / f"graph_{i:02d}_{g.track_id}.pt")

    stats = [graph_summary(g) for g in graphs]
    sims = np.array([s["similarity_edges"] for s in stats])
    print(f"\nGraph statistics (report these in your paper):")
    print(f"  node feature dim : {graphs[0].x.shape[1]} "
          f"(absolute + relative views concatenated)")
    print(f"  avg nodes/graph  : {np.mean([s['num_nodes'] for s in stats]):.1f}")
    print(f"  avg edges/graph  : {np.mean([s['num_edges'] for s in stats]):.1f}")
    print(f"  similarity edges : mean {sims.mean():.1f}  std {sims.std():.1f}  "
          f"min {sims.min()}  max {sims.max()}  (tau={cfg['similarity_tau']})")

    print(f"\nSaved graphs to {PROCESSED / 'gtzan_graphs.pt'}")
    print(f"Saved raw features to {PROCESSED / 'gtzan_features.pt'}")
    print("  -> use tune_tau.py / rebuild_graphs.py to change similarity_tau")
    print("     without re-running this slow audio extraction step.")


if __name__ == "__main__":
    main()
