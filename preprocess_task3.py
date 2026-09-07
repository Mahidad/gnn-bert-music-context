"""
Task 3 preprocessing: MusicCaps audio clips -> segment graphs.

Same graph construction as Task 2, but tuned for 10-second clips instead
of 30-second ones: a 1s window with 0.5s hop yields ~19 nodes per clip,
matching the GTZAN graph size so the two tasks stay comparable.

Text is NOT baked in here. Captions are tokenized at training time, so
switching between the masked and unmasked conditions costs seconds rather
than a full re-preprocess.

Prerequisite: python download_musiccaps.py --all

Run with:
    python preprocess_task3.py
"""

from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.audio_features import extract_segment_features
from src.graph_builder import build_segment_graph, graph_summary
from src.musiccaps_data import load_metadata, build_tag_vocab, masking_report
from src.split_utils import stratified_split, train_feature_stats

AUDIO_DIR = Path("data/raw/musiccaps/audio")
PROCESSED = Path("data/processed")
SPLITS = Path("data/splits")


def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)["task3"]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)

    records = load_metadata()
    print(f"Metadata covers {len(records)} clips with downloaded audio.")

    # ---------- PASS 1: features ----------
    kept, skipped = [], []
    for rec in tqdm(records, desc="Extracting features"):
        wav = AUDIO_DIR / f"{rec['ytid']}.wav"
        if not wav.exists():
            skipped.append(rec["ytid"])
            continue

        features, chords = extract_segment_features(
            str(wav),
            segment_seconds=cfg["segment_seconds"],
            hop_seconds=cfg["hop_seconds"],
        )
        if features is None:
            skipped.append(rec["ytid"])
            continue

        kept.append({"features": features, "chords": chords,
                     "ytid": rec["ytid"], "label": 0})

    print(f"\nUsable clips: {len(kept)}  (skipped {len(skipped)} unreadable/missing)")
    if len(kept) < 100:
        raise RuntimeError(
            "Fewer than 100 usable clips. Re-run download_musiccaps.py --all;\n"
            "it resumes and will retry the clips that failed."
        )

    # ---------- PASS 2: split, then train-only statistics ----------
    # Multi-label data has no single class to stratify on, so this is a
    # plain seeded random split. Iterative stratification would balance
    # rare tags better; note that as a limitation rather than pretending
    # the split is stratified.
    tr, va, te = stratified_split([0] * len(kept), seed=cfg["seed"],
                                  val_frac=cfg["val_frac"], test_frac=cfg["test_frac"])
    print(f"Split -> train {len(tr)}, val {len(va)}, test {len(te)}")

    g_mean, g_std = train_feature_stats(kept, tr)

    # ---------- PASS 3: graphs ----------
    graphs = []
    for rec in kept:
        g = build_segment_graph(
            rec["features"], label=0,
            tau=cfg["similarity_tau"],
            max_similarity_edges=cfg["max_similarity_edges"],
            chords=rec["chords"], track_id=rec["ytid"],
            global_mean=g_mean, global_std=g_std,
        )
        g.ytid = rec["ytid"]
        graphs.append(g)

    torch.save({"graphs": graphs, "splits": {"train": tr, "val": va, "test": te}},
               PROCESSED / "musiccaps_graphs.pt")

    # ---------- tag vocabulary + masking statistics ----------
    have = {g.ytid for g in graphs}
    usable_records = [r for r in records if r["ytid"] in have]
    tag_vocab = build_tag_vocab(usable_records, top_k=cfg["num_tags"])

    import json
    with open(SPLITS / "task3_tag_vocab.json", "w") as f:
        json.dump(tag_vocab, f, indent=2)

    stats = [graph_summary(g) for g in graphs]
    sims = np.array([s["similarity_edges"] for s in stats])
    print(f"\nGraph statistics:")
    print(f"  node feature dim : {graphs[0].x.shape[1]}")
    print(f"  avg nodes/graph  : {np.mean([s['num_nodes'] for s in stats]):.1f}")
    print(f"  similarity edges : mean {sims.mean():.1f}  std {sims.std():.1f}")

    rep = masking_report(usable_records, tag_vocab)
    print(f"\nCaption masking (phrase level) -- report these numbers:")
    print(f"  captions affected  : {rep['pct_affected']:.1%}")
    print(f"  phrases removed    : {rep['avg_phrases_removed']:.2f} per caption")
    print(f"  words before/after : {rep['avg_words_before']:.1f} -> {rep['avg_words_after']:.1f}")

    with open("results/task3_masking_report.json", "w") as f:
        json.dump(rep, f, indent=2)

    print(f"\nSaved graphs to {PROCESSED / 'musiccaps_graphs.pt'}")
    print("Next: python train_task3.py --variant cross_attention")


if __name__ == "__main__":
    main()
