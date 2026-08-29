# GNN-BERT Music Context Understanding

CSE425/EEE474/CSE715 Neural Networks project: a hybrid BERT + Graph Neural
Network system for understanding musical context (genre, mood, structure)
from audio and text.

This repo is built task-by-task, matching the four-task roadmap in the
assignment brief:

| Task | What it does | Dataset | Status |
|------|---------------|---------|--------|
| 1 (Easy) | BERT multi-label tag classifier on MusicCaps captions | MusicCaps | done |
| 2 (Medium) | GNN on GTZAN chroma/segment graphs, genre classification | GTZAN | done |
| 3 (Hard) | GNN-BERT cross-attention fusion | MusicCaps (paired audio+caption) | next |
| 4 (Advanced) | Contrastive dual-encoder retrieval | MusicCaps subset | later |

## Why this dataset choice

The brief lists several dataset options; here is the reasoning behind
picking these two:

- **MusicCaps** is the only dataset in the table that ships *paired*
  audio clips and natural-language captions for the *same* tracks. That
  pairing is exactly what Task 3 (fusion) and Task 4 (contrastive
  retrieval) need, so using it consistently across Tasks 1, 3, and 4
  means we solve "how do I connect this text to this audio" once instead
  of three separate times.
- MusicCaps captions also come with an `aspect_list` field: short,
  tag-like phrases extracted from each caption (e.g. `"mellow piano
  melody"`, `"pop"`). The most frequent 50 of these become our tag
  vocabulary for Task 1. This sidesteps the problem of MagnaTagATune not
  having real natural-language text to feed BERT in the first place.
- **GTZAN** is used only for Task 2, which is audio-structure-only (no
  text involved). It's small (1,000 tracks), well labeled, and it's the
  dataset the brief itself calls the "easy baseline" for genre
  classification.

## Setup (Windows + NVIDIA GPU)

You need a recent NVIDIA driver, and nothing else from NVIDIA. PyTorch
bundles its own CUDA runtime libraries inside the package, so you do
**not** need to install the CUDA Toolkit separately.

### Step 1 — check Python

Open PowerShell in the project folder and run:

```powershell
python --version
```

You want Python 3.10-3.12. If Python isn't installed, get it from
python.org and **tick "Add python.exe to PATH"** during install.

### Step 2 — create a virtual environment

A venv is a private package folder just for this project, so installing
these versions of torch/transformers can't break anything else on your
machine.

```powershell
python -m venv venv
venv\Scripts\activate
```

Your prompt should now start with `(venv)`. If PowerShell blocks the
activate script, run this once, then retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Step 3 — install PyTorch with CUDA (do this BEFORE requirements.txt)

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

The `--index-url` part is the whole point: plain `pip install torch` on
Windows silently installs a CPU-only build, and your 4060 would never be
used. For the newest CUDA build available, check the selector at
https://pytorch.org/get-started/locally/ — cu121 is a safe, well-tested
choice for an RTX 4060.

### Step 4 — install everything else

```powershell
pip install -r requirements.txt
```

### Step 5 — verify the GPU is actually visible

```powershell
python check_gpu.py
```

You should see your RTX 4060 named, ~8 GB VRAM, and a successful test
matmul. If it reports a CPU-only build, the script tells you exactly how
to fix it. Do not move on until this passes.

### Step 6 — run Task 1

```powershell
python train_task1.py
python plot_task1_curves.py
```

MusicCaps caption/tag data downloads automatically on first run. No audio
download is needed for Task 1.

### Running from VS Code

Open the project folder, then Ctrl+Shift+P -> "Python: Select
Interpreter" -> pick the one inside `venv`. After that the built-in
terminal and the Run button both use the right environment automatically.

### A note on Colab

If you previously tried this in Colab and hit "cannot find or open
gnn-bert-music-context.zip": files on your own computer do not
automatically exist inside Colab's VM -- they have to be uploaded to it
first. Also, each `!` cell in Colab runs in its own subprocess, so
`!cd some-folder` does not persist to the next cell (use `%cd` instead).
Running locally avoids both problems along with the session time limits.

## Project structure

```
gnn-bert-music-context/
├── README.md
├── requirements.txt
├── config.yaml            # all hyperparameters live here, not in code
├── check_gpu.py            # run first: confirms CUDA/GPU works
├── data/
│   ├── raw/                # (future) downloaded audio
│   ├── processed/          # (future) graphs, cached embeddings
│   └── splits/             # tag vocabulary + split info (auto-generated)
├── src/
│   ├── data_utils.py       # MusicCaps loading, tag vocab, PyTorch Dataset
│   ├── bert_baseline.py    # BERT model, train/eval loops
│   ├── audio_features.py   # segmentation, chroma/MFCC/mel extraction
│   ├── graph_builder.py    # segment graphs (temporal + similarity edges)
│   └── gnn_model.py        # GraphSAGE encoder + CNN baseline
├── train_task1.py          # entry point for Task 1
├── plot_task1_curves.py    # F1-vs-epoch plot for the report
├── preprocess_task2.py     # GTZAN -> cached graphs (run once)
├── train_task2.py          # GraphSAGE genre classifier
├── train_cnn_baseline.py   # baseline B2 for comparison
├── plot_task2_results.py   # Task 2 curves, confusion matrix, comparison
├── inspect_graphs.py       # inspect saved .pt graphs
│   └── split_utils.py      # shared stratified splits + train-only stats
├── tune_tau.py             # pick similarity_tau from real data
├── rebuild_graphs.py       # rebuild graphs at a new tau (fast, no re-extract)
└── results/                 # metrics.json, saved model, plots land here
```

This mirrors the structure required by the assignment brief, so later
tasks (graph_builder.py, gnn_model.py, fusion_model.py, contrastive.py)
slot in without needing to reorganize anything.

## Task 1: what it does and why

**Goal:** given only a caption describing a piece of music (no audio),
predict which of the top-50 most common aspect tags apply.

**Why this baseline matters:** it tells us how much of "musical context"
is recoverable from language alone, before any audio structure is added.
Tasks 2-4 build on top of this number -- if the Task 3 fusion model
doesn't beat this baseline, that's itself a finding worth discussing in
the report (it would mean the text signal alone was already carrying
most of the tagging power).

**How the pieces fit together:**
- `src/data_utils.py` downloads MusicCaps, parses the `aspect_list`
  string into real tags, keeps the top 50 most frequent ones as the
  label set, and turns each caption into `(tokenized_text, multi-hot
  label vector)` pairs.
- `src/bert_baseline.py` defines the model (DistilBERT -> CLS token ->
  linear layer -> one logit per tag) and the train/evaluate loops.
  Binary cross-entropy is used because tags are independent yes/no
  decisions, not mutually exclusive classes.
- `train_task1.py` wires it together: load data, split it, train for
  N epochs, save the best checkpoint by validation macro-F1, then
  report final test-set metrics and 5 example predictions.

**Config you'll likely want to tweak** (`config.yaml`):
- `epochs`: start at 8; raise it if train/val loss are both still
  dropping when it stops.
- `batch_size`: 16 is safe for a free-tier T4 GPU; lower to 8 if you
  hit an out-of-memory error.
- `model_name`: `distilbert-base-uncased` trains roughly 2x faster than
  full `bert-base-uncased` with a small accuracy cost. Good default
  while iterating; consider switching to `bert-base-uncased` for your
  final reported run if time allows.

**Outputs after running `train_task1.py`:**
- `results/task1_best_model.pt` -- best checkpoint by validation macro-F1
- `results/task1_metrics.json` -- per-epoch loss/F1 history + final test
  metrics
- `results/task1_f1_curve.png` -- after running `plot_task1_curves.py`
- Console output with 5 example predictions vs. ground truth

## Task 2: GNN on music structure graphs

**Goal:** classify GTZAN genre using *only* audio structure -- no text,
no captions. A track is turned into a graph of its own 3-second
segments, and a GraphSAGE encoder learns from how those segments relate.

### Getting GTZAN

GTZAN is not bundled here (it is ~1.2 GB). Download it and extract so
the layout is:

```
data/raw/gtzan/genres_original/blues/blues.00000.wav
data/raw/gtzan/genres_original/classical/classical.00000.wav
... (10 genre folders, 100 tracks each)
```

Common sources: the Kaggle mirror "GTZAN Dataset - Music Genre
Classification", or the `marsyas/gtzan` dataset on Hugging Face. Any copy
works as long as the folder layout above matches.

Known dataset caveats worth stating in your report: GTZAN contains a
handful of exact duplicate tracks, some mislabelled clips, and at least
one corrupted file (`jazz.00054.wav`). The preprocessing script skips
unreadable files and prints how many it skipped -- report that number.

### Running Task 2

```powershell
pip install -r requirements.txt        # picks up librosa + torch_geometric
python preprocess_task2.py             # one-time, ~10-20 min for 1000 tracks
python train_task2.py                  # GraphSAGE genre classifier
python train_cnn_baseline.py           # baseline B2 for comparison
python plot_task2_results.py           # curves, confusion matrix, comparison table
python inspect_graphs.py               # verify the 20 sample .pt graphs
```

Re-running `preprocess_task2.py` after changing `similarity_tau` or the
segment settings **overwrites** the cache in `data/processed/`, so any
ablation over tau means: edit `config.yaml` -> re-run preprocessing ->
re-run training. Changing `num_layers`, `hidden_dim`, or `epochs` only
affects training, so those need no re-preprocessing.


### Choosing `similarity_tau` (do not skip this)

Similarity edges are computed on **standardised** node features. This
matters: the raw feature vector mixes scales spanning three orders of
magnitude (chroma ~0-1, MFCC means ~-400..100, spectral rolloff ~4000).
Cosine similarity measures the angle between vectors, so those few huge
dimensions dominate every vector's direction and *all* segment pairs come
out at ~0.99. The symptom is unmistakable and worth checking for: every
graph in the dataset ends up with an identical node and edge count, the
similarity-edge cap is hit by 100% of tracks, and the GNN scores far
below the CNN baseline because graph topology carries no information at
all about the track.

Because similarity now runs on standardised features, the meaningful tau
range is roughly **0.3-0.9**, not near 1.0. Pick it from data:

```powershell
python tune_tau.py                     # shows the real similarity distribution
python rebuild_graphs.py --tau 0.6     # rebuild graphs in seconds
python train_task2.py
```

`preprocess_task2.py` caches raw features to
`data/processed/gtzan_features.pt`, so `rebuild_graphs.py` can rebuild the
whole graph set at a new tau in seconds rather than re-decoding 1,000
audio files. That makes a tau ablation cheap: rebuild, retrain, record,
repeat. `rebuild_graphs.py` reuses the existing split file so every tau
setting is evaluated on the identical test set.

The health check that matters is **variation**: if every graph has the
same number of similarity edges, the topology is uninformative no matter
what the mean is. `rebuild_graphs.py` and `inspect_graphs.py` both warn
when that happens.

### How the graph is built

Each track becomes one graph:

- **Nodes** = overlapping 3-second segments (3 s window, 1.5 s hop), so a
  30 s GTZAN clip gives ~19 nodes.
- **Node features (56 dims)** = 12 chroma (harmony) + 20 MFCC means
  (timbre) + 20 MFCC stds (timbral movement) + 4 spectral statistics
  (brightness, rolloff, noisiness, energy).
- **Temporal edges** connect segment *i* to segment *i+1* -- the timeline.
- **Similarity edges** connect any two non-adjacent segments whose
  feature cosine similarity exceeds `similarity_tau` -- this is what
  makes *repetition* (a chorus returning, a riff looping) visible to the
  model, and it is the structural signal a plain CNN cannot easily see.

Each segment also gets a coarse chord estimate (chroma matched against 24
major/minor templates). This is **not** a training feature -- it is stored
as graph metadata so Task 3 can ask "which chord regions did the fusion
model attend to when predicting this mood?", which is the interpretability
angle of this project.

### Why these modelling choices

- **GraphSAGE, 2 layers.** SAGEConv implements exactly the update in the
  brief: concatenate a node's own features with the mean of its
  neighbours, then transform. Two layers means each segment ends up aware
  of its neighbours' neighbours. Deeper stacks cause *over-smoothing* --
  every node converges toward the same vector and segments stop being
  distinguishable.
- **Mean pooling readout.** Matches the brief's `g = (1/|V|) sum_i h_i`.
  The resulting `g` is reused directly as the graph branch of the Task 3
  fusion model, which is why `GraphSAGEClassifier.encode()` is a separate
  method.
- **CrossEntropyLoss, not BCE.** Unlike Task 1's multi-label tagging, a
  GTZAN track has exactly one genre, so the classes are mutually
  exclusive.
- **Per-graph feature standardisation.** Tracks differ hugely in loudness
  and recording quality; normalising within each track forces the model
  to learn from *relative* differences between segments rather than
  absolute recording levels.
- **Stratified split, fixed seed.** With only 100 tracks per genre, an
  unstratified random split can badly imbalance the test set and move
  accuracy by several points on its own.
- **Train-only normalisation for the CNN.** Mel statistics are computed
  from training data only. Using whole-dataset statistics would leak
  test information into training -- exactly what the rubric's "no
  leakage" criterion targets.


### Node features: two views, and why both are needed

Each node carries **112 dimensions** -- the same 56 features expressed twice:

- **Absolute view**: z-scored against dataset-wide statistics computed
  from the *training split only*. Answers "is this segment bright, loud,
  dense compared to music in general?" Most genre signal lives here --
  metal is bright and dense, classical is not.
- **Relative view**: z-scored against the track's own mean and standard
  deviation. Answers "how does this segment differ from the rest of THIS
  song?" Good for spotting internal contrast and repeats.

Similarity edges are always computed from the **relative** view, since
repetition is a within-track question.

An earlier version of this pipeline used only the relative view, and it
cost roughly 20 accuracy points. Per-track z-scoring subtracts each
track's own mean, which erases absolute timbre entirely -- the model was
left inferring genre purely from within-track contrasts while the CNN
baseline read absolute values straight off the spectrogram. If you ablate
this, report it: it is a clean demonstration that a normalisation choice
can matter more than architecture.

### Overfitting controls

With ~700 training graphs the GNN memorises quickly (train accuracy 98%,
validation ~45%, validation loss climbing). The current settings push back
on that:

- `hidden_dim: 64` and `weight_decay: 0.005` -- fewer parameters, stronger
  L2 penalty.
- `dropout: 0.5` on hidden units, plus `edge_dropout: 0.2`, which randomly
  hides a fifth of the edges on every training step. That is dropout for
  graph *structure*: it stops the model relying on any one track's exact
  wiring.
- **Early stopping on validation loss** with `early_stopping_patience: 25`.

The checkpoint is selected by validation **loss**, not macro-F1. On a
99-track validation set F1 is noisy enough that its peak often lands on a
badly overfit epoch that guessed luckily -- an earlier run selected epoch
97, where validation loss was near its worst and test loss came out at
3.40. Loss reflects calibration as well as correctness, so it is the more
stable criterion at this dataset size.

### What to expect

GTZAN genre accuracy in the 0.55-0.75 range is normal for models of this
size. The number that matters for your report is not the absolute
accuracy but the **GNN vs CNN comparison on the identical split**, plus
the per-genre confusion matrix (classical and metal are usually easy;
rock and country are usually confused with everything).

### Outputs

- `data/processed/gtzan_graphs.pt` -- all graphs + genre mapping
- `data/processed/graph_samples/` -- 20 example `.pt` graphs (a required
  submission deliverable)
- `data/splits/task2_splits.json` -- the shared split used by BOTH models
- `results/task2_metrics.json` -- history, test metrics, confusion matrix
- `results/task2_cnn_metrics.json` -- baseline results

## Next steps

Once Task 1 has trained and you have a macro-F1 number you're happy
with, the next step is Task 2: building chroma/segment graphs from
GTZAN and training a GraphSAGE encoder. Ask your guide for that
walkthrough when ready.
