# GNN-BERT Music Context Understanding

CSE425/EEE474/CSE715 Neural Networks project: a hybrid BERT + Graph Neural
Network system for understanding musical context (genre, mood, structure)
from audio and text.

This repo is built task-by-task, matching the four-task roadmap in the
assignment brief:

| Task | What it does | Dataset | Status |
|------|---------------|---------|--------|
| 1 (Easy) | BERT multi-label tag classifier on MusicCaps captions | MusicCaps | done |
| 2 (Medium) | GNN on GTZAN chroma/segment graphs, genre classification | GTZAN | next |
| 3 (Hard) | GNN-BERT cross-attention fusion | MusicCaps (paired audio+caption) | later |
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
│   └── bert_baseline.py    # BERT model, train/eval loops
├── train_task1.py          # entry point for Task 1
├── plot_task1_curves.py    # F1-vs-epoch plot for the report
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

## Next steps

Once Task 1 has trained and you have a macro-F1 number you're happy
with, the next step is Task 2: building chroma/segment graphs from
GTZAN and training a GraphSAGE encoder. Ask your guide for that
walkthrough when ready.
