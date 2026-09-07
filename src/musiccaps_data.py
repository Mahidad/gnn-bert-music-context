"""
Task 3 data utilities: paired MusicCaps audio graphs + captions.

The important piece here is caption MASKING.

Task 1 showed that MusicCaps aspect tags are largely lifted from the
caption's own wording -- a caption saying "the song is slow tempo" carries
the tag `slow tempo` verbatim. A model can score well by pattern-matching
phrases rather than understanding anything about the music, and if the
text branch can simply copy the answer, the graph branch has nothing left
to contribute and the whole fusion experiment measures nothing.

Masking removes those give-away phrases from the caption before it reaches
BERT, so the text branch has to INFER the tag from surrounding description
instead of copying it. Running both conditions turns Task 3 from a metrics
dump into an actual experiment: unmasked measures extraction, masked
measures understanding.
"""

import ast
import json
import re
from collections import Counter
from pathlib import Path

import torch

META_PATH = Path("data/raw/musiccaps/metadata.json")

# Words too generic to be worth masking -- removing them would damage the
# caption's grammar without removing any real answer leakage.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "with", "is", "are",
    "this", "that", "it", "its", "to", "for", "as", "at", "by", "song",
    "track", "music", "sound", "sounds", "playing", "played",
}


def load_metadata(path=META_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run `python download_musiccaps.py --all` first -- it writes\n"
            "metadata.json covering exactly the clips whose audio downloaded."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_aspects(raw):
    if isinstance(raw, list):
        return list(raw)
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []


def build_tag_vocab(records, top_k=50):
    counter = Counter()
    for r in records:
        counter.update(parse_aspects(r["aspect_list"]))
    return {tag: i for i, tag in enumerate(t for t, _ in counter.most_common(top_k))}


def make_multi_hot(aspects, tag_to_idx):
    vec = torch.zeros(len(tag_to_idx), dtype=torch.float32)
    for tag in aspects:
        if tag in tag_to_idx:
            vec[tag_to_idx[tag]] = 1.0
    return vec


def mask_caption(caption, tag_vocab, aggressive=False):
    """
    Remove label-bearing phrases from a caption.

    Two levels:
      phrase-level (default) -- removes exact tag phrases, e.g. the caption
        "The song is slow tempo with accordions" becomes
        "The song is [MASK] with accordions".
      aggressive -- additionally removes the individual content words that
        make up multi-word tags, closing paraphrase leaks like "slow" still
        appearing on its own.

    Phrase-level is the default because aggressive masking strips so much
    that captions stop being fluent English, which is itself a confound
    (BERT is pretrained on fluent text). Neither level is airtight -- a
    caption can still hint at "slow tempo" by saying "unhurried" -- so
    report masking as a reduction in leakage, not its elimination.

    Returns (masked_caption, num_phrases_removed).
    """
    masked = caption
    removed = 0

    # Longest tags first, so "mellow piano melody" is removed as a unit
    # before the shorter "piano melody" can match part of it.
    for tag in sorted(tag_vocab, key=len, reverse=True):
        pattern = re.compile(re.escape(tag), re.IGNORECASE)
        masked, n = pattern.subn("[MASK]", masked)
        removed += n

    if aggressive:
        words = set()
        for tag in tag_vocab:
            for w in re.findall(r"[a-z']+", tag.lower()):
                if w not in _STOPWORDS and len(w) > 2:
                    words.add(w)
        for w in sorted(words, key=len, reverse=True):
            pattern = re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE)
            masked, n = pattern.subn("[MASK]", masked)
            removed += n

    masked = re.sub(r"(\[MASK\]\s*)+", "[MASK] ", masked)   # collapse runs
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked, removed


def masking_report(records, tag_vocab, aggressive=False):
    """Statistics for the report: how much text did masking actually remove?"""
    affected, total_removed, orig_len, new_len = 0, 0, 0, 0
    for r in records:
        masked, n = mask_caption(r["caption"], tag_vocab, aggressive)
        if n:
            affected += 1
        total_removed += n
        orig_len += len(r["caption"].split())
        new_len += len(masked.split())
    return {
        "captions": len(records),
        "captions_affected": affected,
        "pct_affected": affected / max(1, len(records)),
        "avg_phrases_removed": total_removed / max(1, len(records)),
        "avg_words_before": orig_len / max(1, len(records)),
        "avg_words_after": new_len / max(1, len(records)),
    }


def attach_text(graphs, records, tokenizer, tag_to_idx, caption_mode="masked",
                max_length=128, aggressive=False):
    """
    Attach tokenized captions and multi-hot labels to cached graphs.

    Text lives here rather than in the cached graph files so that switching
    between masked and unmasked costs a few seconds of tokenizing instead of
    re-running audio preprocessing.

    Tensors are stored with a leading dim of 1 so PyTorch Geometric's
    batching concatenates them into (batch, ...) rather than treating them
    as per-node features.
    """
    by_id = {r["ytid"]: r for r in records}
    out = []

    for g in graphs:
        rec = by_id.get(g.ytid)
        if rec is None:
            continue

        caption = rec["caption"]
        if caption_mode == "masked":
            caption, _ = mask_caption(caption, tag_to_idx, aggressive=aggressive)

        enc = tokenizer(caption, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")

        g.input_ids = enc["input_ids"]
        g.attention_mask = enc["attention_mask"]
        g.y = make_multi_hot(parse_aspects(rec["aspect_list"]), tag_to_idx).unsqueeze(0)
        out.append(g)

    return out
