"""
Download MusicCaps audio clips from YouTube.

MusicCaps ships captions and tags but no audio -- only YouTube IDs and a
start/end timestamp. This script fetches just the 10-second labelled
window from each video, which is far faster than downloading whole videos.

Setup (one time):
    pip install yt-dlp imageio-ffmpeg

imageio-ffmpeg bundles a static ffmpeg binary, so you do NOT need to
install ffmpeg system-wide on Windows -- the script locates it for you.

Run with:
    python download_musiccaps.py                 # 1k genre-balanced subset
    python download_musiccaps.py --all           # all 5,521 clips
    python download_musiccaps.py --limit 200     # quick trial run
    python download_musiccaps.py --workers 2     # slower, gentler on YouTube

The script is RESUMABLE: clips already on disk are skipped, so if it dies
partway (or you stop it), just run it again. Expect a meaningful failure
rate -- videos get deleted, made private, or region-blocked over time.
That is normal and every MusicCaps paper deals with it; the script logs
which IDs failed so you can report the number honestly.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = Path("data/raw/musiccaps/audio")
META_DIR = Path("data/raw/musiccaps")
SAMPLE_RATE = 22050


def find_ffmpeg():
    """Prefer the pip-installed static ffmpeg so no system install is needed."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("imageio-ffmpeg not found; falling back to system ffmpeg.")
        print("If downloads fail, run: pip install imageio-ffmpeg")
        return None


def download_clip(row, ffmpeg_path, sample_rate=SAMPLE_RATE):
    """
    Download only the labelled window of one video.

    --download-sections tells yt-dlp to fetch just the requested time range
    instead of the whole video, which is the difference between a few
    seconds and a few minutes per clip.
    """
    ytid = row["ytid"]
    start, end = int(row["start_s"]), int(row["end_s"])
    out_path = OUT_DIR / f"{ytid}.wav"

    if out_path.exists():
        return ytid, "skipped"

    cmd = [
        sys.executable, "-m", "yt_dlp",
        f"https://www.youtube.com/watch?v={ytid}",
        "--quiet", "--no-warnings", "--no-playlist",
        "--extract-audio",
        "--audio-format", "wav",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "--postprocessor-args", f"ffmpeg:-ar {sample_rate} -ac 1",
        "--output", str(OUT_DIR / f"{ytid}.%(ext)s"),
        "--retries", "2",
        "--socket-timeout", "20",
    ]
    if ffmpeg_path:
        cmd += ["--ffmpeg-location", ffmpeg_path]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        return ytid, "timeout"

    if out_path.exists() and out_path.stat().st_size > 1000:
        return ytid, "ok"

    err = (result.stderr or b"").decode(errors="ignore").strip().splitlines()
    reason = err[-1][:120] if err else f"exit {result.returncode}"
    return ytid, f"failed: {reason}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="download all 5,521 clips instead of the 1k balanced subset")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many clips (useful for a trial run)")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel downloads; lower this if YouTube throttles you")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        print(f"Using bundled ffmpeg: {ffmpeg_path}")

    print("Loading MusicCaps metadata...")
    ds = load_dataset("google/MusicCaps", split="train")
    df = ds.to_pandas()

    if not args.all:
        # is_balanced_subset marks the 1,000-clip genre-balanced subset.
        # For a solo project this is the right scope: it covers the genre
        # range without a multi-hour download, and it keeps the graph
        # dataset comparable in size to GTZAN.
        df = df[df["is_balanced_subset"]].reset_index(drop=True)
        print(f"Using the genre-balanced subset: {len(df)} clips.")
    else:
        print(f"Using the full dataset: {len(df)} clips.")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to first {len(df)} clips.")

    rows = df.to_dict("records")
    existing = {p.stem for p in OUT_DIR.glob("*.wav")}
    todo = [r for r in rows if r["ytid"] not in existing]
    print(f"{len(existing)} already downloaded, {len(todo)} to fetch.\n")

    results = {}
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download_clip, r, ffmpeg_path): r["ytid"] for r in todo}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                ytid, status = fut.result()
                results[ytid] = status

    ok = sum(1 for s in results.values() if s in ("ok", "skipped"))
    failed = {k: v for k, v in results.items() if v not in ("ok", "skipped")}

    total_on_disk = len(list(OUT_DIR.glob("*.wav")))
    print(f"\nNewly downloaded: {ok}")
    print(f"Failed:           {len(failed)}")
    print(f"Total on disk:    {total_on_disk}")

    # Save the caption/tag metadata for exactly the clips we actually have,
    # so later stages never reference a clip whose audio is missing.
    have = {p.stem for p in OUT_DIR.glob("*.wav")}
    kept = [r for r in rows if r["ytid"] in have]
    META_DIR.mkdir(parents=True, exist_ok=True)
    with open(META_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump([{
            "ytid": r["ytid"],
            "caption": r["caption"],
            "aspect_list": r["aspect_list"],
            "start_s": int(r["start_s"]),
            "end_s": int(r["end_s"]),
            "audioset_positive_labels": r["audioset_positive_labels"],
            "is_audioset_eval": bool(r["is_audioset_eval"]),
        } for r in kept], f, indent=2)

    if failed:
        with open(META_DIR / "download_failures.json", "w") as f:
            json.dump(failed, f, indent=2)
        print(f"\nFailure reasons written to {META_DIR / 'download_failures.json'}")
        print("Report the failure count in your paper's dataset section --")
        print("YouTube attrition is a known, citable property of MusicCaps.")

    print(f"\nMetadata for {len(kept)} usable clips -> {META_DIR / 'metadata.json'}")
    print("\nIf many clips failed, re-run this script; it resumes automatically.")


if __name__ == "__main__":
    main()
