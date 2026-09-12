"""
Robust checkpoint saving.

torch.save writes directly to the destination path, which fails on Windows
if anything holds a handle on that file -- antivirus scanning a large
checkpoint, a sync client, or a stale handle from a previous run. The
symptom is `RuntimeError: File ... cannot be opened` partway through
training, losing the run.

It is also unsafe in a subtler way: writing over an existing checkpoint
means a crash mid-write leaves a truncated file and destroys the good
checkpoint that was already there.

safe_save fixes both by writing to a temporary file first and then
atomically replacing the destination. If the write fails, the previous
checkpoint is still intact.
"""

import os
import time
from pathlib import Path

import torch


def safe_save(obj, path, retries=3, delay=1.0):
    """
    Save `obj` to `path` via a temporary file plus atomic replace.

    Retries a few times with a short pause, since the usual cause (a
    scanner holding the file open) clears on its own within a second or two.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    last_error = None
    for attempt in range(retries):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)      # atomic on Windows and POSIX
            return True
        except (RuntimeError, OSError) as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))

    # Never raise: losing a checkpoint should not kill a training run that
    # is otherwise progressing fine.
    print(f"  [warning] could not save checkpoint to {path}: {last_error}")
    print("  Training continues, but this epoch was not checkpointed.")
    print("  If this repeats, add the project folder to your antivirus")
    print("  exclusions, or move it off any synced directory.")
    return False
