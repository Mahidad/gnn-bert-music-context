"""
Task 2 audio feature extraction.

Turns one audio file into a sequence of per-segment feature vectors.
Each segment becomes one node in that track's graph.

Feature design rationale:
- chroma (12 dims)  -> WHAT pitches/harmony are present (key, chords)
- MFCC mean (20)    -> WHAT it sounds like (timbre: distorted guitar vs
                       piano vs voice)
- MFCC std (20)     -> HOW MUCH the timbre moves within the segment
                       (steady pad vs busy drum fill)
- 4 spectral stats  -> brightness, energy, noisiness

Together that's a 56-dim summary of "what is going on musically in these
3 seconds" -- rich enough to distinguish genres, small enough that a
1000-track dataset preprocesses in minutes rather than hours.
"""

import numpy as np
import librosa

SAMPLE_RATE = 22050          # the brief specifies 22,050 Hz
SEGMENT_SECONDS = 3.0        # each node covers 3 seconds of audio
HOP_SECONDS = 1.5            # segments overlap by half, so boundaries
                             # falling mid-phrase don't destroy a feature

# 24 chord templates (12 major + 12 minor), used to give each segment a
# coarse chord guess. This is NOT used as a training feature -- it is
# stored as metadata so that in Task 3 we can ask "which chord regions
# did the fusion model attend to?", which is our interpretability angle.
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _build_chord_templates():
    templates, labels = [], []
    major = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    minor = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    for i, root in enumerate(NOTE_NAMES):
        templates.append(np.roll(major, i))
        labels.append(f"{root}maj")
        templates.append(np.roll(minor, i))
        labels.append(f"{root}min")
    return np.stack(templates), labels


CHORD_TEMPLATES, CHORD_LABELS = _build_chord_templates()


def estimate_chord(chroma_vector):
    """
    Cheapest possible chord estimate: compare the segment's average
    chroma against 24 idealised chord shapes and take the best match.
    Not accurate enough to publish as chord recognition, but plenty good
    enough to label graph regions for qualitative analysis.
    """
    v = chroma_vector / (np.linalg.norm(chroma_vector) + 1e-8)
    norms = CHORD_TEMPLATES / np.linalg.norm(CHORD_TEMPLATES, axis=1, keepdims=True)
    return CHORD_LABELS[int(np.argmax(norms @ v))]


def extract_segment_features(audio_path, sample_rate=SAMPLE_RATE,
                             segment_seconds=SEGMENT_SECONDS,
                             hop_seconds=HOP_SECONDS):
    """
    Returns:
        features : (num_segments, 56) float32 array -- one row per node
        chords   : list of coarse chord labels, one per segment
    Returns (None, None) if the file can't be read (GTZAN ships with at
    least one corrupted file, so callers must handle this).
    """
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception:
        return None, None

    if y.size < sample_rate:                       # shorter than 1 second
        return None, None

    y = y / (np.max(np.abs(y)) + 1e-8)             # per-track normalisation

    seg_len = int(segment_seconds * sample_rate)
    hop_len = int(hop_seconds * sample_rate)

    features, chords = [], []

    for start in range(0, max(1, len(y) - seg_len + 1), hop_len):
        seg = y[start:start + seg_len]
        if seg.size < seg_len // 2:
            continue

        chroma = librosa.feature.chroma_stft(y=seg, sr=sr)
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=20)
        centroid = librosa.feature.spectral_centroid(y=seg, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=seg)
        rms = librosa.feature.rms(y=seg)

        chroma_mean = chroma.mean(axis=1)

        vec = np.concatenate([
            chroma_mean,                 # 12
            mfcc.mean(axis=1),           # 20
            mfcc.std(axis=1),            # 20
            [centroid.mean(), rolloff.mean(), zcr.mean(), rms.mean()],  # 4
        ]).astype(np.float32)

        features.append(vec)
        chords.append(estimate_chord(chroma_mean))

    if len(features) < 2:                          # need >= 2 nodes for an edge
        return None, None

    return np.stack(features), chords


def extract_mel_spectrogram(audio_path, sample_rate=SAMPLE_RATE,
                            n_mels=128, max_frames=1280):
    """
    Used only by the CNN baseline (B2 in the brief). Produces a fixed-size
    log-mel 'image' of the track so a standard 2D CNN can consume it.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception:
        return None

    if y.size < sample_rate:
        return None

    y = y / (np.max(np.abs(y)) + 1e-8)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # Pad or crop to a fixed width so every example is the same shape.
    if log_mel.shape[1] < max_frames:
        pad = max_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant",
                         constant_values=log_mel.min())
    else:
        log_mel = log_mel[:, :max_frames]

    return log_mel
