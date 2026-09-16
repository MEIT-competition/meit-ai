"""Shared constants and helpers used across the pipeline scripts."""
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

BASE_DIR = Path(__file__).parent
MANIFEST_PATH = BASE_DIR / "manifest.csv"
SAVED_MODEL_PATH = BASE_DIR / "saved_model" / "danger_sound_classifier"
EMBED_CACHE_DIR = BASE_DIR / "embed_cache"
CALIBRATION_PATH = BASE_DIR / "calibration.json"
ABLATION_LOG_PATH = BASE_DIR / "ablation_log.csv"

CLASSES = ["horn", "siren", "crash", "normal"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NORMAL_IDX = CLASS_TO_IDX["normal"]

# 위험음을 놓치는 게 더 치명적이라 danger 클래스에 더 큰 가중치를 줌 (crash > siren/horn > normal)
CLASS_WEIGHTS = {
    CLASS_TO_IDX["horn"]: 2.0,
    CLASS_TO_IDX["siren"]: 2.0,
    CLASS_TO_IDX["crash"]: 2.0,
    CLASS_TO_IDX["normal"]: 1.0,
}

YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"


def load_manifest():
    import csv
    rows = []
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "filepath": str(BASE_DIR / r["filepath"]),
                "label": CLASS_TO_IDX[r["label"]],
                "fold": int(r["fold"]),
                "source": r["source"],
            })
    return rows


def load_wav_16k_mono(filepath):
    file_contents = tf.io.read_file(filepath)
    wav, sr = tf.audio.decode_wav(file_contents, desired_channels=1)
    return tf.squeeze(wav, axis=-1)


def extract_embeddings_for_rows(rows, yamnet_model, split_name):
    """YAMNet embeddings per 0.96s frame (one clip -> multiple frames), cached to .npz."""
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EMBED_CACHE_DIR / f"{split_name}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["embeddings"], data["labels"], data["filepaths"], data["clip_id"]

    all_emb, all_labels, all_fp, all_clip_id = [], [], [], []
    for clip_id, row in enumerate(rows):
        wav = load_wav_16k_mono(row["filepath"])
        _, embeddings, _ = yamnet_model(wav)
        n = embeddings.shape[0]
        all_emb.append(embeddings.numpy())
        all_labels.extend([row["label"]] * n)
        all_fp.extend([row["filepath"]] * n)
        all_clip_id.extend([clip_id] * n)
        if clip_id % 50 == 0:
            print(f"  [{split_name}] embedded {clip_id}/{len(rows)} clips")

    embeddings = np.concatenate(all_emb, axis=0)
    labels = np.array(all_labels)
    filepaths = np.array(all_fp)
    clip_id = np.array(all_clip_id)

    np.savez(cache_path, embeddings=embeddings, labels=labels,
             filepaths=filepaths, clip_id=clip_id)
    return embeddings, labels, filepaths, clip_id


def pool_clip_logits(logits, labels, clip_id, extra=None):
    """Reduce frame-level logits/labels to one row per clip, using the same
    pooling as the deployed model: the frame with the single highest peak
    logit wins (not the mean - short impulsive sounds only light up 1-2 of
    ~8 frames, so averaging with the mostly-silent rest washes them out).

    extra: optional dict of other frame-aligned arrays (e.g. filepaths) to
    reduce the same way, returned as a dict keyed the same way.
    Returns (pooled_logits, pooled_labels, unique_clip_ids, pooled_extra).
    """
    unique_clips = np.unique(clip_id)
    pooled_logits = np.zeros((len(unique_clips), logits.shape[1]), dtype=logits.dtype)
    pooled_labels = np.zeros(len(unique_clips), dtype=labels.dtype)
    pooled_extra = {k: np.empty(len(unique_clips), dtype=object) for k in (extra or {})}

    for i, cid in enumerate(unique_clips):
        mask = clip_id == cid
        clip_logits = logits[mask]
        best = np.argmax(np.max(clip_logits, axis=1))
        pooled_logits[i] = clip_logits[best]
        pooled_labels[i] = labels[mask][0]
        for k, arr in (extra or {}).items():
            pooled_extra[k][i] = arr[mask][best]

    return pooled_logits, pooled_labels, unique_clips, pooled_extra


def load_yamnet():
    return hub.load(YAMNET_HANDLE)


def load_head():
    return tf.keras.models.load_model(BASE_DIR / "head_model.keras")
