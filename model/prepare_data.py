"""
Raw dataset -> 16kHz mono, 4.5s clips, class-labeled CSV manifest.

Input: raw/{horn,siren,crash,normal}/*
Output: dataset/{class}/*.wav + manifest.csv (filepath,label,source,fold)

fold is hashed from the original filename so clips split from the same
source file stay in one fold (no train/val/test leakage).
"""
import csv
import hashlib
import math
from pathlib import Path

import soundfile as sf
import librosa
import numpy as np

CLASSES = ["horn", "siren", "crash", "normal"]
TARGET_SR = 16000
CLIP_SEC = 4.5
RAW_DIR = Path(__file__).parent / "raw"
OUT_DIR = Path(__file__).parent / "dataset"
MANIFEST_PATH = Path(__file__).parent / "manifest.csv"
NUM_FOLDS = 5


def fold_for(name: str) -> int:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(h, 16) % NUM_FOLDS


def split_into_clips(y: np.ndarray, sr: int, clip_sec: float):
    clip_len = int(clip_sec * sr)
    n_clips = max(1, math.ceil(len(y) / clip_len))
    clips = []
    for i in range(n_clips):
        start = i * clip_len
        chunk = y[start:start + clip_len]
        if len(chunk) < clip_len * 0.5:  # drop trailing scraps under half a clip
            continue
        if len(chunk) < clip_len:
            chunk = np.pad(chunk, (0, clip_len - len(chunk)))
        clips.append(chunk)
    return clips


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for label in CLASSES:
        src_dir = RAW_DIR / label
        dst_dir = OUT_DIR / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"[skip] no raw/{label}/ yet")
            continue

        for src_path in sorted(src_dir.glob("**/*")):
            if src_path.suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg"):
                continue
            try:
                y, sr = librosa.load(src_path, sr=TARGET_SR, mono=True)
            except Exception as e:
                print(f"[error] {src_path}: {e}")
                continue

            fold = fold_for(src_path.stem)
            for idx, clip in enumerate(split_into_clips(y, TARGET_SR, CLIP_SEC)):
                out_name = f"{label}_{src_path.stem}_{idx:03d}.wav"
                out_path = dst_dir / out_name
                sf.write(out_path, clip, TARGET_SR)
                rows.append({
                    "filepath": str(out_path.relative_to(OUT_DIR.parent)),
                    "label": label,
                    "source": src_path.name,
                    "fold": fold,
                })

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "source", "fold"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} clips to {OUT_DIR}, manifest at {MANIFEST_PATH}")
    for label in CLASSES:
        count = sum(1 for r in rows if r["label"] == label)
        print(f"  {label}: {count} clips")


if __name__ == "__main__":
    main()
