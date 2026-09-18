"""
Compares clip-level accuracy when the real-time buffer is shorter than the
4.5s clips used for training, at two crop strategies: from the start of the
clip, or centered on the highest-energy region (matches the decision-logic
side's adapter.py fit_length()).

Run after train_yamnet.py. Reads test_set_fold4.csv (fold=4 held-out clips).
"""
import csv
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import BASE_DIR, CALIBRATION_PATH, CLASSES, SAVED_MODEL_PATH, load_wav_16k_mono

WINDOW_SECS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.5]
SR = 16000


def crop_front(wav, n):
    if len(wav) >= n:
        return wav[:n]
    return np.pad(wav, (0, n - len(wav)))


def crop_centered(wav, n):
    """Centers the crop on the highest short-time-energy region."""
    if len(wav) == n:
        return wav
    if len(wav) < n:
        return np.pad(wav, (0, n - len(wav)))
    win = SR // 10
    energy = np.convolve(wav ** 2, np.ones(win), mode="same")
    center = int(np.argmax(energy))
    start = max(0, min(center - n // 2, len(wav) - n))
    return wav[start:start + n]


def run(crop_fn):
    import json
    model = tf.saved_model.load(str(SAVED_MODEL_PATH))
    with open(CALIBRATION_PATH) as f:
        temperature = json.load(f)["temperature"]

    rows = list(csv.DictReader(open(BASE_DIR / "test_set_fold4.csv", encoding="utf-8")))
    results = {w: {c: [0, 0] for c in CLASSES} for w in WINDOW_SECS}

    for r in rows:
        wav = load_wav_16k_mono(str(BASE_DIR / r["filepath"])).numpy()
        for w in WINDOW_SECS:
            clip = crop_fn(wav, int(w * SR))
            logits = model(tf.constant(clip, dtype=tf.float32)).numpy()
            probs = tf.nn.softmax(logits / temperature).numpy()
            pred = CLASSES[int(np.argmax(probs))]
            results[w][r["label"]][1] += 1
            if pred == r["label"]:
                results[w][r["label"]][0] += 1
    return results


def print_table(results, title):
    print(f"\n{title}")
    print(f"{'window(s)':>10}", *[f"{c:>16}" for c in CLASSES])
    for w in WINDOW_SECS:
        cells = [f"{results[w][c][0]}/{results[w][c][1]}({100*results[w][c][0]/results[w][c][1]:.1f}%)"
                 for c in CLASSES]
        print(f"{w:>10.1f}", *[f"{v:>16}" for v in cells])


if __name__ == "__main__":
    print_table(run(crop_front), "앞에서부터 크롭")
    print_table(run(crop_centered), "최대 에너지 지점 중심 크롭 (adapter.py fit_length()와 동일)")
