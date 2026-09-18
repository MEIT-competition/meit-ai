"""
Compares crash-class accuracy for "급브레이크"(sudden-braking) clips against
the rest of crash, to check whether brake sounds need a separate class.

Runs on the full manifest, not just held-out test - fold=4 alone has only
one brake clip, not enough for a real comparison. This is a descriptive
check for the class-design question, not a generalization metric.

Run after train_yamnet.py.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CALIBRATION_PATH, CLASSES, SAVED_MODEL_PATH, load_manifest, load_wav_16k_mono


def main():
    model = tf.saved_model.load(str(SAVED_MODEL_PATH))
    with open(CALIBRATION_PATH) as f:
        temperature = json.load(f)["temperature"]

    crash_idx = CLASSES.index("crash")
    crash_rows = [r for r in load_manifest() if r["label"] == crash_idx]

    results = {"brake": [], "other": []}
    for r in crash_rows:
        key = "brake" if "brake" in r["filepath"].lower() else "other"
        wav = load_wav_16k_mono(r["filepath"]).numpy()
        logits = model(tf.constant(wav, dtype=tf.float32)).numpy()
        probs = tf.nn.softmax(logits / temperature).numpy()
        results[key].append(int(np.argmax(probs)))

    for key in ["brake", "other"]:
        preds = results[key]
        n = len(preds)
        if n == 0:
            print(f"{key}: 데이터 없음")
            continue
        correct = sum(1 for p in preds if p == crash_idx)
        print(f"{key} (n={n}): crash로 맞게 예측 {correct}/{n} ({100*correct/n:.1f}%)")
        wrong = [CLASSES[p] for p in preds if p != crash_idx]
        if wrong:
            print(f"  틀렸을 때 예측 분포: {dict(Counter(wrong))}")


if __name__ == "__main__":
    main()
