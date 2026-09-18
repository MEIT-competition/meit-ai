"""
Loads the deployable SavedModel + calibration temperature and turns a single
gating-window audio clip (16kHz mono numpy array) into a decision:
  {class, confidence, is_danger, intensity, all_probs}

Usage:
    python inference.py path/to/clip.wav
or import classify_clip(wav_array) directly.
"""
import json
import sys

import numpy as np
import tensorflow as tf

from common import CALIBRATION_PATH, CLASSES, NORMAL_IDX, SAVED_MODEL_PATH

DECISION_THRESHOLD = 0.4


def load_runtime():
    model = tf.saved_model.load(str(SAVED_MODEL_PATH))
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            temperature = json.load(f)["temperature"]
    else:
        temperature = 1.0
    return model, temperature


def classify_clip(wav_array: np.ndarray, model, temperature, threshold: float = DECISION_THRESHOLD):
    """wav_array: 1-D float32 numpy array, 16kHz mono, values in [-1, 1]."""
    wav_tensor = tf.constant(wav_array, dtype=tf.float32)
    raw_scores = model(wav_tensor).numpy()
    probs = tf.nn.softmax(raw_scores / temperature).numpy()

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    is_danger = pred_idx != NORMAL_IDX and confidence >= threshold

    if not is_danger:
        intensity = 0
    else:
        intensity = 100 if confidence >= 0.7 else 60

    return {
        "class": CLASSES[pred_idx],
        "confidence": round(confidence, 4),
        "is_danger": is_danger,
        "intensity": intensity,
        "all_probs": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py path/to/clip.wav")
        return

    from common import load_wav_16k_mono
    wav = load_wav_16k_mono(sys.argv[1]).numpy()

    model, temperature = load_runtime()
    result = classify_clip(wav, model, temperature)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
