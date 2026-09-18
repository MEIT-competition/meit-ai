import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import librosa

CLASSES = ["horn", "siren", "crash", "normal"]
SR = 16000
CLIP_SEC = 4.5   # 학습과 동일하게 고정

MODEL_PATH = "model/saved_model/danger_sound_classifier"
CALIB_PATH = Path("model/calibration.json")

_infer = None
_temperature = None


def load_model(path=MODEL_PATH):
    global _infer
    if _infer is None:
        _infer = tf.saved_model.load(path).signatures["serving_default"]
    return _infer


def load_temperature():
    """확신도 보정값. model/calibration.py 결과를 그대로 사용."""
    global _temperature
    if _temperature is None:
        if CALIB_PATH.exists():
            _temperature = float(json.loads(CALIB_PATH.read_text())["temperature"])
        else:
            _temperature = 1.0
            print("[경고] calibration.json 없음 — 보정 미적용")
    return _temperature


def fit_length(wav, sec=CLIP_SEC):
    """4.5초로 맞춤 — 짧으면 패딩, 길면 가장 큰 구간 중심으로 자름"""
    n = int(SR * sec)
    if len(wav) == n:
        return wav
    if len(wav) < n:
        return np.pad(wav, (0, n - len(wav)))
    win = SR // 10
    energy = np.convolve(wav ** 2, np.ones(win), mode="same")
    center = int(np.argmax(energy))
    start = max(0, min(center - n // 2, len(wav) - n))
    return wav[start:start + n]


def measure_db(wav):
    """RMS 기반 dBFS"""
    rms = float(np.sqrt(np.mean(wav ** 2)))
    return -100.0 if rms < 1e-10 else float(20 * np.log10(rms))


def predict(audio_path):
    """오디오 파일 경로 → ({클래스: 확신도}, dBFS)"""
    wav, _ = librosa.load(audio_path, sr=SR, mono=True)
    db = measure_db(wav)
    wav = fit_length(wav)

    out = load_model()(audio=tf.constant(wav, dtype=tf.float32))
    logits = out["output_0"].numpy()
    probs = tf.nn.softmax(logits / load_temperature()).numpy()
    return {c: float(p) for c, p in zip(CLASSES, probs)}, db