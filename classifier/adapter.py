import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import librosa

CLASSES = ["horn", "siren", "crash", "normal"]
SR = 16000
CLIP_SEC = 2.5

# 어느 폴더에서 실행하든 같은 파일을 찾도록 이 파일 기준 절대경로로 잡는다
_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = _ROOT / "model" / "saved_model" / "danger_sound_classifier"
CALIB_PATH = _ROOT / "model" / "calibration.json"

_infer = None
_temperature = None


def load_model(path=MODEL_PATH):
    global _infer
    if _infer is None:
        _infer = tf.saved_model.load(str(path)).signatures["serving_default"]
    return _infer


def load_temperature():
    global _temperature
    if _temperature is None:
        if CALIB_PATH.exists():
            _temperature = float(json.loads(CALIB_PATH.read_text())["temperature"])
        else:
            _temperature = 1.0
            print("[경고] calibration.json 없음 — 보정 미적용")
    return _temperature


def fit_length(wav, sec=CLIP_SEC):
    """학습·평가와 동일하게 앞에서부터 자름 (짧으면 뒤를 0으로 채움)"""
    n = int(SR * sec)
    if len(wav) < n:
        return np.pad(wav, (0, n - len(wav)))
    return wav[:n]


def measure_db(wav):
    rms = float(np.sqrt(np.mean(wav ** 2)))
    return -100.0 if rms < 1e-10 else float(20 * np.log10(rms))


def predict_array(wav):
    """16kHz mono float32 배열 → ({클래스: 확신도}, dBFS)"""
    wav = np.asarray(wav, dtype=np.float32)
    db = measure_db(wav)
    wav = fit_length(wav)
    out = load_model()(audio=tf.constant(wav, dtype=tf.float32))
    logits = out["output_0"].numpy()
    probs = tf.nn.softmax(logits / load_temperature()).numpy()
    return {c: float(p) for c, p in zip(CLASSES, probs)}, db


def predict(audio_path):
    wav, _ = librosa.load(audio_path, sr=SR, mono=True)
    return predict_array(wav)