# classifier/adapter.py
import tensorflow as tf
import librosa

CLASSES = ["horn", "siren", "crash", "normal"]
SR = 16000

_model = None
_infer = None


def load_model(path="model/saved_model/danger_sound_classifier"):
    global _model, _infer
    if _infer is None:
        _model = tf.saved_model.load(path)
        _infer = _model.signatures["serving_default"]
    return _infer


def predict(audio_path):
    """오디오 파일 경로 → {클래스: 확신도}"""
    wav, _ = librosa.load(audio_path, sr=SR, mono=True)
    infer = load_model()
    out = infer(audio=tf.constant(wav, dtype=tf.float32))
    logits = out["output_0"].numpy()
    probs = tf.nn.softmax(logits).numpy()
    return {c: float(p) for c, p in zip(CLASSES, probs)}