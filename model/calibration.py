"""
Confidence calibration (temperature scaling) for the danger-sound classifier.

Fits a single scalar T on validation logits to minimize NLL; softmax(logits/T)
becomes the calibrated confidence. Reports ECE before/after.

Run after train_yamnet.py. Writes calibration.json.
"""
import json

import numpy as np
import tensorflow as tf

from common import BASE_DIR, CALIBRATION_PATH, CLASSES, pool_clip_logits


def expected_calibration_error(probs, labels, n_bins=10):
    """ECE: weighted average gap between confidence and accuracy across bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not np.any(in_bin):
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return ece


class TemperatureScaler(tf.Module):
    def __init__(self):
        super().__init__()
        self.temperature = tf.Variable(1.0, trainable=True, dtype=tf.float32)

    def calibrated_probs(self, logits):
        return tf.nn.softmax(logits / self.temperature, axis=-1)

    def fit(self, logits, labels, epochs=300, lr=0.01):
        logits = tf.constant(logits, dtype=tf.float32)
        labels = tf.constant(labels, dtype=tf.int64)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        for _ in range(epochs):
            with tf.GradientTape() as tape:
                scaled = logits / self.temperature
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(labels, scaled, from_logits=True)
                )
            grads = tape.gradient(loss, [self.temperature])
            optimizer.apply_gradients(zip(grads, [self.temperature]))

        return float(self.temperature.numpy())


def main():
    val_logits = np.load(BASE_DIR / "val_logits.npy")
    val_labels = np.load(BASE_DIR / "val_labels.npy")
    val_clip_id = np.load(BASE_DIR / "val_clip_id.npy")
    val_logits, val_labels, _, _ = pool_clip_logits(val_logits, val_labels, val_clip_id)

    raw_probs = tf.nn.softmax(val_logits, axis=-1).numpy()
    ece_before = expected_calibration_error(raw_probs, val_labels)

    scaler = TemperatureScaler()
    T = scaler.fit(val_logits, val_labels)
    calibrated_probs = scaler.calibrated_probs(tf.constant(val_logits, dtype=tf.float32)).numpy()
    ece_after = expected_calibration_error(calibrated_probs, val_labels)

    print(f"Temperature T = {T:.4f}  (T>1 means the raw model was overconfident)")
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration:  {ece_after:.4f}")

    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump({"temperature": T, "ece_before": ece_before, "ece_after": ece_after,
                    "classes": CLASSES}, f, indent=2)
    print(f"Saved {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
