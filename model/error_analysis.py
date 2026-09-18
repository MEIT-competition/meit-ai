"""
Lists misclassified test clips (missed danger sounds first) with filepath,
true/predicted label, and calibrated confidence, for manual review.

Run after train_yamnet.py + calibration.py.
Writes error_analysis.csv - fill in the "reason" column by ear.
"""
import csv
import json

import numpy as np
import tensorflow as tf

from common import BASE_DIR, CALIBRATION_PATH, CLASSES, NORMAL_IDX, pool_clip_logits

OUT_PATH = BASE_DIR / "error_analysis.csv"


def main():
    test_logits = np.load(BASE_DIR / "test_logits.npy")
    test_labels = np.load(BASE_DIR / "test_labels.npy")
    test_filepaths = np.load(BASE_DIR / "test_filepaths.npy", allow_pickle=True)
    test_clip_id = np.load(BASE_DIR / "test_clip_id.npy")

    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            temperature = json.load(f)["temperature"]
    else:
        temperature = 1.0

    # clip-level: one row per clip, matching the deployed model's pooled decision
    pooled_logits, pooled_labels, clip_ids, extra = pool_clip_logits(
        test_logits, test_labels, test_clip_id, extra={"filepath": test_filepaths})
    probs = tf.nn.softmax(pooled_logits / temperature, axis=-1).numpy()
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)

    rows = []
    for i in range(len(pooled_labels)):
        true_idx, pred_idx = pooled_labels[i], pred[i]
        if true_idx == pred_idx:
            continue
        is_dangerous_miss = true_idx != NORMAL_IDX and pred_idx == NORMAL_IDX
        rows.append({
            "priority": "MISSED_DANGER" if is_dangerous_miss else "false_positive_or_confused",
            "clip_id": int(clip_ids[i]),
            "filepath": extra["filepath"][i],
            "true_label": CLASSES[true_idx],
            "predicted_label": CLASSES[pred_idx],
            "confidence": round(float(conf[i]), 4),
            "reason": "",
        })

    rows.sort(key=lambda r: (r["priority"] != "MISSED_DANGER", -r["confidence"]))

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "priority", "clip_id", "filepath", "true_label", "predicted_label",
            "confidence", "reason"])
        writer.writeheader()
        writer.writerows(rows)

    n_missed = sum(1 for r in rows if r["priority"] == "MISSED_DANGER")
    print(f"{len(rows)} misclassified clips ({n_missed} missed-danger cases) written to {OUT_PATH}")
    print("Listen to the MISSED_DANGER rows first and fill in the 'reason' column.")


if __name__ == "__main__":
    main()
