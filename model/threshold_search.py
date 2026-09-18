"""
Sweep decision thresholds (danger = argmax != normal AND confidence >=
threshold) and report recall/precision/false-alarms-per-hour at each.
Also simulates N-of-M consecutive-confirmation gating as a reference point
for the actual gating implementation on the decision-logic side.

Run after train_yamnet.py + calibration.py. Appends to ablation_log.csv.
"""
import csv
import datetime
import json

import numpy as np
import tensorflow as tf

from common import ABLATION_LOG_PATH, BASE_DIR, CALIBRATION_PATH, CLASSES, NORMAL_IDX, pool_clip_logits

GATING_WINDOW_SEC = 0.25  # midpoint of the 200-300ms gating cycle, for the frame-level N-of-M sim
CLIP_DURATION_SEC = 4.5   # each pooled row below is one full clip's decision


def calibrated_probs(logits, temperature):
    return tf.nn.softmax(logits / temperature, axis=-1).numpy()


def metrics_at_threshold(probs, labels, threshold, decision_period_sec):
    pred_class = np.argmax(probs, axis=1)
    pred_conf = np.max(probs, axis=1)

    flagged = (pred_class != NORMAL_IDX) & (pred_conf >= threshold)
    true_danger = labels != NORMAL_IDX

    tp = np.sum(flagged & true_danger)
    fn = np.sum(~flagged & true_danger)
    fp = np.sum(flagged & ~true_danger)
    tn = np.sum(~flagged & ~true_danger)

    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

    normal_windows = np.sum(~true_danger)
    fp_rate_per_window = fp / normal_windows if normal_windows > 0 else float("nan")
    windows_per_hour = 3600 / decision_period_sec
    false_alarms_per_hour = fp_rate_per_window * windows_per_hour

    return {
        "recall": recall, "precision": precision, "f1": f1,
        "false_alarm_per_hour": false_alarms_per_hour,
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }


def simulate_consecutive_confirmation(probs, labels, clip_id, threshold, n_required, m_window):
    """Flags a clip only if n_required of the last m_window frames agree
    on a non-normal class above threshold. Approximate - the real gating
    logic lives elsewhere."""
    pred_class = np.argmax(probs, axis=1)
    pred_conf = np.max(probs, axis=1)
    is_hit = (pred_class != NORMAL_IDX) & (pred_conf >= threshold)

    unique_clips = np.unique(clip_id)
    flagged_clip = {}
    true_danger_clip = {}
    for cid in unique_clips:
        mask = clip_id == cid
        hits = is_hit[mask]
        confirmed = False
        for i in range(len(hits)):
            window = hits[max(0, i - m_window + 1): i + 1]
            if window.sum() >= n_required:
                confirmed = True
                break
        flagged_clip[cid] = confirmed
        true_danger_clip[cid] = bool((labels[mask] != NORMAL_IDX).any())

    flagged = np.array([flagged_clip[c] for c in unique_clips])
    true_danger = np.array([true_danger_clip[c] for c in unique_clips])

    tp = np.sum(flagged & true_danger)
    fn = np.sum(~flagged & true_danger)
    fp = np.sum(flagged & ~true_danger)
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {"recall": recall, "precision": precision, "tp": int(tp), "fn": int(fn), "fp": int(fp)}


def append_ablation_rows(rows):
    file_exists = ABLATION_LOG_PATH.exists()
    with open(ABLATION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "date", "experiment", "parameter", "value", "recall", "precision",
            "f1", "false_alarm_per_hour", "notes"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    test_logits = np.load(BASE_DIR / "test_logits.npy")
    test_labels = np.load(BASE_DIR / "test_labels.npy")
    test_clip_id = np.load(BASE_DIR / "test_clip_id.npy")

    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            temperature = json.load(f)["temperature"]
    else:
        print("No calibration.json found - run calibration.py first. Using T=1.0 for now.")
        temperature = 1.0

    # clip-level: the deployed model pools a clip's frames into one decision
    # before a threshold is ever applied, so that's what the sweep must reflect
    pooled_logits, pooled_labels, _, _ = pool_clip_logits(test_logits, test_labels, test_clip_id)
    pooled_probs = calibrated_probs(pooled_logits, temperature)
    frame_probs = calibrated_probs(test_logits, temperature)  # for the N-of-M sim below

    print(f"\n{'threshold':>10} {'recall':>8} {'precision':>10} {'f1':>8} {'false_alarms/hr':>16}")
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rows = []
    best = None
    for t in thresholds:
        m = metrics_at_threshold(pooled_probs, pooled_labels, t, CLIP_DURATION_SEC)
        print(f"{t:>10.2f} {m['recall']:>8.3f} {m['precision']:>10.3f} {m['f1']:>8.3f} {m['false_alarm_per_hour']:>16.2f}")
        rows.append({
            "date": datetime.date.today().isoformat(),
            "experiment": "threshold_sweep",
            "parameter": "threshold",
            "value": t,
            "recall": round(m["recall"], 4),
            "precision": round(m["precision"], 4),
            "f1": round(m["f1"], 4),
            "false_alarm_per_hour": round(m["false_alarm_per_hour"], 2),
            "notes": "",
        })
        # recall >= 0.90 is the hard constraint; prefer the highest threshold meeting it
        if m["recall"] >= 0.90:
            if best is None or t > best[0]:
                best = (t, m)

    if best:
        print(f"\nRecommended threshold: {best[0]:.2f} "
              f"(recall={best[1]['recall']:.3f}, false_alarms/hr={best[1]['false_alarm_per_hour']:.2f})")
    else:
        print("\nNo threshold reached Recall >= 0.90 on this test split - "
              "need more/better data for the classes with low recall (check classification_report).")

    print("\n--- N-of-M consecutive confirmation (experimental) ---")
    print(f"{'n_of_m':>10} {'threshold':>10} {'recall':>8} {'precision':>10}")
    for n_required, m_window in [(1, 1), (2, 3), (3, 5)]:
        m = simulate_consecutive_confirmation(frame_probs, test_labels, test_clip_id, 0.4, n_required, m_window)
        print(f"{f'{n_required}/{m_window}':>10} {'0.40':>10} {m['recall']:>8.3f} {m['precision']:>10.3f}")
        rows.append({
            "date": datetime.date.today().isoformat(),
            "experiment": "consecutive_confirmation",
            "parameter": "n_of_m",
            "value": f"{n_required}/{m_window}",
            "recall": round(m["recall"], 4),
            "precision": round(m["precision"], 4),
            "f1": "",
            "false_alarm_per_hour": "",
            "notes": "threshold=0.40, experimental only",
        })

    append_ablation_rows(rows)
    print(f"\nAppended results to {ABLATION_LOG_PATH}")


if __name__ == "__main__":
    main()
