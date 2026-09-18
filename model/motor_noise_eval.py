"""
For each off/on recording pair (from add_motor_noise_data.py), runs the
model on both and reports whether motor noise flips the prediction and
how much confidence drops. Writes motor_noise_eval.csv.

Run after add_motor_noise_data.py and train_yamnet.py/calibration.py.
"""
import csv
import json

from common import BASE_DIR, CALIBRATION_PATH, load_wav_16k_mono
from inference import classify_clip, load_runtime

MOTOR_MANIFEST_PATH = BASE_DIR / "motor_noise_manifest.csv"
OUT_PATH = BASE_DIR / "motor_noise_eval.csv"


def main():
    if not MOTOR_MANIFEST_PATH.exists():
        print(f"{MOTOR_MANIFEST_PATH} not found - run add_motor_noise_data.py first.")
        return

    model, temperature = load_runtime()
    rows = list(csv.DictReader(open(MOTOR_MANIFEST_PATH, newline="", encoding="utf-8")))

    results = []
    n_flipped = 0
    conf_drops = []
    for r in rows:
        off_wav = load_wav_16k_mono(str(BASE_DIR.parent / r["off_path"])).numpy()
        on_wav = load_wav_16k_mono(str(BASE_DIR.parent / r["on_path"])).numpy()

        off_result = classify_clip(off_wav, model, temperature)
        on_result = classify_clip(on_wav, model, temperature)

        flipped = off_result["class"] != on_result["class"]
        conf_drop = off_result["confidence"] - on_result["confidence"]
        n_flipped += int(flipped)
        conf_drops.append(conf_drop)

        results.append({
            "take_id": r["take_id"],
            "label": r["label"],
            "off_pred": off_result["class"],
            "off_confidence": off_result["confidence"],
            "on_pred": on_result["class"],
            "on_confidence": on_result["confidence"],
            "confidence_drop": round(conf_drop, 4),
            "prediction_flipped": flipped,
        })

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    avg_drop = sum(conf_drops) / len(conf_drops)
    print(f"{len(results)} pairs evaluated")
    print(f"predictions flipped by motor noise: {n_flipped}/{len(results)} ({100*n_flipped/len(results):.1f}%)")
    print(f"average confidence drop (off -> on): {avg_drop:.4f}")
    print(f"Details written to {OUT_PATH}")
    if n_flipped > 0:
        print("\n-> Flipped cases are worth fixing first: add more motor-noise "
              "training data for those classes, or revisit mic-motor isolation.")


if __name__ == "__main__":
    main()
