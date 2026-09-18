"""
Synthetic background-noise augmentation: mixes each horn/siren/crash clip
with a random normal clip at a few SNR levels, so the model sees noisy
danger sounds even before real motor-noise recordings exist.

Augmented clips inherit their source danger clip's fold (not the normal
clip's) to keep train/val/test leak-free.

Run after prepare_data.py, before train_yamnet.py.
"""
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
MANIFEST_PATH = BASE_DIR / "manifest.csv"

DANGER_CLASSES = ["horn", "siren", "crash"]
SNR_LEVELS_DB = [15, 5]  # light / heavy noise; avoid 0dB, can flip the perceived class
SR = 16000


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-12)


def mix_at_snr(danger, noise, snr_db):
    danger_rms = rms(danger)
    noise_rms = rms(noise)
    target_noise_rms = danger_rms / (10 ** (snr_db / 20))
    scale = target_noise_rms / (noise_rms + 1e-12)
    mixed = danger + noise * scale
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def main():
    random.seed(0)
    rows = list(csv.DictReader(open(MANIFEST_PATH, newline="", encoding="utf-8")))

    normal_rows = [r for r in rows if r["label"] == "normal"]
    danger_rows = [r for r in rows if r["label"] in DANGER_CLASSES]
    if not normal_rows or not danger_rows:
        print("Need both normal and danger clips in manifest.csv first - run prepare_data.py")
        return

    new_rows = []
    for danger_row in danger_rows:
        danger_path = BASE_DIR / danger_row["filepath"]
        danger_audio, sr = sf.read(danger_path)
        assert sr == SR, f"expected {SR}Hz, got {sr} in {danger_path}"

        for snr_db in SNR_LEVELS_DB:
            normal_row = random.choice(normal_rows)
            normal_path = BASE_DIR / normal_row["filepath"]
            normal_audio, _ = sf.read(normal_path)

            if len(normal_audio) < len(danger_audio):
                reps = int(np.ceil(len(danger_audio) / len(normal_audio)))
                normal_audio = np.tile(normal_audio, reps)
            normal_audio = normal_audio[:len(danger_audio)]

            mixed = mix_at_snr(danger_audio, normal_audio, snr_db)

            out_name = f"{danger_row['label']}_aug_snr{snr_db}_{Path(danger_row['filepath']).stem}.wav"
            out_path = DATASET_DIR / danger_row["label"] / out_name
            sf.write(out_path, mixed, SR)

            new_rows.append({
                "filepath": str(out_path.relative_to(BASE_DIR)),
                "label": danger_row["label"],
                "source": f"aug(snr={snr_db}dB,base={danger_row['source']})",
                "fold": danger_row["fold"],
            })

    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "source", "fold"])
        writer.writerows(new_rows)

    print(f"Added {len(new_rows)} noise-augmented clips to manifest.csv "
          f"({len(danger_rows)} danger clips x {len(SNR_LEVELS_DB)} SNR levels)")
    print("Re-run prepare cache: delete embed_cache/*.npz before the next train_yamnet.py run "
          "so the new clips get embedded.")


if __name__ == "__main__":
    main()
