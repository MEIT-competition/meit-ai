"""
Add motor on/off paired danger-sound recordings (same sound, mic position
fixed, motors off vs on). Used to add real motor-noise examples to training
and to measure the sim-to-real accuracy gap (see motor_noise_eval.py).

Input: raw_motor_noise/{horn,siren,crash}/{take_id}_off.wav + _on.wav
Output: dataset_motor_noise/{class}/{take_id}_{off,on}.wav, motor_noise_manifest.csv

Only the "_on" clip gets added to the main manifest.csv for training;
"_off" stays as the clean reference for motor_noise_eval.py.
"""
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from prepare_data import CLIP_SEC, TARGET_SR, fold_for

BASE_DIR = Path(__file__).parent
RAW_MOTOR_DIR = BASE_DIR / "raw_motor_noise"
OUT_DIR = BASE_DIR / "dataset_motor_noise"
MOTOR_MANIFEST_PATH = BASE_DIR / "motor_noise_manifest.csv"
MAIN_MANIFEST_PATH = BASE_DIR / "manifest.csv"
CLASSES = ["horn", "siren", "crash"]


def load_trim_pad(path):
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    clip_len = int(CLIP_SEC * TARGET_SR)
    if len(y) < clip_len:
        y = np.pad(y, (0, clip_len - len(y)))
    else:
        y = y[:clip_len]
    return y


def main():
    if not RAW_MOTOR_DIR.exists():
        print(f"Put paired recordings under {RAW_MOTOR_DIR}/{{class}}/{{take_id}}_off.wav and _on.wav first.")
        return

    motor_rows = []
    main_rows_to_add = []

    for label in CLASSES:
        src_dir = RAW_MOTOR_DIR / label
        if not src_dir.exists():
            continue
        dst_dir = OUT_DIR / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        off_files = {p.stem[:-len("_off")]: p for p in src_dir.glob("*_off.wav")}
        on_files = {p.stem[:-len("_on")]: p for p in src_dir.glob("*_on.wav")}

        take_ids = sorted(set(off_files) & set(on_files))
        missing = sorted(set(off_files) ^ set(on_files))
        if missing:
            print(f"[warn] {label}: unpaired takes ignored (missing off or on): {missing}")

        for take_id in take_ids:
            off_audio = load_trim_pad(off_files[take_id])
            on_audio = load_trim_pad(on_files[take_id])

            off_out = dst_dir / f"{take_id}_off.wav"
            on_out = dst_dir / f"{take_id}_on.wav"
            sf.write(off_out, off_audio, TARGET_SR)
            sf.write(on_out, on_audio, TARGET_SR)

            fold = fold_for(take_id)
            motor_rows.append({
                "take_id": take_id, "label": label,
                "off_path": str(off_out.relative_to(BASE_DIR.parent)),
                "on_path": str(on_out.relative_to(BASE_DIR.parent)),
                "fold": fold,
            })
            main_rows_to_add.append({
                "filepath": str(on_out.relative_to(BASE_DIR.parent)),
                "label": label,
                "source": f"motor_noise_on(take={take_id})",
                "fold": fold,
            })

    if not motor_rows:
        print("No complete off/on pairs found - nothing to add.")
        return

    with open(MOTOR_MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["take_id", "label", "off_path", "on_path", "fold"])
        writer.writeheader()
        writer.writerows(motor_rows)

    file_exists = MAIN_MANIFEST_PATH.exists()
    with open(MAIN_MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "source", "fold"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(main_rows_to_add)

    print(f"Added {len(motor_rows)} motor on/off pairs.")
    print(f"  -> {MOTOR_MANIFEST_PATH} (for motor_noise_eval.py)")
    print(f"  -> {len(main_rows_to_add)} '_on' clips appended to manifest.csv (for training)")
    print("Delete embed_cache/*.npz and re-run train_yamnet.py to include these.")


if __name__ == "__main__":
    main()
