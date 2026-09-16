from pathlib import Path
from datasets import load_dataset, Audio

OUT_DIR = Path(r"C:\Users\rabbi\Desktop\2026\School\competi\MEIT\data\huggingface_urbansound8k_car_horn")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset("danavery/urbansound8K", split="train", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

count = 0
seen = 0
for ex in ds:
    seen += 1
    if ex["class"] != "car_horn":
        continue
    out_path = OUT_DIR / ex["slice_file_name"]
    with open(out_path, "wb") as f:
        f.write(ex["audio"]["bytes"])
    count += 1
    if count % 20 == 0:
        print(f"...{count} car_horn files saved so far (scanned {seen} total rows)")

print(f"DONE. Saved {count} car_horn files to {OUT_DIR} (scanned {seen} total rows)")
