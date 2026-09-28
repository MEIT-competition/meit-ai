import glob
import sys

import librosa
import numpy as np

from classifier.adapter import predict_array

SR = 16000
N = 40960

cls = sys.argv[1]
files = sorted(glob.glob(f"data/{cls}/*.wav"))
print(f"{cls}: {len(files)}개 검사 중...\n")

rows = []
for f in files:
    wav, _ = librosa.load(f, sr=SR, mono=True)
    wav = wav[:N] if len(wav) >= N else np.pad(wav, (0, N - len(wav)))
    probs, db = predict_array(wav)
    rows.append((probs[cls], db, f))

rows.sort(reverse=True)
print(f"{'확신도':>7}  {'dBFS':>7}  파일")
for conf, db, f in rows[:10]:
    mark = "" if db > -50 else "  ← 너무 조용함"
    print(f"{conf:7.3f}  {db:7.1f}  {f}{mark}")