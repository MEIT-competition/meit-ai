import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from classifier.adapter import predict_array

SR = 16000
N = 40960          # CLIP_FRAMES=120 기준 (120 × 1024 ÷ 3), 2.56초

src, name = sys.argv[1], sys.argv[2]

wav, _ = librosa.load(src, sr=SR, mono=True)
wav = wav[:N] if len(wav) >= N else np.pad(wav, (0, N - len(wav)))

probs, db = predict_array(wav)
top = max(probs, key=probs.get)
print(f"{name}: {top} {probs[top]:.3f}  ({db:.1f} dBFS)  ← {Path(src).name}")

sf.write(f"{name}_16k.wav", wav, SR, subtype="PCM_16")

pcm = np.clip(wav * 32768, -32768, 32767).astype("<i2")
with open(f"{name}_pcm.h", "w", encoding="utf-8") as f:
    f.write("#include <stdint.h>\n\n")
    f.write(f"// {name}: 16 kHz mono PCM16, {len(pcm)} samples (2.56 s)\n")
    f.write(f"// source: {Path(src).name}\n")
    f.write(f"// meit-ai result: {top} conf={probs[top]:.3f}, {db:.1f} dBFS\n")
    f.write(f"static const int16_t {name}_pcm[{len(pcm)}] = {{\n")
    for i in range(0, len(pcm), 12):
        f.write("  " + ", ".join(str(v) for v in pcm[i:i + 12]) + ",\n")
    f.write("};\n")

print(f"→ {name}_16k.wav, {name}_pcm.h")