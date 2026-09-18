"""
Measures inference latency at different buffer lengths (see
buffer_length_eval.py) to confirm a shorter buffer doesn't add latency risk
on top of the recall/precision tradeoff.

Run after train_yamnet.py + calibration.py.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference import classify_clip, load_runtime

WINDOW_SECS = [2.0, 2.5, 4.5]
N_WARMUP = 3
N_RUNS = 30
SR = 16000


def main():
    model, temperature = load_runtime()

    for window_sec in WINDOW_SECS:
        n = int(window_sec * SR)
        clips = [np.random.uniform(-0.1, 0.1, size=n).astype(np.float32) for _ in range(N_RUNS)]

        for c in clips[:N_WARMUP]:
            classify_clip(c, model, temperature)

        latencies_ms = []
        for c in clips:
            start = time.perf_counter()
            classify_clip(c, model, temperature)
            latencies_ms.append((time.perf_counter() - start) * 1000)

        latencies_ms = np.array(latencies_ms)
        print(f"window={window_sec}s  mean={latencies_ms.mean():.1f}ms  "
              f"p50={np.percentile(latencies_ms, 50):.1f}ms  "
              f"p95={np.percentile(latencies_ms, 95):.1f}ms  max={latencies_ms.max():.1f}ms")


if __name__ == "__main__":
    main()
