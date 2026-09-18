"""
Measures inference latency against the 200ms gating-window target.
Runs classify_clip() repeatedly on test clips (or random noise if none
exist yet) and reports mean/p50/p95/max latency.
"""
import time

import numpy as np

from common import BASE_DIR
from inference import classify_clip, load_runtime

N_WARMUP = 3
N_RUNS = 30
TARGET_MS = 200


def get_sample_clips(n):
    test_fp_path = BASE_DIR / "test_filepaths.npy"
    if test_fp_path.exists():
        from common import load_wav_16k_mono
        filepaths = np.unique(np.load(test_fp_path, allow_pickle=True))
        chosen = np.random.choice(filepaths, size=min(n, len(filepaths)), replace=False)
        return [load_wav_16k_mono(fp).numpy() for fp in chosen]
    else:
        print("No test_filepaths.npy yet (run train_yamnet.py first) - "
              "benchmarking on random noise instead (latency only, not accuracy).")
        return [np.random.uniform(-0.1, 0.1, size=int(16000 * 4.5)).astype(np.float32) for _ in range(n)]


def main():
    model, temperature = load_runtime()
    clips = get_sample_clips(N_RUNS)

    for clip in clips[:N_WARMUP]:
        classify_clip(clip, model, temperature)

    latencies_ms = []
    for clip in clips:
        start = time.perf_counter()
        classify_clip(clip, model, temperature)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms = np.array(latencies_ms)
    print(f"n={len(latencies_ms)} runs")
    print(f"mean:  {latencies_ms.mean():.1f} ms")
    print(f"p50:   {np.percentile(latencies_ms, 50):.1f} ms")
    print(f"p95:   {np.percentile(latencies_ms, 95):.1f} ms")
    print(f"max:   {latencies_ms.max():.1f} ms")

    over_target = np.mean(latencies_ms > TARGET_MS) * 100
    print(f"\n{over_target:.0f}% of runs exceeded the {TARGET_MS}ms target")
    if over_target > 5:
        print("-> Consider: smaller Dense head, batching frames, or trimming clip length "
              "fed into YAMNet per gating window.")
    else:
        print("-> Within target.")


if __name__ == "__main__":
    main()
