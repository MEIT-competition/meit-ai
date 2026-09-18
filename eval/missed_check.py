# missed_check.py
from pathlib import Path
from classifier.adapter import predict

vals = []
for f in sorted((Path("data") / "crash").glob("*.wav")):
    p = predict(str(f))
    if p["crash"] < 0.4:
        vals.append((p["crash"], f.name))

vals.sort(reverse=True)
print(f"놓친 개수: {len(vals)}\n")

for lo, hi in [(0.3, 0.4), (0.2, 0.3), (0.1, 0.2), (0.0, 0.1)]:
    n = sum(1 for v, _ in vals if lo <= v < hi)
    print(f"{lo:.1f}~{hi:.1f}: {n}개")

print("\n상위 10개:")
for v, name in vals[:10]:
    print(f"  {v:.3f}  {name}")