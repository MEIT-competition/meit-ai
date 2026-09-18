# eval/conf_dist.py
from pathlib import Path
from collections import Counter
from classifier.adapter import predict

bins = Counter()
for cls in ["horn", "siren", "crash", "normal"]:
    for f in sorted((Path("data") / cls).glob("*.wav")):
        probs, _ = predict(str(f))
        top = max(probs, key=probs.get)
        if top == "normal":
            continue
        c = probs[top]
        key = "0.9+" if c >= 0.9 else f"{int(c*10)/10:.1f}"
        bins[key] += 1

for k in sorted(bins):
    print(f"{k}: {bins[k]}")