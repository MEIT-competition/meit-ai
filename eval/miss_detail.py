# miss_detail.py
from pathlib import Path
from collections import Counter
from classifier.adapter import predict

KEYS = ["brake", "skid", "handbrake", "screech", "tire"]
src = Counter()
pred_to = Counter()

for f in sorted((Path("data") / "crash").glob("*.wav")):
    probs = predict(str(f))
    if probs["crash"] >= 0.4:
        continue
    name = f.name.lower()
    if any(k in name for k in KEYS):
        continue

    # 증강 여부
    src["증강(aug)" if "aug" in name else "원본"] += 1
    # 출처
    for tag in ["esc50", "freesound", "fsd", "pixabay"]:
        if tag in name:
            src[tag] += 1
            break
    else:
        src["기타출처"] += 1
    # 뭘로 착각했는지
    pred_to[max(probs, key=probs.get)] += 1

print("놓친 96개 구성:")
for k, v in src.most_common():
    print(f"  {k}: {v}")
print("\n무엇으로 분류됐나:")
for k, v in pred_to.most_common():
    print(f"  {k}: {v}")