# evaluate_val.py
import pandas as pd
from pathlib import Path
from collections import Counter
from classifier.adapter import predict

df = pd.read_csv("model/manifest.csv")
val = df[df["fold"] == 4]
print(f"검증 데이터 {len(val)}개")
print(val["label"].value_counts(), "\n")

CLASSES = ["horn", "siren", "crash", "normal"]
correct = Counter(); total = Counter(); confusion = Counter()
alerted = Counter(); missed = Counter()
skipped = 0

for _, row in val.iterrows():
    cls = row["label"]
    # manifest는 dataset\ 기준, 실제 파일은 data\ 에 있음
    path = Path(str(row["filepath"]).replace("dataset\\", "data\\").replace("dataset/", "data/"))
    if not path.exists():
        skipped += 1
        continue

    probs, _ = predict(str(path))   # 튜플로 바뀌었음
    pred = max(probs, key=probs.get)

    total[cls] += 1
    confusion[(cls, pred)] += 1
    if pred == cls:
        correct[cls] += 1

    danger = {k: v for k, v in probs.items() if k != "normal"}
    alert = max(danger.values()) >= 0.4
    if cls == "normal":
        if alert: alerted["false_alarm"] += 1
    else:
        if alert: alerted[cls] += 1
        else:     missed[cls] += 1

if skipped:
    print(f"파일 없어서 건너뜀: {skipped}개\n")

print("=== 검증 데이터 정확도 ===")
for c in CLASSES:
    if total[c]:
        print(f"{c:7} {correct[c]:4}/{total[c]:4}  ({correct[c]/total[c]*100:.1f}%)")

print("\n=== 혼동 행렬 (행=실제, 열=예측) ===")
print("        " + "".join(f"{c:>8}" for c in CLASSES))
for a in CLASSES:
    print(f"{a:7} " + "".join(f"{confusion[(a, p)]:>8}" for p in CLASSES))

print("\n=== 임계값 0.4 적용 ===")
for c in ["horn", "siren", "crash"]:
    t = alerted[c] + missed[c]
    if t:
        print(f"{c:7} Recall {alerted[c]/t*100:.1f}%  ({alerted[c]}/{t})")
if total["normal"]:
    print(f"오탐: {alerted['false_alarm']}/{total['normal']} ({alerted['false_alarm']/total['normal']*100:.1f}%)")