from pathlib import Path
from collections import Counter
from classifier.adapter import predict
from decision.judge import judge
from logger.csv_logger import EventLogger

log = EventLogger()
CLASSES = ["horn", "siren", "crash", "normal"]

correct = Counter()
total = Counter()
confusion = Counter()   # (실제, 예측) -> 횟수
alerted = Counter()
missed = Counter()

for cls in CLASSES:
    for f in sorted((Path("data") / cls).glob("*.wav")):
        probs, _ = predict(str(f))
        pred = max(probs, key=probs.get)

        total[cls] += 1
        confusion[(cls, pred)] += 1
        if pred == cls:
            correct[cls] += 1

        cmd = judge(probs, direction=0)
        if cls != "normal":
            if cmd:
                alerted[cls] += 1
            else:
                missed[cls] += 1
        else:
            if cmd:
                alerted["normal_false_alarm"] += 1

print("\n=== 클래스별 정확도 ===")
for cls in CLASSES:
    if total[cls]:
        print(f"{cls:7} {correct[cls]:4}/{total[cls]:4}  ({correct[cls]/total[cls]*100:.1f}%)")

print("\n=== 혼동 행렬 (행=실제, 열=예측) ===")
print("        " + "".join(f"{c:>8}" for c in CLASSES))
for a in CLASSES:
    row = "".join(f"{confusion[(a, p)]:>8}" for p in CLASSES)
    print(f"{a:7} {row}")

print("\n=== 임계값 0.4 적용 후 ===")
for cls in ["horn", "siren", "crash"]:
    t = alerted[cls] + missed[cls]
    if t:
        print(f"{cls:7} 알림 {alerted[cls]:4}/{t:4}  (Recall {alerted[cls]/t*100:.1f}%)")
print(f"오탐(normal에 알림): {alerted['normal_false_alarm']}/{total['normal']}")