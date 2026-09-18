from pathlib import Path
from collections import Counter
from classifier.adapter import predict
from decision.judge import judge

stats = Counter()
dbs = []

for cls in ["horn", "siren", "crash", "normal"]:
    for f in sorted((Path("data") / cls).glob("*.wav")):
        probs, db = predict(str(f))
        dbs.append(db)
        cmd = judge(probs, 0, db)
        stats[cmd["intensity"] if cmd else "없음"] += 1

print("세기 분포:")
for k, v in sorted(stats.items(), key=lambda x: str(x[0])):
    print(f"  {k}: {v}")

print(f"\ndBFS 범위: {min(dbs):.1f} ~ {max(dbs):.1f}")
print(f"평균: {sum(dbs)/len(dbs):.1f}")