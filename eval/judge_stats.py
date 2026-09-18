# eval/judge_stats.py
from pathlib import Path
from collections import Counter
from classifier.adapter import predict
from decision.judge import judge

stats = Counter()
for cls in ["horn", "siren", "crash", "normal"]:
    for f in sorted((Path("data") / cls).glob("*.wav")):
        cmd = judge(predict(str(f)), direction=0)
        if cmd:
            stats[(cmd["pattern_name"], cmd["intensity"])] += 1
        else:
            stats[("없음", 0)] += 1

for k, v in stats.most_common():
    print(k, v)