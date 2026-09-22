# brake_check.py
from pathlib import Path
from classifier.adapter import predict

KEYS = ["brake", "skid", "handbrake", "screech", "tire"]

miss_brake = miss_other = ok_brake = ok_other = 0

for f in sorted((Path("data") / "crash").glob("*.wav")):
    probs, _ = predict(str(f))
    p = probs["crash"]
    is_brake = any(k in f.name.lower() for k in KEYS)
    if p < 0.4:
        if is_brake: miss_brake += 1
        else:        miss_other += 1
    else:
        if is_brake: ok_brake += 1
        else:        ok_other += 1

print(f"급브레이크: 맞춤 {ok_brake}, 놓침 {miss_brake}")
if ok_brake + miss_brake:
    print(f"  Recall {ok_brake/(ok_brake+miss_brake)*100:.1f}%")
print(f"그 외 crash: 맞춤 {ok_other}, 놓침 {miss_other}")
if ok_other + miss_other:
    print(f"  Recall {ok_other/(ok_other+miss_other)*100:.1f}%")