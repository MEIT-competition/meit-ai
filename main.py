# main.py
import random
from classifier.adapter import predict
from decision.judge import judge
from logger.csv_logger import EventLogger

log = EventLogger()


def process(audio_path, direction=None):
    probs, db = predict(audio_path)
    if direction is None:
        direction = random.randint(0, 7)   # 하드웨어 없어서 임시

    cmd = judge(probs, direction, db)
    top = max(probs, key=probs.get)
    print(f"{audio_path}\n  → {top} {probs[top]:.3f}  ({db:.1f} dBFS)")

    if cmd:
        print(f"  진동: {cmd['pattern_name']} 세기{cmd['intensity']} 방향{cmd['direction']}")
        log.log(cmd["sound_class"], cmd["confidence"], cmd["direction"],
                cmd["intensity"], cmd["pattern_name"])
    else:
        print("  알림 없음")
    return cmd


if __name__ == "__main__":
    import sys, glob
    files = sys.argv[1:] or sorted(glob.glob("data/*/*.wav"))[:5]
    for f in files:
        process(f)