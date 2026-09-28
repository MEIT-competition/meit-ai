"""
Laptop mic fallback for the live demo: listens on the default input
device, captures CLIP_SEC seconds per loud event via mic_capture.py, and
runs the full classify -> judge pipeline on it.

For use when the belt's own mics aren't working. Ctrl+C to stop.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classifier.adapter import predict_array
from decision.judge import judge
from mic_capture import CLIP_SEC, TRIGGER_DB, capture_events


def main():
    print(f"노트북 마이크로 듣는 중... (트리거 {TRIGGER_DB} dBFS, Ctrl+C로 종료)")
    for clip, level in capture_events():
        print(f"\n[이벤트 감지] {level:.1f} dBFS - {CLIP_SEC}초 캡처 중...")

        probs, db = predict_array(clip)
        top = max(probs, key=probs.get)
        print(f"  → {top} {probs[top]:.3f}  ({db:.1f} dBFS)")

        cmd = judge(probs, direction=-1, db=db)
        if cmd:
            print(f"  진동: {cmd['pattern_name']} 세기{cmd['intensity']}")
        else:
            print("  알림 없음")


if __name__ == "__main__":
    main()
