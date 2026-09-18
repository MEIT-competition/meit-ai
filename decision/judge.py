from decision.intensity import compute_intensity
from decision.patterns import get_pattern

THRESHOLD = 0.4
DB_GATE = -50.0  # 임시값, 실측 후 조정


def judge(probs: dict, direction: int, db: float | None = None):
    """probs: {"horn":0.1,"siren":0.8,"crash":0.05,"normal":0.05}
       direction: 0~7 (0=정면, 시계방향), 판별 불가 시 -1
       db: 수음 음량(dB), None이면 게이트 미적용"""
    if db is not None and db < DB_GATE:
        return None

    top = max(probs, key=probs.get)
    conf = probs[top]

    if top == "normal" or conf < THRESHOLD:
        return None

    return {
        "direction": direction,
        "pattern": get_pattern(top),
        "pattern_name": top,
        "intensity": compute_intensity(conf, db),
        "sound_class": top,
        "confidence": conf,
    }