# decision/judge.py
THRESHOLD = 0.4

PATTERNS = {
    "horn":  "double_short",   # 짧게 2번
    "siren": "repeat",         # 반복
    "crash": "single_strong",  # 강하게 1번
}

def judge(probs: dict, direction: int, db: float | None = None):
    """probs: {"horn":0.1,"siren":0.8,"crash":0.05,"normal":0.05}
       direction: 0~7 (0=정면, 시계방향), 판별 불가 시 -1
       반환: None(알림 없음) 또는 진동 명령 dict"""
    top = max(probs, key=probs.get)
    conf = probs[top]

    if top == "normal" or conf < THRESHOLD:
        return None

    intensity = 100 if conf >= 0.7 else 60
    # TODO: db 기반 세기 조절 (전자팀 PWM 매핑 확정 후)

    return {
        "direction": direction,
        "pattern": PATTERNS[top],
        "intensity": intensity,
        "sound_class": top,
        "confidence": conf,
    }