# 게이팅 주기(ms) 
GATING_MS = 250

# 긴 버전 (게이팅 제약 x)
PATTERNS_FULL = {
    "horn":  [[150, 100], [150, 0]],              # 400ms — 짧게 2번
    "siren": [[300, 150], [300, 150], [300, 0]],  # 1050ms — 반복
    "crash": [[500, 0]],                          # 500ms — 강하게 1번
}

# 짧은 버전 (게이팅 주기 안에 들어가게)
PATTERNS_SHORT = {
    "horn":  [[80, 60], [80, 0]],    # 220ms
    "siren": [[100, 50], [100, 0]],  # 250ms
    "crash": [[250, 0]],             # 250ms
}


def duration(pattern) -> int:
    """패턴 전체 소요 시간(ms)"""
    return sum(on + off for on, off in pattern)


def get_pattern(sound_class: str, gating_ms: int = GATING_MS):
    """게이팅 주기에 맞는 패턴을 반환.
       긴 버전이 주기 안에 들어가면 그대로, 넘으면 짧은 버전."""
    full = PATTERNS_FULL[sound_class]
    if duration(full) <= gating_ms:
        return full
    return PATTERNS_SHORT[sound_class]


if __name__ == "__main__":
    # 패턴 길이 확인용
    for cls in PATTERNS_FULL:
        f, s = PATTERNS_FULL[cls], PATTERNS_SHORT[cls]
        chosen = get_pattern(cls)
        print(f"{cls:6} full={duration(f):5}ms  short={duration(s):4}ms  "
              f"→ {'short' if chosen is s else 'full'} ({duration(chosen)}ms)")