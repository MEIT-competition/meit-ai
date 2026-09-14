# 각 패턴은 [on_ms, off_ms] 쌍의 리스트
PATTERNS = {
    "horn":  [[150, 100], [150, 0]],              # 짧게 2번
    "siren": [[300, 150], [300, 150], [300, 0]],  # 반복
    "crash": [[500, 0]],                          # 강하게 1번
}

def total_duration(sound_class: str) -> int:
    """패턴 전체 소요 시간(ms) — 게이팅 주기 검토용"""
    return sum(on + off for on, off in PATTERNS[sound_class])