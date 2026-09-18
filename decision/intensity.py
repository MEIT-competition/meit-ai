# 데시벨(dBFS) → 진동 세기 매핑
# 임시값. 하드웨어 도착 후 실측하여 조정 필요
DB_MIN = -40.0   # 이 이하는 최소 세기
DB_MAX = -10.0   # 이 이상은 최대 세기
INTENSITY_MIN = 40
INTENSITY_MAX = 100

LOW_CONF_RATIO = 0.6   # 확신도 0.7 미만일 때 감쇠


def db_to_intensity(db):
    if db is None:
        return INTENSITY_MAX          # 데시벨 없으면 최대로 (위음성 최소화)
    ratio = (db - DB_MIN) / (DB_MAX - DB_MIN)
    ratio = max(0.0, min(1.0, ratio))
    return int(INTENSITY_MIN + ratio * (INTENSITY_MAX - INTENSITY_MIN))


def compute_intensity(conf, db):
    base = db_to_intensity(db)
    if conf < 0.7:
        base = int(base * LOW_CONF_RATIO)
    return max(base, INTENSITY_MIN)