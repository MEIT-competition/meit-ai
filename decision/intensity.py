DB_MIN = -40.0
DB_MAX = -10.0
INTENSITY_MIN = 40
INTENSITY_MAX = 100


def db_to_intensity(db):
    if db is None:
        return INTENSITY_MAX
    ratio = (db - DB_MIN) / (DB_MAX - DB_MIN)
    ratio = max(0.0, min(1.0, ratio))
    return int(INTENSITY_MIN + ratio * (INTENSITY_MAX - INTENSITY_MIN))


def compute_intensity(conf, db):
    """세기는 dBFS만으로 결정.
    확신도는 알릴지 말지만 판단한다(decision/judge.py의 THRESHOLD).
    """
    return db_to_intensity(db)