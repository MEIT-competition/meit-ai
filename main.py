from decision.judge import judge
from logger.csv_logger import EventLogger

log = EventLogger()

samples = [
    ({"horn": .05, "siren": .82, "crash": .03, "normal": .10}, 3),
    ({"horn": .55, "siren": .10, "crash": .05, "normal": .30}, 0),
    ({"horn": .05, "siren": .05, "crash": .05, "normal": .85}, 6),  # 무시돼야 함
]

for probs, direction in samples:
    cmd = judge(probs, direction)
    if cmd:
        print(cmd)
        log.log(cmd["sound_class"], cmd["confidence"], cmd["direction"],
                cmd["intensity"], cmd["pattern_name"])