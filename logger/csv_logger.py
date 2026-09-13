# logger/csv_logger.py
import csv
from datetime import datetime
from pathlib import Path

COLUMNS = ["timestamp", "sound_class", "confidence", "direction", "intensity", "pattern"]

class EventLogger:
    def __init__(self, path="logs/events.csv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(COLUMNS)

    def log(self, sound_class, confidence, direction, intensity, pattern):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                sound_class, round(confidence, 3), direction, intensity, pattern
            ])