"""
마이크 이벤트 캡처만 따로 뺀 것. classifier/decision 의존성 없음 - 나중에
ble_receiver.py의 BLE 오디오 수신 부분과 자리만 바꿔 끼우면 되게.

capture_events()는 블로킹이라 asyncio 루프에서 쓰려면 스레드로 돌려야 함.
"""
import time
from typing import Iterator, Tuple

import numpy as np
import sounddevice as sd

SR = 16000
CLIP_SEC = 2.5             # meit-ai's classifier.adapter.CLIP_SEC
TRIGGER_DB = -35.0         # 이 이상 크기의 소리가 나면 이벤트로 판단 (환경 소음 보고 조정)
COOLDOWN_SEC = 1.5         # 이벤트 감지 후 다시 감지하기까지 최소 대기시간
CHECK_WINDOW_SEC = 0.25    # 음량 체크 주기


def rms_dbfs(block: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)) + 1e-12)
    return 20 * np.log10(rms)


def _read_seconds(stream: sd.InputStream, sec: float, sr: int) -> np.ndarray:
    n = int(sec * sr)
    frames, read = [], 0
    while read < n:
        block, _ = stream.read(min(1024, n - read))
        frames.append(block[:, 0])
        read += len(block)
    return np.concatenate(frames)[:n]


def capture_events(
    trigger_db: float = TRIGGER_DB,
    cooldown_sec: float = COOLDOWN_SEC,
    clip_sec: float = CLIP_SEC,
    check_window_sec: float = CHECK_WINDOW_SEC,
    sr: int = SR,
) -> Iterator[Tuple[np.ndarray, float]]:
    """블로킹 제너레이터. 이벤트(큰 소리)가 감지될 때마다
    (clip_sec초짜리 float32 클립, 감지된 dBFS) 튜플을 하나씩 내놓는다.
    외부에서 break/return으로 멈추기 전까지 계속 돈다."""
    check_n = int(check_window_sec * sr)

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=check_n) as stream:
        last_trigger = 0.0
        while True:
            block, _ = stream.read(check_n)
            level = rms_dbfs(block)
            now = time.time()
            if level >= trigger_db and (now - last_trigger) > cooldown_sec:
                last_trigger = now
                # 트리거된 블록부터 이어서 clip_sec초를 채움 (이벤트 시작점부터 잘림)
                rest = _read_seconds(stream, clip_sec - check_window_sec, sr)
                clip = np.concatenate([block[:, 0], rest])[:int(clip_sec * sr)].astype(np.float32)
                yield clip, level
