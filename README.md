# meit-ai

MEIT 5회 전공 융합 프로젝트 — 위험음 방향 인지 웨어러블의 AI 파트

마이크로 들어온 소리를 분류하고, 위험도를 판단해 진동 명령을 생성합니다.
AI 연산은 노트북에서 수행하고, 결과를 블루투스로 MCU에 전달합니다.

## 구조

```
meit-ai/
├─ main.py              # 전체 파이프라인 (오디오 → 진동 명령 → 로그)
├─ classifier/
│   └─ adapter.py       # 모델 로드·추론, 4.5초 클립 처리, dBFS 측정
├─ decision/
│   ├─ judge.py         # 위험도 판단 (임계값, 음량 게이트)
│   ├─ intensity.py     # dBFS → 진동 세기 산출
│   └─ patterns.py      # 클래스별 진동 패턴 정의
├─ logger/
│   └─ csv_logger.py    # 감지 이벤트 CSV 기록
├─ eval/                # 성능 평가·오류 분석 스크립트
├─ model/               # 학습된 모델 (SavedModel) + manifest.csv
├─ data/                # 학습·검증 데이터 (gitignore, 별도 공유)
└─ logs/                # 감지 이벤트 CSV (gitignore)
```

## 설치

```bash
pip install tensorflow librosa numpy pandas
```

## 실행

```bash
python main.py 오디오파일.wav
```

여러 파일을 한 번에 넣을 수도 있습니다.

```bash
python main.py data/horn/a.wav data/siren/b.wav
```

## 출력 포맷

위험음으로 판단되면 아래 형태의 진동 명령이 생성됩니다.

```python
{
    "direction": 0,                        # 0~7 (0=정면, 시계방향), 판별 불가 시 -1
    "intensity": 85,                       # 40~100
    "pattern": [[100, 50], [100, 0]],      # [[켜는 ms, 끄는 ms], ...]
    "pattern_name": "siren",
    "sound_class": "siren",
    "confidence": 0.998,
}
```

일상소음이거나 임계값 미달이면 `None`을 반환합니다.

## 클래스

| 클래스 | 설명 | 진동 패턴 |
|---|---|---|
| horn | 차량 경적 | 짧게 2번 |
| siren | 구급차·경찰차·소방차 | 반복 |
| crash | 쾅 소리, 급브레이크, 유리 파열음 | 강하게 1번 |
| normal | 말소리, 음악, 주행음 등 | 알림 없음 |

## 주요 설정값

실측 후 조정이 필요한 임시값입니다.

| 위치 | 값 | 설명 |
|---|---|---|
| `decision/judge.py` | `THRESHOLD = 0.4` | 알림 임계값 |
| `decision/judge.py` | `DB_GATE = -50.0` | 이보다 작은 소리는 분류 생략 |
| `decision/intensity.py` | `DB_MIN = -40, DB_MAX = -10` | dBFS → 세기 매핑 범위 |
| `decision/patterns.py` | `GATING_MS = 300` | 게이팅 주기, 패턴 길이 자동 선택 기준 |

## 성능

검증 데이터(manifest.csv fold=4, 340개) 기준, 임계값 0.4 적용

| 클래스 | Recall |
|---|---|
| horn | 98.3% (57/58) |
| siren | 98.8% (84/85) |
| crash | 100% (43/43) |
| 오탐 | 4/154 (2.6%) |

평가 스크립트

```bash
python -m eval.evaluate_val    # 검증 데이터 평가
python -m eval.evaluate        # 학습 데이터 전체 평가
python -m eval.judge_stats     # 세기 분포, dBFS 범위
```

## 로그

감지 시 `logs/events.csv`에 아래 컬럼으로 기록됩니다.
timestamp, sound_class, confidence, direction, intensity, pattern


## 남은 작업

- 실제 마이크 기준 dBFS 실측 후 매핑 기준 조정
- 실시간 버퍼 방식 (4.5초 유지하며 게이팅 주기마다 판단)
- 실제 방향값 연동
- microSD 로그 저장 연동
