# meit-ai

MEIT 5회 전공 융합 프로젝트 — 팀 **팔방송이** · 위험음 방향 인지 웨어러블의 **AI 파트**

허리띠 마이크로 들어온 소리를 분류하고, 위험도를 판단해 진동 명령을 만듭니다.
AI 연산은 노트북에서 수행하고, 결과를 블루투스로 MCU에 전달합니다.

```
마이크 4채널
  └→ [전자] 게이팅 주기마다 TDoA 방향 추정 → 해당 구간을 분류 대상으로 전달
       └→ [AI/분류] YAMNet 임베딩 → 커스텀 분류층 → {클래스, 확신도}
            └→ [AI/판단] 위험도 판단 → 진동 세기·패턴 결정 → 모터 구동 + 로그
```

---

## 구조

```
meit-ai/
├─ main.py              # 전체 파이프라인 (오디오 → 진동 명령 → 로그)
├─ classifier/
│   └─ adapter.py       # 모델 로드·추론, 확신도 보정, 4.5초 클립 처리, dBFS 측정
├─ decision/
│   ├─ judge.py         # 위험도 판단 (임계값, 음량 게이트)
│   ├─ intensity.py     # dBFS → 진동 세기 산출
│   └─ patterns.py      # 클래스별 진동 패턴 정의
├─ logger/
│   └─ csv_logger.py    # 감지 이벤트 CSV 기록
├─ eval/                # 파이프라인 성능 평가·오류 분석 스크립트
├─ model/               # 모델 학습 파이프라인 + 배포용 SavedModel (담당: 이주영)
├─ data/                # 학습·검증 클립 (gitignore, 별도 공유)
└─ logs/                # 감지 이벤트 CSV (gitignore)
```

담당 구분 — `model/`은 분류 모델 학습·검증(이주영), 나머지는 판단 로직·통합(이예은).

---

## 설치

```bash
pip install tensorflow librosa numpy pandas
```

모델을 직접 재학습하려면 `model/requirements.txt`를 쓰세요 (tensorflow-hub, scikit-learn, soundfile 포함).

---

## 실행

```bash
python main.py 오디오파일.wav
```

여러 파일을 한 번에 넣을 수도 있습니다.

```bash
python main.py data/horn/a.wav data/siren/b.wav
```

인자를 생략하면 `data/*/*.wav` 중 앞 5개를 처리합니다.
방향값은 하드웨어 연동 전이라 현재 임의값(0~7)이 들어갑니다.

---

## 출력 포맷

위험음으로 판단되면 아래 형태의 진동 명령이 생성됩니다. 전자팀은 이 형식을 받아 모터를 구동합니다.

```python
{
    "direction": 0,                        # 0~7 (0=정면, 시계방향), 판별 불가 시 -1
    "intensity": 85,                       # 40~100
    "pattern": [[100, 50], [100, 0]],      # [[켜는 ms, 끄는 ms], ...]
    "pattern_name": "siren",
    "sound_class": "siren",
    "confidence": 0.949,
}
```

일상소음이거나 임계값·음량 게이트에 걸리면 `None`을 반환합니다.

---

## 클래스

예선 기획안의 "돌발상황"은 정의가 모호하다는 심사 피드백(#4)을 받아 `crash`로 구체화했습니다.

| 클래스 | 설명 | 진동 패턴 |
|---|---|---|
| `horn` | 차량 경적 | 짧게 2번 |
| `siren` | 구급차·경찰차·소방차 | 반복 |
| `crash` | 쾅 소리, 급브레이크, 유리 파열음 | 강하게 1번 |
| `normal` | 말소리, 음악, 주행음 등 | 알림 없음 |

패턴은 게이팅 주기(`GATING_MS`)에 맞춰 긴 버전/짧은 버전이 자동 선택됩니다.
현재 300ms 기준으로는 세 클래스 모두 짧은 버전(220~250ms)이 선택됩니다.

```bash
python -m decision.patterns   # 패턴별 길이와 선택 결과 확인
```

---

## 확신도 보정

`model/calibration.json`의 temperature(2.2445)를 softmax 이전에 적용합니다.
신경망은 과신하는 경향이 있어, 확신도 수치가 실제 정확도를 반영하도록 보정한 값입니다.
파일이 없으면 T=1.0으로 동작하며 경고를 출력합니다.

`model/threshold_search.py`가 검증한 임계값 0.4는 보정된 확신도 기준이므로,
보정을 건너뛰면 튜닝된 적 없는 동작점에서 동작하게 됩니다.

---

## 주요 설정값

실측 후 조정이 필요한 임시값입니다.

| 위치 | 값 | 설명 |
|---|---|---|
| `decision/judge.py` | `THRESHOLD = 0.4` | 알림 임계값 |
| `decision/judge.py` | `DB_GATE = -50.0` | 이보다 작은 소리는 분류 생략 |
| `decision/intensity.py` | `DB_MIN = -40, DB_MAX = -10` | dBFS → 세기 40~100 매핑 범위 |
| `decision/intensity.py` | `LOW_CONF_RATIO = 0.6` | 확신도 0.7 미만일 때 감쇠 비율 |
| `decision/patterns.py` | `GATING_MS = 300` | 게이팅 주기, 패턴 길이 선택 기준 |

단위는 **dBFS**(디지털 기준, 음수)입니다. 일상에서 쓰는 dBSPL(양수)과 다르므로
전자팀과 값을 주고받을 때 단위를 맞춰야 합니다.

---

## 성능

### 분류 모델 (test set = fold 4, 340 클립, 클립 단위)

| 클래스 | Precision | Recall | F1 |
|---|---|---|---|
| horn | 0.875 | 0.966 | 0.918 |
| siren | 1.000 | 0.953 | 0.976 |
| crash | 0.907 | 0.907 | 0.907 |
| normal | 0.987 | 0.974 | 0.980 |

전체 정확도 95.9% · 추론 지연 평균 16.6ms / 최대 18.4ms (200ms 목표 통과)

### 파이프라인 (같은 fold 4, 임계값 0.4 적용, "알림이 울렸는가" 기준)

| 지표 | 값 |
|---|---|
| danger Recall | 98.9% (184/186) |
| danger Precision | 97.9% |
| 오탐 | 4/154 (2.6%) |

목표였던 **Recall 90% 이상**을 달성했습니다.
위 두 표는 채점 기준이 다릅니다 — 앞은 최상위 클래스가 맞았는지(argmax), 뒤는 알림이 울렸는지(binary).
binary 기준 수치는 양쪽이 정확히 일치합니다.

```bash
python -m eval.evaluate_val    # 검증 데이터(fold 4) 평가
python -m eval.evaluate        # data/ 전체 평가
python -m eval.judge_stats     # 세기 분포, dBFS 범위
python -m eval.conf_dist       # 확신도 분포
```

> `data/` 폴더에는 중단된 증강 실험의 잔여 파일(`aug_snr5`, `aug_snr15`)이 남아 있을 수 있습니다.
> 현재 `manifest.csv`(1,739 클립)에는 증강 파일이 포함되어 있지 않으므로,
> 성능 수치는 `manifest.csv` 기준인 `eval.evaluate_val`을 기준으로 보세요.

---

## 모델 학습 (`model/`)

재학습이 필요할 때만 씁니다. 자세한 절차는 `model/MANUAL.md`, 학습 경과와 결정 사항은 `model/TRAINING_REPORT.md`에 있습니다.

| 파일 | 역할 |
|---|---|
| `prepare_data.py` | 원본 → 16kHz mono 4.5초 클립 + `manifest.csv` |
| `augment_data.py` | 배경소음 증강 (현재 미적용 — SNR 과다로 성능 하락) |
| `train_yamnet.py` | YAMNet 임베딩 + Dense head 학습 → SavedModel export |
| `calibration.py` | Temperature scaling → `calibration.json` |
| `threshold_search.py` | 임계값 스윕 + N-of-M 연속확인 실험 |
| `error_analysis.py` | 오분류 사례 → `error_analysis.csv` |
| `latency_benchmark.py` | 추론 지연 측정 |
| `inference.py` | 전자팀 전달용 단일 클립 추론 래퍼 |

구조: YAMNet(고정) → Dense 512 → Dense 4.
클립 내 프레임 중 **peak logit이 가장 높은 프레임**을 클립의 최종 판단으로 씁니다 (max pooling).
짧은 충돌음이 무음 프레임에 희석되는 문제 때문에 mean pooling에서 바꾼 것입니다.

---

## 로그

감지 시 `logs/events.csv`에 아래 컬럼으로 기록됩니다.

```
timestamp, sound_class, confidence, direction, intensity, pattern
```

---

## 남은 작업

- [ ] 실제 MEMS 마이크 기준 dBFS 실측 후 `DB_MIN`/`DB_MAX`/`DB_GATE` 조정
- [ ] 실시간 버퍼 방식 — 최근 4.5초를 유지하면서 게이팅 주기마다 판단
- [ ] 연속 확인(N-of-M) 로직 — 게이팅 주기 확정 후 (`model/ablation_log.csv`에 실험 결과 있음)
- [ ] 실제 방향값 연동 (전자팀 하드웨어 대기)
- [ ] microSD 로그 저장 연동

### 논의 필요 — 확신도 구간별 세기 구분

기획안의 "확신도 0.7 이상 100% / 0.4~0.7 60%" 설계가 실제로는 거의 작동하지 않습니다.
보정 적용 후 위험음 판정 3,169건의 확신도 분포는 아래와 같습니다.

| 구간 | 건수 |
|---|---|
| 0.3–0.4 | 1 |
| 0.5–0.6 | 5 |
| 0.6–0.7 | 13 |
| 0.7–0.8 | 39 |
| 0.8–0.9 | 178 |
| 0.9+ | 2,933 |

0.7 미만이 19건(0.6%)에 불과해, `LOW_CONF_RATIO`를 제거하고 데시벨 기반 세기로
일원화하는 방안을 검토 중입니다. 다만 위 수치는 공개 데이터셋 기준이므로,
실제 마이크 환경에서는 애매한 케이스가 늘어날 수 있어 실측 후 결정이 필요합니다.
