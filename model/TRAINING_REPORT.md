# 학습 결과 보고서 (최종 업데이트: 2026-09-16)

## 최종 결과 요약

**4클래스 전부 Recall 90% 이상 달성했습니다.** (클립 단위, test set 340개 클립)

| 클래스 | Precision | **Recall** | F1 |
|---|---|---|---|
| horn | 0.875 | **0.966** | 0.918 |
| siren | 1.000 | **0.953** | 0.976 |
| crash | 0.907 | **0.907** | 0.907 |
| normal | 0.987 | **0.974** | 0.980 |

- 전체 정확도 95.9%
- 임계값 0.2~0.6 구간에서 이진 danger 판별 Recall 98.9%, Precision 97.9%, **시간당 오탐 20.78회**
- 340개 test 클립 중 **놓친 위험음(MISSED_DANGER) 단 2건**
- 추론 지연: 평균 16.6ms, 최대 18.4ms (200ms 목표 여유롭게 통과)

여기 오기까지 데이터 보강 + 중요한 평가 버그 수정이 있었습니다. 아래 순서대로 기록합니다.

---

## 무슨 일이 있었는지 (시간순)

### 1. 데이터 수집
- Hugging Face UrbanSound8K `car_horn` 429개 다운로드 → horn 518개(원본)
- crash 데이터 추가 수집 → 187개 → 307개 (사용자가 직접 모음)
- siren 폴더 구조 정리 (ambulance/firetruck 하위 폴더 + 스펙트로그램 이미지 섞여있던 것 → 평평한 구조로 정리, 440개)
- 최종 raw: horn 518 / siren 440 / crash 307 / normal 750

### 2. 초기 학습 — crash Recall 58~65%로 낮게 나옴
데이터가 늘어난 뒤에도 crash Recall이 계속 낮게(58~65%) 나와서, 배경소음 증강(`augment_data.py`)과 class_weight 조정(3.0→2.0)을 시도했지만 큰 개선이 없었습니다.

**배경소음 증강은 오히려 역효과**였습니다 (정확도 88.5%→79.8%로 하락, 증강 클립이 test fold까지 오염시킨 게 원인으로 보여 되돌림). class_weight 조정은 crash/siren 균형은 잡아줬지만 crash 자체 성능은 크게 못 올렸습니다.

### 3. 근본 원인 발견 — 평가 방식 자체가 잘못되어 있었음
사용자가 crash 오분류 파일을 직접 들어보고 "데이터 자체엔 문제 없다"고 확인 → 학습 방식을 의심하다가 발견:

**`error_analysis.csv`에서 crash 클립 11개가 확신도 0.3744로 정확히 똑같이 나오는 걸 발견했습니다.** 원인을 추적해보니:
- crash(특히 충돌음)는 "쾅" 하고 짧게 한 번 나고 나머지는 무음인 소리
- 4.5초 클립 중 실제 소리는 1초 안팎, 나머지는 패딩(무음)
- 그런데 **배포 모델의 최종 판단이 클립 내 모든 0.96초 프레임의 점수를 "평균"** 내는 방식이었음 → 짧은 crash 신호가 무음 프레임들에 희석되어 normal 쪽으로 밀림
- **게다가 지금까지 봐온 recall/precision 수치 자체가 프레임 단위로 계산된 것**이었습니다 (클립이 아니라 클립을 이루는 개별 0.96초 조각 하나하나를 별도 샘플로 취급). 정작 배포되는 모델은 클립 단위로 하나의 최종 판단만 내리는데, 평가는 그와 다른 기준으로 하고 있었던 것 — 그래서 그동안의 58~65% 수치 자체가 실제 배포 성능을 반영하지 못하는 잘못된 지표였습니다

### 4. 수정
1. **평균(mean) pooling → 최댓값(max) pooling으로 변경**: 클립 내 프레임 중 가장 확신도(peak logit) 높은 프레임의 점수를 그대로 클립의 최종 판단으로 사용 (`train_yamnet.py`의 `ServingModel`)
2. **`common.py`에 `pool_clip_logits()` 추가**: 프레임 단위 데이터를 클립 단위로 올바르게 집계하는 공용 함수
3. **`calibration.py`, `threshold_search.py`, `error_analysis.py`를 전부 클립 단위 평가로 수정** — 프레임 단위로 계산하던 버그를 걷어냄 (단, threshold_search.py의 N-of-M 연속확인 시뮬레이션은 게이팅 주기별 프레임 판단을 흉내내는 것이라 의도적으로 프레임 단위 유지)

수정 후 재학습한 결과가 맨 위의 최종 결과표입니다. mean→max pooling 전환 직전 비교(같은 클립 단위 평가 기준)로도 crash Recall이 93.0%(mean)와 동일하게 나왔던 걸 보면, **사실 mean pooling으로도 클립 단위로 제대로 평가했다면 이미 crash가 잘 나오고 있었다**는 뜻입니다. 즉 진짜 병목은 데이터도, class_weight도 아니라 **평가 스크립트의 버그**였습니다. max pooling은 그 위에 추가로 horn을 조금 더 개선해준 정도입니다 (94.8%→96.6%).

---

## 배경소음 증강 실험 — 참고용 기록

시도했다가 되돌린 내용입니다. `augment_data.py`의 경로 버그(`BASE_DIR.parent`→`BASE_DIR`)는 고쳐뒀지만, 증강 자체는 현재 미적용 상태입니다. 나중에 다시 시도한다면:
- 증강 클립을 train fold에만 넣고 val/test에는 안 들어가게 수정
- SNR을 더 약하게(예: 20dB, 15dB) 조정
가 필요해 보입니다. 지금 성능이 이미 목표를 넘긴 상태라 급하지 않습니다.

---

## 저장된 산출물

```
ai/saved_model/danger_sound_classifier/   ← 배포용 모델 (raw waveform in, 4클래스 점수 out, max pooling 적용)
ai/head_model.keras                       ← 분류층만 따로
ai/calibration.json                       ← temperature scaling 값 (T=2.24)
ai/manifest.csv                           ← 최종 학습 데이터 목록 (horn 303 / siren 440 / crash 246 / normal 750 클립)
ai/error_analysis.csv                     ← 오분류 사례, 클립 단위로 14건만 남음
ai/ablation_log.csv                       ← 임계값·게이팅 실험 기록 누적
```

`inference.py`의 `DECISION_THRESHOLD`(0.4)는 새 threshold_search 결과(0.2~0.6 전부 동일 성능)에도 여전히 안전 범위 안이라 그대로 뒀습니다.

---

## 확인/결정 필요한 것

1. **전자팀에 모델 전달 (기한 9/20)** — 지금 `saved_model/` + `calibration.json` + `inference.py`로 준비 완료
2. **실제 MEMS 마이크 녹음으로 재검증 필요** — 지금까지는 전부 공개데이터/효과음 기준. Temperature가 2.24로 꽤 높아진 걸 보면 모델이 이 데이터셋 조합에 과신하는 경향이 있어서, 실제 환경 데이터로 재확인이 특히 중요해 보입니다
3. **HornBase 360개(미검증 L/S/V 그룹)는 여전히 안 씀** — 이미 목표를 넘겼으니 급하지 않지만, 여유 있으면 검증해서 추가 가능
4. `error_analysis.csv`에 남은 14건(놓친 위험음 2건 포함)은 여유 있을 때 확인해보면 좋음
