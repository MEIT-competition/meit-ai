# AI 파트 사용 설명서

위험음 방향 인지 웨어러블 — horn(경적)/siren(사이렌)/crash(충돌음)/normal(일상소음) 분류 모델을 만들고 검증하는 과정입니다.

## 0. 준비

```bash
cd "C:/Users/rabbi/Desktop/2026/School/competi/MEIT/ai"
pip install -r requirements.txt
```

## 1. 공개 데이터셋 넣기

다운로드한 파일을 클래스별로 그냥 던져 넣으면 됩니다. 파일 형식(wav/mp3 등)이나 길이는 신경 안 써도 됩니다.

```
ai/raw/horn/    ← 경적 관련 원본 파일
ai/raw/siren/   ← 사이렌 관련 원본 파일
ai/raw/crash/   ← 충돌음 관련 원본 파일
ai/raw/normal/  ← 일상소음 관련 원본 파일
```

```bash
python prepare_data.py
```

16kHz mono, 4.5초 클립으로 자동 정리하고 `manifest.csv`(클립별 라벨/fold)를 만들어줍니다. 실행 후 콘솔에 클래스별 클립 개수가 찍히니 부족한 클래스는 바로 보여요.

## 2. (선택) 배경 소음 증강

```bash
python augment_data.py
```

danger 클래스 클립마다 normal 소리를 섞은 버전을 추가로 만들어 manifest.csv에 등록합니다. 참고로 실제로 써보니 SNR을 너무 세게(5dB) 잡으면 오히려 성능이 떨어졌어서, 지금 최종 모델에는 미적용 상태입니다. 다시 시도한다면 SNR을 더 약하게 잡고 train fold에만 섞이게 해야 해요.

다시 실행하면 중복으로 추가되니, 재실행 전엔 manifest.csv를 백업하거나 augmented 행을 지우고 실행하세요.

## 3. 모델 학습

```bash
python train_yamnet.py
```

YAMNet은 고정해두고 Dense(512)+Dense(4) 헤드만 학습합니다. 첫 실행은 임베딩 추출 때문에 좀 걸리고, 이후엔 `embed_cache/`에 캐시된 걸 씁니다. 끝나면 test set 기준 클래스별 Recall/Precision/F1을 클립 단위로 출력해요 (실제 배포 모델이 클립 하나당 하나의 판단만 내리기 때문에, 여기서도 클립 단위로 채점하는 게 맞습니다 — 예전엔 프레임 단위로 채점하던 버그가 있었어요, TRAINING_REPORT.md 참고).

데이터를 추가하거나 다시 학습하고 싶으면 **`embed_cache/` 폴더를 지우고** 다시 실행하세요. 안 지우면 예전 캐시를 그대로 씁니다.

## 4. 확신도 보정 (Calibration)

```bash
python calibration.py
```

Temperature scaling으로 모델 확신도가 실제 정확도를 더 잘 반영하도록 보정합니다. ECE 전후 수치가 출력되고 `calibration.json`에 저장돼서 이후 스크립트들이 자동으로 불러다 씁니다.

## 5. 임계값 결정

```bash
python threshold_search.py
```

임계값 0.2~0.8을 다 시도해서 Recall/Precision/시간당 오탐 횟수를 표로 보여주고, Recall 90% 이상 중 오탐이 제일 적은 임계값을 추천해줍니다. N-of-M 연속확인 시뮬레이션도 같이 출력되는데 이건 참고용이고 실제 게이팅 로직은 이예은님 구현에 맞춰야 해요. 결과는 `ablation_log.csv`에 자동으로 쌓입니다.

추천받은 임계값을 [inference.py](inference.py)의 `DECISION_THRESHOLD`에 반영하세요.

## 6. 오류 분석

```bash
python error_analysis.py
```

`error_analysis.csv`가 생성됩니다. `MISSED_DANGER`로 표시된 행(위험음을 놓친 경우)부터 원본 파일을 직접 들어보고 reason 칸에 왜 틀렸는지 적어두면 좋습니다.

## 7. 추론 지연 측정

```bash
python latency_benchmark.py
```

게이팅 주기(200~300ms) 안에 드는지 mean/p50/p95/max 지연시간을 출력합니다.

## 8. 단일 클립 테스트 / 전자팀 전달

```bash
python inference.py path/to/clip.wav
```

`{class, confidence, is_danger, intensity}` 형태의 JSON이 나옵니다. 전자팀에 넘길 때는 `saved_model/`, `calibration.json`, `inference.py`를 같이 전달하면 됩니다 (기한 9/20).

## 9. 모터 on/off 녹음 추가 (허리띠 시제품 나온 뒤)

기획안에 있던 "노이즈 억제용 데이터" — 같은 위험음을 모터 끈 상태(off)/켠 상태(on)에서 같은 마이크 위치로 녹음해서 쌍으로 만드는 겁니다.

같은 danger 소리를 모터 off 상태에서 1번, on 상태에서 1번, 같은 위치/크기로 녹음해서 아래처럼 넣으세요:

```
ai/raw_motor_noise/horn/take001_off.wav
ai/raw_motor_noise/horn/take001_on.wav
ai/raw_motor_noise/siren/take001_off.wav
ai/raw_motor_noise/siren/take001_on.wav
ai/raw_motor_noise/crash/take001_off.wav
ai/raw_motor_noise/crash/take001_on.wav
```

take002, take003... 계속 추가하면 되는데, off/on 이름이 하나라도 안 맞으면 그 take는 무시되고 콘솔에 경고가 뜹니다.

```bash
python add_motor_noise_data.py
```

off/on 쌍을 정리해서 `dataset_motor_noise/`에 저장하고, `_on` 버전만 학습용 manifest.csv에 자동 추가합니다 (`_off`는 비교용 기준값으로만 씀). 끝나면 `embed_cache/` 지우고 `train_yamnet.py` 다시 돌리세요.

```bash
python motor_noise_eval.py
```

같은 소리의 off/on 쌍을 모델에 넣어서 예측이 뒤집히는지, 확신도가 얼마나 떨어지는지 비교해 `motor_noise_eval.csv`로 저장합니다.

## 자주 막히는 부분

| 증상 | 원인/해결 |
|---|---|
| `manifest.csv is empty` | `prepare_data.py`를 먼저 실행 안 함 |
| 데이터 추가했는데 결과가 그대로임 | `embed_cache/` 폴더 지우고 재학습 안 함 |
| `No calibration.json found` 경고 | `calibration.py` 아직 안 돌림 (없어도 동작은 함, 정확도만 떨어짐) |
| 특정 take가 모터 데이터에 안 잡힘 | off/on 파일명이 정확히 일치하는지 확인 |
