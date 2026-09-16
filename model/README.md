# AI 파트 — 위험음 분류 모델 (horn / siren / crash / normal)

## 폴더 구조
```
ai/
  raw/{horn,siren,crash,normal}/     # 원본 다운로드 파일 (git에 커밋하지 말 것 - 용량)
  dataset/{horn,siren,crash,normal}/ # prepare_data.py 결과 (16kHz mono, 4.5s 클립)
  manifest.csv                       # 클립별 라벨/fold 매니페스트
  embed_cache/{train,val,test}.npz   # YAMNet 임베딩 캐시 (재실행 시 재계산 안 함)
  saved_model/                       # 배포용 SavedModel (raw wav in -> 4클래스 점수 out)
  head_model.keras                   # Dense head만 저장 (다른 스크립트에서 재사용)
  val_logits.npy / val_labels.npy    # calibration.py용
  test_logits.npy / test_labels.npy / test_filepaths.npy / test_clip_id.npy  # threshold_search, error_analysis용
  calibration.json                   # temperature scaling 결과
  ablation_log.csv                   # 실험 기록
  error_analysis.csv                 # 오분류 사례 (직접 들어보고 reason 채우기)

  common.py            # 공통 상수/함수 (CLASSES, 경로, 임베딩 추출 등)
  prepare_data.py       # 1단계: 원본 -> 정리된 클립
  augment_data.py        # 1.5단계(선택): danger 클립에 normal 소리 섞어 노이즈 강건성 증강
  train_yamnet.py        # 2단계: 임베딩 추출 + 학습 + export
  calibration.py         # 3단계: confidence calibration
  threshold_search.py    # 4단계: 임계값 스윕 + N-of-M 연속확인 실험
  error_analysis.py      # 5단계: 오분류 사례 정리
  inference.py            # 전자팀에 전달할 추론 래퍼 (모델+추론 코드, 9/20 기한)
  latency_benchmark.py    # 추론 지연이 200ms 목표 안에 드는지 측정
```

## 실행 순서
```bash
cd "C:/Users/rabbi/Desktop/2026/School/competi/MEIT/ai"
pip install -r requirements.txt
```
1. `raw/{class}/` 아래에 다운로드한 데이터셋 파일을 클래스별로 넣기
2. `python prepare_data.py` — 16kHz mono, 4.5초 클립으로 정리 + manifest.csv 생성
3. (선택, 권장) `python augment_data.py` — danger 클립마다 normal 소리를 섞은 버전을 추가로 만들어 manifest.csv에 등록 (실사용 잡음 대응력 강화, 하드웨어 없이 지금 바로 가능)
4. `python train_yamnet.py` — YAMNet 임베딩 추출(캐시됨) → 클래스 가중치 적용한 Dense head 학습 → 평가 → SavedModel/head_model export
5. `python calibration.py` — temperature scaling으로 confidence 보정, ECE 전후 비교, calibration.json 저장
6. `python threshold_search.py` — Recall 90% 이상을 만족하는 임계값 중 오탐(시간당 false alarm)이 제일 적은 지점 추천, N-of-M 연속확인 실험 결과도 함께 출력, ablation_log.csv에 자동 기록
7. `python error_analysis.py` — 오분류 사례(특히 위험음을 놓친 경우 우선) 정리해서 error_analysis.csv 생성 → 직접 들어보고 reason 칸 채우기
8. `python latency_benchmark.py` — 추론 지연이 200ms 목표 안에 드는지 측정 (mean/p50/p95/max)
9. `python inference.py path/to/clip.wav` — 단일 클립 분류 테스트 (전자팀에 넘길 때 이 파일 기준으로 연동)

`augment_data.py`를 나중에 다시 돌리면 중복 추가되니, 재실행 전엔 `manifest.csv`를 백업하거나 augmented 행을 지우고 실행하세요. 또한 데이터를 추가/증강한 뒤에는 `embed_cache/*.npz`를 삭제하고 `train_yamnet.py`를 다시 돌려야 새 클립이 반영됩니다.
