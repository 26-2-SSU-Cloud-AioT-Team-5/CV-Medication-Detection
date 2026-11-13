# CV-Medication-Detection

스마트 알약통 CV 모듈. 칸에 알약이 **있는지/없는지**만 판별한다.

## ResNet-18 이진 분류 학습

Python 3.9 이상에서 프로젝트 루트 기준으로 의존성을 설치한다.

```bash
python3 -m pip install -r requirements.txt
```

데이터셋은 아래처럼 준비한다. `filled=1`은 알약 있음, `not_filled=0`은 알약 없음을 의미한다.
train/val 분리는 사용자가 준비하며, 동일한 이미지나 같은 촬영에서 나온 유사 프레임이
두 split에 섞이지 않도록 한다. 각 클래스에는 적어도 한 장의 이미지가 필요하다.

```text
training/dataset/
├── train/
│   ├── filled/       # 알약이 보이는 이미지
│   └── not_filled/   # 알약이 없는 이미지
└── val/
    ├── filled/
    └── not_filled/
```

JPG/JPEG, PNG, BMP, WEBP, TIFF 이미지를 지원하며 클래스 폴더 내부의 하위 폴더도 탐색한다.
각 클래스의 `.gitignore`는 이미지가 Git에 올라가지 않도록 한다.

**설정은 `training/train.py` 상단 전역 변수 섹션에서 변경한다. CLI 인자는 사용하지 않는다.**

| 변수 | 기본값 | 역할 |
|---|---|---|
| `DATASET_ROOT` | `training/dataset` | train/val 폴더가 있는 경로 |
| `OUTPUT_DIR` | `training/outputs` | 실행별 결과 폴더를 생성할 위치 |
| `EPOCHS` | `20` | 전체 학습 epoch 수 |
| `BATCH_SIZE` | `32` | 배치당 이미지 수 |
| `LEARNING_RATE` | `1e-4` | Adam 학습률 |
| `WEIGHT_DECAY` | `1e-4` | 가중치 감쇠 |
| `IMAGE_SIZE` | `224` | 입력 이미지 한 변의 크기 |
| `NUM_WORKERS` | `0` | 데이터 로딩 프로세스 수 |
| `VAL_EVERY_STEPS` | `0` | 0이면 매 epoch 종료 시 검증, 양수면 N step마다 추가 검증 |
| `LOG_EVERY_STEPS` | `10` | 학습 진행 로그 출력 주기 |
| `PRETRAINED` | `True` | ImageNet 가중치로 초기화 |
| `THRESHOLD` | `0.5` | 알약 있음 판정 확률 임계값 |
| `DEVICE` | `"auto"` | CUDA → MPS → CPU 자동 선택 또는 직접 지정 |
| `SEED` | `42` | 난수 시드 |

경로에는 `Path` 또는 문자열을 사용할 수 있다. 상대 경로는 프로젝트 루트 기준으로 해석한다.
`PRETRAINED=True`인 최초 실행에는 가중치 다운로드를 위한 인터넷 연결이 필요할 수 있다.

```bash
python3 training/train.py
# 또는
python3 -m training.train
```

`architecture/model.py`의 `PillClassifier`는 ResNet-18의 FC 출력을 1개로 변경하고
이미지당 하나의 logit을 반환한다. `BCEWithLogitsLoss`와 Adam으로 모든 레이어를 학습한다.
`sigmoid(logit) >= THRESHOLD`이면 알약 검출(`filled`)이다.

`training/dataset_sampler.py`는 RGB 변환, 전체 이미지 리사이즈, ImageNet 정규화를 수행한다.
학습에만 약한 밝기·대비 증강과 셔플을 적용하며, 검증은 증강 없이 모든 이미지를 평가한다.
검증 accuracy는 전체 정답 수 / 전체 이미지 수이고, loss도 이미지 수로 가중 평균한다.
중간 검증 후에는 학습 모드로 복귀한다.

결과는 `training/outputs/<실행 시각>/` 아래에 저장되며 기본 출력 경로는 Git에서 무시한다.

- `best.pt`: validation accuracy가 가장 높은 체크포인트. 동률이면 먼저 저장한 모델 유지.
- `last.pt`: 가장 최근 검증 시점의 체크포인트. 정상 완료 시 마지막 학습 step과 일치.
- `config.json`: 해당 실행의 설정과 클래스 매핑, 전처리 정보.
- `metrics.jsonl`: 검증 시점별 epoch/step, train/validation loss와 accuracy.

체크포인트에는 `model_state_dict`, `optimizer_state_dict`, epoch/step,
검증 지표 및 `config`가 포함된다. 저장한 모델을 다시 사용할 때는
`PillClassifier(pretrained=False)`를 생성해 `model_state_dict`를 로드하고,
저장된 입력 크기와 정규화 설정으로 전처리한 뒤 `model.eval()`에서 추론한다.

임시 이미지와 무작위 초기화 모델만 사용하는 오프라인 검증:

```bash
python3 -m unittest discover -s tests -v
```

## 설치 / 실행

```bash
pip install -r requirements.txt
python -c "from detector.interface import detect; print(detect(1))"
pytest
```

## 연동 (RPi SW)

`detect(compartment_id) -> DetectResult(compartment_id, has_pill, confidence, captured_at)` 만 호출한다.
시그니처와 필드는 고정. 백엔드는 `detector/config.yaml`의 `detector.backend` (`dummy` / `baseline` / `model`)로 바꾼다.
기본 `dummy`는 항상 `has_pill=True, confidence=1.0`. `baseline`/`model`은 구현 전이라 `NotImplementedError`출력
이미지는 디스크에 저장하지 않는다.

## 할 일

- [ ] `detector/camera.py` — `capture` (webcam / picamera2 / file 모드)
- [ ] `samples/` — 빈 칸 / 알약 있는 칸 사진 수집
- [ ] `detector/preprocess.py` — `crop_roi`, `resize`, `normalize_brightness`
- [ ] `detector/baseline.py` — `register_baseline`, `compare` (OpenCV 기준 이미지 비교 판별)
- [ ] 평가 스크립트 (신규) — `samples/empty`, `samples/filled`로 정확도 측정
- [ ] `detector/config.yaml` — 칸별 ROI, 임계값 조정
- [ ] (추후) `detector/model.py` — `load_model`, `predict` (규칙 기반이 부족할 때만)

## 작업 우선순위

1. 뼈대 (완료)
2. `camera.py` + 사진 수집
3. `baseline.py` + 평가
4. (선택) `model.py`
