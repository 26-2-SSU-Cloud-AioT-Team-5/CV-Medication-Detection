# CV-Medication-Detection

스마트 알약통 CV 모듈. 칸에 알약이 **있는지/없는지**만 판별한다.

## 설치 / 실행

```bash
pip install -r requirements.txt
python -c "from detector.interface import detect; print(detect(1))"
pytest
```

## 연동 (RPi SW)

`detect(compartment_id) -> DetectResult(compartment_id, has_pill, confidence, captured_at)` 만 호출한다.
시그니처와 필드는 고정. 백엔드는 `detector/config.yaml`의 `detector.backend` (`dummy` / `baseline` / `model`)로 바꾼다.
기본 `dummy`는 항상 `has_pill=True, confidence=1.0`. `baseline`과 카메라/전처리는 구현 전이라 `NotImplementedError`출력
이미지는 디스크에 저장하지 않는다.

서버 전송은 `detector.publisher.publish(detect(1))`로 한다. 브로커와 토픽은 `config.yaml`의 `mqtt`에서 정하고, QoS 1로 아래 JSON을 보낸다.

```json
{"compartment_id": 1, "has_pill": true, "confidence": 0.97, "captured_at": 1794214800.0}
```

`confidence`는 판정 결과에 대한 확신도다.

## 모델 규격 (학습 → 배포)

RPi 기기 내 추론은 ONNX Runtime을 쓴다. RPi OS는 64-bit여야 한다 (PyPI에 32-bit ARM용 `onnxruntime` 없음).

- 모델: ResNet18 이진 분류. 파일은 `.onnx` 한 개, 경로는 `config.yaml`의 `model.path`
- 입력: `float32`, `NCHW` (배치 1), **RGB 3채널**, 값 범위 0~1. 크기는 `preprocess.size` (기본 224×224)
  - 기기 쪽은 칸 ROI 자르기 → 리사이즈 → BGR→RGB → `/255`까지만 한다. 밝기 보정은 하지 않는다
  - mean/std 정규화 등 그 외 전처리는 **모델 안에 넣어서** export한다
- 출력: 로짓 (sigmoid/softmax는 넣지 않는다). 둘 중 하나
  - 로짓 1개, shape `(1,)` 또는 `(1, 1)`: "있음" 로짓 (`BCEWithLogitsLoss`로 학습한 경우. `architecture/model.py`의 `PillClassifier`)
  - shape `(1, 2)`: `[없음, 있음]` 로짓 (`CrossEntropyLoss`로 학습한 경우)

학습 체크포인트는 `training/export_onnx.py`로 위 규격의 ONNX로 바꾼다. 체크포인트의 `normalization_mean/std`를 모델 안에 넣고, 변환 결과가 PyTorch와 같은지 확인까지 한다.

```bash
python -m training.export_onnx training/checkpoints/<실행>/best.pt   # -> 같은 폴더에 best.onnx
```

## 할 일

- [ ] `detector/camera.py` — `capture` (webcam / picamera2 / file 모드)
- [ ] `samples/` — 빈 칸 / 알약 있는 칸 사진 수집
- [ ] `detector/preprocess.py` — `crop_roi`, `resize`, `normalize_brightness`
- [ ] `detector/baseline.py` — `register_baseline`, `compare` (OpenCV 기준 이미지 비교 판별)
- [ ] 평가 스크립트 (신규) — `samples/empty`, `samples/filled`로 정확도 측정
- [ ] `detector/config.yaml` — 칸별 ROI, 임계값 조정
- [x] `detector/model.py` — `load_model`, `predict` (ONNX Runtime)


