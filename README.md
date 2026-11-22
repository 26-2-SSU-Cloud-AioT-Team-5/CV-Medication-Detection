# CV-Medication-Detection

스마트 알약통 CV 모듈. **통 전체 사진 + 연 칸 번호 → 그 칸의 알약 개수(0~7)**를 낸다.
학습 파트가 만든 ONNX 모델을 RPi 기기에서 실행한다. 복용 판정(전 − 후 비교)은 서버가 한다.

## 설치 / 실행

```bash
pip install -r requirements.txt
pytest
```

## 연동 (RPi SW)

RPi SW는 `count_pills(image, compartment_id)`만 호출한다. 촬영과 서버 전송(MQTT)은 RPi SW 담당이다.

```python
from detector.interface import count_pills

result = count_pills(image, 2)  # image: 통 전체 BGR (H, W, 3, uint8), 2: 뚜껑을 연 칸 번호(1~4)
# CountResult(compartment_id=2, count=3, confidence=0.97)
```

- image의 shape/dtype이 맞지 않거나, 칸 번호가 config에 없거나, ROI가 이미지 밖이면 `ValueError`
- `config.yaml`의 `model.path` 또는 해당 칸의 `rois`가 비어 있으면 `RuntimeError`
- 촬영 시각은 RPi SW가 관리한다. 이미지는 디스크에 저장하지 않는다.
- image는 학습 사진과 같은 카메라 위치·해상도여야 한다 (ROI가 픽셀 좌표).

## 칸 ROI 보정

카메라나 통이 움직이면 통 전체를 다시 찍고 ROI를 새로 잡는다. (PC에서는 `pip install opencv-python` 필요)

```bash
python -m detector.calibrate 사진.jpg          # 칸 1→4 순서로 드래그 → Enter, 끝나면 Esc → config.yaml rois 저장
python -m detector.calibrate 사진.jpg --show   # 지금 rois를 사진에 그려 <사진>_rois.jpg로 확인만
```

## 모델 규격 (학습 → 배포)

RPi 기기 내 추론은 ONNX Runtime을 쓴다. RPi OS는 64-bit여야 한다 (PyPI에 32-bit ARM용 `onnxruntime` 없음).

- 모델: `architecture/model.py`의 `build_model` (ResNet-18, 로짓 8개, 인덱스 = 알약 개수 0~7)
- 변환: `python -m training.export_onnx <체크포인트.pt>` → 옆에 `.onnx` 생성. 그 경로를 `detector/config.yaml`의 `model.path`에 적는다.
  - `best.pt`(state_dict)와 `last.pt`(`model_state_dict`) 둘 다 된다. ImageNet mean/std 정규화는 모델 안에 들어간다.
- 전처리 (`detector/model.py`): `rois`의 `[x, y, w, h]`로 칸 크롭 → BGR→RGB → PIL LANCZOS로 224×224 → `/255` → `float32` `NCHW`
  - 학습 데이터(`resize_test.py`)와 같은 리사이즈를 쓴다. 비율이 바뀌어도 그대로 늘린다.
- 출력: `(1, K)` 로짓. softmax 최댓값의 인덱스 = 개수, 그 확률 = confidence. 다른 shape이면 `ValueError`

## 남은 일

- [ ] 학습 파트에서 체크포인트 받기 → ONNX 변환
- [ ] 칸별 ROI 좌표 재기 (학습 때 크롭 좌표가 저장되지 않음) → `config.yaml`의 `rois`
- [ ] 실제 RPi 사진으로 정확도 확인 (학습 검증은 합성 데이터만)
- [ ] RPi에서 추론 시간 측정 (목표 5초 이내)
