# CV-Medication-Detection

스마트 알약통 CV 모듈. **통 전체 사진 → 칸별(4칸) 알약 개수**를 낸다.
학습 파트가 만든 ONNX 모델을 RPi 기기에서 실행한다. 복용 판정(전 − 후 비교)은 서버가 한다.

## 설치 / 실행

```bash
pip install -r requirements.txt
pytest
```

## 연동 (RPi SW)

RPi SW는 `count_pills(image)`만 호출한다. 촬영과 서버 전송(MQTT)은 RPi SW 담당이다.

```python
from detector.interface import count_pills

results = count_pills(image)  # image: 통 전체 BGR (H, W, 3, uint8)
# {1: CountResult(compartment_id=1, count=2, confidence=0.97), ..., 4: CountResult(...)}
```

- 입력이 None이거나 shape/dtype이 맞지 않으면 `ValueError`
- `config.yaml`의 `model.path`가 비어 있으면 `RuntimeError`
- 촬영 시각은 RPi SW가 관리한다. 이미지는 디스크에 저장하지 않는다.

## 모델 규격 (학습 → 배포) — ⚠️ 학습 파트 확정 필요

RPi 기기 내 추론은 ONNX Runtime을 쓴다. RPi OS는 64-bit여야 한다 (PyPI에 32-bit ARM용 `onnxruntime` 없음).
모델 파일은 `.onnx` 한 개, 경로는 `detector/config.yaml`의 `model.path`. 칸 자르기나 ROI 같은 전처리는 없다.

아래는 임시 가정이다 (`detector/model.py`의 `TODO(학습 파트 확정)`).

- 입력: `model.input_size`(기본 224×224)로 리사이즈 → BGR→RGB → `/255` → `float32` `NCHW` (배치 1)
  - mean/std 정규화는 **모델 안에 넣어서** export한다
- 출력: shape `(1, 4, K)` 칸별 클래스 로짓 (softmax는 넣지 않는다)
  - 클래스 인덱스 = 개수 (0 ~ K−1), softmax 최댓값 = confidence
  - 순서 i = 칸 i+1
  - 다른 shape이면 `ValueError`
