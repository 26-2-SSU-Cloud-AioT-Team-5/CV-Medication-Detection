# CV-Medication-Detection

스마트 알약통 CV 모듈. **통 전체 사진 + 연 칸 번호 → 그 칸의 알약 개수(0~7)**를 낸다.
학습 파트가 만든 ONNX 모델을 RPi 기기에서 실행한다. 복용 판정(전 − 후 비교)은 서버가 한다.

## 설치 / 실행

```bash
pip install -r requirements.txt
pytest
```

## 연동 (RPi SW)

뚜껑 센서가 없어서 CV가 사진으로 칸 열림/닫힘을 판단하고, **뚜껑이 닫힐 때 칸별 개수를 확정**한다.
RPi SW는 약 1초마다 통 전체를 찍어 `update`를 부르고, 나온 이벤트를 MQTT로 보내기만 한다.

```python
from dataclasses import asdict
from detector.watcher import PillboxWatcher

watcher = PillboxWatcher()   # 프로세스당 한 번 (모델도 한 번만 로드)
while True:
    for event in watcher.update(camera.capture(), datetime.now().astimezone()):
        mqtt.publish(topic, json.dumps(asdict(event), default=str))
    time.sleep(1)
# CountEvent(compartment_id=2, count=2, confidence=0.94, status="ok", captured_at=...)
```

- 칸마다 검은 바닥판이 보이면 열림, 흰 뚜껑이 보이면 닫힘. `switch_frames`장 연속이어야 상태를 바꾼다.
- 열려 있는 동안 매 장 개수를 세고, 닫히면 **같은 개수가 `stable_frames`장 연속 `min_confidence` 이상인 가장 최근 구간**을 결과로 낸다.
  손이 가린 사진이나 닫히는 중인 뚜껑은 이렇게 걸러진다.
- 그런 구간이 없으면 `status="uncertain"`, `count=None`. 서버는 판정에 쓰지 않는다.
- 서버는 칸별로 이번 `count`와 지난 `count`의 차이로 복용(줄어듦)/채움(늘어남)을 판정한다.
- 기준값은 `config.yaml`의 `watcher`에서 바꾼다.

한 칸만 바로 세고 싶으면 `count_pills(image, compartment_id) -> CountResult(compartment_id, count, confidence)`.

- image의 shape/dtype이 맞지 않거나, 칸 번호가 config에 없거나, ROI가 이미지 밖이면 `ValueError`
- `config.yaml`의 `model.path` 또는 해당 칸의 `rois`가 비어 있으면 `RuntimeError`
- 이미지는 디스크에 저장하지 않는다. image는 ROI를 잡은 사진과 같은 카메라 위치·해상도여야 한다.

## RPi 설치 / 속도 측정

Raspberry Pi OS **64-bit** 기준 (32-bit에는 `onnxruntime`이 없음).

```bash
sudo apt install -y python3-picamera2            # 카메라 (apt로만 설치됨)
python3 -m venv --system-site-packages .venv     # apt로 깐 picamera2를 venv에서 쓰기 위해
. .venv/bin/activate
pip install onnxruntime numpy Pillow PyYAML
```

모델이 아직 없으면 PC에서 랜덤 가중치 ONNX를 만들어 복사한다 (속도만 같고 개수는 의미 없음).

```bash
python -m training.export_onnx --random bench.onnx   # PC (torch 필요)
scp bench.onnx pi@<약통 Pi>:~/CV-Medication-Detection/
```

```bash
python -m detector.bench bench.onnx --camera     # RPi: 촬영 + 크롭 + 추론 시간, CPU 온도
```

`최악 1회(촬영 + 4칸 열림 update)`가 1000 ms 안이면 1초 간격 호출이 가능하다.
PC(데스크톱 CPU, 촬영 제외)에서는 약 210 ms였다.

## 칸 ROI 보정

카메라나 통이 움직이면 통 전체를 다시 찍고 ROI를 새로 잡는다. (PC에서는 `pip install opencv-python` 필요)
기준: 칸마다 **검은 바닥판을 딱 덮는 박스** (흰 벽은 거의 넣지 않음). 칸 번호는 사진 왼쪽부터 1~4.

```bash
python -m detector.calibrate 사진.jpg          # 칸 1→4 순서로 드래그 → Enter, 끝나면 Esc → config.yaml rois 저장
python -m detector.calibrate 사진.jpg --show   # 지금 rois를 사진에 그려 <사진>_rois.jpg로 확인만
```

## 모델 규격 (학습 → 배포)

RPi 기기 내 추론은 ONNX Runtime을 쓴다. RPi OS는 64-bit여야 한다 (PyPI에 32-bit ARM용 `onnxruntime` 없음).

- 모델: `architecture/model.py`의 `PillStateClassifier` (Phase 2, ResNet-18 + 개수 head 8 + 상태 head 3). Phase 1 `build_model`(개수만)도 된다.
- 변환: `python -m training.export_onnx models/phase2_best.pt` → 옆에 `.onnx` 생성. 그 경로를 `detector/config.yaml`의 `model.path`에 적는다.
  - 체크포인트와 `.onnx`는 git에 올리지 않는다 (`models/`). Pi에는 scp로 복사한다.
  - ImageNet mean/std 정규화는 모델 안에 들어간다.
- 전처리 (`detector/model.py`): `rois`의 `[x, y, w, h]`로 칸 크롭 → BGR→RGB → PIL LANCZOS로 224×224 → `/255` → `float32` `NCHW`
  - 학습 데이터(`resize_test.py`)와 같은 리사이즈를 쓴다. 비율이 바뀌어도 그대로 늘린다.
- 출력: `count_logits (1, 8)` (인덱스 = 개수), `status_logits (1, 3)` (`countable`, `hand_occlusion`, `lid_closed`)
  - 상태가 `countable`이면 `(개수, 개수 확률)`, 아니면 `(None, 상태 확률)`. watcher는 `None` 사진을 안정 구간에서 뺀다.

## 남은 일

- [x] 학습 파트에서 체크포인트 받기 → ONNX 변환 (Phase 2)
- [ ] 칸별 ROI 좌표 재기 (학습 때 크롭 좌표가 저장되지 않음) → `config.yaml`의 `rois`
- [ ] 실제 RPi 사진으로 정확도 확인 (학습 검증은 합성 데이터만)
- [x] RPi에서 추론 시간 측정 (랜덤 가중치: 1칸 열림 약 0.3초, 4칸 0.9초)
