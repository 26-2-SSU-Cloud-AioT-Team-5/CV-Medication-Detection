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
