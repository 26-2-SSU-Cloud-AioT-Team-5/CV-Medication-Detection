# RPi SW가 호출하는 유일한 진입점: detect(compartment_id).

# config.yaml의 detector.backend 값으로 구현을 고른다.
# - dummy    : 항상 has_pill=True, confidence=1.0 (연동용)
# - baseline : 기준 이미지 비교 (baseline.py, 아직 stub)
# - model    : 학습 모델 (model.py, 아직 stub)

import time
from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")
BACKENDS = ("dummy", "baseline", "model")


@dataclass
class DetectResult:
    compartment_id: int
    has_pill: bool        # 알약 유무
    confidence: float     # 0~1
    captured_at: float    # time.time() 기기 시각


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect(compartment_id: int) -> DetectResult:
    #촬영 -> 전처리 -> 판별
    cfg = _load_config()
    backend = cfg["detector"]["backend"]
    if backend not in BACKENDS:
        raise ValueError(f"unknown detector.backend: {backend!r} (choose from {BACKENDS})")

    captured_at = time.time()
    if backend == "dummy":
        return DetectResult(compartment_id, True, 1.0, captured_at)

    # 아래 모듈은 아직 stub -> NotImplementedError가 나는 게 정상
    from . import camera, preprocess

    image = camera.capture(compartment_id, cfg["camera"])
    image = preprocess.crop_roi(image, cfg["compartments"][compartment_id]["roi"])
    image = preprocess.resize(image, tuple(cfg["preprocess"]["size"]))
    image = preprocess.normalize_brightness(image)

    if backend == "baseline":
        from . import baseline
        has_pill, confidence = baseline.compare(
            compartment_id, image, cfg["baseline"]["threshold"]
        )
    else:
        from . import model
        has_pill, confidence = model.predict(
            model.load_model(cfg["model"]["path"]), image
        )

    del image  
    return DetectResult(compartment_id, bool(has_pill), float(confidence), captured_at)
