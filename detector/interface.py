# RPi SW가 호출하는 유일한 진입점: count_pills(image).
# 촬영과 서버 전송은 RPi SW 담당. 이미지는 디스크에 저장하지 않는다.
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import model

CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass
class CountResult:
    compartment_id: int   # 1~4
    count: int
    confidence: float     # 0~1


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_pills(image: np.ndarray) -> dict[int, CountResult]:
    # image: 통 전체 BGR 이미지 (H, W, 3, uint8)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 \
            or image.shape[2] != 3 or 0 in image.shape:
        desc = None if image is None else (getattr(image, "shape", type(image)), getattr(image, "dtype", None))
        raise ValueError(f"image는 (H, W, 3) uint8 BGR ndarray여야 함: {desc}")

    cfg = _load_config()["model"]
    if not cfg.get("path"):
        raise RuntimeError("config.yaml의 model.path가 비어 있음. 학습 파트가 준 .onnx 경로를 적을 것")

    preds = model.predict(
        model.load_model(cfg["path"]), image, tuple(cfg["input_size"]), cfg["num_compartments"]
    )
    return {cid: CountResult(cid, count, conf) for cid, (count, conf) in preds.items()}
