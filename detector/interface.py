# RPi SW가 호출하는 유일한 진입점: count_pills(image, compartment_id).
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
    count: int            # 0~7
    confidence: float     # 0~1


def _load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_image(image) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3             or image.shape[2] != 3 or 0 in image.shape:
        desc = None if image is None else (getattr(image, "shape", type(image)), getattr(image, "dtype", None))
        raise ValueError(f"image는 (H, W, 3) uint8 BGR ndarray여야 함: {desc}")


def _crop(image: np.ndarray, rois: dict, compartment_id: int) -> np.ndarray:
    if compartment_id not in rois:
        raise ValueError(f"compartment_id는 {sorted(rois)} 중 하나여야 함: {compartment_id!r}")
    roi = rois[compartment_id]
    if roi is None:
        raise RuntimeError(f"config.yaml의 rois.{compartment_id}가 비어 있음. 칸 크롭 좌표 [x, y, w, h]를 적을 것")
    x, y, w, h = roi
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError(f"rois.{compartment_id} {roi}가 이미지 {image.shape[1]}x{image.shape[0]} 밖으로 나감")
    return image[y:y + h, x:x + w]


def _count(cfg: dict, crop: np.ndarray) -> tuple[int, float]:
    if not cfg["model"].get("path"):
        raise RuntimeError("config.yaml의 model.path가 비어 있음. export_onnx.py로 만든 .onnx 경로를 적을 것")
    return model.predict(model.load_model(cfg["model"]["path"]), crop, tuple(cfg["model"]["input_size"]))


def count_pills(image: np.ndarray, compartment_id: int) -> CountResult:
    # image: 통 전체 BGR 이미지 (H, W, 3, uint8), compartment_id: 뚜껑을 연 칸 번호
    _check_image(image)
    cfg = _load_config()
    count, conf = _count(cfg, _crop(image, cfg["rois"], compartment_id))
    return CountResult(compartment_id, count, conf)
