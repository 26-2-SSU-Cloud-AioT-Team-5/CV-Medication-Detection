# RPi SW가 호출하는 유일한 진입점: count_pills(image, compartment_id).
# 촬영과 서버 전송은 RPi SW 담당. 이미지는 디스크에 저장하지 않는다.
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from . import model

CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass
class CountResult:
    compartment_id: int   # 1~4
    count: Optional[int]  # 0~7. 손 가림/뚜껑 닫힘으로 셀 수 없으면 None
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


def _count(cfg: dict, crop: np.ndarray) -> tuple[Optional[int], float]:
    if not cfg["model"].get("path"):
        raise RuntimeError("config.yaml의 model.path가 비어 있음. export_onnx.py로 만든 .onnx 경로를 적을 것")
    return model.predict(model.load_model(cfg["model"]["path"]), crop, tuple(cfg["model"]["input_size"]))


# 자동 보정: 통이 조금 밀려도 config rois를 매 사진마다 검은 바닥판 위로 옮긴다.
# ponytail: 칸마다 평행이동만 찾음 (회전·확대 없음). 크기의 MAX_SHIFT 넘게 움직이면 calibrate로 다시 잡을 것
MAX_SHIFT = 0.2       # 칸 크기 대비 찾아보는 최대 이동 비율 (이웃 칸 바닥판에 걸리지 않을 만큼만)
PLATEAU = 0.02        # 최고 점수와 이 차이 안의 위치들의 가운데를 고른다 (roi가 바닥판보다 작아 꽉 차는 위치가 여러 개)


def align_rois(image: np.ndarray, cfg: dict) -> dict:
    # 칸마다 바닥판 픽셀 비율이 가장 큰 위치로 roi를 옮긴다. 뚜껑이 닫혀 바닥판이 안 보이면 config 값 그대로
    w = cfg.get("watcher", {})
    level, need = w.get("dark_level", 100), w.get("open_dark_ratio", 0.5)
    sizes = [min(r[2], r[3]) for r in cfg["rois"].values() if r]
    s = max(1, min(sizes, default=1) // 60)  # 칸 한 변을 약 60칸으로 줄여 계산 (RPi 속도)
    dark = image[::s, ::s].mean(axis=2) < level
    integral = np.pad(dark.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return {cid: roi and _align(integral, s, roi, image.shape, need) for cid, roi in cfg["rois"].items()}


def _align(integral: np.ndarray, s: int, roi: list, shape: tuple, need: float) -> list:
    x, y, w, h = roi
    bw, bh = w // s, h // s
    if bw == 0 or bh == 0:
        return roi
    dxs = np.arange(-int(w * MAX_SHIFT), int(w * MAX_SHIFT) + 1, s)
    dys = np.arange(-int(h * MAX_SHIFT), int(h * MAX_SHIFT) + 1, s)
    dxs = dxs[(x + dxs >= 0) & (x + dxs + w <= shape[1])]
    dys = dys[(y + dys >= 0) & (y + dys + h <= shape[0])]
    if dxs.size == 0 or dys.size == 0:
        return roi  # 이미지 밖 roi는 _crop이 에러로 알린다
    x0, y0 = (x + dxs) // s, (y + dys) // s
    x1 = np.minimum(x0 + bw, integral.shape[1] - 1)
    y1 = np.minimum(y0 + bh, integral.shape[0] - 1)
    score = (integral[np.ix_(y1, x1)] - integral[np.ix_(y0, x1)]
             - integral[np.ix_(y1, x0)] + integral[np.ix_(y0, x0)]) / (bw * bh)
    if score.max() < need:
        return roi  # 닫힘: 옮길 기준이 없음
    iy, ix = np.nonzero(score >= score.max() - PLATEAU)
    return [int(x + np.median(dxs[ix])), int(y + np.median(dys[iy])), w, h]


def count_pills(image: np.ndarray, compartment_id: int) -> CountResult:
    # image: 통 전체 BGR 이미지 (H, W, 3, uint8), compartment_id: 뚜껑을 연 칸 번호
    _check_image(image)
    cfg = _load_config()
    count, conf = _count(cfg, _crop(image, align_rois(image, cfg), compartment_id))
    return CountResult(compartment_id, count, conf)
