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


# 자동 보정: 통이 밀려도 config rois를 매 사진마다 검은 바닥판 위로 옮긴다. 두 단계로 찾는다.
#   1) 통 전체 이동: 4칸을 한 덩어리로 사진 크기의 ±GLOBAL_SHIFT 안에서 찾는다 (칸끼리 바뀌지 않게)
#   2) 칸별 미세 이동: 1)에서 옮긴 자리 주변 칸 크기의 ±MAX_SHIFT
# ponytail: 평행이동만 찾음 (회전·확대 없음). 카메라를 바꾸거나 크게 돌렸으면 calibrate로 다시 잡을 것
GLOBAL_SHIFT = 0.25   # 사진 가로·세로 대비 통 전체 이동 탐색 범위 (4608x2592 기준 약 ±1150, ±650 px)
MAX_SHIFT = 0.2       # 칸 크기 대비 칸별 미세 이동 범위 (이웃 칸 바닥판에 걸리지 않을 만큼만)
PLATEAU = 0.02        # 최고 점수와 이 차이 안의 위치들 중에서 고른다 (roi가 바닥판보다 작아 꽉 차는 위치가 여러 개)
BAND = (0.10, 0.25)   # 바닥판 위·아래 흰 벽을 보는 띠 (칸 높이 대비 거리). 검은 배경(책상, 벽)에 박스가 붙지 않게


def align_rois(image: np.ndarray, cfg: dict) -> dict:
    # 바닥판 위치로 roi를 옮긴다. 뚜껑이 다 닫혀 바닥판이 안 보이면 config 값 그대로
    w = cfg.get("watcher", {})
    level, need = w.get("dark_level", 100), w.get("open_dark_ratio", 0.5)
    rois = cfg["rois"]
    sizes = [min(r[2], r[3]) for r in rois.values() if r]
    s = max(1, min(sizes, default=1) // 60)  # 칸 한 변을 약 60칸으로 줄여 계산 (RPi 속도)
    dark = image[::s, ::s].mean(axis=2) < level
    integral = np.pad(dark.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    gx, gy = _global_shift(integral, s, rois, image.shape, need)
    moved = {cid: r and [r[0] + gx, r[1] + gy, r[2], r[3]] for cid, r in rois.items()}
    out = {cid: r and _align(integral, s, r, image.shape, need) for cid, r in moved.items()}
    # 닫힌 칸(미세 이동 못 함)은 열린 칸들이 실제로 움직인 평균만큼 같이 옮긴다
    found = [(o[0] - r[0], o[1] - r[1]) for cid, r in rois.items() if r and (o := out[cid]) != moved[cid]]
    if found:
        mx, my = (int(round(v)) for v in np.mean(found, axis=0))
        out = {cid: r and (out[cid] if out[cid] != moved[cid] else [r[0] + mx, r[1] + my, r[2], r[3]])
               for cid, r in rois.items()}
    return out


def _box_mean(integral: np.ndarray, x0, y0, bw: int, bh: int) -> np.ndarray:
    # 줄인 좌표의 왼쪽 위 (y0 x x0 격자)에서 bw x bh 박스의 바닥판 비율. 사진 밖으로 나간 부분은 0으로 센다
    H, W = integral.shape[0] - 1, integral.shape[1] - 1
    xa, xb = np.clip(x0, 0, W), np.clip(x0 + bw, 0, W)
    ya, yb = np.clip(y0, 0, H), np.clip(y0 + bh, 0, H)
    total = (integral[np.ix_(yb, xb)] - integral[np.ix_(ya, xb)]
             - integral[np.ix_(yb, xa)] + integral[np.ix_(ya, xa)])
    return total / max(1, bw * bh)


def _global_shift(integral: np.ndarray, s: int, rois: dict, shape: tuple, need: float) -> tuple[int, int]:
    # 통 전체 이동 (dx, dy). 점수 = 칸마다 (박스 안 바닥판 비율 - 위아래 띠의 바닥판 비율)의 합
    boxes = [r for r in rois.values() if r and r[2] // s and r[3] // s]
    if not boxes:
        return 0, 0
    dxs = np.arange(-int(shape[1] * GLOBAL_SHIFT), int(shape[1] * GLOBAL_SHIFT) + 1, s)
    dys = np.arange(-int(shape[0] * GLOBAL_SHIFT), int(shape[0] * GLOBAL_SHIFT) + 1, s)
    # 모든 박스가 사진 안에 남는 이동만 본다
    dxs = dxs[np.all([(x + dxs >= 0) & (x + dxs + w <= shape[1]) for x, y, w, h in boxes], axis=0)]
    dys = dys[np.all([(y + dys >= 0) & (y + dys + h <= shape[0]) for x, y, w, h in boxes], axis=0)]
    if dxs.size == 0 or dys.size == 0:
        return 0, 0
    score = np.zeros((dys.size, dxs.size))
    found = np.zeros((dys.size, dxs.size), int)  # 바닥판으로 보이는 칸 수
    for x, y, w, h in boxes:
        bw, bh = w // s, h // s
        x0, y0 = (x + dxs) // s, (y + dys) // s
        near, far = int(bh * BAND[0]), int(bh * BAND[1])
        inside = _box_mean(integral, x0, y0, bw, bh)
        band = (_box_mean(integral, x0, y0 - far, bw, far - near)
                + _box_mean(integral, x0, y0 + bh + near, bw, far - near)) / 2
        score += inside - band
        found += (inside >= need) & (band <= 0.3)
    # 바닥판이 2칸 이상 맞아야 크게 옮긴다. 1칸만 보이면 사진 어디든 어두운 곳에 붙을 수 있어서 칸별 미세 이동만 한다
    # ponytail: 뚜껑이 1개만 열린 사진에서는 ±MAX_SHIFT까지만 따라감
    if found.max() < 2:
        return 0, 0
    best = (found == found.max()) & (score >= score[found == found.max()].max() - 0.1 * len(boxes))
    # 그중 덜 움직인 쪽 (일부 칸만 열렸을 때 칸 한 개만큼 건너뛴 자리로 가지 않게)
    dist = np.where(best, dys[:, None] ** 2 + dxs[None, :] ** 2, np.inf)
    iy, ix = np.unravel_index(dist.argmin(), dist.shape)
    return int(dxs[ix]), int(dys[iy])


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
