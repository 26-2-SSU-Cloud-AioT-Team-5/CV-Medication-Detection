# 뚜껑 열림/닫힘 판단 + 닫힐 때 칸별 개수 확정. 뚜껑 센서가 없어서 사진으로 판단한다.
#
#   watcher = PillboxWatcher()
#   while True:                                    # SW: 촬영 주기(약 1초)와 MQTT 전송 담당
#       for event in watcher.update(camera.capture(), now()):
#           mqtt.publish(topic, json.dumps(asdict(event), default=str))
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from .interface import _check_image, _count, _crop, _load_config, align_rois


@dataclass
class CountEvent:
    compartment_id: int
    count: Optional[int]          # status가 "uncertain"이면 None
    confidence: Optional[float]
    status: str                   # "ok" | "uncertain" (서버는 uncertain을 판정에 쓰지 않음)
    captured_at: Optional[datetime]  # 확정에 쓴 마지막 사진 시각 (uncertain이면 닫힘 확정 시각)


@dataclass
class _Compartment:
    is_open: bool = False
    streak: int = 0    # 지금 상태와 다른 판단이 연속으로 나온 장 수
    history: list = field(default_factory=list)  # 열려 있는 동안의 (개수, conf, 시각)


def is_open(crop: np.ndarray, dark_level: int, open_dark_ratio: float) -> bool:
    # 열린 칸은 검은 바닥판, 닫힌 칸은 흰 뚜껑이 보인다
    # ponytail: 밝기 비율만 봄. 바닥판 색이 바뀌거나 조명이 어두워 틀리면 닫힌 뚜껑 기준 사진과의 차이로 바꿀 것
    return bool((crop.mean(axis=2) < dark_level).mean() >= open_dark_ratio)


def stable_result(history: list, frames: int, min_conf: float) -> Optional[tuple]:
    # 같은 개수가 frames장 연속 conf >= min_conf로 나온 구간 중 가장 최근 것의 마지막 (개수, conf, 시각)
    best, run = None, 0
    # 셀 수 없는 사진(count None: 손 가림 등)은 구간을 끊는다
    for i, (count, conf, _) in enumerate(history):
        if count is None or conf < min_conf:
            run = 0
            continue
        run = run + 1 if run and history[i - 1][0] == count else 1
        if run >= frames:
            best = history[i]
    return best


class PillboxWatcher:
    def __init__(self, config: Optional[dict] = None) -> None:
        self.cfg = config or _load_config()
        self.compartments = {cid: _Compartment() for cid in self.cfg["rois"]}

    def update(self, image: np.ndarray, captured_at: datetime) -> list[CountEvent]:
        # 통 전체 BGR 사진 한 장 -> 이번 사진에서 닫힘이 확정된 칸들의 결과 (보통 빈 리스트)
        _check_image(image)
        w = self.cfg["watcher"]
        events = []
        rois = align_rois(image, self.cfg)  # 통이 밀려도 매 사진 자동 보정
        for cid, c in self.compartments.items():
            crop = _crop(image, rois, cid)
            now_open = is_open(crop, w["dark_level"], w["open_dark_ratio"])
            if now_open:
                count, conf = _count(self.cfg, crop)
                c.history.append((count, conf, captured_at))

            c.streak = c.streak + 1 if now_open != c.is_open else 0
            if c.streak >= w["switch_frames"]:  # 상태 전환 확정
                c.is_open, c.streak = now_open, 0
                if not now_open:
                    events.append(self._confirm(cid, c.history, captured_at))
                    c.history = []
            elif not c.is_open and not now_open:
                c.history = []  # 닫힌 상태에서 한두 장 튄 열림은 버린다
        return events

    def _confirm(self, cid: int, history: list, closed_at: datetime) -> CountEvent:
        w = self.cfg["watcher"]
        best = stable_result(history, w["stable_frames"], w["min_confidence"])
        if best is None:
            return CountEvent(cid, None, None, "uncertain", closed_at)
        count, conf, at = best
        return CountEvent(cid, count, conf, "ok", at)
