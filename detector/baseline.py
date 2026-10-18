# 기준(빈 칸) 이미지와 비교해 알약 유무 판별
import numpy as np


def register_baseline(compartment_id: int, image: np.ndarray) -> None:
    raise NotImplementedError("TODO: 칸별 기준 등록")


def compare(compartment_id: int, image: np.ndarray, threshold: float) -> tuple[bool, float]:
    # 기준과 현재 이미지의 차이로 (has_pill, confidence 0~1)를 반환한다.
    # 차이 점수 > threshold 이면 알약이 있다고 본다.

    raise NotImplementedError("TODO: 기준 이미지 비교 판별")
