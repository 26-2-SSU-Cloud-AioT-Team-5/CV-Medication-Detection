#전처리
import numpy as np


def crop_roi(image: np.ndarray, roi: list[int]) -> np.ndarray:
    #roi = [x, y, w, h] (픽셀) 영역만 잘라 반환한다.
    raise NotImplementedError("TODO: ROI 자르기")


def resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    #size = (width, height)로 크기를 맞춘다.
    raise NotImplementedError("TODO: 리사이즈")


def normalize_brightness(image: np.ndarray) -> np.ndarray:
    #조명 차이를 줄이도록 밝기를 정규화한다 (예: 그레이 + 히스토그램 평활화/CLAHE).
    raise NotImplementedError("TODO: 밝기 정규화")
