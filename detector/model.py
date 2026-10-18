#학습 모델 판별. 추후 (규칙 기반이 부족할 때만).
import numpy as np


def load_model(path: str):
    #path의 모델을 로드해 반환한다.
    raise NotImplementedError("TODO: 모델 로딩")


def predict(model, image: np.ndarray) -> tuple[bool, float]:
    #전처리된 이미지로 (has_pill, confidence 0~1)를 반환한다.
    raise NotImplementedError("TODO: 모델 추론")
