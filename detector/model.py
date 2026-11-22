# 학습 모델 실행 (ONNX Runtime, RPi 기기 내 추론). 모델 규격은 README 참고.
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).parent.parent


@lru_cache(maxsize=1)  # 호출마다 다시 읽지 않도록 프로세스당 한 번만 로드
def load_model(path: str) -> ort.InferenceSession:
    # 상대 경로는 저장소 루트 기준 (실행 위치와 무관)
    return ort.InferenceSession(str(ROOT / path), providers=["CPUExecutionProvider"])


def preprocess(crop: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    # 칸 크롭 BGR -> 1x3xHxW float32 RGB 0~1. mean/std는 모델 안에 있음
    # 학습 데이터(resize_test.py)와 같은 PIL LANCZOS 리사이즈 (cv2 LANCZOS4는 축소 시 결과가 다름)
    rgb = Image.fromarray(np.ascontiguousarray(crop[..., ::-1]))
    x = np.asarray(rgb.resize(tuple(input_size), Image.Resampling.LANCZOS), np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]


def predict(session: ort.InferenceSession, crop: np.ndarray,
            input_size: tuple[int, int] = (224, 224)) -> tuple[Optional[int], float]:
    # 칸 하나의 BGR 크롭 -> (개수, confidence 0~1).
    # 출력 0: 개수 로짓 (1, K), 인덱스 = 개수. 출력 1(Phase 2 모델만): 상태 로짓 (1, 3)
    # 상태가 countable(0)이 아니면(손 가림, 뚜껑 닫힘) 개수 None, confidence는 그 상태의 확률
    outputs = session.run(None, {session.get_inputs()[0].name: preprocess(crop, input_size)})
    if len(outputs) > 1:
        status = _softmax(outputs[1])
        if status.argmax() != 0:
            return None, float(status.max())
    p = _softmax(outputs[0])
    count = int(p.argmax())
    return count, float(p[count])


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(f"모델 출력 shape이 (1, K)가 아님: {logits.shape}")
    z = logits[0].astype(np.float64)
    p = np.exp(z - z.max())  # 최댓값을 빼서 overflow 방지
    return p / p.sum()
