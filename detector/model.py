# 학습 모델 실행 (ONNX Runtime, RPi 기기 내 추론). 모델 규격은 README 참고.
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).parent.parent


@lru_cache(maxsize=1)  # 호출마다 다시 읽지 않도록 프로세스당 한 번만 로드
def load_model(path: str) -> ort.InferenceSession:
    # 상대 경로는 저장소 루트 기준 (실행 위치와 무관)
    return ort.InferenceSession(str(ROOT / path), providers=["CPUExecutionProvider"])


def predict(
    session: ort.InferenceSession,
    image: np.ndarray,
    input_size: tuple[int, int] = (224, 224),
    num_compartments: int = 4,
) -> dict[int, tuple[int, float]]:
    # 통 전체 BGR 이미지 -> {칸 번호: (개수, confidence 0~1)}
    # TODO(학습 파트 확정): 입력 변환. 리사이즈 -> BGR→RGB -> /255 -> float32 NCHW, mean/std는 모델 안에 있다고 가정
    x = cv2.resize(image, tuple(input_size), interpolation=cv2.INTER_AREA)
    x = x[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    logits = session.run(None, {session.get_inputs()[0].name: x})[0]
    # TODO(학습 파트 확정): 출력 해석. (1, 칸 수, K) 칸별 클래스 로짓, 클래스 인덱스 = 개수, 순서 i = 칸 i+1
    if logits.ndim != 3 or logits.shape[:2] != (1, num_compartments):
        raise ValueError(f"모델 출력 shape이 (1, {num_compartments}, K)가 아님: {logits.shape}")

    z = logits[0].astype(np.float64)
    p = np.exp(z - z.max(axis=1, keepdims=True))  # 최댓값을 빼서 overflow 방지
    p /= p.sum(axis=1, keepdims=True)
    counts = p.argmax(axis=1)
    return {i + 1: (int(counts[i]), float(p[i, counts[i]])) for i in range(num_compartments)}
