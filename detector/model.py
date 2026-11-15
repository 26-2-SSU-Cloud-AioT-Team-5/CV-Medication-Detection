#학습 모델 판별 (ResNet18 이진 분류, ONNX Runtime, RPi 기기 내 추론). 모델 규격은 README 참고.
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).parent.parent


@lru_cache(maxsize=1)  # detect()마다 호출되므로 프로세스당 한 번만 읽는다
def load_model(path: str) -> ort.InferenceSession:
    #path의 모델을 로드해 반환한다. 상대 경로는 저장소 루트 기준 (실행 위치와 무관).
    return ort.InferenceSession(str(ROOT / path), providers=["CPUExecutionProvider"])


def predict(model: ort.InferenceSession, image: np.ndarray, threshold: float = 0.5) -> tuple[bool, float]:
    #ROI 자르기/리사이즈를 거친 BGR 이미지(H, W, 3, uint8)로 (has_pill, confidence 0~1)를 반환한다.
    #mean/std 정규화는 모델 안에서 하므로 여기서는 RGB 0~1로만 바꿔 넘긴다.
    x = image[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR HWC -> RGB NCHW
    logits = model.run(None, {model.get_inputs()[0].name: x})[0].reshape(-1)
    # 출력이 1개면 "있음" 로짓(sigmoid), 2개면 [없음, 있음] 로짓(softmax). 둘 다 같은 식으로 귀결된다.
    z = logits[0] if logits.size == 1 else logits[1] - logits[0]
    p_filled = float(0.5 * (1.0 + np.tanh(z / 2.0)))  # sigmoid, 큰 로짓에서도 overflow 없음
    has_pill = p_filled >= threshold
    return has_pill, p_filled if has_pill else 1.0 - p_filled
