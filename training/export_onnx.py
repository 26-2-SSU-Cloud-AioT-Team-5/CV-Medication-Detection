"""학습 체크포인트(.pt)를 RPi용 ONNX로 변환한다.

모델: Phase 2 체크포인트면 architecture/model.PillStateClassifier
(출력 count_logits [1,8] 인덱스 = 알약 개수, status_logits [1,3] = STATUS_NAMES),
Phase 1이면 build_model (count_logits만).
ImageNet 정규화를 모델 안에 넣어 detector/model.py 규격(RGB 0~1 입력, 1x3x224x224)에 맞춘다.
변환 후 PyTorch(학습 방식 전처리)와 ONNX Runtime 결과가 같은지 스스로 확인한다.

사용 (학습 환경, torch/torchvision/onnxruntime 필요):
    python -m training.export_onnx training/checkpoints/<파일>.pt
    python -m training.export_onnx --random bench.onnx   # 가중치 없이 RPi 속도 측정용
결과는 체크포인트 옆 .onnx. 그 경로를 detector/config.yaml의 model.path에 적는다.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))  # python training/export_onnx.py 로도 실행되도록

from architecture.model import PillStateClassifier, build_model, load_model_state  # noqa: E402  (학습 파트 코드)

# training/dataset_sampler.py의 NORMALIZATION_MEAN/STD와 같은 값 (ImageNet)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SIZE = 224


class Normalized(nn.Module):
    # 0~1 RGB 입력에 학습 때와 같은 mean/std 정규화를 적용한 뒤 모델을 돌린다.
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        return self.model((x - self.mean) / self.std)


def export(checkpoint_path, onnx_path: Path) -> None:
    # None = 랜덤 가중치 (RPi 속도 측정용, 개수는 의미 없음)
    state = None if checkpoint_path is None else load_model_state(checkpoint_path)
    dual = state is None or any(k.startswith("status_head.") for k in state)
    model = PillStateClassifier() if dual else build_model(pretrained=False)
    if state is not None:
        model.load_state_dict(state, strict=True)
    model.eval()
    names = ["count_logits", "status_logits"] if dual else ["count_logits"]
    wrapped = Normalized(model).eval()

    x = torch.rand(1, 3, SIZE, SIZE)
    # ponytail: 구 exporter(dynamo=False)는 추가 패키지가 필요 없다. torch에서 제거되면 dynamo=True + `pip install onnxscript`로 교체
    torch.onnx.export(wrapped, x, str(onnx_path), input_names=["image"], output_names=names,
                      opset_version=17, dynamo=False)

    # 확인: 학습 방식(모델 밖 정규화)으로 돌린 PyTorch 결과 == 정규화를 품은 ONNX 결과
    with torch.no_grad():
        expected = model((x - wrapped.mean) / wrapped.std)
    expected = [t.numpy() for t in (expected if dual else [expected])]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": x.numpy()})
    assert [a.shape for a in actual] == [(1, 8), (1, 3)][:len(names)], [a.shape for a in actual]
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=1e-3, atol=1e-4)
    print(f"saved {onnx_path} (input 1x3x{SIZE}x{SIZE}, outputs {names}, ONNX == torch)")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--random":
        export(None, Path(sys.argv[2]))
    elif len(sys.argv) == 2:
        ckpt = Path(sys.argv[1])
        export(ckpt, ckpt.with_suffix(".onnx"))
    else:
        sys.exit("usage: python -m training.export_onnx <checkpoint.pt>\n"
                 "       python -m training.export_onnx --random <out.onnx>   # 속도 측정용 랜덤 가중치")
