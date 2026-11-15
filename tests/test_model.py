import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from detector import model as m  # noqa: E402


def make_model(path, n_out):
    # 1x3x2x2 RGB 입력에서 R 채널만 본다: 있음 로짓 z = 5*sum(R) - 10 (R 전부 1이면 +10, 전부 0이면 -10)
    # n_out=1이면 [z], n_out=2이면 [0, z]를 낸다 (ResNet18 이진 분류의 두 가지 출력 형태)
    w_filled = np.repeat(np.array([5.0, 0.0, 0.0], np.float32), 4)[:, None]  # Flatten 순서는 채널 우선
    weights = w_filled if n_out == 1 else np.hstack([np.zeros_like(w_filled), w_filled])
    bias = np.array([-10.0] if n_out == 1 else [0.0, -10.0], np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Flatten", ["x"], ["flat"]),
            helper.make_node("MatMul", ["flat", "w"], ["scores"]),
            helper.make_node("Add", ["scores", "b"], ["logits"]),
        ],
        "t",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, n_out])],
        [numpy_helper.from_array(weights, "w"), numpy_helper.from_array(bias, "b")],
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.save(proto, str(path))
    m.load_model.cache_clear()
    return m.load_model(str(path))


def bgr(channel, value):
    image = np.zeros((2, 2, 3), np.uint8)
    image[..., channel] = value
    return image


@pytest.mark.parametrize("n_out", [1, 2])
def test_predict(tmp_path, n_out):
    session = make_model(tmp_path / "m.onnx", n_out)

    has_pill, confidence = m.predict(session, bgr(2, 255))  # BGR의 2번 = R
    assert has_pill is True and confidence > 0.99

    has_pill, confidence = m.predict(session, bgr(0, 255))  # B만 밝음 -> R은 0
    assert has_pill is False and confidence > 0.99


def test_threshold(tmp_path):
    session = make_model(tmp_path / "m.onnx", 1)
    image = bgr(2, 128)  # 알약 있음 확률이 0.5를 살짝 넘는 입력
    assert m.predict(session, image, 0.5)[0] is True
    assert m.predict(session, image, 0.99)[0] is False


def test_load_model_is_cached(tmp_path):
    path = tmp_path / "m.onnx"
    assert make_model(path, 1) is m.load_model(str(path))
