# pytest가 저장소 루트를 sys.path에 넣도록 두는 파일 (tests에서 `import detector` 가능) + 공용 가짜 모델
import numpy as np
import pytest


@pytest.fixture
def make_model(tmp_path):
    # 1x3x2x2 RGB 입력 -> 로짓 = flatten(x) @ w + b 를 out_shape으로 reshape한 가짜 ONNX 모델. 경로를 반환
    # status를 주면 두 번째 출력으로 그 상수 상태 로짓 (1, 3)을 낸다 (Phase 2 모델 흉내)
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    def make(w, b, out_shape, status=None):
        extra_nodes, extra_outputs = [], []
        if status is not None:
            extra_nodes = [helper.make_node("Constant", [], ["status"], value=numpy_helper.from_array(
                np.asarray([status], np.float32)))]
            extra_outputs = [helper.make_tensor_value_info("status", TensorProto.FLOAT, None)]
        graph = helper.make_graph(
            extra_nodes + [
                helper.make_node("Flatten", ["x"], ["flat"]),
                helper.make_node("MatMul", ["flat", "w"], ["scores"]),
                helper.make_node("Add", ["scores", "b"], ["sum"]),
                helper.make_node("Reshape", ["sum", "shape"], ["logits"]),
            ],
            "t",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 2, 2])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, None)] + extra_outputs,
            [
                numpy_helper.from_array(np.asarray(w, np.float32), "w"),
                numpy_helper.from_array(np.asarray(b, np.float32), "b"),
                numpy_helper.from_array(np.array(out_shape, np.int64), "shape"),
            ],
        )
        path = tmp_path / "m.onnx"
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8), str(path))
        from detector import model
        model.load_model.cache_clear()
        return str(path)

    return make
