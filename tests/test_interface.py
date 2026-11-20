import numpy as np
import pytest

from detector import interface
from detector.interface import CountResult, count_pills

K = 8
# 왼쪽 절반(칸1)은 R이 밝고 오른쪽 절반(칸2)은 검정
IMAGE = np.zeros((6, 8, 3), np.uint8)
IMAGE[:, :4, 2] = 255
ROIS = {1: [0, 0, 4, 6], 2: [4, 0, 4, 6], 3: None, 4: [6, 0, 4, 6]}


def use_config(monkeypatch, path, rois=ROIS):
    cfg = {"model": {"path": path, "input_size": [2, 2]}, "rois": rois}
    monkeypatch.setattr(interface, "_load_config", lambda: cfg)


def test_count_pills_crops_roi(monkeypatch, make_model):
    w = np.zeros((12, K))
    w[:4, 3] = 5.0   # R이 밝으면 개수 3
    b = np.zeros(K)
    b[0] = 10.0      # 아니면 0
    use_config(monkeypatch, make_model(w, b, [1, K]))

    r1 = count_pills(IMAGE, 1)
    assert isinstance(r1, CountResult) and (r1.compartment_id, r1.count) == (1, 3)
    assert isinstance(r1.confidence, float) and 0.99 < r1.confidence <= 1.0
    assert count_pills(IMAGE, 2).count == 0  # 다른 칸은 다른 영역을 본다


@pytest.mark.parametrize("image", [
    None,
    [[0, 0, 0]],
    np.zeros((4, 4), np.uint8),
    np.zeros((4, 4, 4), np.uint8),
    np.zeros((4, 4, 3), np.float32),
    np.zeros((0, 4, 3), np.uint8),
])
def test_bad_image(monkeypatch, image):
    use_config(monkeypatch, "unused.onnx")
    with pytest.raises(ValueError):
        count_pills(image, 1)


@pytest.mark.parametrize("cid", [0, 5, "1", None])
def test_bad_compartment_id(monkeypatch, cid):
    use_config(monkeypatch, "unused.onnx")
    with pytest.raises(ValueError, match="compartment_id"):
        count_pills(IMAGE, cid)


def test_roi_unset(monkeypatch):
    use_config(monkeypatch, "unused.onnx")
    with pytest.raises(RuntimeError, match="rois.3"):
        count_pills(IMAGE, 3)


def test_roi_outside_image(monkeypatch):
    use_config(monkeypatch, "unused.onnx")
    with pytest.raises(ValueError, match="rois.4"):
        count_pills(IMAGE, 4)


def test_model_path_unset(monkeypatch):
    use_config(monkeypatch, None)
    with pytest.raises(RuntimeError, match="model.path"):
        count_pills(IMAGE, 1)
