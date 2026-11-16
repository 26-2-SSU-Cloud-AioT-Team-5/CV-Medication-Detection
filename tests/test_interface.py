import numpy as np
import pytest

from detector import interface
from detector.interface import CountResult, count_pills

K = 4


def use_config(monkeypatch, path):
    cfg = {"model": {"path": path, "input_size": [2, 2], "num_compartments": 4}}
    monkeypatch.setattr(interface, "_load_config", lambda: cfg)


def test_count_pills(monkeypatch, make_model):
    rng = np.random.default_rng(0)
    use_config(monkeypatch, make_model(rng.normal(size=(12, 4 * K)), rng.normal(size=4 * K), [1, 4, K]))

    out = count_pills(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    assert sorted(out) == [1, 2, 3, 4]
    for cid, r in out.items():
        assert isinstance(r, CountResult) and r.compartment_id == cid
        assert isinstance(r.count, int) and 0 <= r.count < K
        assert isinstance(r.confidence, float) and 1 / K <= r.confidence <= 1.0


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
        count_pills(image)


def test_model_path_unset(monkeypatch):
    use_config(monkeypatch, None)
    with pytest.raises(RuntimeError, match="model.path"):
        count_pills(np.zeros((4, 4, 3), np.uint8))
