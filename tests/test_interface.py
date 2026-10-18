from detector import interface
from detector.interface import DetectResult, detect


def test_dummy_detect(monkeypatch):
    # config.yaml 값과 무관하게 dummy로 고정
    monkeypatch.setattr(interface, "_load_config", lambda: {"detector": {"backend": "dummy"}})
    r = detect(1)
    assert isinstance(r, DetectResult)
    assert r.compartment_id == 1
    assert isinstance(r.has_pill, bool)
    assert isinstance(r.confidence, float) and 0.0 <= r.confidence <= 1.0
    assert isinstance(r.captured_at, float)
