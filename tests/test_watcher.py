import numpy as np
import pytest

from detector import watcher as wm
from detector.watcher import PillboxWatcher, is_open, stable_result

CFG = {
    "model": {"path": "unused.onnx", "input_size": [2, 2]},
    "watcher": {"dark_level": 100, "open_dark_ratio": 0.5, "switch_frames": 2,
                "stable_frames": 2, "min_confidence": 0.8},
    "rois": {1: [0, 0, 4, 4], 2: [4, 0, 4, 4]},
}


def frame(c1=None, c2=None):
    # 칸 하나: None = 닫힘(흰 뚜껑), (개수, conf) = 열림(어두운 바닥, 가짜 모델이 B=개수, G=conf*100을 읽음)
    image = np.full((4, 8, 3), 255, np.uint8)
    for x, cell in ((0, c1), (4, c2)):
        if cell is not None:
            image[:, x:x + 4] = (cell[0], round(cell[1] * 100), 0)
    return image


@pytest.fixture
def watcher(monkeypatch):
    calls = []

    def fake_count(cfg, crop):
        calls.append(crop)
        count = int(crop[0, 0, 0])
        return (None if count == 9 else count), crop[0, 0, 1] / 100  # B=9 -> 셀 수 없음(손 가림)

    monkeypatch.setattr(wm, "_count", fake_count)
    w = PillboxWatcher(CFG)
    w.calls = calls
    return w


def run(w, frames):
    return {t: w.update(img, t) for t, img in enumerate(frames)}


def test_confirms_latest_stable_count_on_close(watcher):
    # 칸 2: 알약 3개 중 1개 꺼냄. 손(3~5초)과 닫히는 뚜껑(9초)은 걸러져야 한다
    c2 = [None, (3, .95), (3, .96), (5, .4), (1, .5), (2, .6), (2, .93), (2, .95), (2, .94), (6, .7), None, None]
    out = run(watcher, [frame(None, c) for c in c2])
    events = [e for t in out for e in out[t]]
    assert len(events) == 1 and out[11] == events
    e = events[0]
    assert (e.compartment_id, e.count, e.confidence, e.status, e.captured_at) == (2, 2, .94, "ok", 8)
    assert len(watcher.calls) == 9  # 모델은 열린 사진에만, 닫힌 칸 1에는 안 돌린다


def test_hand_frame_breaks_stable_run(watcher):
    # 손 가림(개수 None) 사진끼리 연속 2장이어도 확정하지 않는다
    out = run(watcher, [frame(c2=c) for c in [(3, .95), (9, .99), (9, .99), None, None]])
    (e,) = out[4]
    assert e.status == "uncertain"


def test_uncertain_when_never_stable(watcher):
    out = run(watcher, [frame(c2=c) for c in [(3, .95), (4, .95), (4, .5), None, None]])
    (e,) = out[4]
    assert (e.compartment_id, e.count, e.confidence, e.status, e.captured_at) == (2, None, None, "uncertain", 4)


def test_two_compartments_independent(watcher):
    frames = [frame((1, .9), (4, .9))] * 3 + [frame(None, (4, .9)), frame(None, (4, .9)), frame(), frame()]
    out = run(watcher, frames)
    assert [(e.compartment_id, e.count) for e in out[4]] == [(1, 1)]
    assert [(e.compartment_id, e.count) for e in out[6]] == [(2, 4)]
    assert sum(map(len, out.values())) == 2


def test_single_frame_flicker_is_ignored(watcher):
    # 닫힌 상태에서 한 장만 열림 -> 이벤트 없음, 그 결과도 다음 세션에 섞이지 않음
    frames = [frame(c2=(7, .99)), frame()] + [frame(c2=(1, .9))] * 3 + [frame(), frame()]
    out = run(watcher, frames)
    (e,) = out[6]
    assert sum(map(len, out.values())) == 1 and e.count == 1


def test_stable_result():
    h = [(3, .9, 0), (3, .9, 1), (3, .9, 2), (2, .9, 3), (2, .5, 4), (2, .9, 5), (2, .9, 6)]
    assert stable_result(h, 3, .8) == (3, .9, 2)        # 2는 low conf로 끊겨서 3장 연속이 안 됨
    assert stable_result(h, 2, .8) == (2, .9, 6)        # 가장 최근 구간
    assert stable_result(h[:2], 3, .8) is None


def test_is_open():
    dark, white = np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 230, np.uint8)
    assert is_open(dark, 100, .5) and not is_open(white, 100, .5)
    half = white.copy()
    half[:2] = 0
    assert is_open(half, 100, .5) and not is_open(half, 100, .6)
