import pytest

from detector.calibrate import write_rois

CONFIG = """model:
  input_size: [224, 224]  # 주석
# rois 설명
rois:
  1: null
  2: [1, 2, 3, 4]
"""


def test_write_rois_keeps_comments():
    out = write_rois(CONFIG, {1: [10, 20, 30, 40], 2: [5, 6, 7, 8]})
    assert "  1: [10, 20, 30, 40]\n  2: [5, 6, 7, 8]\n" in out
    assert "# 주석" in out and "# rois 설명" in out and "input_size: [224, 224]" in out


def test_write_rois_unknown_id():
    with pytest.raises(ValueError, match="칸 3"):
        write_rois(CONFIG, {3: [1, 1, 1, 1]})
