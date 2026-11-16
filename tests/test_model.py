import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from detector import model as m  # noqa: E402

K = 5  # 칸당 클래스 수 (개수 0~4)


def count_model(make_model):
    # 칸 2~4는 bias로 개수 2, 1, 3 고정. 칸 1은 R 채널이 밝으면 1개, 아니면 0개 (BGR→RGB 변환 확인용)
    w = np.zeros((12, 4 * K))
    w[:4, 1] = 5.0             # Flatten은 채널 우선 -> 앞 4개가 R. R 전부 1이면 칸1 클래스1 = 20
    b = np.zeros(4 * K)
    b[0] = 10.0                # 칸1 클래스0
    for i, c in enumerate([2, 1, 3], start=1):
        b[i * K + c] = 10.0
    return m.load_model(make_model(w, b, [1, 4, K]))


def bgr(channel):
    image = np.zeros((6, 8, 3), np.uint8)  # 입력 크기(2x2)와 다르게 해서 리사이즈도 거치게 한다
    image[..., channel] = 255
    return image


def test_predict_counts(make_model):
    session = count_model(make_model)

    out = m.predict(session, bgr(2), (2, 2), 4)  # BGR의 2번 = R
    assert {k: v[0] for k, v in out.items()} == {1: 1, 2: 2, 3: 1, 4: 3}
    assert all(0.99 < conf <= 1.0 for _, conf in out.values())

    assert m.predict(session, bgr(0), (2, 2), 4)[1][0] == 0  # B만 밝음 -> R은 0


@pytest.mark.parametrize("out_shape", [[1, 4 * K], [1, 2, 2 * K]])
def test_wrong_output_shape(make_model, out_shape):
    session = m.load_model(make_model(np.zeros((12, 4 * K)), np.zeros(4 * K), out_shape))
    with pytest.raises(ValueError, match="shape"):
        m.predict(session, bgr(0), (2, 2), 4)


def test_load_model_is_cached(make_model):
    path = make_model(np.zeros((12, 4 * K)), np.zeros(4 * K), [1, 4, K])
    assert m.load_model(path) is m.load_model(path)
