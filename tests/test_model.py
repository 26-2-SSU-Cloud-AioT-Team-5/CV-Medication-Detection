import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from detector import model as m  # noqa: E402

K = 8  # 클래스 수 (개수 0~7)


def r_model(make_model):
    # R 채널이 밝으면 개수 3, 아니면 0 (BGR→RGB 변환 확인용)
    w = np.zeros((12, K))
    w[:4, 3] = 5.0   # Flatten은 채널 우선 -> 앞 4개가 R. R 전부 1이면 클래스3 = 20
    b = np.zeros(K)
    b[0] = 10.0      # 기본은 클래스0
    return m.load_model(make_model(w, b, [1, K]))


def bgr(channel):
    image = np.zeros((6, 8, 3), np.uint8)  # 입력 크기(2x2)와 다르게 해서 리사이즈도 거치게 한다
    image[..., channel] = 255
    return image


def test_preprocess():
    x = m.preprocess(bgr(2), (2, 2))
    assert x.shape == (1, 3, 2, 2) and x.dtype == np.float32
    assert np.allclose(x[0, 0], 1.0) and np.allclose(x[0, 1:], 0.0)  # BGR의 R -> RGB 0번 채널, 0~1


def test_predict_count(make_model):
    session = r_model(make_model)
    count, conf = m.predict(session, bgr(2), (2, 2))
    assert (count, isinstance(conf, float)) == (3, True) and 0.99 < conf <= 1.0
    assert m.predict(session, bgr(0), (2, 2))[0] == 0  # B만 밝음 -> R은 0


@pytest.mark.parametrize("out_shape", [[K], [1, 2, K // 2]])
def test_wrong_output_shape(make_model, out_shape):
    session = m.load_model(make_model(np.zeros((12, K)), np.zeros(K), out_shape))
    with pytest.raises(ValueError, match="shape"):
        m.predict(session, bgr(0), (2, 2))


def test_load_model_is_cached(make_model):
    path = make_model(np.zeros((12, K)), np.zeros(K), [1, K])
    assert m.load_model(path) is m.load_model(path)
