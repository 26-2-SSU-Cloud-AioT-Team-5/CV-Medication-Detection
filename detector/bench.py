"""RPi 약통에서 CV 처리 시간 측정. 목표: watcher.update 한 번이 1초(촬영 간격) 안.

    python -m detector.bench 모델.onnx                       # 가짜 4608x2592 사진으로
    python -m detector.bench 모델.onnx --photos 열림.jpg 닫힘.jpg   # 실제 사진으로
    python -m detector.bench 모델.onnx --camera              # picamera2 촬영 시간도 측정

모델이 아직 없으면 PC에서 `python -m training.export_onnx --random models/bench.onnx`로 만들어 복사한다
(랜덤 가중치라 개수는 의미 없고 속도만 같다).
"""
import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import model
from .interface import _load_config
from .watcher import PillboxWatcher

SIZE = (2592, 4608)  # Camera Module 3 원본 (H, W)


def timed(fn, n: int) -> tuple[float, float]:
    # n번 실행 -> (중앙값, 최댓값) ms
    ms = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - start) * 1000)
    return float(np.median(ms)), max(ms)


def fake_photos(rois: dict) -> tuple[np.ndarray, np.ndarray]:
    # 닫힘: 흰 뚜껑, 열림: 칸 영역만 어두운 바닥판(노이즈)
    closed = np.full((*SIZE, 3), 200, np.uint8)
    opened = closed.copy()
    rng = np.random.default_rng(0)
    for x, y, w, h in rois.values():
        opened[y:y + h, x:x + w] = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
    return opened, closed


def load_photo(path: str) -> np.ndarray:
    from PIL import Image
    return np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB"))[..., ::-1])  # RGB -> BGR


def system_info() -> None:
    import onnxruntime as ort
    print(f"platform  : {platform.platform()} ({platform.machine()}), Python {platform.python_version()}")
    print(f"onnxruntime {ort.__version__}, numpy {np.__version__}")
    if platform.machine() not in ("aarch64", "arm64", "AMD64", "x86_64"):
        print("  !! 64-bit OS가 아님. onnxruntime은 64-bit Raspberry Pi OS에서만 설치됨")
    try:
        temp = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
        throttled = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
        print(f"CPU 온도   : {temp:.1f}°C, {throttled} (0x0이 아니면 전원 부족/과열로 속도 저하)")
    except (OSError, ValueError):
        pass  # RPi가 아니면 생략


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("--photos", nargs=2, metavar=("OPEN", "CLOSED"))
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("-n", type=int, default=20, help="반복 횟수")
    args = ap.parse_args()

    system_info()
    cfg = _load_config()
    cfg["model"]["path"] = str(Path(args.onnx).resolve())

    start = time.perf_counter()
    session = model.load_model(cfg["model"]["path"])
    print(f"\n모델 로드        : {(time.perf_counter() - start) * 1000:7.0f} ms (프로세스당 한 번)")

    opened, closed = (load_photo(p) for p in args.photos) if args.photos else fake_photos(cfg["rois"])
    x, y, w, h = cfg["rois"][1]
    crop = opened[y:y + h, x:x + w]
    size = tuple(cfg["model"]["input_size"])
    model.predict(session, crop, size)  # 첫 실행은 느려서 제외
    rows = [
        ("전처리 (칸 1개)", lambda: model.preprocess(crop, size)),
        ("추론 (칸 1개)", lambda: model.predict(session, crop, size)),
    ]
    # 상태가 바뀌지 않게 매번 새 watcher: 모두 닫힘(평소) / 모두 열림(최악, 추론 4번)
    for name, img in (("update 모두 닫힘", closed), ("update 모두 열림", opened)):
        w_ = PillboxWatcher(cfg)
        rows.append((name, lambda w_=w_, img=img: w_.update(img, 0)))

    if args.camera:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(main={"format": "RGB888", "size": SIZE[::-1]}))
        cam.start()
        time.sleep(2)  # 노출 안정화
        rows.insert(0, ("촬영 (원본)", cam.capture_array))

    worst = 0.0
    for name, fn in rows:
        med, mx = timed(fn, args.n)
        print(f"{name:<16} : {med:7.0f} ms (최대 {mx:.0f})")
        if name in ("촬영 (원본)", "update 모두 열림"):
            worst += mx
    if args.camera:
        cam.stop()

    verdict = "OK" if worst < 1000 else "느림 -> 촬영 간격을 늘리거나 최적화 필요"
    print(f"\n최악 1회(촬영 + 4칸 열림 update) : {worst:.0f} ms / 목표 1000 ms -> {verdict}")
    if not args.camera:
        print("  (촬영 시간 미포함. RPi에서 --camera로 다시 확인)")


if __name__ == "__main__":
    sys.exit(main())
