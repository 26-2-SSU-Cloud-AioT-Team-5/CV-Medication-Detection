# Pi: Enter마다 한 장 찍어 4칸 개수와 처리 시간 출력
import time
from picamera2 import Picamera2
from detector.interface import count_pills
cam = Picamera2()
cam.configure(cam.create_still_configuration(main={"format": "RGB888", "size": (4608, 2592)}))
cam.start(); time.sleep(1.5)
while True:
    input("\n약통 세팅 후 Enter (끝내려면 Ctrl+C)")
    cam.autofocus_cycle()
    img = cam.capture_array()
    t = time.perf_counter()
    for cid in (1, 2, 3, 4):
        r = count_pills(img, cid)
        print(f"{cid}번 칸: {'닫힘/가림' if r.count is None else f'{r.count}개'} (신뢰도 {r.confidence:.2f})")
    print(f"4칸 처리 {time.perf_counter() - t:.1f}초")
