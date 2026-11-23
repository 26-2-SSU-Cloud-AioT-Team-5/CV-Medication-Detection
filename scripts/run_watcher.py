# Pi: 약 1초마다 찍어 watcher가 확정한 칸별 개수 이벤트 출력
import time
from datetime import datetime
from picamera2 import Picamera2
from detector.watcher import PillboxWatcher
cam = Picamera2()
cam.configure(cam.create_still_configuration(main={"format": "RGB888", "size": (4608, 2592)}))
cam.start(); time.sleep(1.5); cam.autofocus_cycle()
w = PillboxWatcher()
print("뚜껑을 열고 4초 이상 두었다가 닫아 보세요 (Ctrl+C로 종료)")
while True:
    for e in w.update(cam.capture_array(), datetime.now()):
        print(e)
    time.sleep(1)
