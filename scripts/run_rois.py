# Pi: 한 장 찍어 자동 보정된 칸 박스를 그린 capture_rois.jpg 저장 (개수는 세지 않음)
import time
from picamera2 import Picamera2
from PIL import Image, ImageDraw
from detector.interface import _load_config, align_rois
cam = Picamera2()
cam.configure(cam.create_still_configuration(main={"format": "RGB888", "size": (4608, 2592)}))
cam.start(); time.sleep(1.5); cam.autofocus_cycle()
img = cam.capture_array(); cam.stop()                # BGR
cfg = _load_config()
rois = align_rois(img, cfg)                          # count_pills가 쓰는 보정된 박스
photo = Image.fromarray(img[..., ::-1].copy())
photo.save("capture.jpg")
d = ImageDraw.Draw(photo)
for cid, (x, y, w, h) in rois.items():
    d.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=8)
    ox, oy = cfg["rois"][cid][:2]
    print(f"{cid}번 칸: {[x, y, w, h]}  (config에서 {x - ox:+}, {y - oy:+} 이동)")
photo.save("capture_rois.jpg")
print("저장: capture.jpg, capture_rois.jpg")
