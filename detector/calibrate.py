"""칸 ROI 보정 도구. 카메라가 움직이면 새로 찍은 통 전체 사진으로 다시 돌린다.

    python -m detector.calibrate 사진.jpg          # 칸 1→4 순서로 박스를 그려 config.yaml rois에 저장
    python -m detector.calibrate 사진.jpg --show   # 지금 rois를 사진에 그려 확인만

박스 그리기: 드래그 → Enter(또는 Space)로 확정, 칸 수만큼 반복 → Esc로 끝.
두 경우 모두 사진 옆에 <사진>_rois.jpg 미리보기를 저장한다.
PC에서 돌릴 때는 opencv-python 필요 (RPi 추론에는 필요 없음).
"""
import re
import sys
from pathlib import Path

from .interface import CONFIG_PATH, _load_config

MAX_VIEW_WIDTH = 1600  # 원본(4608x2592)은 화면보다 커서 줄여서 보여준다


def write_rois(text: str, rois: dict[int, list[int]]) -> str:
    # config.yaml의 주석을 살리려고 "  N: ..." 줄만 바꾼다
    for cid, roi in rois.items():
        text, n = re.subn(rf"^(  {cid}:).*$", rf"\1 {list(roi)}", text, flags=re.M)
        if n != 1:
            raise ValueError(f"config.yaml rois에서 칸 {cid} 줄을 찾지 못함")
    return text


def draw(img, rois):
    import cv2
    out = img.copy()
    for cid, roi in rois.items():
        if roi:
            x, y, w, h = roi
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 8)
            cv2.putText(out, str(cid), (x + 15, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)
    return out


def main(photo: str, show_only: bool) -> None:
    import cv2
    img = cv2.imread(photo)
    if img is None:
        sys.exit(f"사진을 읽을 수 없음: {photo}")
    rois = _load_config()["rois"]

    if not show_only:
        scale = min(1.0, MAX_VIEW_WIDTH / img.shape[1])
        view = cv2.resize(draw(img, rois), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"칸 {sorted(rois)} 순서로 박스를 그리세요. 드래그 → Enter, 다 그리면 Esc. (빨간 박스 = 지금 값)")
        boxes = cv2.selectROIs("calibrate", view, showCrosshair=False)
        cv2.destroyAllWindows()
        if len(boxes) != len(rois):
            sys.exit(f"박스 {len(boxes)}개를 그림. 칸 수({len(rois)})와 달라서 저장하지 않음")
        rois = {cid: [round(v / scale) for v in box] for cid, box in zip(sorted(rois), boxes)}
        CONFIG_PATH.write_text(write_rois(CONFIG_PATH.read_text(encoding="utf-8"), rois), encoding="utf-8")
        print(f"저장: {CONFIG_PATH} rois = {rois}")

    preview = Path(photo).with_name(Path(photo).stem + "_rois.jpg")
    cv2.imwrite(str(preview), draw(img, rois))
    print(f"미리보기: {preview}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--show"]
    if len(args) != 1:
        sys.exit("usage: python -m detector.calibrate <사진> [--show]")
    main(args[0], "--show" in sys.argv)
