"""Browse images in a directory, draw crop boxes, and save 224x224 RGB images.

Run: python3 training/resize_test.py
Requires tkinter and Pillow. HEIC/HEIF files use macOS's sips when needed.
On macOS, an old system Tk is replaced with a compatible Python on PATH.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from PIL import Image, ImageOps, ImageTk, UnidentifiedImageError

OUTPUT_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
Box = tuple[float, float, float, float]


def find_gui_python() -> Optional[Path]:
    """Find an existing Python with modern Tk and Pillow, without opening a GUI."""
    checked = {Path(sys.executable).resolve()}
    probe = (
        "import tkinter; from PIL import Image, ImageTk; "
        "raise SystemExit(0 if tkinter.TkVersion >= 8.6 else 1)"
    )
    for name in ("pythonw", "python", "python3", "python3.14", "python3.13", "python3.12", "python3.11", "python3.10"):
        executable = shutil.which(name)
        if not executable:
            continue
        candidate = Path(executable).resolve()
        if candidate in checked:
            continue
        checked.add(candidate)
        try:
            result = subprocess.run([str(candidate), "-c", probe], capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate
    return None


def ensure_gui_runtime() -> None:
    """Avoid the blank-window problem of Apple's deprecated Tk 8.5 on macOS."""
    if sys.platform != "darwin" or tk.TkVersion >= 8.6:
        return
    executable = find_gui_python()
    if executable is None:
        raise SystemExit(
            "macOS 기본 Tk 8.5에서는 창 내용이 표시되지 않을 수 있습니다.\n"
            "Tk 8.6 이상과 Pillow가 설치된 Python으로 이 파일을 실행해주세요."
        )
    print(f"[resize_test] Tk 8.6 이상을 지원하는 Python으로 실행합니다: {executable}", flush=True)
    os.execv(str(executable), [str(executable), str(Path(__file__).resolve()), *sys.argv[1:]])


def load_image(path: Path) -> Image.Image:
    """Load an independent RGB image with the camera's EXIF rotation applied."""
    path = Path(path).expanduser().resolve()
    try:
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except UnidentifiedImageError:
        if path.suffix.lower() not in {".heic", ".heif"}:
            raise

    sips = shutil.which("sips")
    if sips is None:
        raise ValueError("HEIC/HEIF 이미지는 JPG 또는 PNG로 변환한 후 선택해주세요.")
    with tempfile.TemporaryDirectory(prefix="pill_crop_") as temporary:
        converted = Path(temporary) / "converted.png"
        subprocess.run(
            [sips, "-s", "format", "png", str(path), "--out", str(converted)],
            check=True,
            capture_output=True,
            timeout=30,
        )
        with Image.open(converted) as source:
            return ImageOps.exif_transpose(source).convert("RGB")


def pixel_box(image_size: tuple[int, int], selection: Box) -> tuple[int, int, int, int]:
    """Normalize either drag direction and clamp the crop to image boundaries."""
    width, height = image_size
    x1, y1, x2, y2 = selection
    left, right = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    top, bottom = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    if right <= left or bottom <= top:
        raise ValueError("마우스를 드래그해 너비와 높이가 있는 영역을 선택해주세요.")
    return math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)


def crop_and_resize(image: Image.Image, selection: Box) -> Image.Image:
    """Crop the original pixels, then resize exactly to 224x224."""
    box = pixel_box(image.size, selection)
    return image.crop(box).convert("RGB").resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)


class CropResizeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.directory: Optional[Path] = None
        self.image_paths: list[Path] = []
        self.image_index = -1
        self.source_path: Optional[Path] = None
        self.source_image: Optional[Image.Image] = None
        self.result_image: Optional[Image.Image] = None
        self.selection: Optional[Box] = None
        self.drag_start: Optional[tuple[float, float]] = None
        self.display_size = (1, 1)
        self.offset = (0.0, 0.0)
        self.source_photo: Optional[ImageTk.PhotoImage] = None
        self.result_photo: Optional[ImageTk.PhotoImage] = None

        root.title("알약 이미지 크롭 · 224 × 224")
        width = min(1180, root.winfo_screenwidth() - 80)
        height = min(820, root.winfo_screenheight() - 100)
        root.geometry(f"{width}x{height}")
        root.minsize(760, 480)

        toolbar = ttk.Frame(root, padding=10)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="1. 폴더 선택", command=self.open_directory).pack(side="left")
        self.reset_button = ttk.Button(
            toolbar, text="선택 초기화", command=self.clear_selection, state="disabled"
        )
        self.reset_button.pack(side="left", padx=8)
        self.crop_button = ttk.Button(
            toolbar, text="3. 크롭 → 224×224", command=self.make_crop, state="disabled"
        )
        self.crop_button.pack(side="left")
        self.save_button = ttk.Button(
            toolbar, text="결과 저장", command=self.save_result, state="disabled"
        )
        self.save_button.pack(side="left", padx=8)

        navigation = ttk.Frame(root, padding=(10, 0, 10, 8))
        navigation.pack(fill="x")
        self.previous_button = ttk.Button(
            navigation, text="← 이전", command=lambda: self.show_image(self.image_index - 1), state="disabled"
        )
        self.previous_button.pack(side="left")
        self.next_button = ttk.Button(
            navigation, text="다음 →", command=lambda: self.show_image(self.image_index + 1), state="disabled"
        )
        self.next_button.pack(side="left", padx=8)
        self.image_position = tk.StringVar(master=root, value="0 / 0")
        ttk.Label(navigation, textvariable=self.image_position, wraplength=600).pack(side="left")

        ttk.Label(
            root, text="2. 왼쪽 이미지에서 마우스를 드래그해 사각형 영역을 선택하세요.",
            padding=(10, 0, 10, 10),
        ).pack(anchor="w")
        body = ttk.Frame(root, padding=(10, 0, 10, 0))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(body, background="#252830", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        sidebar = ttk.Frame(body, padding=(16, 0, 0, 0))
        sidebar.grid(row=0, column=1, sticky="n")
        ttk.Label(sidebar, text="224 × 224 결과 미리보기").pack(pady=(0, 10))
        self.preview = tk.Canvas(
            sidebar, width=224, height=224, background="#252830", highlightthickness=0
        )
        self.preview.pack()
        ttk.Label(
            sidebar, text="선택한 영역 전체를 224×224로 변환합니다.\n직사각형 영역은 가로·세로 비율이 바뀝니다.",
            wraplength=224, justify="left",
        ).pack(anchor="w", pady=12)
        self.status = tk.StringVar(master=root, value="폴더 선택 버튼을 눌러 시작하세요.")
        ttk.Label(root, textvariable=self.status, padding=10, wraplength=1000).pack(fill="x")

        self.canvas.bind("<Configure>", self.redraw_image)
        self.canvas.bind("<ButtonPress-1>", self.start_selection)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.finish_selection)
        root.bind("<Escape>", lambda event: self.clear_selection())
        root.bind("<Left>", lambda event: self.show_image(self.image_index - 1))
        root.bind("<Right>", lambda event: self.show_image(self.image_index + 1))

    def open_directory(self) -> None:
        directory_name = filedialog.askdirectory(
            parent=self.root,
            title="이미지가 들어 있는 폴더 선택",
            initialdir=str(self.directory if self.directory else Path(__file__).parent),
            mustexist=True,
        )
        if not directory_name:
            return
        directory = Path(directory_name).resolve()
        try:
            paths = sorted(
                (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
                key=lambda path: path.name.casefold(),
            )
        except OSError as exc:
            messagebox.showerror("폴더 열기 실패", str(exc), parent=self.root)
            return
        if not paths:
            messagebox.showinfo("이미지 없음", "선택한 폴더에 지원하는 이미지가 없습니다.", parent=self.root)
            return
        self.directory = directory
        self.image_paths = paths
        self.root.title(f"알약 이미지 크롭 · 224 × 224 — {directory.name}")
        self.show_image(0)

    def show_image(self, index: int) -> None:
        if not 0 <= index < len(self.image_paths):
            return
        self.image_index = index
        self.source_path = self.image_paths[index].resolve()
        self.source_image = None
        self.source_photo = None
        self.clear_selection()
        self.reset_button.configure(state="disabled")
        self.redraw_image()
        self.previous_button.configure(state="normal" if index > 0 else "disabled")
        self.next_button.configure(state="normal" if index < len(self.image_paths) - 1 else "disabled")
        self.image_position.set(f"{index + 1} / {len(self.image_paths)} · {self.source_path.name}")
        try:
            image = load_image(self.source_path)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            messagebox.showerror("이미지 열기 실패", f"이미지를 열 수 없습니다.\n{exc}", parent=self.root)
            self.status.set(f"{self.source_path.name} 열기 실패. 이전/다음 버튼으로 다른 이미지를 선택하세요.")
            return
        self.source_image = image
        self.reset_button.configure(state="normal")
        self.redraw_image()
        self.status.set(f"{self.source_path.name} | 원본 {image.width}×{image.height} | 영역을 드래그하세요.")

    def clear_result(self) -> None:
        self.result_image = None
        self.result_photo = None
        self.preview.delete("all")
        self.save_button.configure(state="disabled")

    def clear_selection(self) -> None:
        self.selection = None
        self.drag_start = None
        self.canvas.delete("selection")
        self.crop_button.configure(state="disabled")
        self.clear_result()
        self.status.set("이미지에서 새 영역을 드래그하세요." if self.source_image else "폴더를 선택하세요.")

    def redraw_image(self, event=None) -> None:
        self.canvas.delete("all")
        if self.source_image is None:
            return
        width, height = self.source_image.size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        scale = min(max(1, canvas_width - 20) / width, max(1, canvas_height - 20) / height, 1.0)
        self.display_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        shown_width, shown_height = self.display_size
        self.offset = ((canvas_width - shown_width) / 2, (canvas_height - shown_height) / 2)
        displayed = self.source_image.resize(self.display_size, Image.Resampling.LANCZOS)
        self.source_photo = ImageTk.PhotoImage(displayed, master=self.root)
        self.canvas.create_image(*self.offset, image=self.source_photo, anchor="nw")
        self.draw_selection()

    def source_point(self, x: float, y: float, clamp: bool = False) -> Optional[tuple[float, float]]:
        """Map canvas coordinates to original pixels, including letterbox offsets."""
        if self.source_image is None:
            return None
        x -= self.offset[0]
        y -= self.offset[1]
        width, height = self.display_size
        if not clamp and not (0 <= x <= width and 0 <= y <= height):
            return None
        return (
            max(0.0, min(width, x)) * self.source_image.width / width,
            max(0.0, min(height, y)) * self.source_image.height / height,
        )

    def draw_selection(self) -> None:
        self.canvas.delete("selection")
        if self.selection is None or self.source_image is None:
            return
        x1, y1, x2, y2 = self.selection
        scale_x = self.display_size[0] / self.source_image.width
        scale_y = self.display_size[1] / self.source_image.height
        self.canvas.create_rectangle(
            self.offset[0] + x1 * scale_x, self.offset[1] + y1 * scale_y,
            self.offset[0] + x2 * scale_x, self.offset[1] + y2 * scale_y,
            outline="#56e39f", width=2, tags="selection",
        )

    def start_selection(self, event) -> None:
        point = self.source_point(event.x, event.y)
        if point is None:
            return
        self.clear_selection()
        self.drag_start = point

    def update_selection(self, event) -> None:
        if self.drag_start is None:
            return
        point = self.source_point(event.x, event.y, clamp=True)
        self.selection = (*self.drag_start, *point)
        self.draw_selection()

    def finish_selection(self, event) -> None:
        if self.drag_start is None:
            return
        self.update_selection(event)
        self.drag_start = None
        try:
            box = pixel_box(self.source_image.size, self.selection)
        except ValueError as exc:
            self.clear_selection()
            self.status.set(str(exc))
            return
        self.selection = box
        self.draw_selection()
        self.crop_button.configure(state="normal")
        self.status.set(
            f"선택: ({box[0]}, {box[1]})–({box[2]}, {box[3]}) | "
            f"{box[2] - box[0]}×{box[3] - box[1]} → 크롭 버튼을 누르세요."
        )

    def make_crop(self) -> None:
        if self.source_image is None or self.selection is None:
            return
        self.result_image = crop_and_resize(self.source_image, self.selection)
        self.result_photo = ImageTk.PhotoImage(self.result_image, master=self.root)
        self.preview.delete("all")
        self.preview.create_image(0, 0, image=self.result_photo, anchor="nw")
        self.save_button.configure(state="normal")
        self.status.set("224×224 RGB 이미지 생성 완료. 결과 저장 버튼으로 PNG/JPG를 저장하세요.")

    def save_result(self) -> None:
        if self.result_image is None or self.source_path is None:
            return
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="224×224 이미지 저장",
            initialdir=str(self.source_path.parent),
            initialfile=f"{self.source_path.stem}_crop_224.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg")],
        )
        if not filename:
            return
        if Path(filename).resolve() == self.source_path:
            messagebox.showerror("저장 실패", "원본과 다른 파일 이름으로 저장해주세요.", parent=self.root)
            return
        try:
            self.result_image.save(filename)
        except (OSError, ValueError) as exc:
            messagebox.showerror("저장 실패", str(exc), parent=self.root)
            return
        self.status.set(f"저장 완료: {filename} (224×224)")


def main() -> None:
    ensure_gui_runtime()
    root = tk.Tk()
    root.withdraw()
    CropResizeApp(root)
    root.update_idletasks()
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
