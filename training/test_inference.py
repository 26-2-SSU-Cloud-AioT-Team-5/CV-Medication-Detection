"""Evaluate class0..class7 test images, then browse cached predictions.

Run from the project root: python3 -m training.test_inference
Edit the globals below; no command-line arguments are needed.
The GUI can use a separate Python with Tk 8.6+ and Pillow on macOS.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# =============================================================================
# Configuration: relative paths resolve against PROJECT_ROOT.
# =============================================================================
TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"

CHECKPOINT_NAME = "261119_class07_best.pt"

TEST_DIR = TRAINING_DIR / "dataset" / "test"

BATCH_SIZE = 32
DEVICE = "auto"  # "auto", "cuda", "mps", "cpu"

SHOW_GUI = True
GUI_PYTHON = None  # Optional Python/pythonw path with Tk 8.6+ and Pillow.

CLASS_NAMES = tuple(f"class{label}" for label in range(8))
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"}
GUI_PAYLOAD_ENV = "PILL_TEST_INFERENCE_GUI_PAYLOAD"

# =============================================================================

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_path(value) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def resolve_device(preference: str):
    import torch

    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }
    if preference == "auto":
        return torch.device(next(name for name in ("cuda", "mps", "cpu") if available[name]))
    if preference not in available:
        raise ValueError('DEVICE must be "auto", "cuda", "mps", or "cpu".')
    if not available[preference]:
        raise RuntimeError(f"Requested device {preference!r} is unavailable.")
    return torch.device(preference)


def collect_samples(test_dir: Path) -> tuple[list[tuple[Path, int]], list[dict]]:
    """Keep folder-number labels fixed even when only class1 has images."""
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory is missing: {test_dir}")
    samples = []
    stats = []
    for label, name in enumerate(CLASS_NAMES):
        directory = test_dir / name
        paths = sorted(
            (path for path in directory.rglob("*")
             if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda path: path.relative_to(test_dir).as_posix().casefold(),
        ) if directory.is_dir() else []
        samples.extend((path, label) for path in paths)
        stats.append({
            "class": name, "label": label, "found": len(paths),
            "evaluated": 0, "skipped": 0, "correct": 0, "wrong": 0,
            "accuracy": None, "status": "보류",
        })
    return samples, stats


def load_model(checkpoint_path: Path, device):
    import torch
    from collections.abc import Mapping
    from architecture.model import build_model

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected a state_dict or checkpoint dictionary: {checkpoint_path}")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Empty or invalid model_state_dict: {checkpoint_path}")
    model = build_model(pretrained=False)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint does not match the 8-class ResNet-18: {checkpoint_path}\n{exc}") from exc
    return model.to(device).eval()


def evaluate_samples(model, samples: list[tuple[Path, int]], stats: list[dict],
                     device, test_dir: Path, batch_size: int) -> tuple[list[dict], list[dict]]:
    """Decode one batch at a time; cache eight probabilities per valid image."""
    import torch
    from training.dataset_sampler import CLASS_TO_IDX, build_transforms, load_rgb

    if CLASS_TO_IDX != {name: label for label, name in enumerate(CLASS_NAMES)}:
        raise ValueError("Training and inference class mappings differ.")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("BATCH_SIZE must be a positive integer.")
    transform = build_transforms()
    records, skipped = [], []
    tensors, batch_samples = [], []
    model.eval()

    def predict_batch() -> None:
        logits = model(torch.stack(tensors).to(device))
        if tuple(logits.shape) != (len(batch_samples), len(CLASS_NAMES)):
            raise ValueError(f"Expected [batch, 8] logits, got {tuple(logits.shape)}.")
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("Model produced non-finite logits; evaluation stopped.")
        probability_tensor = torch.softmax(logits, dim=1).cpu()
        probabilities = probability_tensor.tolist()
        predictions = probability_tensor.argmax(dim=1).tolist()
        for (path, label), scores, prediction in zip(batch_samples, probabilities, predictions):
            correct = prediction == label
            stats[label]["evaluated"] += 1
            stats[label]["correct"] += int(correct)
            records.append({
                "path": str(path), "relative_path": path.relative_to(test_dir).as_posix(),
                "label": label, "prediction": prediction, "probabilities": scores,
                "correct": correct,
            })
        print(f"Inferred {len(records)} / {len(samples)} images", flush=True)
        tensors.clear()
        batch_samples.clear()

    with torch.inference_mode():
        for path, label in samples:
            try:
                tensor = transform(load_rgb(str(path)))
            except (OSError, ValueError) as exc:
                stats[label]["skipped"] += 1
                skipped.append({"path": str(path), "label": label, "reason": str(exc)})
                continue
            tensors.append(tensor)
            batch_samples.append((path, label))
            if len(tensors) == batch_size:
                predict_batch()
        if tensors:
            predict_batch()

    for row in stats:
        total = row["evaluated"]
        row["wrong"] = total - row["correct"]
        row["accuracy"] = row["correct"] / total if total else None
        row["status"] = "평가 완료" if total else "평가 불가" if row["found"] else "보류"
    return records, skipped


def print_accuracy_table(stats: list[dict], skipped: list[dict]) -> None:
    for item in skipped:
        print(f"[SKIP] {item['path']}\n       {item['reason']}")
    print("\nClass   | Found | Tested | Skip | Correct | Wrong | Accuracy | Status")
    print("-" * 80)
    for row in stats:
        accuracy = f"{row['accuracy']:.2%}" if row["accuracy"] is not None else "N/A"
        print(
            f"{row['class']:<7} | {row['found']:>5} | {row['evaluated']:>6} | "
            f"{row['skipped']:>4} | {row['correct']:>7} | {row['wrong']:>5} | "
            f"{accuracy:>8} | {row['status']}"
        )
    total = sum(row["evaluated"] for row in stats)
    correct = sum(row["correct"] for row in stats)
    accuracy = f"{correct / total:.2%}" if total else "N/A"
    print("-" * 80)
    print(f"Total: {correct}/{total} correct; accuracy={accuracy}; skipped={len(skipped)}")
    evaluated_classes = [row["class"] for row in stats if row["evaluated"]]
    print(f"Evaluated classes: {', '.join(evaluated_classes) or 'none'}", flush=True)


def run_evaluation() -> dict:
    if type(BATCH_SIZE) is not int or BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be a positive integer.")
    test_dir = resolve_path(TEST_DIR)
    checkpoint_path = resolve_path(CHECKPOINT_DIR) / CHECKPOINT_NAME
    samples, stats = collect_samples(test_dir)
    records, skipped = [], []
    device_name = "not used"
    if samples:
        device = resolve_device(DEVICE)
        device_name = str(device)
        print(f"Checkpoint: {checkpoint_path}\nTest directory: {test_dir}\nDevice: {device}", flush=True)
        print("Input: RGB PNG 224x224; training ImageNet normalization; no resize/crop.", flush=True)
        model = load_model(checkpoint_path, device)
        records, skipped = evaluate_samples(model, samples, stats, device, test_dir, BATCH_SIZE)
    print_accuracy_table(stats, skipped)
    return {
        "schema_version": 1, "checkpoint": str(checkpoint_path), "test_dir": str(test_dir),
        "device": device_name, "classes": list(CLASS_NAMES), "class_stats": stats,
        "records": records, "skipped": skipped,
    }


def find_gui_python() -> Path:
    """The viewer only needs Tk/Pillow, not the inference process's PyTorch."""
    probe = (
        "import tkinter; from PIL import Image, ImageTk; "
        "raise SystemExit(0 if tkinter.TkVersion >= 8.6 else 1)"
    )
    candidates = [shutil.which(str(GUI_PYTHON)) or str(resolve_path(GUI_PYTHON))] if GUI_PYTHON else [
        shutil.which(name) for name in
        ("pythonw", "python", "python3", "python3.14", "python3.13", "python3.12", "python3.11", "python3.10")
    ]
    seen = set()
    for executable in candidates:
        if not executable:
            continue
        executable = shutil.which(executable) or str(Path(executable).expanduser())
        path = Path(executable).resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            result = subprocess.run([str(path), "-c", probe], capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return path
    raise RuntimeError("Tk 8.6 이상과 Pillow가 있는 Python을 GUI_PYTHON에 설정해주세요.")


class InferenceViewer:
    """Browse immutable inference records; image changes never rerun the model."""

    def __init__(self, root, payload: dict) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.payload = payload
        self.records = payload["records"]
        self.index = 0
        self.photo = None
        self.probability_rows = []
        root.title("알약 개수 · 테스트 추론 결과")
        width = min(1100, root.winfo_screenwidth() - 80)
        height = min(780, root.winfo_screenheight() - 100)
        root.geometry(f"{width}x{height}")
        root.minsize(850, 600)

        header = ttk.Frame(root, padding=12)
        header.pack(fill="x")
        ttk.Label(header, text=f"가중치: {Path(payload['checkpoint']).name}").pack(anchor="w")
        navigation = ttk.Frame(header, padding=(0, 10, 0, 0))
        navigation.pack(fill="x")
        self.previous_button = ttk.Button(navigation, text="← 이전", command=lambda: self.move(-1))
        self.previous_button.pack(side="left")
        self.next_button = ttk.Button(navigation, text="다음 →", command=lambda: self.move(1))
        self.next_button.pack(side="left", padx=8)
        self.position = tk.StringVar(master=root)
        ttk.Label(navigation, textvariable=self.position).pack(side="left", padx=8)

        body = ttk.Frame(root, padding=(12, 0, 12, 0))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        image_panel = ttk.Frame(body)
        image_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        self.image_label = tk.Label(image_panel, background="#252830", foreground="white")
        self.image_label.pack(fill="both", expand=True)
        self.filename = tk.StringVar(master=root)
        ttk.Label(image_panel, textvariable=self.filename, wraplength=440).pack(anchor="w", pady=10)
        self.truth = tk.StringVar(master=root)
        ttk.Label(image_panel, textvariable=self.truth, font=("Helvetica", 16, "bold")).pack(anchor="w")

        scores_panel = ttk.Frame(body)
        scores_panel.grid(row=0, column=1, sticky="nsew")
        scores_panel.columnconfigure(1, weight=1)
        self.prediction = tk.StringVar(master=root)
        ttk.Label(scores_panel, textvariable=self.prediction, font=("Helvetica", 18, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        self.verdict = tk.Label(scores_panel, font=("Helvetica", 22, "bold"), anchor="w")
        self.verdict.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 20))
        ttk.Label(scores_panel, text="모든 클래스의 softmax 확률").grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        for label in range(len(CLASS_NAMES)):
            name = ttk.Label(scores_panel, text=f"class{label}", width=14)
            bar = ttk.Progressbar(scores_panel, maximum=100, length=180)
            value = ttk.Label(scores_panel, width=9, anchor="e")
            name.grid(row=3 + label, column=0, sticky="w", pady=8)
            bar.grid(row=3 + label, column=1, sticky="ew", padx=6)
            value.grid(row=3 + label, column=2, sticky="e")
            self.probability_rows.append((name, bar, value))

        total = len(self.records)
        correct = sum(record["correct"] for record in self.records)
        evaluated = ", ".join(row["class"] for row in payload["class_stats"] if row["evaluated"])
        ttk.Label(
            root, padding=12,
            text=f"평가 클래스: {evaluated}  |  전체 {correct}/{total} 정답 ({correct / total:.2%})  |  ← / → 이동",
            wraplength=1000,
        ).pack(fill="x")
        root.bind("<Left>", lambda event: self.move(-1))
        root.bind("<Right>", lambda event: self.move(1))
        self.show_record()

    def move(self, delta: int) -> None:
        index = self.index + delta
        if 0 <= index < len(self.records):
            self.index = index
            self.show_record()

    def show_record(self) -> None:
        from PIL import Image, ImageTk

        record = self.records[self.index]
        label, prediction = record["label"], record["prediction"]
        self.position.set(f"{self.index + 1} / {len(self.records)}")
        self.filename.set(record["relative_path"])
        self.truth.set(f"정답: class{label} · 알약 {label}개")
        self.prediction.set(f"예측: class{prediction} · 알약 {prediction}개")
        self.verdict.configure(
            text="O · 정답" if record["correct"] else "X · 오답",
            foreground="#219653" if record["correct"] else "#dc3545",
        )
        for index, (name, bar, value) in enumerate(self.probability_rows):
            selected = index == prediction
            name.configure(text=f"class{index}" + (" ← 예측" if selected else ""),
                           font=("Helvetica", 12, "bold" if selected else "normal"))
            bar["value"] = record["probabilities"][index] * 100
            value.configure(text=f"{record['probabilities'][index]:.2%}")
        self.previous_button.configure(state="normal" if self.index else "disabled")
        self.next_button.configure(state="normal" if self.index + 1 < len(self.records) else "disabled")
        try:
            with Image.open(record["path"]) as source:
                image = source.convert("RGB")
            image = image.resize((400, 400), Image.Resampling.NEAREST)
            self.photo = ImageTk.PhotoImage(image, master=self.root)
            self.image_label.configure(image=self.photo, text="")
        except (OSError, ValueError) as exc:
            self.photo = None
            self.image_label.configure(image="", text=f"이미지를 열 수 없습니다.\n{exc}", wraplength=400)


def show_gui(payload: dict) -> None:
    import tkinter as tk

    if tk.TkVersion < 8.6:
        raise RuntimeError("GUI requires Tk 8.6 or newer; set GUI_PYTHON to a compatible interpreter.")
    if not payload.get("records"):
        raise ValueError("There are no evaluated images to display.")
    root = tk.Tk()
    root.withdraw()
    viewer = InferenceViewer(root, payload)
    root.update_idletasks()
    root.deiconify()
    root.mainloop()
    # Keep the PhotoImage-owning viewer alive throughout the main loop.
    del viewer


def launch_gui(payload: dict) -> None:
    if GUI_PYTHON is None:
        try:
            import tkinter as tk
            from PIL import ImageTk
            if tk.TkVersion >= 8.6:
                show_gui(payload)
                return
        except ImportError:
            pass
    executable = find_gui_python()
    print(f"GUI runtime: {executable} (cached results; no additional model inference)", flush=True)
    with tempfile.TemporaryDirectory(prefix="pill-test-viewer-") as temporary:
        path = Path(temporary) / "predictions.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        environment = dict(os.environ)
        environment[GUI_PAYLOAD_ENV] = str(path)
        subprocess.run([str(executable), str(Path(__file__).resolve())], env=environment, check=True)


def main() -> Optional[dict]:
    payload_path = os.environ.get(GUI_PAYLOAD_ENV)
    if payload_path:
        payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("classes") != list(CLASS_NAMES):
            raise ValueError("Viewer payload has an incompatible schema or class mapping.")
        show_gui(payload)
        return None
    payload = run_evaluation()
    if not payload["records"]:
        print("No valid test images were evaluated; GUI is not opened.", flush=True)
    elif SHOW_GUI:
        launch_gui(payload)
    return payload


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc
