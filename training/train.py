"""Eight-class pill-count training. Edit globals below, then run this file.

python training/train.py --smoke-test   # short, offline run in a temporary directory
python training/train.py               # full configured training

Existing outputs are never reused. For another run, set BOTH output directories
to fresh paths. Synthetic validation measures synthetic performance only; an
independently captured real-world test set is needed before deployment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ============================================================
# Path Configuration (relative overrides resolve from PROJECT_ROOT)
# ============================================================
TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DATASET_DIR = TRAINING_DIR / "dataset"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
RESULTS_DIR = TRAINING_DIR / "results"

# ============================================================
# Model / Dataset Configuration
# ============================================================
NUM_CLASSES = 8
USE_PRETRAINED = True

NUM_VAL_SAMPLES = 1600
SPLIT_SEED = 42

# ============================================================
# Training Configuration
# ============================================================
NUM_EPOCHS = 30

TRAIN_BATCH_SIZE = 64

VAL_BATCH_SIZE = 64

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MIN_LEARNING_RATE = 1e-6

NUM_WORKERS = 4

TRAIN_SEED = 42

# ============================================================
# Early Stopping Configuration
# ============================================================
ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 7

# ============================================================
# Device / Precision Configuration
# ============================================================
DEVICE = "auto"  # CUDA -> MPS -> CPU, or explicitly "cuda", "mps", "cpu"
ENABLE_AMP = True  # CUDA only; CPU and MPS always use FP32
DETERMINISTIC_TRAINING = True

# Both direct execution and `python -m training.train` work from any directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from architecture.model import build_model
from training.dataset_sampler import CLASS_TO_IDX, PILL_TYPES, PREPROCESSING, DatasetSampler

HISTORY_FIELDS = (
    "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_mae",
    "false_empty_rate", "lr", "epoch_time_sec", "best_val_acc",
)
CONFIG_NAMES = (
    "NUM_CLASSES", "USE_PRETRAINED", "NUM_VAL_SAMPLES", "SPLIT_SEED",
    "NUM_EPOCHS", "TRAIN_BATCH_SIZE", "VAL_BATCH_SIZE", "LEARNING_RATE",
    "WEIGHT_DECAY", "MIN_LEARNING_RATE", "NUM_WORKERS", "TRAIN_SEED",
    "ENABLE_EARLY_STOPPING", "EARLY_STOPPING_PATIENCE", "DEVICE",
    "ENABLE_AMP", "DETERMINISTIC_TRAINING",
)


def resolve_path(path) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def validate_settings(config: dict) -> None:
    for name in ("NUM_EPOCHS", "TRAIN_BATCH_SIZE", "VAL_BATCH_SIZE", "EARLY_STOPPING_PATIENCE"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if type(config["NUM_WORKERS"]) is not int or config["NUM_WORKERS"] < 0:
        raise ValueError("NUM_WORKERS must be a nonnegative integer.")
    if config["NUM_CLASSES"] != len(CLASS_TO_IDX):
        raise ValueError("NUM_CLASSES must be 8 for the fixed count labels and model head.")
    for name in ("LEARNING_RATE", "MIN_LEARNING_RATE", "WEIGHT_DECAY"):
        if not math.isfinite(config[name]) or config[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if not 0 < config["MIN_LEARNING_RATE"] <= config["LEARNING_RATE"]:
        raise ValueError("Require 0 < MIN_LEARNING_RATE <= LEARNING_RATE.")
    for name in ("SPLIT_SEED", "TRAIN_SEED"):
        if type(config[name]) is not int or not 0 <= config[name] < 2**32:
            raise ValueError(f"{name} must be an integer in [0, 2**32).")
    for name in ("USE_PRETRAINED", "ENABLE_EARLY_STOPPING", "ENABLE_AMP", "DETERMINISTIC_TRAINING"):
        if type(config[name]) is not bool:
            raise ValueError(f"{name} must be a boolean.")
    if (type(config["NUM_VAL_SAMPLES"]) is not int or config["NUM_VAL_SAMPLES"] <= 0
            or config["NUM_VAL_SAMPLES"] % 32):
        raise ValueError("NUM_VAL_SAMPLES must be a positive multiple of 32.")


def resolve_device(preference: str) -> torch.device:
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if preference == "auto":
        return torch.device("cuda" if cuda else "mps" if mps else "cpu")
    if preference not in {"cuda", "mps", "cpu"}:
        raise ValueError('DEVICE must be "auto", "cuda", "mps", or "cpu".')
    if (preference == "cuda" and not cuda) or (preference == "mps" and not mps):
        raise RuntimeError(f"Requested device {preference!r} is unavailable.")
    return torch.device(preference)


def seed_training(seed: int, deterministic: bool) -> None:
    # Must be set before CUDA context / cuBLAS initialization.
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    # Unsupported deterministic kernels must fail visibly, including with AMP.
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def create_loaders(sampler: DatasetSampler, device: torch.device, config: dict,
                   smoke_test: bool = False) -> tuple:
    train_indices, val_indices = sampler.train_indices, sampler.val_indices
    train_dataset, val_dataset = sampler.get_train_dataset(), sampler.get_val_dataset()
    if smoke_test:
        # One sample per class/type, including empty: 29 samples per split.
        def representatives(indices):
            seen, selected = set(), []
            for index in indices:
                meta = sampler.metadata[index]
                key = (meta["label"], meta["pill_type"])
                if key not in seen:
                    selected.append(index)
                    seen.add(key)
            return selected
        train_indices, val_indices = representatives(train_indices), representatives(val_indices)
        train_dataset = Subset(sampler.dataset, train_indices)
        val_dataset = Subset(sampler.dataset, val_indices)
    common = {
        "num_workers": config["NUM_WORKERS"], "pin_memory": device.type == "cuda",
        "drop_last": False, "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset, batch_size=config["TRAIN_BATCH_SIZE"], shuffle=True,
        generator=torch.Generator().manual_seed(config["TRAIN_SEED"]), **common,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["VAL_BATCH_SIZE"], shuffle=False,
        generator=torch.Generator().manual_seed(config["TRAIN_SEED"] + 1), **common,
    )
    return train_loader, val_loader, [sampler.metadata[i] for i in val_indices]


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler) -> dict:
    model.train()
    loss_sum = correct = total = 0
    for images, labels in tqdm(train_loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, dtype=torch.long, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite training loss; stopping without saving this epoch.")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        count = labels.numel()
        loss_sum += loss.item() * count
        correct += (logits.detach().argmax(1) == labels).sum().item()
        total += count
    if not total:
        raise ValueError("Training loader is empty.")
    return {"loss": loss_sum / total, "accuracy": correct / total, "total": total}


@torch.inference_mode()
def validate(model, val_loader, criterion, device, metadata=None) -> dict:
    """Evaluate in FP32; optional metadata follows the unshuffled loader order."""
    model.eval()
    loss_sum = absolute_error = false_empty = nonempty = total = 0
    confusion = torch.zeros((len(CLASS_TO_IDX), len(CLASS_TO_IDX)), dtype=torch.long)
    type_correct = {pill: 0 for pill in PILL_TYPES}
    type_total = {pill: 0 for pill in PILL_TYPES}
    for images, labels in tqdm(val_loader, desc="Validate", leave=False):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, dtype=torch.long, non_blocking=device.type == "cuda")
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite validation loss.")
        predictions = logits.argmax(dim=1)
        count = labels.numel()
        loss_sum += loss.item() * count
        absolute_error += (predictions - labels).abs().sum().item()
        nonempty += (labels > 0).sum().item()
        false_empty += ((labels > 0) & (predictions == 0)).sum().item()
        truth, predicted = labels.cpu(), predictions.cpu()
        confusion += torch.bincount(truth * len(CLASS_TO_IDX) + predicted,
                                    minlength=len(CLASS_TO_IDX)**2).reshape_as(confusion)
        if metadata is not None:
            batch_meta = metadata[total:total + count]
            if len(batch_meta) != count:
                raise ValueError("Validation metadata length does not match loader.")
            for meta, label, prediction in zip(batch_meta, truth.tolist(), predicted.tolist()):
                if meta["label"] != label:
                    raise ValueError("Validation metadata order does not match labels.")
                pill = meta["pill_type"]
                if label > 0:
                    type_total[pill] += 1
                    type_correct[pill] += label == prediction
        total += count
    if not total:
        raise ValueError("Validation loader is empty.")
    if metadata is not None and len(metadata) != total:
        raise ValueError("Unused validation metadata; loader did not evaluate the whole subset.")
    class_totals = confusion.sum(dim=1)
    return {
        "loss": loss_sum / total, "accuracy": confusion.diag().sum().item() / total,
        "mae": absolute_error / total, "false_empty_rate": false_empty / nonempty if nonempty else 0.0,
        "false_empty_count": false_empty, "nonempty_count": nonempty, "total": total,
        "class_accuracy": {f"class{i}": confusion[i, i].item() / n if n else None
                           for i, n in enumerate(class_totals.tolist())},
        "class_samples": dict(zip(CLASS_TO_IDX, class_totals.tolist())),
        "pill_type_accuracy": {pill: type_correct[pill] / n if n else None
                               for pill, n in type_total.items()} if metadata is not None else None,
        "pill_type_samples": type_total if metadata is not None else None,
        "confusion_matrix": confusion.tolist(),
    }


def atomic_save(value, path: Path, *, as_json: bool = False) -> None:
    """Replace only after the complete file is flushed in the same directory."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            if as_json:
                stream.write((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())
            else:
                torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_output_dirs(checkpoint_dir: Path, results_dir: Path) -> None:
    if checkpoint_dir == results_dir:
        raise ValueError("CHECKPOINT_DIR and RESULTS_DIR must be distinct fresh directories.")
    for directory in (checkpoint_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)
        existing = [path.name for path in directory.iterdir() if path.name not in {".gitignore", ".gitkeep"}]
        if existing:
            raise FileExistsError(
                f"Existing run files in {directory}: {existing[:5]}. "
                "Set CHECKPOINT_DIR and RESULTS_DIR to fresh directories; existing files are preserved."
            )


def append_history(path: Path, record: dict) -> None:
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(record)
        stream.flush()
        os.fsync(stream.fileno())


def plot_results(history: list[dict], metrics: dict, best_epoch: int, results_dir: Path) -> None:
    # Lazy import keeps worker processes and --help independent of plotting.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pill-count-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    for filename, train_key, val_key, ylabel, scale in (
        ("accuracy_curve.png", "train_acc", "val_acc", "Accuracy (%)", 100),
        ("loss_curve.png", "train_loss", "val_loss", "Cross Entropy Loss", 1),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [row[train_key] * scale for row in history], marker="o", markersize=4, label="Training")
        ax.plot(epochs, [row[val_key] * scale for row in history], marker="o", markersize=4, label="Validation")
        if len(epochs) == 1:
            ax.set_xticks(epochs)
            ax.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
        if train_key == "train_acc":
            ax.axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch: {best_epoch}")
            ax.set_ylim(0, 100)
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(results_dir / filename, dpi=150)
        plt.close(fig)
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    color = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(color, ax=ax)
    ticks = list(range(len(CLASS_TO_IDX)))
    ax.set(xticks=ticks, yticks=ticks, xlabel="Prediction", ylabel="Ground truth",
           title=f"Best model (epoch {best_epoch})")
    for row in ticks:
        for col in ticks:
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def smoke_check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Smoke test failed: {message}")


def snapshot_state(model) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def run_training(sampler: DatasetSampler, config: dict, device: torch.device,
                 checkpoint_dir: Path, results_dir: Path, *, smoke_test: bool = False) -> dict:
    started = time.perf_counter()
    prepare_output_dirs(checkpoint_dir, results_dir)
    sampler.save_validation_split(checkpoint_dir / "validation_split.json")
    train_loader, val_loader, val_metadata = create_loaders(sampler, device, config, smoke_test)
    model = build_model(pretrained=config["USE_PRETRAINED"]).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Parameters: {parameter_count:,}; trainable: {trainable_count:,}")
    print(f"Active loader samples: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")
    amp_enabled = config["ENABLE_AMP"] and device.type == "cuda"
    print(f"AMP active: {amp_enabled}; requested: {config['ENABLE_AMP']}; precision: {'mixed float16' if amp_enabled else 'FP32'}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["NUM_EPOCHS"], eta_min=config["MIN_LEARNING_RATE"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if smoke_test:
        smoke_check(parameter_count == trainable_count, "all backbone parameters must be trainable")
        dummy = torch.randn(4, 3, 224, 224, device=device)
        logits = model(dummy)
        smoke_check(tuple(logits.shape) == (4, len(CLASS_TO_IDX)), "forward output shape")
        criterion(logits, torch.arange(4, device=device)).backward()
        smoke_check(model.conv1.weight.grad is not None and model.fc.weight.grad is not None, "backbone/head backward")
        smoke_check(all(torch.isfinite(p).all().item() for p in model.parameters()), "finite parameters")
        optimizer.zero_grad(set_to_none=True)
        del dummy, logits

    history = []
    best_val_acc, best_epoch, patience_counter = -1.0, 0, 0
    for epoch in range(1, config["NUM_EPOCHS"] + 1):
        epoch_start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]  # actual LR used during this epoch
        before_training = snapshot_state(model) if smoke_test else None
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        if smoke_test:
            after_training = snapshot_state(model)
            for key in ("conv1.weight", "fc.weight"):
                smoke_check(not torch.equal(before_training[key], after_training[key]), f"optimizer changed {key}")
            gradients = {name: p.grad.detach().clone() if p.grad is not None else None
                         for name, p in model.named_parameters()}
            def inference_hook(module, args, output):
                smoke_check(not torch.is_grad_enabled() and torch.is_inference_mode_enabled(), "validation inference mode")
                smoke_check(not output.requires_grad, "validation output has no gradient")
            handle = model.register_forward_hook(inference_hook)
        val_metrics = validate(model, val_loader, criterion, device)
        if smoke_test:
            handle.remove()
            smoke_check(all(torch.equal(value, after_training[key])
                            for key, value in snapshot_state(model).items()), "validation changed weights or BatchNorm buffers")
            smoke_check(all((p.grad is None and gradients[name] is None)
                            or (p.grad is not None and gradients[name] is not None
                                and torch.equal(p.grad, gradients[name]))
                            for name, p in model.named_parameters()), "validation changed gradients")
            smoke_check(train_metrics["total"] == len(train_loader.dataset)
                        and val_metrics["total"] == len(val_loader.dataset), "sample coverage including final batches")
        improved = val_metrics["accuracy"] > best_val_acc
        if improved:
            best_val_acc, best_epoch, patience_counter = val_metrics["accuracy"], epoch, 0
            atomic_save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            patience_counter += 1
        # After optimizer steps, once per epoch. last.pt stores the next LR state.
        scheduler.step()
        last = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc, "best_epoch": best_epoch,
            "split_seed": config["SPLIT_SEED"], "train_seed": config["TRAIN_SEED"],
            "early_stopping_counter": patience_counter, "config": config,
            "dataset_fingerprint": sampler.fingerprint, "preprocessing": PREPROCESSING,
        }
        if scaler.is_enabled():
            last["scaler_state_dict"] = scaler.state_dict()
        atomic_save(last, checkpoint_dir / "last.pt")
        elapsed = time.perf_counter() - epoch_start
        record = dict(zip(HISTORY_FIELDS, (
            epoch, train_metrics["loss"], train_metrics["accuracy"], val_metrics["loss"],
            val_metrics["accuracy"], val_metrics["mae"], val_metrics["false_empty_rate"],
            lr, elapsed, best_val_acc,
        )))
        append_history(results_dir / "training_history.csv", record)
        history.append(record)
        print(
            f"\nEpoch [{epoch:02d}/{config['NUM_EPOCHS']:02d}]\n" + "-" * 40 + "\n"
            f"Train Loss       : {train_metrics['loss']:.4f}\n"
            f"Train Accuracy   : {train_metrics['accuracy']:.2%}\n"
            f"Val Loss         : {val_metrics['loss']:.4f}\n"
            f"Val Accuracy     : {val_metrics['accuracy']:.2%}\n"
            f"Val MAE          : {val_metrics['mae']:.4f}\n"
            f"False Empty Rate : {val_metrics['false_empty_rate']:.2%}\n"
            f"Learning Rate    : {lr:.8f}\nEpoch Time       : {int(elapsed)//60:02d}:{int(elapsed)%60:02d}\n"
            f"Best Val Acc     : {best_val_acc:.2%}\nBest Epoch       : {best_epoch}\n"
            + ("[CHECKPOINT] best.pt saved\n" if improved else "") + "-" * 40, flush=True,
        )
        if config["ENABLE_EARLY_STOPPING"] and patience_counter >= config["EARLY_STOPPING_PATIENCE"]:
            print(f"Early stopping: {patience_counter} consecutive epochs without improvement.", flush=True)
            break

    # Reconstruct without ImageNet downloads, using only the best raw state_dict.
    best_state = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=True)
    best_model = build_model(pretrained=False)
    best_model.load_state_dict(best_state, strict=True)
    if smoke_test:
        smoke_check(all(torch.equal(best_model.state_dict()[key], value) for key, value in best_state.items()), "CPU checkpoint parameter round trip")
        model.eval()
        best_model.eval()
        images, labels = next(iter(val_loader))
        smoke_check(tuple(images.shape[1:]) == (3, 224, 224) and labels.dtype == torch.long, "loader tensor shape/dtype")
        # A one-epoch smoke run guarantees that the current model is the best.
        with torch.inference_mode():
            expected = model(images.to(device)).cpu()
            actual = best_model(images)
        smoke_check(torch.allclose(expected, actual, rtol=1e-4, atol=1e-5), "checkpoint output round trip")
        saved_last = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=True)
        smoke_check(saved_last["epoch"] == len(history)
                    and saved_last["scheduler_state_dict"]["last_epoch"] == len(history), "last checkpoint scheduler/epoch")
        smoke_check(saved_last["optimizer_state_dict"]["state"], "optimizer state saved")
        with (results_dir / "training_history.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        smoke_check(len(rows) == len(history) and float(rows[0]["lr"]) == config["LEARNING_RATE"], "CSV rows and actual learning rate")
    best_model = best_model.to(device)
    final_metrics = validate(best_model, val_loader, criterion, device, val_metadata)
    summary = {
        "architecture": "ResNet-18 / Linear(512, 8) / raw logits",
        "best_epoch": best_epoch, "best_val_acc": best_val_acc,
        "evaluation": final_metrics, "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset), "full_dataset_samples": len(sampler.dataset),
        "full_train_samples": len(sampler.train_indices), "full_val_samples": len(sampler.val_indices),
        "split_seed": config["SPLIT_SEED"], "train_seed": config["TRAIN_SEED"],
        "preprocessing": PREPROCESSING, "config": config, "smoke_test": smoke_test,
        "device": str(device), "platform": platform.platform(), "torch_version": str(torch.__version__),
        "amp_active": amp_enabled, "parameters": parameter_count, "trainable_parameters": trainable_count,
        "dataset_fingerprint": sampler.fingerprint, "duplicate_check": sampler.duplicate_summary,
        "validation_scope": "Synthetic dataset only; not measured Raspberry Pi real-world accuracy.",
    }
    plot_results(history, final_metrics, best_epoch, results_dir)
    summary["total_time_sec"] = time.perf_counter() - started
    atomic_save(summary, results_dir / "evaluation_summary.json", as_json=True)
    print(f"Best-model evaluation: {json.dumps(final_metrics, indent=2)}")
    print(f"Checkpoints: {checkpoint_dir}\nResults: {results_dir}", flush=True)
    return summary


def main(smoke_test: bool = False) -> dict:
    config = {name: globals()[name] for name in CONFIG_NAMES}
    if smoke_test:
        config.update(NUM_EPOCHS=1, TRAIN_BATCH_SIZE=8, VAL_BATCH_SIZE=8,
                      NUM_WORKERS=0, USE_PRETRAINED=False)
    validate_settings(config)
    seed_training(config["TRAIN_SEED"], config["DETERMINISTIC_TRAINING"])
    device = resolve_device(config["DEVICE"])
    print(f"Device: {device}; platform: {platform.platform()}; PyTorch: {torch.__version__}", flush=True)
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(f"GPU: {properties.name}; VRAM: {properties.total_memory / 2**30:.1f} GiB; CUDA: {torch.version.cuda}")
    elif device.type == "mps":
        print("GPU backend: Apple Metal (MPS)")
    print(f"Configuration: {json.dumps(config, indent=2)}", flush=True)
    print("Deterministic kernels are enforced when requested; cross-hardware bitwise equality is not guaranteed.")
    print("Synthetic validation does not establish real-world Raspberry Pi accuracy.", flush=True)
    sampler = DatasetSampler(resolve_path(DATASET_DIR), config["NUM_VAL_SAMPLES"], config["SPLIT_SEED"])
    print(json.dumps({key: value for key, value in sampler.split_manifest().items()
                      if key != "validation_paths"}, indent=2), flush=True)
    if smoke_test:
        # Global training RNG activity must not affect the split-local generator.
        random.random()
        np.random.rand()
        torch.rand(3)
        second = DatasetSampler(resolve_path(DATASET_DIR), config["NUM_VAL_SAMPLES"], config["SPLIT_SEED"])
        smoke_check(sampler.val_indices == second.val_indices and sampler.fingerprint == second.fingerprint,
                    "same seed/dataset reproduces the validation split after training RNG use")
        del second
        seed_training(config["TRAIN_SEED"], config["DETERMINISTIC_TRAINING"])
        temporary = Path(tempfile.mkdtemp(prefix="pill-count-smoke-"))
        print(f"Smoke artifacts (separate from formal training): {temporary}", flush=True)
        summary = run_training(sampler, config, device, temporary / "checkpoints",
                               temporary / "results", smoke_test=True)
        print("[PASS] CPU-loadable checkpoint, forward/backward, optimizer, validation isolation, CSV and plots.")
        print(f"AMP hardware verification: {'PASS' if summary['amp_active'] else 'NOT VERIFIED (CUDA AMP inactive)'}")
        return summary
    return run_training(sampler, config, device, resolve_path(CHECKPOINT_DIR), resolve_path(RESULTS_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke-test", action="store_true", help="offline one-epoch miniature run; artifacts go only to a system temporary directory")
    arguments = parser.parse_args()
    try:
        main(smoke_test=arguments.smoke_test)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Training failed: {exc}\nDevice={DEVICE}, deterministic={DETERMINISTIC_TRAINING}, "
            f"AMP requested={ENABLE_AMP}. Unsupported deterministic/AMP operations are never silently retried."
        ) from exc
