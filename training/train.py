"""Train the pill classifier. Edit the settings below, then run this file."""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

import torch
from torch import nn
from torch.utils.data import DataLoader

#===================================================================================
# 학습 설정: CLI 인자 대신 이 섹션의 전역 변수를 수정하세요.
# 상대 DATASET_ROOT / OUTPUT_DIR 경로는 프로젝트 루트를 기준으로 해석합니다.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "training" / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "training" / "checkpoints"

EPOCHS = 20

BATCH_SIZE = 32

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

IMAGE_SIZE = 224

NUM_WORKERS = 0  # macOS에서도 바로 실행할 수 있도록 기본값은 0입니다.

VAL_EVERY_STEPS = 0  # 0: epoch 종료 시에만 검증. 양수: N step마다 추가 검증.
LOG_EVERY_STEPS = 10

PRETRAINED = True  # 최초 실행 시 ImageNet 가중치를 다운로드할 수 있습니다.

THRESHOLD = 0.5  # sigmoid(logit) >= THRESHOLD이면 filled(알약 검출).

DEVICE = "auto"  # "auto", "cuda", "mps", "cpu"

SEED = 42

#===================================================================================

# Support both `python training/train.py` and `python -m training.train`.
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from architecture.model import PillClassifier
from training.dataset_sampler import (
    CLASS_TO_IDX,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
    create_dataloaders,
)

def resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def resolve_device(preference: str) -> torch.device:
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if preference == "auto":
        return torch.device("cuda" if cuda_available else "mps" if mps_available else "cpu")
    if preference not in {"cuda", "mps", "cpu"}:
        raise ValueError('DEVICE must be "auto", "cuda", "mps", or "cpu".')
    if preference == "cuda" and not cuda_available:
        raise RuntimeError("DEVICE='cuda' was requested, but CUDA is unavailable.")
    if preference == "mps" and not mps_available:
        raise RuntimeError("DEVICE='mps' was requested, but Apple MPS is unavailable.")
    return torch.device(preference)


def validate_settings() -> None:
    for name, value in (
        ("EPOCHS", EPOCHS),
        ("BATCH_SIZE", BATCH_SIZE),
        ("IMAGE_SIZE", IMAGE_SIZE),
        ("LOG_EVERY_STEPS", LOG_EVERY_STEPS),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    for name, value in (("NUM_WORKERS", NUM_WORKERS), ("VAL_EVERY_STEPS", VAL_EVERY_STEPS)):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    if LEARNING_RATE <= 0 or WEIGHT_DECAY < 0:
        raise ValueError("LEARNING_RATE must be positive and WEIGHT_DECAY non-negative.")
    if not 0.0 < THRESHOLD < 1.0:
        raise ValueError("THRESHOLD must be between 0 and 1 (exclusive).")

#===================================================================================

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate every sample, then restore the previous model mode.
    """
    was_training = model.training

    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0
    try:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)

            #criterion : BCEwithlogitloss
            loss_sum += criterion(logits, targets).item() * targets.numel()
            predictions = torch.sigmoid(logits) >= threshold

            correct += (predictions == targets.bool()).sum().item()
            total += targets.numel()
    finally:
        model.train(was_training)

    if total == 0:
        raise ValueError("Validation loader contains no images.")
    
    return {"loss": loss_sum / total, "accuracy": correct / total, "correct": correct, "total": total}

#-------------------------------------------------------

def validate_and_save(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    train_metrics: dict,
    best_accuracy: float,
    settings: dict,
    run_dir: Path,
    metrics_file: TextIO,
) -> float:
    
    val_metrics = evaluate(model, val_loader, criterion, device, settings["threshold"])
    record = {
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
    }
    print(
        f"[val] epoch={epoch} step={global_step} "
        f"train_loss={record['train_loss']:.4f} train_acc={record['train_accuracy']:.2%} "
        f"val_loss={record['val_loss']:.4f} val_acc={record['val_accuracy']:.2%} "
        f"({val_metrics['correct']}/{val_metrics['total']})",
        flush=True,
    )
    metrics_file.write(json.dumps(record) + "\n")
    metrics_file.flush()

    improved = val_metrics["accuracy"] > best_accuracy
    best_accuracy = max(best_accuracy, val_metrics["accuracy"])
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_accuracy": best_accuracy,
        "val_metrics": val_metrics,
        "config": settings,
    }
    torch.save(checkpoint, run_dir / "last.pt")
    if improved:
        torch.save(checkpoint, run_dir / "best.pt")

    return best_accuracy

#===================================================================================

def main() -> Path:
    validate_settings()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = resolve_device(DEVICE)
    dataset_root = resolve_path(DATASET_ROOT)
    train_loader, val_loader = create_dataloaders(
        dataset_root,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        pin_memory=device.type == "cuda",
    )
    print(f"Device: {device}", flush=True)
    print(f"Labels: {CLASS_TO_IDX}", flush=True)
    print(f"Train images: {train_loader.dataset.class_counts}", flush=True)
    print(f"Validation images: {val_loader.dataset.class_counts}", flush=True)

    #------------------------------

    model = PillClassifier(pretrained=PRETRAINED).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    settings = {
        "architecture": "resnet18",
        "dataset_root": str(dataset_root),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "image_size": IMAGE_SIZE,
        "num_workers": NUM_WORKERS,
        "val_every_steps": VAL_EVERY_STEPS,
        "log_every_steps": LOG_EVERY_STEPS,
        "pretrained": PRETRAINED,
        "threshold": THRESHOLD,
        "device": str(device),
        "seed": SEED,
        "class_to_idx": dict(CLASS_TO_IDX),
        "normalization_mean": list(NORMALIZATION_MEAN),
        "normalization_std": list(NORMALIZATION_STD),
        "train_augmentation": {"brightness": 0.15, "contrast": 0.15},
    }
    run_dir = resolve_path(OUTPUT_DIR) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    print(f"Outputs: {run_dir}", flush=True)

    #------------------------------

    global_step = 0
    last_validation_step = -1
    best_accuracy = -1.0
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics_file:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            loss_sum = 0.0
            correct = 0
            total = 0
            for batch_index, (images, targets) in enumerate(train_loader, start=1):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                logits = model(images)
                loss = criterion(logits, targets)

                loss.backward()
                optimizer.step()

                global_step += 1
                loss_sum += loss.item() * targets.numel()
                predictions = torch.sigmoid(logits.detach()) >= THRESHOLD
                correct += (predictions == targets.bool()).sum().item()
                total += targets.numel()
                train_metrics = {"loss": loss_sum / total, "accuracy": correct / total}
                if global_step % LOG_EVERY_STEPS == 0:
                    print(
                        f"[train] epoch={epoch}/{EPOCHS} batch={batch_index}/{len(train_loader)} "
                        f"step={global_step} loss={train_metrics['loss']:.4f} "
                        f"accuracy={train_metrics['accuracy']:.2%}",
                        flush=True,
                    )

                if VAL_EVERY_STEPS > 0 and global_step % VAL_EVERY_STEPS == 0:
                    best_accuracy = validate_and_save(
                        model, val_loader, criterion, optimizer, device, epoch, global_step,
                        train_metrics, best_accuracy, settings, run_dir, metrics_file,
                    )
                    last_validation_step = global_step

                #--------------------

            # Always validate at epoch end, without repeating the same step.
            if last_validation_step != global_step:
                best_accuracy = validate_and_save(
                    model, val_loader, criterion, optimizer, device, epoch, global_step,
                    train_metrics, best_accuracy, settings, run_dir, metrics_file,
                )
                last_validation_step = global_step

    #--------------------

    print(f"Training complete. Best validation accuracy: {best_accuracy:.2%}", flush=True)
    print(f"Best checkpoint: {run_dir / 'best.pt'}", flush=True)

    return run_dir

#===================================================================================

if __name__ == "__main__":
    main()