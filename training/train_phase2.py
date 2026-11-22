"""Train the frame-status head on features from a frozen Phase 1 checkpoint."""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================================

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
DATASET_DIR = TRAINING_DIR / "dataset" / "train_n_val"
VALIDATION_DIR = TRAINING_DIR / "dataset" / "test"
PHASE1_CHECKPOINT = TRAINING_DIR / "checkpoints" / "phase1" / "best.pt"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints" / "phase2"
RESULTS_DIR = TRAINING_DIR / "results" / "phase2"

SAMPLE_SEED = 42
TRAIN_SEED = 42

NUM_EPOCHS = 30

BATCH_SIZE = 16

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = "auto"

# ==============================================================================================

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from architecture.model import STATUS_NAMES, PillStateClassifier, load_model_state
from training.dataset_sampler import (
    BLOCKED_FOLDERS, CLASS_TO_IDX, PREPROCESSING, VALIDATION_COUNT_FOLDERS,
    LabeledImages, inspect_image_splits, png_paths,
)
from training.train_phase1 import (
    atomic_save, prepare_output_dirs, resolve_device, resolve_path, seed_training, smoke_check, snapshot_state,
)


def collect_status_samples(root: Path, *, validation: bool, seed: int) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    count_folders = VALIDATION_COUNT_FOLDERS if validation else CLASS_TO_IDX
    quota = 2 if validation else 4
    samples = []
    for folder in count_folders:
        paths = png_paths(root / folder, minimum=quota)
        samples.extend((path, 0) for path in sorted(rng.sample(paths, quota)))

    for label, folder in enumerate(BLOCKED_FOLDERS, start=1):
        paths = png_paths(root / folder, minimum=12 if validation else 1)
        if validation and len(paths) != 12:
            raise ValueError(f"Expected 12 validation PNG images in {root / folder}, found {len(paths)}.")
        samples.extend((path, label) for path in paths)
    return samples


@torch.no_grad()
def extract_features(model, samples, device):
    """Cache embeddings once because the backbone and preprocessing are fixed."""
    loader = DataLoader(LabeledImages(samples), batch_size=BATCH_SIZE)
    features, targets = [], []
    for images, labels in loader:
        features.append(model.backbone(images.to(device)).cpu())
        targets.append(labels)
    return TensorDataset(torch.cat(features), torch.cat(targets))


@torch.inference_mode()
def validate_status(head, loader, criterion, device) -> dict:
    head.eval()
    size = len(STATUS_NAMES)
    confusion = torch.zeros((size, size), dtype=torch.long)
    loss_sum = 0.0
    for features, labels in loader:
        logits = head(features.to(device))
        loss = criterion(logits, labels.to(device))
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite Phase 2 validation loss.")
        loss_sum += loss.item() * labels.numel()
        indices = labels * size + logits.argmax(1).cpu()
        confusion += torch.bincount(indices, minlength=size**2).reshape(size, size)
    totals = confusion.sum(dim=1)
    recall = confusion.diag() / totals
    false_accept = confusion[1:, 0].sum().item()
    false_reject = confusion[0, 1:].sum().item()
    total = totals.sum().item()
    return {
        "loss": loss_sum / total,
        "accuracy": confusion.diag().sum().item() / total,
        "balanced_accuracy": recall.mean().item(),
        "status_recall": dict(zip(STATUS_NAMES, recall.tolist())),
        "status_samples": dict(zip(STATUS_NAMES, totals.tolist())),
        "false_accept_count": false_accept,
        "false_accept_rate": false_accept / totals[1:].sum().item(),
        "false_reject_count": false_reject,
        "false_reject_rate": false_reject / totals[0].item(),
        "confusion_matrix": confusion.tolist(),
    }


def run_training(model, train_samples, val_samples, manifest, device, checkpoint_dir, results_dir,
                 *, smoke_test: bool = False) -> dict:
    prepare_output_dirs(checkpoint_dir, results_dir)
    atomic_save(manifest, checkpoint_dir / "data_selection.json", as_json=True, exclusive=True)
    model.to(device).requires_grad_(False).eval()
    head = model.status_head.requires_grad_(True)
    if smoke_test:
        frozen_before = snapshot_state(model.backbone)
        count_before = snapshot_state(model.count_head)
        status_before = snapshot_state(head)
    train_loader = DataLoader(
        extract_features(model, train_samples, device), batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(TRAIN_SEED),
    )
    val_loader = DataLoader(extract_features(model, val_samples, device), batch_size=BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = []
    best_key = (-1.0, 0)
    epochs = 1 if smoke_test else NUM_EPOCHS
    for epoch in range(1, epochs + 1):
        head.train()
        loss_sum = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features.to(device)), labels.to(device))
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite Phase 2 training loss.")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
        metrics = validate_status(head, val_loader, criterion, device)
        record = {"epoch": epoch, "train_loss": loss_sum / len(train_samples), "validation": metrics}
        history.append(record)
        selection_key = (metrics["balanced_accuracy"], -metrics["false_accept_count"])
        if selection_key > best_key:
            best_key, best = selection_key, record
            atomic_save({
                "phase": 2, "model_state_dict": model.state_dict(), "status_names": list(STATUS_NAMES),
                "epoch": epoch, "validation": metrics, "phase1_checkpoint": manifest["phase1_checkpoint"],
                "dataset_fingerprint": manifest["dataset_fingerprint"], "preprocessing": PREPROCESSING,
            }, checkpoint_dir / "best.pt")
        print(f"Epoch {epoch:02d}/{epochs:02d} | train loss {record['train_loss']:.4f} | "
              f"val balanced accuracy {metrics['balanced_accuracy']:.2%} | "
              f"false accepts {metrics['false_accept_count']} | false rejects {metrics['false_reject_count']}", flush=True)

    if smoke_test:
        smoke_check(all(torch.equal(value, model.backbone.state_dict()[key].cpu())
                        for key, value in frozen_before.items()), "frozen backbone weights and BatchNorm buffers")
        smoke_check(all(torch.equal(value, model.count_head.state_dict()[key].cpu())
                        for key, value in count_before.items()), "frozen count head")
        smoke_check(any(not torch.equal(value, head.state_dict()[key].cpu())
                        for key, value in status_before.items()), "status head weights changed")
        reloaded = PillStateClassifier()
        reloaded.load_state_dict(load_model_state(checkpoint_dir / "best.pt"), strict=True)
        print("[PASS] Frozen backbone and count head, trained status head, and loadable dual-head checkpoint.")

    summary = {
        "best_epoch": best["epoch"], "evaluation": best["validation"],
        "train_samples": len(train_samples), "validation_samples": len(val_samples),
        "validation_count_folders": manifest["validation_count_folders"],
        "phase1_checkpoint": manifest["phase1_checkpoint"], "sample_seed": manifest["sample_seed"],
        "device": str(device), "smoke_test": smoke_test,
    }
    atomic_save(history, results_dir / "training_history.json", as_json=True)
    atomic_save(summary, results_dir / "evaluation_summary.json", as_json=True)
    return summary


def main(smoke_test: bool = False, phase1_checkpoint=None) -> dict:
    if not smoke_test and NUM_EPOCHS < 1:
        raise ValueError("NUM_EPOCHS must be positive.")
    source = resolve_path(PHASE1_CHECKPOINT if phase1_checkpoint is None else phase1_checkpoint)
    if not source.is_file():
        raise FileNotFoundError(f"Phase 1 checkpoint is missing: {source}. Run train_phase1.py first.")
    seed_training(TRAIN_SEED, deterministic=True)
    device = resolve_device(DEVICE)
    model = PillStateClassifier(load_model_state(source))
    train_root, val_root = resolve_path(DATASET_DIR), resolve_path(VALIDATION_DIR)
    train_samples = collect_status_samples(train_root, validation=False, seed=SAMPLE_SEED)
    val_samples = collect_status_samples(val_root, validation=True, seed=SAMPLE_SEED)
    manifest = {
        "schema_version": 1, "sample_seed": SAMPLE_SEED, "train_seed": TRAIN_SEED,
        "phase1_checkpoint": str(source), "status_names": list(STATUS_NAMES),
        "train_root": str(train_root), "validation_root": str(val_root),
        "validation_count_folders": list(VALIDATION_COUNT_FOLDERS),
        "train_samples": {path.relative_to(train_root).as_posix(): STATUS_NAMES[label] for path, label in train_samples},
        "validation_samples": {path.relative_to(val_root).as_posix(): STATUS_NAMES[label] for path, label in val_samples},
        **inspect_image_splits(train_root, train_samples, val_root, val_samples),
    }
    print(f"Train: {len(train_samples)} images. Validation: {len(val_samples)} images "
          f"(normal counts class0–class6). Device: {device}.", flush=True)
    if smoke_test:
        random.random()
        smoke_check(train_samples == collect_status_samples(train_root, validation=False, seed=SAMPLE_SEED)
                    and val_samples == collect_status_samples(val_root, validation=True, seed=SAMPLE_SEED),
                    "sample selection is reproducible and independent of the global RNG")
        with tempfile.TemporaryDirectory(prefix="pill-status-smoke-") as workdir:
            temporary = Path(workdir)
            return run_training(model, train_samples, val_samples, manifest, device,
                                temporary / "checkpoints", temporary / "results", smoke_test=True)
    return run_training(model, train_samples, val_samples, manifest, device,
                        resolve_path(CHECKPOINT_DIR), resolve_path(RESULTS_DIR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="train one epoch and use temporary output directories")
    parser.add_argument("--phase1-checkpoint", help="override the Phase 1 best.pt path")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test, phase1_checkpoint=args.phase1_checkpoint)
