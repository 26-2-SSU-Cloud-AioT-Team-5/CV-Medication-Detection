"""Train the frame-status head on features from a frozen Phase 1 checkpoint.

Edit the globals below. For a new run, use fresh --checkpoint-dir and
--results-dir paths; existing run files are never overwritten.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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

NUM_EPOCHS = 100

TRAIN_EMPTY_SAMPLES = 256
TRAIN_COUNT_SAMPLES_PER_CLASS = 4  # Each of class1 through class7.

ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-4  # Minimum validation-loss improvement.

BATCH_SIZE = 16

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DEVICE = "auto"

# ==============================================================================================

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from architecture.model import STATUS_NAMES, PillStateClassifier, load_model_state
from training.dataset_sampler import (
    BLOCKED_FOLDERS, CLASS_TO_IDX, PREPROCESSING,
    LabeledImages, inspect_image_splits, png_paths,
)
from training.train_phase1 import (
    atomic_save, prepare_output_dirs, resolve_device, resolve_path, seed_training, smoke_check, snapshot_state,
)


def collect_status_samples(root: Path, *, validation: bool, seed: int) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    samples = []
    for folder, count in CLASS_TO_IDX.items():
        quota = TRAIN_EMPTY_SAMPLES if count == 0 else TRAIN_COUNT_SAMPLES_PER_CLASS
        paths = png_paths(root / folder, minimum=0 if validation else quota)
        selected = paths if validation else sorted(rng.sample(paths, quota))
        samples.extend((path, 0) for path in selected)
    if not samples:
        raise ValueError(f"Need at least one countable PNG image in {root}.")

    for label, folder in enumerate(BLOCKED_FOLDERS, start=1):
        paths = png_paths(root / folder)
        samples.extend((path, label) for path in paths)
    return samples


def build_status_sampler(labels, *, seed: int) -> WeightedRandomSampler:
    """Give each status equal expected exposure despite unequal file counts.

Draw one dataset-length epoch with replacement; minority examples may repeat
and majority examples may be omitted in an individual epoch.
"""
    counts = torch.bincount(labels, minlength=len(STATUS_NAMES))
    if (counts == 0).any().item():
        raise ValueError("Phase 2 training requires examples of all three statuses.")
    return WeightedRandomSampler(
        weights=counts.double().reciprocal()[labels], num_samples=len(labels), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


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
    train_features = extract_features(model, train_samples, device)
    train_loader = DataLoader(
        train_features, batch_size=BATCH_SIZE,
        sampler=build_status_sampler(train_features.tensors[1], seed=TRAIN_SEED),
        generator=torch.Generator().manual_seed(TRAIN_SEED),
    )
    val_loader = DataLoader(extract_features(model, val_samples, device), batch_size=BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = []
    best_key = (-1.0, 0, -float("inf"))
    lowest_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False
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
        selection_key = (metrics["balanced_accuracy"], -metrics["false_accept_count"], -metrics["loss"])
        if selection_key > best_key:
            best_key, best = selection_key, record
            atomic_save({
                "phase": 2, "model_state_dict": model.state_dict(), "status_names": list(STATUS_NAMES),
                "epoch": epoch, "validation": metrics, "phase1_checkpoint": manifest["phase1_checkpoint"],
                "dataset_fingerprint": manifest["dataset_fingerprint"], "preprocessing": PREPROCESSING,
                "training_config": manifest["training_config"],
            }, checkpoint_dir / "best.pt")
        print(f"Epoch {epoch:02d}/{epochs:02d} | train loss {record['train_loss']:.4f} | "
              f"val balanced accuracy {metrics['balanced_accuracy']:.2%} | "
              f"val loss {metrics['loss']:.4f} | "
              f"false accepts {metrics['false_accept_count']} | false rejects {metrics['false_reject_count']}", flush=True)
        if metrics["loss"] < lowest_loss - EARLY_STOPPING_MIN_DELTA:
            lowest_loss = metrics["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if ENABLE_EARLY_STOPPING and not smoke_test and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            print(f"Early stopping: validation loss did not improve by {EARLY_STOPPING_MIN_DELTA:g} "
                  f"for {EARLY_STOPPING_PATIENCE} epochs.", flush=True)
            break

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
        "train_status_samples": {
            name: sum(label == index for _, label in train_samples) for index, name in enumerate(STATUS_NAMES)
        },
        "training_config": manifest["training_config"],
        "epochs_completed": len(history), "stopped_early": stopped_early,
        "device": str(device), "smoke_test": smoke_test,
    }
    atomic_save(history, results_dir / "training_history.json", as_json=True)
    atomic_save(summary, results_dir / "evaluation_summary.json", as_json=True)
    return summary


def main(smoke_test: bool = False, phase1_checkpoint=None, checkpoint_dir=None, results_dir=None) -> dict:
    for name in ("NUM_EPOCHS", "TRAIN_EMPTY_SAMPLES", "TRAIN_COUNT_SAMPLES_PER_CLASS",
                 "BATCH_SIZE", "EARLY_STOPPING_PATIENCE"):
        value = globals()[name]
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not math.isfinite(EARLY_STOPPING_MIN_DELTA) or EARLY_STOPPING_MIN_DELTA < 0:
        raise ValueError("EARLY_STOPPING_MIN_DELTA must be finite and nonnegative.")
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
        "schema_version": 2, "sample_seed": SAMPLE_SEED, "train_seed": TRAIN_SEED,
        "phase1_checkpoint": str(source), "status_names": list(STATUS_NAMES),
        "train_root": str(train_root), "validation_root": str(val_root),
        "validation_count_folders": sorted({path.relative_to(val_root).parts[0]
                                            for path, label in val_samples if label == 0}),
        "training_config": {
            "num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
            "train_empty_samples": TRAIN_EMPTY_SAMPLES,
            "train_count_samples_per_class": TRAIN_COUNT_SAMPLES_PER_CLASS,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "sampling": "inverse_status_frequency_with_replacement",
            "samples_per_epoch": len(train_samples), "validation_sampling": "all_available_images",
            "best_selection": ["balanced_accuracy_desc", "false_accept_count_asc", "loss_asc"],
            "early_stopping": ENABLE_EARLY_STOPPING, "early_stopping_monitor": "validation_loss",
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        },
        "train_samples": {path.relative_to(train_root).as_posix(): STATUS_NAMES[label] for path, label in train_samples},
        "validation_samples": {path.relative_to(val_root).as_posix(): STATUS_NAMES[label] for path, label in val_samples},
        **inspect_image_splits(train_root, train_samples, val_root, val_samples),
    }
    print(f"Train: {len(train_samples)} images. Validation: {len(val_samples)} images "
          f"(all available images). Device: {device}.", flush=True)
    print("Train status samples: " + ", ".join(
        f"{name}={sum(label == index for _, label in train_samples)}"
        for index, name in enumerate(STATUS_NAMES)
    ) + "; equal expected sampling per status.", flush=True)
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
                        resolve_path(CHECKPOINT_DIR if checkpoint_dir is None else checkpoint_dir),
                        resolve_path(RESULTS_DIR if results_dir is None else results_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="train one epoch and use temporary output directories")
    parser.add_argument("--phase1-checkpoint", help="override the Phase 1 best.pt path")
    parser.add_argument("--checkpoint-dir", help="fresh output directory for checkpoints and data selection")
    parser.add_argument("--results-dir", help="fresh output directory for training history and metrics")
    args = parser.parse_args()
    main(smoke_test=args.smoke_test, phase1_checkpoint=args.phase1_checkpoint,
         checkpoint_dir=args.checkpoint_dir, results_dir=args.results_dir)
