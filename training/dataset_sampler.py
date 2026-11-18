"""Fixed, stratified ImageFolder splits with decoded-image duplicate protection."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_TO_IDX = {f"class{label}": label for label in range(8)}
PILL_TYPES = ("A", "B", "C", "D")
IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)
PREPROCESSING = {
    "color_mode": "RGB", "image_size": list(IMAGE_SIZE),
    "tensor_shape": [3, *IMAGE_SIZE], "tensor_dtype": "float32",
    "pixel_scale": "uint8 / 255", "mean": list(NORMALIZATION_MEAN),
    "std": list(NORMALIZATION_STD), "resize": False, "crop": False,
    "augmentation": False, "label_dtype": "torch.long",
}


def load_rgb(path: str) -> Image.Image:
    """Reject malformed inputs rather than silently changing their geometry."""
    with Image.open(path) as source:
        if source.size != IMAGE_SIZE:
            raise ValueError(f"Expected 224x224 pixels, got {source.size}: {path}")
        if source.format != "PNG":
            raise ValueError(f"Expected PNG image: {path}")
        return source.convert("RGB")


def build_transforms() -> transforms.Compose:
    """The same ImageNet normalization is used for both subsets."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ])


def select_whole_groups(groups: list[list[int]], target: int,
                        rng: random.Random, description: str) -> list[int]:
    """Exact subset sum over duplicate groups; never split identical pixels.

    Randomize group order with a split-local RNG, then retain the first exact
    solution. Sorting before this call makes the result independent of traversal.
    """
    groups = list(groups)
    rng.shuffle(groups)
    parents = {0: None}
    for group_index, group in enumerate(groups):
        for subtotal in sorted(parents, reverse=True):
            candidate = subtotal + len(group)
            if candidate <= target and candidate not in parents:
                parents[candidate] = (subtotal, group_index)
        if target in parents:
            selected = []
            subtotal = target
            while subtotal:
                subtotal, group_index = parents[subtotal]
                selected.extend(groups[group_index])
            return selected
    sizes = dict(sorted(Counter(map(len, groups)).items()))
    raise ValueError(
        f"Cannot allocate exactly {target} validation images for {description} "
        f"while keeping duplicate images together. Group-size counts: {sizes}. "
        "Dataset and sampling quotas were not changed."
    )


class DatasetSampler:
    """Inspect all images once and partition indices without changing files.

    Relative dataset paths resolve against the project root. Validation size
    must be a positive multiple of 32: each class receives N/8 images and each
    nonempty class/type stratum receives N/32 images. Conflicting metadata on
    identical pixels is rejected explicitly instead of weakening stratification.
    """

    def __init__(self, dataset_dir, num_val_samples: int, split_seed: int):
        if type(num_val_samples) is not int or num_val_samples <= 0 or num_val_samples % 32:
            raise ValueError("num_val_samples must be a positive multiple of 32.")
        if type(split_seed) is not int:
            raise ValueError("split_seed must be an integer.")
        root = Path(dataset_dir).expanduser()
        self.dataset_dir = (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
        self.num_val_samples = num_val_samples
        self.split_seed = split_seed
        self.dataset = datasets.ImageFolder(
            self.dataset_dir, transform=build_transforms(), loader=load_rgb,
        )
        if self.dataset.class_to_idx != CLASS_TO_IDX:
            raise ValueError(f"Expected class mapping {CLASS_TO_IDX}; got {self.dataset.class_to_idx}")
        # ImageFolder sorts already; explicitly bind every index to a POSIX path.
        self.dataset.samples.sort(key=lambda sample: Path(sample[0]).relative_to(self.dataset_dir).as_posix())
        self.dataset.imgs = self.dataset.samples
        self.dataset.targets = [label for _, label in self.dataset.samples]
        if num_val_samples >= len(self.dataset):
            raise ValueError("Validation size must leave a nonempty training dataset.")

        self.metadata = []
        self.image_hashes = []
        hash_groups = defaultdict(list)
        fingerprint = hashlib.sha256()
        for index, (filename, label) in enumerate(self.dataset.samples):
            path = Path(filename)
            pattern = r"empty_\d{6}\.png" if label == 0 else r"[ABCD]_\d{6}\.png"
            if not re.fullmatch(pattern, path.name):
                raise ValueError(f"Unexpected filename for class{label}: {path}")
            pill_type = None if label == 0 else path.name[0]
            relative_path = path.relative_to(self.dataset_dir).as_posix()
            try:
                with Image.open(path) as source:
                    if source.mode != "RGB" or source.format != "PNG" or source.size != IMAGE_SIZE:
                        raise ValueError(
                            f"Expected RGB PNG 224x224: {path}; "
                            f"got {source.mode}, {source.format}, {source.size}"
                        )
                    digest = hashlib.sha256(source.tobytes()).hexdigest()
            except OSError as exc:
                raise ValueError(f"Unreadable image: {path}") from exc
            self.metadata.append({"relative_path": relative_path, "label": label, "pill_type": pill_type})
            self.image_hashes.append(digest)
            hash_groups[digest].append(index)
            fingerprint.update((relative_path + "\0" + digest + "\n").encode("utf-8"))

        self.fingerprint = fingerprint.hexdigest()
        strata = defaultdict(list)
        for indices in hash_groups.values():
            keys = {(self.metadata[i]["label"], self.metadata[i]["pill_type"]) for i in indices}
            if len(keys) != 1:
                paths = [self.metadata[i]["relative_path"] for i in indices]
                raise ValueError(f"Identical pixels have conflicting class/type metadata: {paths}")
            strata[next(iter(keys))].append(indices)
        self.duplicate_summary = {
            "unique_images": len(hash_groups),
            "duplicate_groups": sum(len(group) > 1 for group in hash_groups.values()),
            "extra_duplicate_files": len(self.dataset) - len(hash_groups),
            "duplicate_groups_by_class": dict(Counter(
                f"class{self.metadata[group[0]]['label']}"
                for group in hash_groups.values() if len(group) > 1
            )),
            "hash_algorithm": "SHA-256 of decoded RGB pixels (224x224)",
        }
        rng = random.Random(split_seed)
        val_indices = []
        for label in range(len(CLASS_TO_IDX)):
            for pill_type in ((None,) if label == 0 else PILL_TYPES):
                target = num_val_samples // (8 if label == 0 else 32)
                groups = strata[(label, pill_type)]
                available = sum(map(len, groups))
                description = f"class{label}/{pill_type or 'empty'}"
                if available < target:
                    raise ValueError(f"Insufficient images in {description}: need {target}, found {available}.")
                val_indices.extend(select_whole_groups(groups, target, rng, description))

        self.val_indices = sorted(val_indices)
        val_set = set(self.val_indices)
        self.train_indices = [i for i in range(len(self.dataset)) if i not in val_set]
        train_set = set(self.train_indices)
        if (len(val_indices) != num_val_samples or len(val_set) != num_val_samples
                or train_set & val_set or train_set | val_set != set(range(len(self.dataset)))):
            raise RuntimeError("Split indices are not an exact, disjoint partition.")
        train_hashes = {self.image_hashes[i] for i in self.train_indices}
        val_hashes = {self.image_hashes[i] for i in self.val_indices}
        if train_hashes & val_hashes:
            raise RuntimeError("Identical RGB images leaked between train and validation.")
        self.duplicate_summary["cross_split_duplicate_hashes"] = 0
        self.train_dataset = Subset(self.dataset, self.train_indices)
        self.val_dataset = Subset(self.dataset, self.val_indices)

    def get_train_dataset(self) -> Subset:
        return self.train_dataset

    def get_val_dataset(self) -> Subset:
        return self.val_dataset

    def distribution(self, indices: list[int]) -> dict:
        class_counts = {name: 0 for name in CLASS_TO_IDX}
        by_class = {f"class{label}": {pill: 0 for pill in PILL_TYPES} for label in range(1, 8)}
        for index in indices:
            meta = self.metadata[index]
            class_counts[f"class{meta['label']}"] += 1
            if meta["pill_type"] is not None:
                by_class[f"class{meta['label']}"][meta["pill_type"]] += 1
        return {"classes": class_counts, "pill_types_by_class": by_class,
                "pill_types": {pill: sum(row[pill] for row in by_class.values()) for pill in PILL_TYPES}}

    def split_manifest(self) -> dict:
        return {
            "schema_version": 1,
            "dataset_path": Path(os.path.relpath(self.dataset_dir, PROJECT_ROOT)).as_posix(),
            "dataset_path_base": "project_root",
            "total_samples": len(self.dataset), "train_samples": len(self.train_indices),
            "num_val_samples": self.num_val_samples, "split_seed": self.split_seed,
            "class_to_idx": CLASS_TO_IDX, "dataset_fingerprint": self.fingerprint,
            "fingerprint_definition": "SHA-256 of sorted relative path + NUL + RGB pixel hash + newline",
            "train_distribution": self.distribution(self.train_indices),
            "val_distribution": self.distribution(self.val_indices),
            "duplicate_check": self.duplicate_summary,
            "validation_paths": [self.metadata[i]["relative_path"] for i in self.val_indices],
            "preprocessing": PREPROCESSING,
        }

    def save_validation_split(self, path) -> None:
        """Atomically create a portable manifest, refusing an existing run."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(self.split_manifest(), stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Exclusive publication also reserves the checkpoint directory for
            # one run. Unlike replace(), link() never overwrites an older split.
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
