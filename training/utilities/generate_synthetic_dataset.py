"""Generate count-labelled pills, controlled contact, and whole-scene augmentation.

Requires only NumPy and OpenCV; torchvision is optional for the loader check.
Run modes in order: assets, masks, rotation, feasibility, pilot, full, validate.
Inspect masks/contact_sheet.png and pilot/contact_sheet.png before using
``--mode full --reviewed-previews``. All debug files live outside ImageFolder.
Existing nonempty datasets are never resumed, deleted, or overwritten: use
validate to audit them, or choose a fresh --output-dir / --debug-dir for another
run. A pre-existing, completely empty output directory may be used.
Use --replace-generated to stage and validate a replacement, then retain the
verified previous generated dataset as a backup. Real/unknown data is refused.
Use --replace-count-classes for an explicitly authorized class0..class7
replacement without a prior manifest: retain a full backup and preserve all
other classes byte-for-byte. The sibling dataset/test is never a target.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Optional

import cv2
import numpy as np


# ============================================================
# Path Configuration (independent of the working directory)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKGROUND_DIR = PROJECT_ROOT / "training/dataset_synth_prep/background"
PILL_ASSET_DIR = PROJECT_ROOT / "training/dataset_synth_prep/pill_type/cropped_resized224"
OUTPUT_DIR = PROJECT_ROOT / "training/dataset/train_n_val"
DEBUG_OUTPUT_DIR = PROJECT_ROOT / "training/dataset_synth_prep/generation_debug"

# ============================================================
# Dataset Configuration
# ============================================================
IMAGE_SIZE = 224
PILL_TYPES = ("A", "B", "C", "D")
MAX_PILL_COUNT = 7
TARGET_CLASSES = tuple(f"class{count}" for count in range(MAX_PILL_COUNT + 1))
EXPECTED_BACKGROUNDS = 16
EXPECTED_ASSETS_PER_TYPE = 8
NUM_SAMPLES_PER_TYPE = 500
NUM_EMPTY_SAMPLES = NUM_SAMPLES_PER_TYPE * 4
PILOT_NUM_SAMPLES_PER_TYPE = 10

# ============================================================
# Mask / Placement Configuration
# ============================================================
# Measured floor pixels are dark and neutral/blue; walls are green/brown.
# Otsu alone (45..112 on these inputs) includes walls on several backgrounds.
DARK_THRESHOLD = 50
MIN_BLUE_MINUS_GREEN = -2
CLOSING_KERNEL_SIZE = 9
BOUNDARY_MARGIN = 3  # Chebyshev distance: erode with a 7 x 7 square.
MIN_PILL_GAP = 0  # Forbidden occupancy is dilated by this many pixels.
ALPHA_THRESHOLD = 0  # Every nonzero alpha pixel participates, even alpha=1.
MAX_PLACEMENT_ATTEMPTS = 200
MAX_IMAGE_RETRIES = 50
MAX_LAYOUT_BACKTRACKS = 20  # Rotation only: undo prior positions, never angles/assets.
FEASIBILITY_TRIALS = 20
# Exact fraction within every class2..7/type stratum. Contact scenes contain
# at least one edge-adjacent pair; the remaining scenes have no 8-neighbor contact.
CONTACT_IMAGE_FRACTION = 0.30
# Optional replacement floor polygons, before safety erosion, keyed by filename.
# No overrides were needed for the supplied 16 backgrounds.
MASK_POLYGONS: dict[str, list[tuple[int, int]]] = {}

# ============================================================
# Augmentation / Execution Configuration
# ============================================================
BRIGHTNESS_RANGE = (0.85, 1.15)
CONTRAST_RANGE = (0.9, 1.1)
ENABLE_PILL_ROTATION = True
PILL_ROTATION_RANGE = (0.0, 360.0)
# Original manual crop coordinates were not saved. None explicitly selects
# ordinary image-plane rotation as an APPROXIMATION, authorized for these data.
# If a verified common source/target ROI size becomes available, set (W, H):
# T = S @ R @ inv(S), S = diag(224/W, 224/H). No guessed scale is used.
PILL_ROI_SIZE: Optional[tuple[int, int]] = None
ROTATION_PREVIEW_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)
# These affect the COMPLETED 224x224 scene, including background and all pills.
# 0.75 nonzero quarter-turn probability makes 0/90/180/270 equally likely.
SCENE_ROTATION_PROB = 0.75
SCENE_HORIZONTAL_FLIP_PROB = 0.50
SCENE_LIGHTING_PROB = 0.50
LIGHTING_GRADIENT_RANGE = (0.04, 0.10)  # Smooth directional gain +/-4..10%.
SCENE_SHADOW_PROB = 0.35
SHADOW_STRENGTH_RANGE = (0.10, 0.25)  # At most 25% local attenuation.
SHADOW_RADIUS_RANGE = (0.30, 0.75)  # Gaussian sigma in normalized [-1,1] image coordinates.
SCENE_GAIN_LIMITS = (0.65, 1.30)  # Combined brightness/lighting/shadow gain budget.
SCENE_GRID_X, SCENE_GRID_Y = np.meshgrid(
    np.linspace(-1.0, 1.0, IMAGE_SIZE, dtype=np.float32),
    np.linspace(-1.0, 1.0, IMAGE_SIZE, dtype=np.float32),
)
SEED = 42
PNG_COMPRESSION = 3
LOG_EVERY = 100


def require(condition: bool, message: str) -> None:
    """Raise a real exception even when Python is invoked with -O."""
    if not condition:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_once(path: Path, data: bytes, allow_identical: bool = False) -> None:
    """Exclusively create a file; never replace existing content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
    except FileExistsError:
        if not allow_identical or path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite {path}; select a fresh output directory.")


def encode_png(image: np.ndarray) -> bytes:
    """Encode OpenCV BGR/BGRA or grayscale pixels as a lossless PNG."""
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    require(ok, "PNG encoding failed")
    return encoded.tobytes()


def read_image(path: Path) -> np.ndarray:
    """Decode unchanged, preserving channel count and bit depth for validation."""
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    require(image is not None, f"Cannot decode image: {path}")
    return image


def image_paths(directory: Path) -> list[Path]:
    require(directory.is_dir(), f"Missing asset directory: {directory}")
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in
                  {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})


def load_backgrounds(directory: Path = BACKGROUND_DIR) -> list[dict]:
    """Load exactly 16 uint8 224x224 three-channel backgrounds once."""
    paths = image_paths(directory)
    require(len(paths) == EXPECTED_BACKGROUNDS, f"Expected 16 backgrounds, found {len(paths)}")
    backgrounds = []
    for path in paths:
        image = read_image(path)
        require(image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3) and image.dtype == np.uint8,
                f"Background must be uint8 224x224 BGR: {path}, {image.shape}, {image.dtype}")
        backgrounds.append({"id": path.name, "path": path, "image": image,
                            "sha256": sha256(path.read_bytes())})
    return backgrounds


def load_pill_assets(directory: Path = PILL_ASSET_DIR) -> dict[str, list[dict]]:
    """Validate RGBA assets and cache native pixels, float alpha, and alpha>0 masks."""
    pools = {}
    manifest_path = directory / "manifest.csv"
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8", newline="") as stream:
            manifest = {r["asset"]: r for r in csv.DictReader(stream)}
    for pill_type in PILL_TYPES:
        paths = image_paths(directory / pill_type)
        require(len(paths) == EXPECTED_ASSETS_PER_TYPE,
                f"Expected 8 {pill_type} assets, found {len(paths)}")
        pool = []
        for path in paths:
            image = read_image(path)
            require(path.suffix.lower() == ".png" and image.ndim == 3 and image.shape[2] == 4
                    and image.dtype == np.uint8, f"Expected uint8 RGBA PNG: {path}")
            h, w = image.shape[:2]
            require(0 < h <= IMAGE_SIZE and 0 < w <= IMAGE_SIZE, f"Invalid asset dimensions: {path}")
            alpha = image[:, :, 3]
            mask = alpha > ALPHA_THRESHOLD
            require(bool(mask.any()) and bool((alpha == 0).any()) and int(alpha.max()) == 255,
                    f"Empty or invalid alpha channel: {path}")
            key = f"{pill_type}/{path.name}"
            if manifest:
                require(key in manifest, f"Asset missing from ROI manifest: {key}")
                row = manifest[key]
                require((w, h) == (int(row["width"]), int(row["height"]))
                        and (int(row["source_width"]), int(row["source_height"])) == (224, 224),
                        f"Asset geometry differs from ROI manifest: {key}")
            pool.append({"id": key, "type": pill_type, "path": path, "image": image,
                         "alpha": alpha[:, :, None].astype(np.float32) / 255.0,
                         "foreground": image[:, :, :3].astype(np.float32),
                         "mask": mask, "area": int(mask.sum()), "width": w, "height": h,
                         "sha256": sha256(path.read_bytes())})
        pools[pill_type] = pool
    return pools


def configuration(backgrounds: list[dict], pools: dict, samples: int, empty: int) -> dict:
    """Bind all validation evidence to exact inputs, code, parameters, and libraries."""
    return {
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "opencv": cv2.__version__, "numpy": np.__version__,
        "image_size": IMAGE_SIZE, "types": list(PILL_TYPES), "max_count": MAX_PILL_COUNT,
        "samples_per_type": samples, "empty_samples": empty,
        "pilot_samples_per_type": PILOT_NUM_SAMPLES_PER_TYPE,
        "dark_threshold": DARK_THRESHOLD, "min_blue_minus_green": MIN_BLUE_MINUS_GREEN,
        "closing_kernel": CLOSING_KERNEL_SIZE, "boundary_margin": BOUNDARY_MARGIN,
        "min_pill_gap": MIN_PILL_GAP, "alpha_threshold": ALPHA_THRESHOLD,
        "mask_polygons": MASK_POLYGONS, "max_placement_attempts": MAX_PLACEMENT_ATTEMPTS,
        "max_image_retries": MAX_IMAGE_RETRIES, "feasibility_trials": FEASIBILITY_TRIALS,
        "max_layout_backtracks": MAX_LAYOUT_BACKTRACKS,
        "brightness": list(BRIGHTNESS_RANGE), "contrast": list(CONTRAST_RANGE),
        "enable_pill_rotation": ENABLE_PILL_ROTATION,
        "pill_rotation_range": list(PILL_ROTATION_RANGE), "pill_roi_size": PILL_ROI_SIZE,
        "rotation_method": "S_R_inverse_S" if PILL_ROI_SIZE else "image_plane_2d_approximation",
        "rotation_interpolation": "one float32 premultiplied-alpha INTER_LINEAR affine",
        "contact_image_fraction": CONTACT_IMAGE_FRACTION, "contact_definition": "8-neighbor alpha>0 adjacency, edge contact forced",
        "scene_rotation_prob": SCENE_ROTATION_PROB, "scene_horizontal_flip_prob": SCENE_HORIZONTAL_FLIP_PROB,
        "scene_lighting_prob": SCENE_LIGHTING_PROB, "lighting_gradient_range": LIGHTING_GRADIENT_RANGE,
        "scene_shadow_prob": SCENE_SHADOW_PROB, "shadow_strength_range": SHADOW_STRENGTH_RANGE,
        "shadow_radius_range": SHADOW_RADIUS_RANGE, "scene_gain_limits": SCENE_GAIN_LIMITS,
        "seed": SEED, "png_compression": PNG_COMPRESSION,
        "backgrounds": {b["id"]: b["sha256"] for b in backgrounds},
        "assets": {a["id"]: a["sha256"] for pool in pools.values() for a in pool},
    }


def rotation_linear_matrix(angle: float) -> np.ndarray:
    """Return determinant-one R, or S R S^-1 with a verified common ROI size.

    Positive angles use the standard 2D matrix; with image y pointing down they
    appear clockwise. Source and target coordinates remain the same ROI system.
    """
    require(math.isfinite(angle), "Rotation angle must be finite")
    radians = math.radians(angle % 360.0)
    c, s = math.cos(radians), math.sin(radians)
    matrix = np.array([[c, -s], [s, c]], np.float64)
    matrix[np.abs(matrix) < 1e-12] = 0.0
    if PILL_ROI_SIZE is not None:
        require(len(PILL_ROI_SIZE) == 2 and all(v > 0 for v in PILL_ROI_SIZE), "Invalid verified ROI size")
        scales = np.diag([IMAGE_SIZE / PILL_ROI_SIZE[0], IMAGE_SIZE / PILL_ROI_SIZE[1]])
        matrix = scales @ matrix @ np.linalg.inv(scales)
    require(abs(float(np.linalg.det(matrix)) - 1.0) < 1e-9, "Rotation must preserve continuous area")
    return matrix


def rotate_pill_asset(asset: dict, angle: float) -> dict:
    """Warp native premultiplied BGRA once on an expanded canvas, then trim alpha.

    Zero degrees is an exact byte-preserving identity. Masks, alpha blending,
    placement and replay all use the same quantized alpha>0 footprint.
    """
    matrix = rotation_linear_matrix(angle)
    if angle % 360.0 == 0:
        return {**asset, "angle": float(angle)}
    h, w = asset["height"], asset["width"]
    center = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    # Include the full bilinear support outside the first/last pixel centers,
    # plus a 2-pixel guard. The guard is checked before transparent trimming.
    corners = np.array([[-1, -1], [w, -1], [w, h], [-1, h]], np.float64)
    transformed = (corners - center) @ matrix.T
    lower = np.floor(transformed.min(axis=0)) - 2
    upper = np.ceil(transformed.max(axis=0)) + 2
    size = tuple(map(int, upper - lower + 1))
    affine = np.column_stack((matrix, -matrix @ center - lower))
    if PILL_ROI_SIZE is None and angle % 90.0 == 0:
        # Avoid a needless half-pixel blur for exact quarter-turns when source
        # width and height have different parity. Translation only sets the
        # output canvas origin; it does not alter the centered linear rotation.
        affine[:, 2] = np.rint(affine[:, 2])
    premultiplied = np.concatenate((asset["foreground"] * asset["alpha"], asset["alpha"]), axis=2)
    warped = cv2.warpAffine(premultiplied, affine, size, flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    require(not bool(warped[[0, -1], :, 3].any()) and not bool(warped[:, [0, -1], 3].any()),
            f"Rotation canvas clips alpha: {asset['id']}, {angle}")
    alpha_float = np.clip(warped[:, :, 3], 0.0, 1.0)
    alpha_u8 = np.clip(np.rint(alpha_float * 255.0), 0, 255).astype(np.uint8)
    mask = alpha_u8 > ALPHA_THRESHOLD
    require(bool(mask.any()), f"Rotation produced empty alpha: {asset['id']}")
    straight = np.zeros_like(warped[:, :, :3])
    np.divide(warped[:, :, :3], alpha_float[:, :, None], out=straight,
              where=alpha_float[:, :, None] > 0)
    image = np.dstack((np.clip(np.rint(straight), 0, 255).astype(np.uint8), alpha_u8))
    image[~mask, :3] = 0
    x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
    image = image[y:y+height, x:x+width].copy()
    alpha = image[:, :, 3]
    return {**asset, "image": image, "angle": float(angle), "width": width, "height": height,
            "alpha": alpha[:, :, None].astype(np.float32) / 255.0,
            "foreground": image[:, :, :3].astype(np.float32),
            "mask": alpha > ALPHA_THRESHOLD, "area": int((alpha > ALPHA_THRESHOLD).sum())}


def generate_rotation_previews(pools: dict, directory: Path, fingerprint: str) -> dict:
    """Check identity, alpha mass, and affine area; export native-size RGBA examples."""
    tiles, results = [], []
    for pill_type, pool in pools.items():
        source = pool[0]
        for label, angle in [("original", None)] + [(f"{a} deg", a) for a in ROTATION_PREVIEW_ANGLES]:
            asset = source if angle is None else rotate_pill_asset(source, angle)
            ratio = float(asset["alpha"].sum() / source["alpha"].sum())
            require(0.97 < ratio < 1.03, f"Unexpected alpha mass change: {source['id']} {angle}")
            if angle == 0:
                require(np.array_equal(asset["image"], source["image"]), "Zero-degree identity failed")
            require(np.array_equal(asset["mask"], asset["image"][:, :, 3] > 0), "Transformed mask differs from alpha")
            native = asset["image"]
            if angle is not None:
                write_once(directory / f"{pill_type}_{angle:03d}.png", encode_png(native), True)
            canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 85, np.uint8)
            canvas[::16, :] = 115
            canvas[:, ::16] = 115
            require(asset["width"] <= IMAGE_SIZE and asset["height"] <= IMAGE_SIZE, "Preview does not fit")
            alpha_blend(canvas, asset, (IMAGE_SIZE - asset["width"]) // 2, (IMAGE_SIZE - asset["height"]) // 2)
            tiles.append((f"{pill_type} {label} {asset['width']}x{asset['height']}", canvas))
            results.append({"asset": source["id"], "angle": angle, "width": asset["width"],
                            "height": asset["height"], "alpha_mass_ratio": ratio,
                            "identity_exact": bool(np.array_equal(native, source["image"])) if angle == 0 else None})
    report = {"status": "PASS", "fingerprint": fingerprint, "examples": results,
              "method": "S_R_inverse_S" if PILL_ROI_SIZE else "image_plane_2d_approximation",
              "physical_roi_correction": "NOT VERIFIED: original ROI dimensions unavailable" if PILL_ROI_SIZE is None else "configured common ROI",
              "visual_quality": "NOT VERIFIED by automation; inspect contact_sheet.png",
              "angle_quantization_degrees": None}
    write_once(directory / "contact_sheet.png", encode_png(contact_sheet(tiles, 9)), True)
    write_once(directory / "validation.json", json_bytes(report), True)
    print(f"[rotation] PASS identity/alpha/canvas; method={report['method']}: {directory}", flush=True)
    return report


def generate_placement_mask(background: dict) -> np.ndarray:
    """Extract the dark neutral/blue floor, fill its convex hull, then erode safely."""
    image = background["image"]
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    if background["id"] in MASK_POLYGONS:
        polygon = np.asarray(MASK_POLYGONS[background["id"]], np.int32)
        require(polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon) >= 3
                and bool(((polygon >= 0) & (polygon < IMAGE_SIZE)).all()), "Invalid floor polygon")
        cv2.fillPoly(mask, [polygon], 255)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blue_minus_green = image[:, :, 0].astype(np.int16) - image[:, :, 1].astype(np.int16)
        dark = ((gray <= DARK_THRESHOLD) & (blue_minus_green >= MIN_BLUE_MINUS_GREEN)).astype(np.uint8)
        closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                                 np.ones((CLOSING_KERNEL_SIZE, CLOSING_KERNEL_SIZE), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
        require(count > 1, f"No floor component: {background['id']}")
        # A floor component must intersect the central lower ROI; isolated outer
        # shadows cannot win solely because they are dark.
        candidates = [i for i in range(1, count)
                      if bool((labels[80:200, 80:145] == i).any())]
        require(bool(candidates), f"No central floor component: {background['id']}")
        component = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        points = cv2.findNonZero((labels == component).astype(np.uint8))
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    if BOUNDARY_MARGIN:
        mask = cv2.erode(mask, np.ones((2 * BOUNDARY_MARGIN + 1,) * 2, np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0)
    fraction = np.count_nonzero(mask) / mask.size
    require(0.2 < fraction < 0.8, f"Suspicious floor area ({fraction:.1%}): {background['id']}")
    return mask > 0


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    overlay[mask] = np.rint(0.65 * image[mask] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
    return overlay


def contact_sheet(tiles: list[tuple[str, np.ndarray]], columns: int) -> np.ndarray:
    """Arrange native-size preview tiles with labels; never transform training pixels."""
    height = IMAGE_SIZE + 30
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full((rows * height, columns * IMAGE_SIZE, 3), 245, np.uint8)
    for i, (label, tile) in enumerate(tiles):
        y, x = (i // columns) * height, (i % columns) * IMAGE_SIZE
        sheet[y + 30:y + height, x:x + IMAGE_SIZE] = tile
        cv2.putText(sheet, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, (20, 20, 20), 1, cv2.LINE_AA)
    return sheet


def generate_mask_previews(backgrounds: list[dict], directory: Path) -> None:
    """Cache each floor mask once; export original, binary, overlay, and sheets."""
    tiles = []
    mask_report = []
    for background in backgrounds:
        stem = Path(background["id"]).stem
        mask_path = directory / f"{stem}_mask.png"
        if mask_path.exists():
            mask_image = read_image(mask_path)
            require(mask_image.shape == (IMAGE_SIZE, IMAGE_SIZE)
                    and set(np.unique(mask_image)).issubset({0, 255}), f"Invalid cached mask: {mask_path}")
            background["mask"] = mask_image > 0
        else:
            background["mask"] = generate_placement_mask(background)
            write_once(mask_path, encode_png(background["mask"].astype(np.uint8) * 255))
        mask = background["mask"]
        overlay = overlay_mask(background["image"], mask)
        binary_bgr = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        write_once(directory / f"{stem}_original.png", encode_png(background["image"]), True)
        write_once(directory / f"{stem}_overlay.png", encode_png(overlay), True)
        short_name = stem.replace("_crop_224", "")
        tiles.extend([(short_name + " original", background["image"]),
                      (short_name + " mask", binary_bgr), (short_name + " overlay", overlay)])
        mask_report.append({"background": background["id"], "area": int(mask.sum()),
                            "mask_sha256": sha256(mask.astype(np.uint8).tobytes()),
                            "manual_polygon": background["id"] in MASK_POLYGONS})
    write_once(directory / "contact_sheet.png", encode_png(contact_sheet(tiles, 6)), True)
    overlays = [(b["id"].replace("_crop_224", ""), overlay_mask(b["image"], b["mask"]))
                for b in backgrounds]
    write_once(directory / "overlay_contact_sheet.png", encode_png(contact_sheet(overlays, 4)), True)
    write_once(directory / "report.json", json_bytes(mask_report), True)
    print(f"[masks] {len(backgrounds)} cached floor masks: {directory}", flush=True)


def candidate_positions(background: dict, asset: dict, cache: dict) -> np.ndarray:
    """Cache every integer top-left position whose entire alpha footprint fits."""
    key = (background["id"], asset["id"], asset.get("angle", 0.0))
    if key not in cache:
        # anchor=(0,0) makes output[y,x] test input[y+dy,x+dx] for each
        # nonzero kernel pixel. The zero border forbids partial image clipping.
        valid = cv2.erode(background["mask"].astype(np.uint8), asset["mask"].astype(np.uint8),
                          anchor=(0, 0), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        h, w = asset["height"], asset["width"]
        cache[key] = (np.argwhere(valid[:IMAGE_SIZE-h+1, :IMAGE_SIZE-w+1] > 0).astype(np.int16)
                      if h <= IMAGE_SIZE and w <= IMAGE_SIZE else np.empty((0, 2), np.int16))
    return cache[key]


def is_valid_placement(placement_mask: np.ndarray, occupancy_mask: np.ndarray,
                       pill_mask: np.ndarray, x: int, y: int) -> bool:
    """Check full-image bounds, every alpha pixel, and pixel-level occupancy."""
    h, w = pill_mask.shape
    if x < 0 or y < 0 or x + w > placement_mask.shape[1] or y + h > placement_mask.shape[0]:
        return False
    allowed = placement_mask[y:y+h, x:x+w]
    occupied = occupancy_mask[y:y+h, x:x+w]
    return not bool((pill_mask & ~allowed).any() or (pill_mask & occupied).any())


def update_occupancy_mask(occupancy: np.ndarray, pill_mask: np.ndarray, x: int, y: int) -> None:
    """Union actual alpha pixels; compute the gap separately without repeated dilation."""
    h, w = pill_mask.shape
    occupancy[y:y+h, x:x+w] |= pill_mask


def alpha_blend(image: np.ndarray, asset: dict, x: int, y: int) -> None:
    """Blend native or transformed BGRA into BGR, retaining every nonzero alpha."""
    h, w = asset["height"], asset["width"]
    roi = image[y:y+h, x:x+w]
    blended = asset["foreground"] * asset["alpha"] + roi.astype(np.float32) * (1.0 - asset["alpha"])
    roi[:] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def sample_scene_augmentation(rng: np.random.Generator) -> dict:
    """Draw count/type-independent whole-scene effects, once per saved scene."""
    scene = {"quarter_turns": 0, "horizontal_flip": False, "lighting": None, "shadow": None}
    if SCENE_ROTATION_PROB > 0 and rng.random() < SCENE_ROTATION_PROB:
        scene["quarter_turns"] = int(rng.integers(1, 4))
    if SCENE_HORIZONTAL_FLIP_PROB > 0:
        scene["horizontal_flip"] = bool(rng.random() < SCENE_HORIZONTAL_FLIP_PROB)
    if SCENE_LIGHTING_PROB > 0 and rng.random() < SCENE_LIGHTING_PROB:
        scene["lighting"] = {"angle": float(rng.uniform(0, 2 * math.pi)),
                             "strength": float(rng.uniform(*LIGHTING_GRADIENT_RANGE))}
    if SCENE_SHADOW_PROB > 0 and rng.random() < SCENE_SHADOW_PROB:
        scene["shadow"] = {"angle": float(rng.uniform(0, 2 * math.pi)),
                           "cx": float(rng.uniform(-1, 1)), "cy": float(rng.uniform(-1, 1)),
                           "rx": float(rng.uniform(*SHADOW_RADIUS_RANGE)),
                           "ry": float(rng.uniform(*SHADOW_RADIUS_RANGE)),
                           "strength": float(rng.uniform(*SHADOW_STRENGTH_RANGE))}
    return scene


def validate_scene_parameters(scene: dict) -> None:
    require(type(scene["quarter_turns"]) is int and 0 <= scene["quarter_turns"] <= 3,
            "Scene rotation must be an integer quarter-turn")
    require(type(scene["horizontal_flip"]) is bool, "Scene flip must be boolean")
    for name in ("lighting", "shadow"):
        params = scene[name]
        if params is None:
            continue
        require(all(type(v) in (int, float) and math.isfinite(v) for v in params.values()), "Nonfinite scene parameter")
        require(0 <= params["angle"] <= 2 * math.pi, "Invalid scene light/shadow angle")
        limits = LIGHTING_GRADIENT_RANGE if name == "lighting" else SHADOW_STRENGTH_RANGE
        require(limits[0] <= params["strength"] <= limits[1], "Invalid scene strength")
        if name == "shadow":
            require(-1 <= params["cx"] <= 1 and -1 <= params["cy"] <= 1
                    and all(SHADOW_RADIUS_RANGE[0] <= params[key] <= SHADOW_RADIUS_RANGE[1] for key in ("rx", "ry")),
                    "Invalid soft shadow geometry")


def apply_scene_geometry(array: np.ndarray, scene: dict) -> np.ndarray:
    """Permute image or mask pixels exactly; positive quarter-turns are clockwise."""
    result = np.rot90(array, -scene["quarter_turns"])
    if scene["horizontal_flip"]:
        result = result[:, ::-1]
    return np.ascontiguousarray(result)


def scene_gain_map(scene: dict, brightness: float) -> np.ndarray:
    """Return a smooth bounded exposure field, never an opaque occluding object."""
    gain = np.ones((IMAGE_SIZE, IMAGE_SIZE), np.float32)
    light = scene["lighting"]
    if light is not None:
        c, s = math.cos(light["angle"]), math.sin(light["angle"])
        projection = (c * SCENE_GRID_X + s * SCENE_GRID_Y) / (abs(c) + abs(s))
        gain *= 1.0 + light["strength"] * projection
    shadow = scene["shadow"]
    if shadow is not None:
        c, s = math.cos(shadow["angle"]), math.sin(shadow["angle"])
        dx, dy = SCENE_GRID_X - shadow["cx"], SCENE_GRID_Y - shadow["cy"]
        radius_squared = ((c * dx + s * dy) / shadow["rx"]) ** 2 + ((-s * dx + c * dy) / shadow["ry"]) ** 2
        gain *= 1.0 - shadow["strength"] * np.exp(-0.5 * radius_squared)
    return np.clip(gain * brightness, *SCENE_GAIN_LIMITS)


def apply_augmentation(image: np.ndarray, brightness: float, contrast: float,
                       scene: Optional[dict] = None) -> np.ndarray:
    """Apply whole-scene contrast and exposure once, clipping only the final RGB."""
    gray_mean = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
    adjusted = (image.astype(np.float32) - gray_mean) * contrast + gray_mean
    if scene is None or (scene["lighting"] is None and scene["shadow"] is None):
        adjusted *= brightness
    else:
        adjusted *= scene_gain_map(scene, brightness)[:, :, None]
    return np.clip(np.rint(adjusted), 0, 255).astype(np.uint8)


def find_contact_pairs(masks: list[np.ndarray]) -> list[list[int]]:
    """Count instance adjacency, not connected components, using every alpha>0 pixel."""
    pairs = []
    for i, mask in enumerate(masks):
        band = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        for j in range(i + 1, len(masks)):
            if bool((band & masks[j]).any()):
                pairs.append([i, j])
    return pairs


def replay_record(record: dict, backgrounds: dict, assets: dict, audit: Optional[dict] = None) -> np.ndarray:
    """Audit geometry/labels from metadata and independently reconstruct pixels."""
    count = record["count"]
    pill_type = record["pill_type"]
    require(type(count) is int and 0 <= count <= MAX_PILL_COUNT, "Invalid count label")
    require(len(record["placements"]) == count, "Label / placement count mismatch")
    require((count == 0 and pill_type is None) or (count > 0 and pill_type in PILL_TYPES), "Invalid pill type")
    brightness, contrast = record["brightness"], record["contrast"]
    require(BRIGHTNESS_RANGE[0] <= brightness <= BRIGHTNESS_RANGE[1]
            and CONTRAST_RANGE[0] <= contrast <= CONTRAST_RANGE[1], "Invalid augmentation")
    background = backgrounds[record["background"]]
    image = background["image"].copy()
    occupancy = np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool)
    instance_masks = []
    contact_mode = record.get("contact_requested")
    require(contact_mode is None or type(contact_mode) is bool, "Invalid contact mode")
    require(not contact_mode or count >= 2, "Contact requires at least two pills")
    # Dilate the union of placed pixels once per check, avoiding cumulative
    # dilation of older pills when a positive gap is configured.
    for placement in record["placements"]:
        asset = assets[placement["asset"]]
        require(asset["type"] == pill_type, "Mixed pill types in one image")
        if ENABLE_PILL_ROTATION:
            angle = placement["angle"]
            require(math.isfinite(angle) and PILL_ROTATION_RANGE[0] <= angle <= PILL_ROTATION_RANGE[1],
                    "Missing or out-of-range rotation angle")
            asset = rotate_pill_asset(asset, angle)
            require(placement["transformed_size"] == [asset["width"], asset["height"]]
                    and placement["transform_sha256"] == sha256(asset["image"].tobytes()),
                    "Reconstructed affine asset differs from placement metadata")
        else:
            require(placement.get("angle", 0.0) == 0.0, "Nonzero rotation when rotation disabled")
        x, y = placement["x"], placement["y"]
        require(type(x) is int and type(y) is int, "Noninteger placement coordinate")
        forbidden = exclusion_mask(occupancy)
        require(is_valid_placement(background["mask"], forbidden, asset["mask"], x, y),
                f"Invalid alpha boundary / overlap / gap: {record.get('file')}")
        update_occupancy_mask(occupancy, asset["mask"], x, y)
        full_mask = np.zeros_like(occupancy)
        full_mask[y:y+asset["height"], x:x+asset["width"]] = asset["mask"]
        instance_masks.append(full_mask)
        alpha_blend(image, asset, x, y)
    contacts = find_contact_pairs(instance_masks)
    if contact_mode is not None:
        require(bool(contacts) == contact_mode, "Saved layout does not match requested contact mode")
    scene = record.get("scene", {"quarter_turns": 0, "horizontal_flip": False, "lighting": None, "shadow": None})
    validate_scene_parameters(scene)
    image = apply_scene_geometry(image, scene)
    allowed = apply_scene_geometry(background["mask"], scene)
    transformed_masks = [apply_scene_geometry(mask, scene) for mask in instance_masks]
    transformed_occupancy = np.zeros_like(occupancy)
    for mask in transformed_masks:
        require(not bool((mask & ~allowed).any()) and not bool((mask & transformed_occupancy).any()),
                "Scene transform changed boundary or collision validity")
        transformed_occupancy |= mask
    require(find_contact_pairs(transformed_masks) == contacts, "Scene transform changed contact relationships")
    result = apply_augmentation(image, brightness, contrast, scene)
    gain = scene_gain_map(scene, brightness)
    render_audit = {"contact_pairs": contacts, "gain_min": float(gain.min()), "gain_max": float(gain.max()),
                    "clipped_channel_fraction": float(((result == 0) | (result == 255)).mean())}
    if "render_audit" in record:
        require(record["render_audit"] == render_audit, "Saved render audit differs from replay")
    if audit is not None:
        audit.update(render_audit)
    return result


def exclusion_mask(occupancy: np.ndarray, gap: Optional[int] = None) -> np.ndarray:
    gap = MIN_PILL_GAP if gap is None else gap
    if not gap:
        return occupancy
    return cv2.dilate(occupancy.astype(np.uint8),
                      np.ones((2 * gap + 1,) * 2, np.uint8)) > 0


def place_rotated_assets(background: dict, selected: list[dict], rng: np.random.Generator,
                         cache: dict, stats: Counter, contact_mode: Optional[bool] = None) -> Optional[list[dict]]:
    """Search exact free alpha positions with bounded backtracking of locations.

    Assets and angles stay fixed. Each pill has at most 200 coordinate attempts
    across the entire layout, and at most 20 locations are undone per layout.
    This avoids both random rejection of tiny valid spaces and greedy dead ends.
    """
    attempts = [0] * len(selected)
    remaining_backtracks = [MAX_LAYOUT_BACKTRACKS]

    def search(depth: int, occupancy: np.ndarray) -> Optional[list[dict]]:
        if depth == len(selected):
            return []
        asset = selected[depth]
        forbidden = exclusion_mask(occupancy, max(MIN_PILL_GAP, 1) if contact_mode is False else MIN_PILL_GAP)
        if depth == 0:
            positions = candidate_positions(background, asset, cache)
        else:
            # A fresh local cache is essential: the free mask changes whenever
            # a preceding position is undone. Boundary masks remain unchanged.
            free_background = {"id": background["id"], "mask": background["mask"] & ~forbidden}
            positions = candidate_positions(free_background, asset, {})
            stats["free_position_searches"] += 1
        if contact_mode and depth == 1 and len(positions):
            # Require an edge-adjacent second instance, without sharing any alpha
            # pixel. The remaining pills may form additional contact pairs.
            band = cv2.dilate(occupancy.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
            touching = cv2.dilate(band, asset["mask"].astype(np.uint8), anchor=(0, 0),
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
            positions = positions[touching[positions[:, 0], positions[:, 1]] > 0]
            stats["contact_position_searches"] += 1
        if len(positions) == 0:
            stats["no_boundary_candidate" if depth == 0 else "occupancy_dead_ends"] += 1
            return None
        while attempts[depth] < MAX_PLACEMENT_ATTEMPTS:
            attempts[depth] += 1
            stats["position_attempts"] += 1
            y, x = map(int, positions[int(rng.integers(len(positions)))])
            require(is_valid_placement(background["mask"], forbidden, asset["mask"], x, y),
                    "Free-position search disagrees with direct alpha collision check")
            next_occupancy = occupancy.copy()
            update_occupancy_mask(next_occupancy, asset["mask"], x, y)
            tail = search(depth + 1, next_occupancy)
            if tail is not None:
                return [{"asset": asset["id"], "x": x, "y": y, "angle": asset.get("angle", 0.0),
                         "transformed_size": [asset["width"], asset["height"]],
                         "transform_sha256": sha256(asset["image"].tobytes())}] + tail
            if remaining_backtracks[0] <= 0:
                return None
            remaining_backtracks[0] -= 1
            stats["backtracks"] += 1
        stats["pill_attempt_exhaustions"] += 1
        return None

    return search(0, np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool))


def generate_single_image(background: dict, pool: list[dict], count: int,
                          rng: np.random.Generator, cache: dict,
                          contact_mode: Optional[bool] = None) -> tuple[Optional[dict], Counter]:
    """Sample one type with replacement; restart bounded layouts, never relax constraints."""
    stats = Counter()
    selected = [pool[int(rng.integers(len(pool)))] for _ in range(count)]
    if ENABLE_PILL_ROTATION:
        start = time.perf_counter()
        selected = [rotate_pill_asset(asset, float(rng.uniform(*PILL_ROTATION_RANGE))) for asset in selected]
        stats["rotation_transform_seconds"] = time.perf_counter() - start
        stats["rotation_transforms"] = len(selected)
    # Continuous angles have no reusable cross-image cache; bound memory by
    # retaining candidates only for this image, including its layout retries.
    position_cache = {} if ENABLE_PILL_ROTATION else cache
    selected.sort(key=lambda asset: asset["area"], reverse=True)
    # Background and sampled assets stay fixed during all retries: difficult
    # backgrounds or large assets are never silently replaced by easier ones.
    for _ in range(MAX_IMAGE_RETRIES):
        stats["layout_attempts"] += 1
        occupancy = np.zeros((IMAGE_SIZE, IMAGE_SIZE), bool)
        placements = []
        use_search = ENABLE_PILL_ROTATION or contact_mode is not None
        if use_search:
            result = place_rotated_assets(background, selected, rng, position_cache, stats, contact_mode)
            placements = result if result is not None else []
        for asset in (() if use_search else selected):
            positions = candidate_positions(background, asset, position_cache)
            if len(positions) == 0:
                stats["no_boundary_candidate"] += 1
                break
            forbidden = exclusion_mask(occupancy)
            placed = False
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                stats["position_attempts"] += 1
                y, x = map(int, positions[int(rng.integers(len(positions)))])
                if not is_valid_placement(background["mask"], forbidden, asset["mask"], x, y):
                    stats["collision_rejections"] += 1
                    continue
                update_occupancy_mask(occupancy, asset["mask"], x, y)
                placement = {"asset": asset["id"], "x": x, "y": y}
                if ENABLE_PILL_ROTATION:
                    placement.update({"angle": asset["angle"],
                                      "transformed_size": [asset["width"], asset["height"]],
                                      "transform_sha256": sha256(asset["image"].tobytes())})
                placements.append(placement)
                placed = True
                break
            if not placed:
                stats["pill_attempt_exhaustions"] += 1
                break
        if len(placements) == count:
            stats["images_succeeded"] += 1
            return {"background": background["id"], "count": count,
                    "pill_type": pool[0]["type"] if count else None,
                    "placements": placements,
                    "brightness": float(rng.uniform(*BRIGHTNESS_RANGE)),
                    "contrast": float(rng.uniform(*CONTRAST_RANGE)),
                    "contact_requested": contact_mode, "scene": sample_scene_augmentation(rng)}, stats
        stats["failed_layouts"] += 1
        if stats["no_boundary_candidate"]:
            break
    stats["images_failed"] += 1
    return None, stats


def check_placement_feasibility(backgrounds: list[dict], pools: dict, cache: dict,
                                directory: Path, fingerprint: str) -> dict:
    """Exercise all 64 background/type pairs in both contact modes, 20 trials each."""
    report_path = directory / "feasibility.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(report["fingerprint"] == fingerprint, "Stale feasibility report")
        return report
    rng = np.random.default_rng(SEED + 1)
    combinations = []
    witness_tiles = []
    bg_index = {b["id"]: b for b in backgrounds}
    asset_index = {a["id"]: a for pool in pools.values() for a in pool}
    for background in backgrounds:
        for pill_type, contact_mode in [(t, mode) for t in PILL_TYPES for mode in (False, True)]:
            pool = pools[pill_type]
            stats = Counter()
            successes = 0
            first = None
            for _ in range(FEASIBILITY_TRIALS):
                record, trial_stats = generate_single_image(background, pool, MAX_PILL_COUNT, rng, cache, contact_mode)
                stats.update(trial_stats)
                if record is not None:
                    replay_record(record, bg_index, asset_index)
                    successes += 1
                    if first is None:
                        first = record
            candidates = {a["id"]: len(candidate_positions(background, a, cache)) for a in pool}
            smallest_area = min(a["area"] for a in pool) * MAX_PILL_COUNT
            reason = "none"
            if successes < FEASIBILITY_TRIALS:
                reason = ("proven area shortage" if smallest_area > int(background["mask"].sum()) else
                          "sampled footprint has no boundary-valid position" if stats["no_boundary_candidate"] else
                          "bounded random packing exhausted; geometric impossibility NOT proven")
            row = {"background": background["id"], "pill_type": pill_type, "contact_requested": contact_mode,
                   "trials": FEASIBILITY_TRIALS, "successes": successes,
                   "success_rate": successes / FEASIBILITY_TRIALS,
                   "mean_layout_attempts": stats["layout_attempts"] / FEASIBILITY_TRIALS,
                   "mean_position_attempts": stats["position_attempts"] / FEASIBILITY_TRIALS,
                   "failure_reason": reason, "stats": dict(stats),
                   "candidate_counts": candidates, "witness": first}
            combinations.append(row)
            if first:
                witness_tiles.append((background["id"].replace("_crop_224", "") + f" {pill_type}:7 {'touch' if contact_mode else 'apart'}",
                                      replay_record(first, bg_index, asset_index)))
            print(f"[feasibility] {background['id']} {pill_type} contact={contact_mode}: {successes}/{FEASIBILITY_TRIALS}, "
                  f"mean layouts={row['mean_layout_attempts']:.2f}, "
                  f"positions={row['mean_position_attempts']:.1f}", flush=True)
    passed = len(combinations) == EXPECTED_BACKGROUNDS * len(PILL_TYPES) * 2 and all(
        row["successes"] == FEASIBILITY_TRIALS and min(row["candidate_counts"].values()) > 0
        for row in combinations)
    report = {"fingerprint": fingerprint, "status": "PASS" if passed else "FAIL",
              "combinations": combinations,
              "limitation": "Random trials demonstrate witnesses, not exhaustive packability of every asset multiset."}
    write_once(report_path, json_bytes(report))
    if witness_tiles:
        write_once(directory / "feasibility_contact_sheet.png", encode_png(contact_sheet(witness_tiles, 8)))
    return report


def generate_dataset(backgrounds: list[dict], pools: dict, cache: dict, output: Path,
                     evidence: Path, samples: int, empty: int, fingerprint: str,
                     seed: int) -> None:
    """Stream PNGs and placement JSONL; any exhausted image makes the run incomplete."""
    require(not output.exists() or (output.is_dir() and not output.is_symlink() and not any(output.iterdir())),
            f"Dataset is not empty: {output}. Use --mode validate or a new path.")
    require(not evidence.exists(), f"Run evidence already exists: {evidence}. Use a new --debug-dir.")
    require(output != evidence and output not in evidence.parents and evidence not in output.parents,
            "Dataset and evidence must be disjoint")
    generation_start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=False)
    for count in range(MAX_PILL_COUNT + 1):
        (output / f"class{count}").mkdir()
    contact_samples = int(math.floor(samples * CONTACT_IMAGE_FRACTION + 0.5))
    spec = {"fingerprint": fingerprint, "output": str(output), "samples_per_type": samples,
            "contact_samples_per_type": contact_samples,
            "empty_samples": empty, "seed": seed, "status": "INCOMPLETE until validation.json is PASS"}
    write_once(evidence / "run.json", json_bytes(spec))
    rng = np.random.default_rng(seed)
    contact_rng = np.random.default_rng(seed + 1000)
    stats = Counter()
    bg_index = {b["id"]: b for b in backgrounds}
    asset_index = {a["id"]: a for pool in pools.values() for a in pool}
    total = empty + MAX_PILL_COUNT * len(PILL_TYPES) * samples
    written = 0
    with (evidence / "placements.jsonl").open("x", encoding="utf-8") as manifest:
        for count in range(MAX_PILL_COUNT + 1):
            for pill_type in (PILL_TYPES if count else (None,)):
                amount = samples if count else empty
                contact_flags = np.zeros(amount, bool)
                if count >= 2 and contact_samples:
                    contact_flags[contact_rng.choice(amount, contact_samples, replace=False)] = True
                for index in range(1, (samples if count else empty) + 1):
                    background = backgrounds[int(rng.integers(len(backgrounds)))]
                    record, image_stats = generate_single_image(
                        background, pools[pill_type] if count else [], count, rng, cache, bool(contact_flags[index - 1]))
                    stats.update(image_stats)
                    if record is None:
                        write_once(evidence / "failure.json", json_bytes({"class": count, "type": pill_type,
                                   "index": index, "background": background["id"], "stats": dict(stats)}))
                        raise RuntimeError(f"Placement exhausted for class{count}/{pill_type}, "
                                           f"{background['id']}; retained incomplete run, no partial label saved.")
                    name = f"{pill_type or 'empty'}_{index:06d}.png"
                    record["file"] = f"class{count}/{name}"
                    # This independent audit also renders the image. It checks
                    # actual sampled native masks, never just the requested count.
                    render_audit = {}
                    image = replay_record(record, bg_index, asset_index, render_audit)
                    record["render_audit"] = render_audit
                    data = encode_png(image)
                    record["png_sha256"] = sha256(data)
                    write_once(output / record["file"], data)
                    manifest.write(json.dumps(record, sort_keys=True) + "\n")
                    manifest.flush()
                    written += 1
                    if written % LOG_EVERY == 0 or written == total:
                        print(f"[generate] {written}/{total}, failed layouts={stats['failed_layouts']}, "
                              f"collision rejections={stats['collision_rejections']}", flush=True)
    write_once(evidence / "stats.json", json_bytes(dict(stats)))
    write_once(evidence / "timing.json", json_bytes({"generation_seconds": time.perf_counter() - generation_start,
               "note": "Includes sampled transforms, placement, independent render audit, PNG encoding/writes; excludes final validation."}))


def rotation_angle_statistics(groups: dict[str, list[float]]) -> dict:
    """Summarize only angles of pills actually saved, by type/background/pair."""
    summary = {}
    for name, values in sorted(groups.items()):
        angles = np.asarray(values, np.float64) % 360.0
        histogram, edges = np.histogram(angles, bins=36, range=(0, 360))
        quadrants, _ = np.histogram(angles, bins=4, range=(0, 360))
        radians = np.deg2rad(angles)
        summary[name] = {"count": len(values), "histogram_10_degrees": histogram.tolist(),
                         "bin_edges_degrees": edges.tolist(), "quadrant_counts": quadrants.tolist(),
                         "min_angle": float(angles.min()), "max_angle": float(angles.max()),
                         "circular_resultant_length": float(abs(np.exp(1j * radians).mean())),
                         "max_quadrant_deviation_fraction": float(np.max(np.abs(quadrants / len(values) - 0.25)))}
    return {"enabled": ENABLE_PILL_ROTATION, "angle_interval_degrees": None,
            "sampling": "Independent continuous uniform draws, fixed through all position/layout retries",
            "population": "Successfully saved placements only", "groups": summary}


def validate_dataset(backgrounds: list[dict], pools: dict, output: Path, evidence: Path,
                     fingerprint: str) -> dict:
    """Decode and replay EVERY saved image, audit balance and exact ImageFolder layout."""
    spec = json.loads((evidence / "run.json").read_text(encoding="utf-8"))
    require(spec["fingerprint"] == fingerprint, "Input/code/config changed since generation")
    if Path(spec["output"]) != output:
        publication = json.loads((evidence / "publication.json").read_text(encoding="utf-8"))
        require(publication["fingerprint"] == fingerprint and Path(publication["output"]) == output
                and publication["staging_output"] == spec["output"], "Run evidence belongs to another dataset")
    samples, empty = spec["samples_per_type"], spec["empty_samples"]
    require(empty == samples * len(PILL_TYPES), "Class0 does not balance the other classes")
    expected = {f"class{c}/{t}_{i:06d}.png"
                for c in range(MAX_PILL_COUNT + 1)
                for t in (PILL_TYPES if c else ("empty",))
                for i in range(1, (samples if c else empty) + 1)}
    preservation_path = evidence / "preservation.json"
    preservation = json.loads(preservation_path.read_text()) if preservation_path.exists() else {"files": {}, "directories": []}
    for relative, digest in preservation["files"].items():
        path = output / relative
        require(path.is_file() and not path.is_symlink() and sha256(path.read_bytes()) == digest,
                f"Preserved non-target file changed: {path}")
    allowed_roots = set(TARGET_CLASSES) | {Path(p).parts[0] for p in preservation["files"]} | {
        Path(p).parts[0] for p in preservation["directories"]}
    require(set(p.name for p in output.iterdir() if p.name not in {".DS_Store", ".gitkeep"}) <= allowed_roots,
            "Unexpected content outside target classes and registered preserved data")
    actual = set()
    for class_dir in (output / name for name in TARGET_CLASSES):
        require(class_dir.is_dir() and not class_dir.is_symlink(), f"Invalid class directory: {class_dir}")
        for path in class_dir.iterdir():
            if path.relative_to(output).as_posix() in preservation["files"]:
                continue
            if path.name in {".DS_Store", ".gitkeep"} and path.is_file() and not path.is_symlink():
                continue
            require(path.is_file() and not path.is_symlink() and path.suffix == ".png",
                    f"Unexpected nested directory or non-PNG: {path}")
            actual.add(path.relative_to(output).as_posix())
    require(actual == expected, f"File inventory mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}")
    all_files = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
                 and p.name not in {".DS_Store", ".gitkeep"}}
    registered = expected | set(preservation["files"])
    require(all_files <= registered and not any(p.is_symlink() for p in output.rglob("*")),
            "Unregistered file or symlink in final dataset")
    bg_index = {b["id"]: b for b in backgrounds}
    asset_index = {a["id"]: a for pool in pools.values() for a in pool}
    seen = set()
    counts = Counter()
    types = Counter()
    backgrounds_used = Counter()
    angle_groups: dict[str, list[float]] = {}
    contact_counts = Counter()
    effect_counts = Counter()
    effects_by_class = {name: Counter() for name in TARGET_CLASSES}
    gain_extrema = [float("inf"), float("-inf")]
    with (evidence / "placements.jsonl").open(encoding="utf-8") as manifest:
        for line in manifest:
            record = json.loads(line)
            name = record["file"]
            require(name in expected and name not in seen, f"Unknown or duplicate manifest path: {name}")
            seen.add(name)
            count, pill_type = record["count"], record["pill_type"]
            require(Path(name).parent.name == f"class{count}" and
                    Path(name).name.startswith(f"{pill_type or 'empty'}_"), "Filename / label mismatch")
            path = output / name
            data = path.read_bytes()
            require(data[:8] == b"\x89PNG\r\n\x1a\n" and sha256(data) == record["png_sha256"],
                    f"Saved PNG changed or invalid format: {path}")
            # PNG IHDR colour type 2 means actual three-channel truecolour RGB,
            # not a palette/grayscale PNG that OpenCV could silently expand.
            require(data[25] == 2 and data[24] == 8, f"Expected 8-bit RGB PNG: {path}")
            image = read_image(path)
            require(image.shape == (IMAGE_SIZE, IMAGE_SIZE, 3) and image.dtype == np.uint8,
                    f"Invalid output shape/channels/depth: {path}")
            replay = replay_record(record, bg_index, asset_index)
            require(np.array_equal(image, replay), f"Placement replay differs from saved pixels: {path}")
            counts[f"class{count}"] += 1
            types[f"class{count}/{pill_type or 'empty'}"] += 1
            backgrounds_used[record["background"]] += 1
            contacts = record["render_audit"]["contact_pairs"]
            if contacts:
                contact_counts[f"class{count}/{pill_type}"] += 1
            scene = record["scene"]
            effect_counts[f"rotation_{scene['quarter_turns'] * 90}"] += 1
            effect_counts[f"geometry_r{scene['quarter_turns']}_h{int(scene['horizontal_flip'])}"] += 1
            for key, present in (("horizontal_flip", scene["horizontal_flip"]),
                                 ("lighting", scene["lighting"] is not None), ("shadow", scene["shadow"] is not None),
                                 ("contact", bool(contacts))):
                effect_counts[key] += int(present)
                effects_by_class[f"class{count}"][key] += int(present)
            gain_extrema[0] = min(gain_extrema[0], record["render_audit"]["gain_min"])
            gain_extrema[1] = max(gain_extrema[1], record["render_audit"]["gain_max"])
            if ENABLE_PILL_ROTATION:
                for placement in record["placements"]:
                    for group in ("overall", f"type/{pill_type}", f"background/{record['background']}",
                                  f"pair/{record['background']}/{pill_type}"):
                        angle_groups.setdefault(group, []).append(placement["angle"])
            if len(seen) % 1000 == 0:
                print(f"[validate] replayed {len(seen)}/{len(expected)}", flush=True)
    require(seen == expected, "Placement manifest does not cover all saved images")
    require(all(counts[f"class{c}"] == empty for c in range(MAX_PILL_COUNT + 1)), "Class imbalance")
    require(all(types[f"class{c}/{t}"] == samples for c in range(1, MAX_PILL_COUNT + 1) for t in PILL_TYPES),
            "Pill type imbalance")
    require(all(contact_counts[f"class{c}/{t}"] == spec["contact_samples_per_type"]
                for c in range(2, MAX_PILL_COUNT + 1) for t in PILL_TYPES), "Contact ratio quota mismatch")
    # Structural compatibility is mandatory. Exercise actual torchvision when
    # installed, without introducing it as a generator dependency.
    loader_check = "NOT VERIFIED (torchvision unavailable); exact ImageFolder structure PASS"
    try:
        from torchvision.datasets import ImageFolder
    except (ImportError, RuntimeError, OSError) as exc:
        loader_check += f"; {type(exc).__name__}: {exc}"
    else:
        class CountImageFolder(ImageFolder):
            def find_classes(self, directory):
                return list(TARGET_CLASSES), {name: i for i, name in enumerate(TARGET_CLASSES)}

        dataset = CountImageFolder(str(output))
        require(dataset.class_to_idx == {f"class{c}": c for c in range(MAX_PILL_COUNT + 1)}
                and len(dataset) == len(expected), "ImageFolder label map/count mismatch")
        for c in range(MAX_PILL_COUNT + 1):
            index = next(i for i, (_, label) in enumerate(dataset.samples) if label == c)
            image, label = dataset[index]
            require(image.mode == "RGB" and image.size == (224, 224) and label == c,
                    "ImageFolder image decode mismatch")
        loader_check = "PASS: torchvision ImageFolder with explicit class0..class7 selection, class map, total, RGB decode/class"
    report = {"fingerprint": fingerprint, "status": "PASS", "total": len(seen),
              "class_counts": dict(sorted(counts.items())), "type_counts": dict(sorted(types.items())),
              "background_counts": dict(sorted(backgrounds_used.items())), "imagefolder": loader_check,
              "contact_counts": dict(sorted(contact_counts.items())), "contact_samples_per_type": spec["contact_samples_per_type"],
              "scene_effect_counts": dict(sorted(effect_counts.items())),
              "scene_effects_by_class": {name: dict(counts) for name, counts in effects_by_class.items()},
              "observed_gain_range": gain_extrema, "preserved_files": len(preservation["files"]),
              "checks": {key: "PASS" for key in ["exact_class_directories", "class_balance", "type_balance",
                         "png_decode", "rgb_224x224", "unique_relative_paths", "single_type_per_image",
                         "count_matches_label", "full_alpha_boundary", "alpha_nonoverlap", "minimum_gap",
                         "no_arbitrary_scaling", "saved_pixels_match_metadata_replay", "transformed_alpha_validity",
                         "controlled_contact_quota", "scene_mask_geometry", "scene_parameter_ranges", "preserved_non_target_data"]},
              "rotation_method": "S_R_inverse_S" if PILL_ROI_SIZE else "image_plane_2d_approximation",
              "visual_quality": "NOT VERIFIED by automation; inspect saved contact sheets",
              "filename_policy": "Indices repeat across class directories as specified; relative paths are unique."}
    write_once(evidence / "validation.json", json_bytes(report), True)
    write_once(evidence / "rotation_statistics.json", json_bytes(rotation_angle_statistics(angle_groups)), True)
    print(f"[validate] PASS {len(seen)} images; {loader_check}", flush=True)
    return report


def verify_generated_dataset_for_replacement(output: Path, debug: Path) -> dict:
    """Prove ownership using a complete prior manifest and exact current PNG hashes.

    Unknown files, edited images, symlinks and real capture data are refused.
    Finder metadata stays in the backup; .gitkeep markers are also carried to
    the new dataset unchanged, so replacing images never removes tracked files.
    """
    require(output.is_dir(), "Replacement requires an existing generated dataset")
    require({p.name for p in output.iterdir()} <= {f"class{c}" for c in range(8)} | {".DS_Store", ".gitkeep"},
            "Unknown files/directories in existing dataset; replacement refused")
    actual = set()
    for c in range(MAX_PILL_COUNT + 1):
        directory = output / f"class{c}"
        require(directory.is_dir() and not directory.is_symlink(), f"Invalid existing class: {directory}")
        for path in directory.iterdir():
            require(path.is_file() and not path.is_symlink(), f"Unknown nested content: {path}")
            if path.name in {".DS_Store", ".gitkeep"}:
                continue
            require(path.suffix == ".png", f"Unknown data in existing dataset: {path}")
            actual.add(path.relative_to(output).as_posix())
    candidates = list(debug.glob("*/full/run.json")) + list(debug.glob("*/run.json"))
    candidates.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for run_path in candidates:
        evidence = run_path.parent
        if not (evidence / "validation.json").exists() or not (evidence / "placements.jsonl").exists():
            continue
        spec = json.loads(run_path.read_text(encoding="utf-8"))
        publication_path = evidence / "publication.json"
        published = json.loads(publication_path.read_text(encoding="utf-8")) if publication_path.exists() else {}
        if str(output) not in (spec.get("output"), published.get("output")):
            continue
        validation = json.loads((evidence / "validation.json").read_text(encoding="utf-8"))
        if validation.get("status") != "PASS" or validation.get("fingerprint") != spec.get("fingerprint"):
            continue
        entries = [json.loads(line) for line in (evidence / "placements.jsonl").read_text(encoding="utf-8").splitlines()]
        names = {record["file"] for record in entries}
        if names != actual or len(entries) != len(names) or len(entries) != validation.get("total"):
            continue
        # Validate membership before joining paths; no manifest path can escape output.
        if all(sha256((output / r["file"]).read_bytes()) == r["png_sha256"] for r in entries):
            markers = {}
            for directory in [output] + [output / f"class{c}" for c in range(MAX_PILL_COUNT + 1)]:
                marker = directory / ".gitkeep"
                if marker.exists():
                    require(marker.is_file() and not marker.is_symlink(), f"Invalid tracked marker: {marker}")
                    markers[marker.relative_to(output).as_posix()] = sha256(marker.read_bytes())
            return {"evidence": str(evidence), "fingerprint": spec["fingerprint"], "images": len(entries),
                    "preserved_markers": markers,
                    "status": "PASS", "verification": "Every current PNG matches a prior generated manifest"}
    raise ValueError("No complete matching generation manifest; existing dataset is protected")


def snapshot_count_dataset(output: Path) -> dict:
    """Snapshot an explicitly authorized target, without inferring image provenance.

    Every existing file is retained in the full backup. Only images in the eight
    named count classes are replaced; other classes and nonimage files survive.
    """
    require(output.is_dir() and not output.is_symlink(), "Count-class replacement requires a real directory")
    require(all((output / name).is_dir() for name in TARGET_CLASSES), "Missing authorized count-class directories")
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    files, preserved, directories = {}, {}, []
    for path in sorted(output.rglob("*")):
        require(not path.is_symlink(), f"Symlink in replacement target: {path}")
        relative = path.relative_to(output).as_posix()
        if path.is_dir():
            directories.append(relative)
        elif path.is_file():
            digest = sha256(path.read_bytes())
            files[relative] = digest
            if Path(relative).parts[0] not in TARGET_CLASSES or path.suffix.lower() not in image_extensions:
                preserved[relative] = digest
        else:
            raise ValueError(f"Unsupported filesystem entry: {path}")
    return {"mode": "count_classes", "output": str(output), "files": files, "directories": directories,
            "preserved_files": preserved,
            "preserved_directories": [d for d in directories if Path(d).parts[0] not in TARGET_CLASSES],
            "note": "Explicit --replace-count-classes authorization; old image provenance is not inferred."}


def preserve_non_target_content(output: Path, staging: Path, evidence: Path, snapshot: dict) -> None:
    """Copy registered non-target files exactly, before validating/publishing staging."""
    for relative in snapshot["preserved_directories"]:
        (staging / relative).mkdir(parents=True, exist_ok=True)
    for relative, digest in snapshot["preserved_files"].items():
        source, target = output / relative, staging / relative
        require(sha256(source.read_bytes()) == digest, f"Non-target data changed during generation: {source}")
        require(not target.exists(), f"Protected file collides with generated data: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(sha256(target.read_bytes()) == digest, f"Non-target copy mismatch: {target}")
    write_once(evidence / "preservation.json", json_bytes({"files": snapshot["preserved_files"],
               "directories": snapshot["preserved_directories"]}))


def publish_validated_replacement(staging: Path, output: Path, evidence: Path,
                                  fingerprint: str, previous: dict, debug: Path) -> None:
    """Retain the entire old dataset, move the validated new one, roll back on error."""
    backup = evidence.parent / "previous_dataset"
    require(not backup.exists(), f"Backup already exists: {backup}")
    require(os.stat(staging).st_dev == os.stat(output.parent).st_dev, "Staging must be on the same filesystem")
    validation = json.loads((evidence / "validation.json").read_text(encoding="utf-8"))
    require(validation["status"] == "PASS" and validation["fingerprint"] == fingerprint, "Staging is not validated")
    # Recheck just before swapping, so edits made during generation are protected.
    latest = (snapshot_count_dataset(output) if previous.get("mode") == "count_classes" else
              verify_generated_dataset_for_replacement(output, debug))
    require(latest == previous, "Existing dataset changed during regeneration")
    journal = {"fingerprint": fingerprint, "output": str(output), "staging_output": str(staging),
               "previous_dataset": str(backup), "previous_generation": previous,
               "method": "validated staging -> old directory retained -> new directory rename; rollback on failure"}
    write_once(evidence / "publication.json", json_bytes(journal))
    output.rename(backup)
    try:
        staging.rename(output)
    except BaseException:
        backup.rename(output)
        raise
    print(f"[replace] Published {output}; previous dataset retained at {backup}", flush=True)


def generate_dataset_previews(output: Path, evidence: Path,
                              backgrounds: Optional[list[dict]] = None, pools: Optional[dict] = None) -> None:
    tiles = []
    for count in range(MAX_PILL_COUNT + 1):
        for pill_type in (PILL_TYPES if count else ("empty",) * len(PILL_TYPES)):
            # Four distinct empty examples, one per other sheet column.
            index = len(tiles) + 1 if count == 0 else 1
            path = output / f"class{count}/{pill_type}_{index:06d}.png"
            tiles.append((f"class{count} {pill_type} #{index}", read_image(path)))
    write_once(evidence / "contact_sheet.png", encode_png(contact_sheet(tiles, 4)), True)
    # Show every pilot image, so small-sample visual review can cover all 320.
    if evidence.name == "pilot":
        for pill_type in PILL_TYPES:
            tiles = []
            for count in range(1, MAX_PILL_COUNT + 1):
                for path in sorted((output / f"class{count}").glob(f"{pill_type}_*.png")):
                    tiles.append((f"class{count} {path.stem}", read_image(path)))
            write_once(evidence / f"all_{pill_type}.png", encode_png(contact_sheet(tiles, PILOT_NUM_SAMPLES_PER_TYPE)), True)
    chosen = {}
    base_record = None
    with (evidence / "placements.jsonl").open(encoding="utf-8") as manifest:
        for line in manifest:
            record = json.loads(line)
            if record["count"] in (2, 7):
                key = (record["count"], record["pill_type"], record["contact_requested"])
                chosen.setdefault(key, record)
            if base_record is None and record["count"] == 3 and record["pill_type"] == "C":
                base_record = record
    tiles = [(f"c{c} {t} {'touch' if mode else 'apart'}", read_image(output / record["file"]))
             for (c, t, mode), record in sorted(chosen.items())]
    if tiles:
        write_once(evidence / "contact_preview.png", encode_png(contact_sheet(tiles, 4)), True)
    if base_record is not None and backgrounds is not None and pools is not None:
        bg_index = {b["id"]: b for b in backgrounds}
        asset_index = {a["id"]: a for pool in pools.values() for a in pool}
        identity = {"quarter_turns": 0, "horizontal_flip": False, "lighting": None, "shadow": None}
        light = {"angle": 0.0, "strength": LIGHTING_GRADIENT_RANGE[1]}
        shadow = {"angle": math.pi / 4, "cx": 0.0, "cy": 0.0, "rx": 0.5, "ry": 0.5,
                  "strength": SHADOW_STRENGTH_RANGE[1]}
        for key in ("rx", "ry"):
            shadow[key] = float(np.clip(shadow[key], *SHADOW_RADIUS_RANGE))
        variants = [("base", identity), ("scene clockwise 90", {**identity, "quarter_turns": 1}),
                    ("scene clockwise 180", {**identity, "quarter_turns": 2}),
                    ("scene clockwise 270", {**identity, "quarter_turns": 3}),
                    ("scene horizontal flip", {**identity, "horizontal_flip": True}),
                    ("directional lighting", {**identity, "lighting": light}),
                    ("soft local shadow", {**identity, "shadow": shadow}),
                    ("rotation + flip + light/shadow", {**identity, "quarter_turns": 1,
                     "horizontal_flip": True, "lighting": light, "shadow": shadow})]
        tiles, details = [], []
        for label, scene in variants:
            record = {key: value for key, value in base_record.items() if key != "render_audit"}
            record["scene"] = scene
            record["brightness"] = float(np.clip(1.0, *BRIGHTNESS_RANGE))
            record["contrast"] = float(np.clip(1.0, *CONTRAST_RANGE))
            audit = {}
            tiles.append((label, replay_record(record, bg_index, asset_index, audit)))
            details.append({"label": label, "source_scene": record["file"], "scene": scene, "audit": audit})
        write_once(evidence / "scene_effects_preview.png", encode_png(contact_sheet(tiles, 4)), True)
        write_once(evidence / "scene_effects_preview.json", json_bytes(details), True)


def resolve_path(value: Path) -> Path:
    value = value.expanduser()
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("assets", "masks", "rotation", "feasibility", "pilot", "full", "validate"), default="masks")
    parser.add_argument("--target", choices=("pilot", "full"), default="full", help="Dataset audited by validate")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--debug-dir", type=Path, default=DEBUG_OUTPUT_DIR)
    parser.add_argument("--samples-per-type", type=int, default=NUM_SAMPLES_PER_TYPE, help="Full dataset samples per type/count")
    parser.add_argument("--reviewed-previews", action="store_true", help="Assert mask and pilot previews have been visually reviewed")
    parser.add_argument("--replace-generated", action="store_true", help="Stage/validate a replacement; retain the verified old dataset as backup")
    parser.add_argument("--replace-count-classes", action="store_true",
                        help="Explicitly replace only class0..class7; retain full backup and preserve other classes")
    args = parser.parse_args()
    require(args.samples_per_type > 0, "--samples-per-type must be positive")
    require(BOUNDARY_MARGIN >= 0 and MIN_PILL_GAP >= 0 and ALPHA_THRESHOLD == 0,
            "Margins must be nonnegative; alpha threshold must remain zero")
    require(CLOSING_KERNEL_SIZE > 0 and CLOSING_KERNEL_SIZE % 2 == 1, "Closing kernel must be positive and odd")
    require(all(v > 0 for v in (MAX_PLACEMENT_ATTEMPTS, MAX_IMAGE_RETRIES, FEASIBILITY_TRIALS,
                               PILOT_NUM_SAMPLES_PER_TYPE)), "Retry/trial/sample limits must be positive")
    require(MAX_LAYOUT_BACKTRACKS >= 0, "Backtracking limit must be nonnegative")
    require(0 < BRIGHTNESS_RANGE[0] <= BRIGHTNESS_RANGE[1]
            and 0 < CONTRAST_RANGE[0] <= CONTRAST_RANGE[1], "Invalid brightness/contrast ranges")
    require(0.0 <= PILL_ROTATION_RANGE[0] < PILL_ROTATION_RANGE[1] <= 360.0, "Invalid rotation interval")
    require(not args.replace_generated or args.mode == "full", "--replace-generated is only valid with --mode full")
    require(not args.replace_count_classes or args.mode == "full", "--replace-count-classes is only valid with --mode full")
    require(not (args.replace_count_classes and args.replace_generated), "Choose one replacement policy")
    require(0 <= CONTACT_IMAGE_FRACTION <= 1, "Contact fraction must be in [0,1]")
    require(not CONTACT_IMAGE_FRACTION or MIN_PILL_GAP == 0, "Contact scenes require MIN_PILL_GAP=0")
    require(all(0 <= p <= 1 for p in (SCENE_ROTATION_PROB, SCENE_HORIZONTAL_FLIP_PROB,
                                     SCENE_LIGHTING_PROB, SCENE_SHADOW_PROB)), "Scene probabilities must be in [0,1]")
    require(0 <= LIGHTING_GRADIENT_RANGE[0] <= LIGHTING_GRADIENT_RANGE[1] < 1
            and 0 <= SHADOW_STRENGTH_RANGE[0] <= SHADOW_STRENGTH_RANGE[1] < 1
            and 0 < SHADOW_RADIUS_RANGE[0] <= SHADOW_RADIUS_RANGE[1], "Invalid scene lighting/shadow ranges")
    require(0 < SCENE_GAIN_LIMITS[0] <= BRIGHTNESS_RANGE[0] <= BRIGHTNESS_RANGE[1] <= SCENE_GAIN_LIMITS[1],
            "Gain budget must contain the configured brightness range")
    output, debug = resolve_path(args.output_dir), resolve_path(args.debug_dir)
    protected_test = (PROJECT_ROOT / "training/dataset/test").resolve()
    require(all(path != protected_test and protected_test not in path.parents and path not in protected_test.parents
                for path in (output, debug)), "The existing test dataset and its ancestors are protected")
    require(output != debug and output not in debug.parents and debug not in output.parents,
            "Debug artifacts must be outside the ImageFolder dataset")
    for source in (BACKGROUND_DIR.resolve(), PILL_ASSET_DIR.resolve()):
        require(output != source and source not in output.parents and output not in source.parents,
                "Output must not overlap source assets")
        require(debug != source and source not in debug.parents and debug not in source.parents,
                "Debug output must not overlap source assets")
    random.seed(SEED)
    np.random.seed(SEED)
    # Keep execution serial and reproducible; OpenCV's implicit worker pool is unnecessary here.
    cv2.setNumThreads(1)
    backgrounds = load_backgrounds()
    pools = load_pill_assets()
    empty = NUM_EMPTY_SAMPLES if args.samples_per_type == NUM_SAMPLES_PER_TYPE else args.samples_per_type * len(PILL_TYPES)
    require(empty == args.samples_per_type * len(PILL_TYPES), "NUM_EMPTY_SAMPLES must preserve class balance")
    config = configuration(backgrounds, pools, args.samples_per_type, empty)
    fingerprint = sha256(json_bytes(config))
    run_dir = debug / fingerprint[:16]
    write_once(run_dir / "config.json", json_bytes(config), True)
    inventory = {"backgrounds": [{"id": b["id"], "shape": list(b["image"].shape), "sha256": b["sha256"]}
                                  for b in backgrounds],
                 "assets": [{"id": a["id"], "shape": list(a["image"].shape), "alpha_min": int(a["image"][:, :, 3].min()),
                             "alpha_max": int(a["image"][:, :, 3].max()), "alpha_pixels": a["area"],
                             "sha256": a["sha256"]} for pool in pools.values() for a in pool], "status": "PASS"}
    write_once(run_dir / "asset_validation.json", json_bytes(inventory), True)
    print(f"[assets] PASS: 16 RGB backgrounds, 8 RGBA assets/type (32 total)\n[evidence] {run_dir}", flush=True)
    if args.mode == "assets":
        return
    generate_mask_previews(backgrounds, run_dir / "masks")
    if args.mode == "masks":
        return
    if args.mode != "validate":
        generate_rotation_previews(pools, run_dir / "rotation", fingerprint)
    if args.mode == "rotation":
        return
    cache = {}
    if args.mode in {"feasibility", "pilot", "full"}:
        feasibility = check_placement_feasibility(backgrounds, pools, cache, run_dir, fingerprint)
        require(feasibility["status"] == "PASS", "Feasibility FAIL: inspect feasibility.json; generation blocked")
        if args.mode == "feasibility":
            return
    pilot_output, pilot_evidence = run_dir / "pilot_dataset", run_dir / "pilot"
    if args.mode == "pilot":
        generate_dataset(backgrounds, pools, cache, pilot_output, pilot_evidence,
                         PILOT_NUM_SAMPLES_PER_TYPE, PILOT_NUM_SAMPLES_PER_TYPE * len(PILL_TYPES), fingerprint, SEED + 2)
        validate_dataset(backgrounds, pools, pilot_output, pilot_evidence, fingerprint)
        generate_dataset_previews(pilot_output, pilot_evidence, backgrounds, pools)
    elif args.mode == "full":
        require(args.reviewed_previews, "Inspect mask, rotation and pilot previews, then pass --reviewed-previews")
        # Re-audit the entire pilot rather than trusting a stale PASS file.
        validate_dataset(backgrounds, pools, pilot_output, pilot_evidence, fingerprint)
        generate_dataset_previews(pilot_output, pilot_evidence, backgrounds, pools)
        evidence = run_dir / "full"
        full_start = time.perf_counter()
        previous = (snapshot_count_dataset(output) if args.replace_count_classes else
                    verify_generated_dataset_for_replacement(output, debug) if args.replace_generated else None)
        if previous:
            write_once(run_dir / "replacement_snapshot.json", json_bytes(previous), True)
        generation_output = run_dir / "staging_dataset" if previous else output
        generate_dataset(backgrounds, pools, cache, generation_output, evidence,
                         args.samples_per_type, empty, fingerprint, SEED)
        if previous and previous.get("mode") == "count_classes":
            preserve_non_target_content(output, generation_output, evidence, previous)
        elif previous:
            for relative, digest in previous["preserved_markers"].items():
                data = (output / relative).read_bytes()
                require(sha256(data) == digest, "Tracked dataset marker changed during generation")
                write_once(generation_output / relative, data)
        validate_dataset(backgrounds, pools, generation_output, evidence, fingerprint)
        if previous:
            publish_validated_replacement(generation_output, output, evidence, fingerprint, previous, debug)
            validate_dataset(backgrounds, pools, output, evidence, fingerprint)
        generate_dataset_previews(output, evidence, backgrounds, pools)
        write_once(evidence / "preview_review.json", json_bytes({"fingerprint": fingerprint,
                   "mask_rotation_and_pilot_review_asserted": True, "method": "--reviewed-previews",
                   "note": "This records operator review, not an automated visual-quality judgment."}))
        write_once(evidence / "regeneration_timing.json", json_bytes({"full_regeneration_seconds": time.perf_counter() - full_start,
                   "includes": "Ownership checks, generation, validation, optional directory replacement and post-publish validation; excludes prerequisite pilot/feasibility."}))
    elif args.mode == "validate":
        selected_output = pilot_output if args.target == "pilot" else output
        selected_evidence = pilot_evidence if args.target == "pilot" else run_dir / "full"
        validate_dataset(backgrounds, pools, selected_output, selected_evidence, fingerprint)
        generate_dataset_previews(selected_output, selected_evidence, backgrounds, pools)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"[ERROR] {error}", file=sys.stderr, flush=True)
        sys.exit(1)
