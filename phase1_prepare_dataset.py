"""
phase1_prepare_dataset.py
Prepare tiled datasets for SegFormer training from NEON + ReforesTree sources.

Pipeline:
  1. Discover image-mask pairs from neon_rgb/ and reforestree/
  2. Split ORIGINAL IMAGES into train / val / test (no leakage)
  3. Tile each split independently
  4. Smart Sampling — keep partial negatives for context
  5. Save tiles to train/val/test directories
  6. Generate dataset_split.json for Phase 2

Reads from:
    dataset/neon_rgb/{images,masks}/
    dataset/reforestree/{images,masks}/

Writes to:
    dataset/{train,val,test}/{images,masks}/
    dataset/dataset_split.json
"""

import json
import logging
import math
import os
import random
import shutil
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Smart Sampling constants
# ---------------------------------------------------------------------------
# Fraction of ZERO-FOREGROUND tiles to keep per image (0.0 = discard all)
KEEP_ZERO_FOREGROUND: float = 0.20
# Fraction of LOW-FOREGROUND tiles to keep per image (0 < fg < 0.5%)
KEEP_LOW_FOREGROUND: float = 0.50
# Threshold separating "low" from "full" foreground
LOW_FG_THRESHOLD: float = 0.005


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _find_images(directory: str) -> List[str]:
    """Return sorted list of image paths in *directory*."""
    files: List[str] = []
    for ext in config.IMAGE_EXTENSIONS:
        files.extend(glob(os.path.join(directory, f"*{ext}")))
    return sorted(files)


def _strip_known_extensions(filename: str) -> str:
    """Remove up to three levels of known image extensions."""
    name = filename
    for _ in range(3):
        _, ext = os.path.splitext(name)
        if ext.lower() in config.IMAGE_EXTENSIONS:
            name = name[: -len(ext)]
        else:
            break
    return name


def discover_neon_pairs() -> List[Dict]:
    """Find image-mask pairs in dataset/neon_rgb/.

    Mask naming convention: same filename as image (e.g., img.png → mask.png).
    Both .tif.tiff and .png exist in images/; masks/ are .png only.
    We match by stem (ignoring all known extensions).
    """
    img_dir = config.NEON_IMG_DIR
    mask_dir = config.NEON_MASK_DIR

    if not os.path.isdir(img_dir):
        logger.warning("NEON image directory not found: %s", img_dir)
        return []

    source_images = _find_images(img_dir)
    if not source_images:
        logger.warning("No images found in %s", img_dir)
        return []

    # Build mask lookup by stem
    mask_stems: Dict[str, str] = {}
    for mpath in glob(os.path.join(mask_dir, "*")):
        if not os.path.isfile(mpath):
            continue
        stem = _strip_known_extensions(os.path.basename(mpath))
        mask_stems[stem] = mpath

    pairs = []
    for img_path in source_images:
        stem = _strip_known_extensions(os.path.basename(img_path))
        if stem in mask_stems:
            pairs.append({
                "dataset": "neon_rgb",
                "image_path": img_path,
                "mask_path": mask_stems[stem],
                "image_id": stem,
            })

    logger.info("NEON: %d pairs found (from %d images)",
                len(pairs), len(source_images))
    return pairs


def discover_reforestree_pairs() -> List[Dict]:
    """Find image-mask pairs in dataset/reforestree/.

    Mask naming convention: {image_stem}_mask.png
    """
    img_dir = os.path.join(config.DATASET_DIR, "reforestree", "images")
    mask_dir = os.path.join(config.DATASET_DIR, "reforestree", "masks")

    if not os.path.isdir(img_dir):
        logger.warning("ReforesTree image directory not found: %s", img_dir)
        return []

    source_images = _find_images(img_dir)
    if not source_images:
        logger.warning("No images found in %s", img_dir)
        return []

    # Build mask lookup by stem + "_mask"
    mask_lookup: Dict[str, str] = {}
    for mpath in glob(os.path.join(mask_dir, "*")):
        if not os.path.isfile(mpath):
            continue
        mname = os.path.basename(mpath)
        # Strip _mask suffix then known extensions
        if mname.lower().endswith("_mask.png"):
            stem = mname[:-len("_mask.png")]
        else:
            stem = _strip_known_extensions(mname)
        mask_lookup[stem] = mpath

    pairs = []
    for img_path in source_images:
        stem = _strip_known_extensions(os.path.basename(img_path))
        if stem in mask_lookup:
            pairs.append({
                "dataset": "reforestree",
                "image_path": img_path,
                "mask_path": mask_lookup[stem],
                "image_id": f"reforestree_{stem}",
            })

    logger.info("ReforesTree: %d pairs found (from %d images)",
                len(pairs), len(source_images))
    return pairs


# ---------------------------------------------------------------------------
# Split at image level
# ---------------------------------------------------------------------------

def split_images(pairs: List[Dict]) -> Dict[str, List[Dict]]:
    """Assign each original image to exactly one split (train/val/test).

    Returns ``{"train": [...], "val": [...], "test": [...]}``.
    """
    # Group by image_id to avoid leakage
    by_image: Dict[str, List[Dict]] = {}
    for p in pairs:
        by_image.setdefault(p["image_id"], []).append(p)

    image_ids = list(by_image.keys())
    rng = random.Random(config.RANDOM_SEED)
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * config.TRAIN_SPLIT)
    n_val = int(n * config.VAL_SPLIT)

    splits = {"train": [], "val": [], "test": []}
    for i, img_id in enumerate(image_ids):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        splits[split].extend(by_image[img_id])

    for name, items in splits.items():
        n_img = len(set(p["image_id"] for p in items))
        logger.info("  %s: %d images → %d tiles (before sampling)",
                     name, n_img, len(items))

    return splits


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def tile_image(img_path: str, mask_path: str,
               image_id: str, dataset: str
               ) -> List[Dict]:
    """Crop an image+mask pair into non-overlapping tiles.

    Returns a list of tile metadata dicts.
    """
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    w, h = img.size
    tile_size = config.TILE_SIZE
    tiles = []
    tile_id = 0

    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile_img = img.crop((x, y, x + tile_size, y + tile_size))
            tile_mask = mask.crop((x, y, x + tile_size, y + tile_size))

            mask_np = np.array(tile_mask)
            fg_ratio = float((mask_np > 0).mean())

            tiles.append({
                "image": tile_img,
                "mask": tile_mask,
                "dataset": dataset,
                "image_id": image_id,
                "tile_id": f"{image_id}_t{tile_id}",
                "foreground_ratio": fg_ratio,
            })
            tile_id += 1

    return tiles


def tile_split(pairs: List[Dict], split_name: str) -> List[Dict]:
    """Tile all images in a split. Returns flat list of tile metadata."""
    all_tiles = []
    for p in tqdm(pairs, desc=f"Tiling {split_name}"):
        try:
            tiles = tile_image(
                p["image_path"], p["mask_path"],
                p["image_id"], p["dataset"],
            )
            all_tiles.extend(tiles)
        except Exception as exc:
            logger.warning("Failed to tile %s: %s", p["image_path"], exc)
    return all_tiles


# ---------------------------------------------------------------------------
# Smart Sampling
# ---------------------------------------------------------------------------

def smart_sample(tiles: List[Dict]) -> List[Dict]:
    """Apply smart sampling: keep partial negatives for context.

    For each image group:
      - foreground == 0%      → keep KEEP_ZERO_FOREGROUND fraction
      - 0 < foreground < 0.5% → keep KEEP_LOW_FOREGROUND fraction
      - foreground >= 0.5%    → keep all
    """
    # Group tiles by image_id
    by_image: Dict[str, List[Dict]] = {}
    for t in tiles:
        by_image.setdefault(t["image_id"], []).append(t)

    sampled = []
    for img_id, img_tiles in by_image.items():
        zero = [t for t in img_tiles if t["foreground_ratio"] == 0.0]
        low = [t for t in img_tiles
               if 0.0 < t["foreground_ratio"] < LOW_FG_THRESHOLD]
        full = [t for t in img_tiles
                if t["foreground_ratio"] >= LOW_FG_THRESHOLD]

        rng = random.Random(config.RANDOM_SEED + hash(img_id) % (2**31))

        n_keep_zero = max(1, int(len(zero) * KEEP_ZERO_FOREGROUND))
        n_keep_low = max(1, int(len(low) * KEEP_LOW_FOREGROUND))

        if zero:
            sampled.extend(rng.sample(zero, min(n_keep_zero, len(zero))))
        if low:
            sampled.extend(rng.sample(low, min(n_keep_low, len(low))))
        sampled.extend(full)

    return sampled


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(tiles: List[Dict], label: str) -> None:
    """Print dataset statistics for a list of tiles."""
    if not tiles:
        logger.warning("No tiles to report for %s", label)
        return

    fg = np.array([t["foreground_ratio"] for t in tiles])
    n_pos = int((fg > 0).sum())
    n_neg = int((fg == 0).sum())

    logger.info("=== %s Statistics ===", label)
    logger.info("  Total tiles    : %d", len(tiles))
    logger.info("  Positive tiles : %d (%.1f%%)", n_pos, 100 * n_pos / len(tiles))
    logger.info("  Negative tiles : %d (%.1f%%)", n_neg, 100 * n_neg / len(tiles))
    logger.info("  Avg fg ratio   : %.4f", fg.mean())
    logger.info("  Median fg      : %.4f", np.median(fg))
    logger.info("  Max fg         : %.4f", fg.max())

    # Histogram buckets
    bins = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.01]
    counts, _ = np.histogram(fg, bins=bins)
    logger.info("  Foreground ratio histogram:")
    for i in range(len(bins) - 1):
        bar = "#" * min(50, int(50 * counts[i] / max(counts.max(), 1)))
        logger.info("    [%5.1f%% - %5.1f%%]: %5d  %s",
                     bins[i] * 100, bins[i + 1] * 100, counts[i], bar)


# ---------------------------------------------------------------------------
# Save tiles
# ---------------------------------------------------------------------------

def save_tiles(tiles: List[Dict], split_name: str) -> int:
    """Write tile images and masks to dataset/{split}/{images,masks}/."""
    img_dir = os.path.join(config.DATASET_DIR, split_name, "images")
    mask_dir = os.path.join(config.DATASET_DIR, split_name, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    saved = 0
    for t in tqdm(tiles, desc=f"Saving {split_name}"):
        fname = f"{t['tile_id']}.png"
        t["image"].save(os.path.join(img_dir, fname))
        t["mask"].save(os.path.join(mask_dir, fname))
        saved += 1

    return saved


# ---------------------------------------------------------------------------
# Generate dataset_split.json
# ---------------------------------------------------------------------------

def generate_split_json(all_tiles: Dict[str, List[Dict]]) -> str:
    """Write dataset_split.json consumed by phase2_train_segformer.py."""
    records = []
    for split_name, tiles in all_tiles.items():
        for t in tiles:
            tile_id = t["tile_id"]
            records.append({
                "dataset": t["dataset"],
                "split": split_name,
                "image": os.path.join(
                    config.DATASET_DIR, split_name, "images",
                    f"{tile_id}.png"),
                "mask": os.path.join(
                    config.DATASET_DIR, split_name, "masks",
                    f"{tile_id}.png"),
                "foreground_ratio": round(t["foreground_ratio"], 6),
                "image_id": t["image_id"],
                "tile_id": tile_id,
            })

    out_path = os.path.join(config.DATASET_DIR, "dataset_split.json")
    with open(out_path, "w") as f:
        json.dump({"train": [r for r in records if r["split"] == "train"],
                    "val":   [r for r in records if r["split"] == "val"],
                    "test":  [r for r in records if r["split"] == "test"]},
                   f, indent=2)

    logger.info("Wrote %d records to %s", len(records), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Tile size: %d x %d", config.TILE_SIZE, config.TILE_SIZE)

    # Clean existing split directories
    for d in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        if os.path.isdir(d):
            logger.info("Cleaning: %s", d)
            shutil.rmtree(d, ignore_errors=True)

    # ------------------------------------------------------------------
    # Step 1: Discover pairs
    # ------------------------------------------------------------------
    logger.info("Step 1 — Discovering image-mask pairs")
    all_pairs: List[Dict] = []

    neon = discover_neon_pairs()
    all_pairs.extend(neon)

    ref = discover_reforestree_pairs()
    all_pairs.extend(ref)

    if not all_pairs:
        logger.error("No image-mask pairs found. Aborting.")
        sys.exit(1)

    total_images = len(set(p["image_id"] for p in all_pairs))
    logger.info("Total: %d pairs from %d unique images",
                len(all_pairs), total_images)

    # ------------------------------------------------------------------
    # Step 2: Split at image level
    # ------------------------------------------------------------------
    logger.info("Step 2 — Splitting images (no leakage)")
    split_pairs = split_images(all_pairs)

    # ------------------------------------------------------------------
    # Step 3: Tile each split independently
    # ------------------------------------------------------------------
    logger.info("Step 3 — Tiling each split")
    split_tiles: Dict[str, List[Dict]] = {}
    for split_name in ["train", "val", "test"]:
        split_tiles[split_name] = tile_split(split_pairs[split_name],
                                             split_name)

    # ------------------------------------------------------------------
    # Step 4: Smart Sampling
    # ------------------------------------------------------------------
    logger.info("Step 4 — Smart Sampling")
    for split_name in ["train", "val", "test"]:
        before = len(split_tiles[split_name])
        split_tiles[split_name] = smart_sample(split_tiles[split_name])
        after = len(split_tiles[split_name])
        logger.info("  %s: %d → %d tiles", split_name, before, after)

    # ------------------------------------------------------------------
    # Step 5: Print statistics
    # ------------------------------------------------------------------
    logger.info("Step 5 — Statistics")
    for split_name in ["train", "val", "test"]:
        print_stats(split_tiles[split_name], split_name)

    # ------------------------------------------------------------------
    # Step 6: Save tiles
    # ------------------------------------------------------------------
    logger.info("Step 6 — Saving tiles")
    for split_name in ["train", "val", "test"]:
        saved = save_tiles(split_tiles[split_name], split_name)
        logger.info("  %s: %d tiles saved", split_name, saved)

    # ------------------------------------------------------------------
    # Step 7: Generate dataset_split.json
    # ------------------------------------------------------------------
    logger.info("Step 7 — Generating dataset_split.json")
    json_path = generate_split_json(split_tiles)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = sum(len(v) for v in split_tiles.values())
    print(f"\nDataset preparation complete")
    print(f"  Source pairs     : {len(all_pairs)}")
    print(f"  Unique images    : {total_images}")
    print(f"  Total tiles      : {total}")
    for name in ["train", "val", "test"]:
        n = len(split_tiles[name])
        n_img = len(set(t["image_id"] for t in split_tiles[name]))
        print(f"  {name:>5}: {n:>5} tiles from {n_img} images")
    print(f"  JSON             : {json_path}")


if __name__ == "__main__":
    main()
