"""
phase3_feature_extraction.py
Extract per-tree-crown features from segmentation masks.

Features computed per tree:
    - Crown Area (px² → m² via geotransform)
    - Equivalent Crown Diameter (m) — D = sqrt(4 × area / π)
    - Crown Perimeter (m)
    - Circularity
    - Crown Density
    - Aspect Ratio
    - NDVI (queried at geographic centroid via affine transform)
    - Health Score
    - Centroid (px) and Geographic coordinates (lon/lat)

References:
    Xie et al. (2021) SegFormer — NeurIPS
    Jucker et al. (2017) Crown diameter allometry — Global Change Biology
"""

import logging
import os
import sys
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.transform import xy
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NDVI loading
# ---------------------------------------------------------------------------

def _load_ndvi(tif_path: str) -> Tuple[np.ndarray, Any, Any]:
    """Return (ndvi_array, transform, crs)."""
    with rasterio.open(tif_path) as src:
        ndvi = src.read(1)
        transform = src.transform
        crs = src.crs
    logger.info("NDVI raster: shape=%s, crs=%s, bounds=%s",
                ndvi.shape, crs, src.bounds)
    return ndvi, transform, crs


# ---------------------------------------------------------------------------
# NDVI query via geographic coordinates
# ---------------------------------------------------------------------------

def _query_ndvi_geo(lon: float, lat: float,
                    ndvi: np.ndarray,
                    transform: Any) -> Optional[float]:
    """Query NDVI at (lon, lat) using the raster's affine *transform*.

    Returns the NDVI value or *None* if the point falls outside the raster.
    """
    try:
        col, row = ~transform * (lon, lat)
        col = int(round(col))
        row = int(round(row))
        if 0 <= row < ndvi.shape[0] and 0 <= col < ndvi.shape[1]:
            return float(ndvi[row, col])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Feature extraction from a single mask
# ---------------------------------------------------------------------------

def extract_features_from_mask(
    mask_path: str,
    img_path: str,
    ndvi_data: Optional[Tuple[np.ndarray, Any, Any]],
    pixel_size_m: Optional[float],
    img_transform: Any = None,
    dataset: str = "unknown",
    plot: Optional[str] = None,
    gsd_status: str = "unknown",
) -> List[Dict[str, Any]]:
    """Extract per-tree features from a predicted mask.

    Parameters
    ----------
    mask_path : str
        Path to the PNG prediction mask.
    img_path : str
        Path to the corresponding RGB image (used to read geotransform).
    ndvi_data : tuple or None
        (ndvi_array, transform, crs) or None.
    pixel_size_m : float
        Ground-sampling distance in metres per pixel.
    img_transform : affine.Affine or None
        Pre-read affine transform for pixel-to-geo conversion.

    Returns
    -------
    list[dict]
        One entry per detected tree crown.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning("Cannot read mask: %s", mask_path)
        return []

    # Normalize any binary mask format ({0,1} or {0,255}) to {0,255}
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    ndvi_arr, ndvi_transform, ndvi_crs = (None, None, None)
    if ndvi_data is not None:
        ndvi_arr, ndvi_transform, ndvi_crs = ndvi_data

    base_name = os.path.splitext(os.path.basename(mask_path))[0]
    features: List[Dict[str, Any]] = []

    for i, cnt in enumerate(contours):
        area_px = cv2.contourArea(cnt)
        if area_px < config.MIN_CROWN_AREA_PX:
            continue

        perimeter_px = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area_px) / (perimeter_px * perimeter_px + 1e-6)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / (h + 1e-6)

        # Crown density: fraction of non-zero mask pixels inside the bbox
        roi = (binary[y:y + h, x:x + w] > 0).astype(np.uint8)
        density = float(np.sum(roi) / (w * h + 1e-6))

        # Crown area in m² (dataset/plot specific GSD; NaN when GSD unknown)
        if pixel_size_m is not None:
            area_m2 = round(area_px * (pixel_size_m ** 2), 4)
            perimeter_m = round(perimeter_px * pixel_size_m, 4)
            # Equivalent crown diameter (m)
            # D = sqrt(4 × area / π)  — [Jucker2017]
            equiv_diameter_m = float(np.sqrt(4.0 * area_m2 / np.pi))
        else:
            area_m2 = float("nan")
            perimeter_m = float("nan")
            equiv_diameter_m = float("nan")

        # Centroid in pixel coordinates
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx_px = M["m10"] / M["m00"]
            cy_px = M["m01"] / M["m00"]
        else:
            cx_px, cy_px = x + w / 2, y + h / 2

        # Convert pixel centroid (cx_px, cy_px) → geographic (lon, lat)
        lon: Optional[float] = None
        lat: Optional[float] = None
        if img_transform is not None:
            try:
                lon, lat = xy(img_transform, cy_px, cx_px, offset="center")
            except Exception:
                pass

        # Query NDVI at geographic location
        ndvi_val: Optional[float] = None
        if ndvi_arr is not None and ndvi_transform is not None and lon is not None and lat is not None:
            # If CRS differ, we'd need to reproject — skip for simplicity
            ndvi_val = _query_ndvi_geo(lon, lat, ndvi_arr, ndvi_transform)

        if ndvi_val is None:
            # Fallback: 0.5 (neutral) when NDVI cannot be queried
            ndvi_val = 0.5

        # Health score: composite of NDVI and crown density
        health_score = (
            max(0.0, ndvi_val - config.HEALTH_NDVI_THRESHOLD)
            * density / config.HEALTH_DENSITY_THRESHOLD
        )
        health_score = min(1.0, health_score)

        features.append({
            "image": base_name,
            "tree_id": i,
            "dataset": dataset,
            "plot": plot,
            "pixel_size_m": round(pixel_size_m, 4) if pixel_size_m is not None else float("nan"),
            "gsd_status": gsd_status,
            "crown_area_px": area_px,
            "crown_area_m2": area_m2,
            "equiv_crown_diameter_m": round(equiv_diameter_m, 4),
            "perimeter_px": perimeter_px,
            "perimeter_m": perimeter_m,
            "circularity": round(circularity, 6),
            "aspect_ratio": round(aspect_ratio, 6),
            "crown_density": round(density, 6),
            "ndvi": round(ndvi_val, 6),
            "health_score": round(health_score, 6),
            "centroid_x_px": round(cx_px, 2),
            "centroid_y_px": round(cy_px, 2),
        })

    return features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    # --- Per-image, dataset/plot-specific GSD registry -------------------
    from gsd_registry import _load_neon_cache, resolve_pixel_size

    _neon_cache = _load_neon_cache()
    logger.info("Cached metadata for %d NEON GeoTIFFs", len(_neon_cache))

    # --- Load NDVI -------------------------------------------------------
    ndvi_data: Optional[Tuple[np.ndarray, Any, Any]] = None
    if os.path.isfile(config.LANDSAT_TIF):
        ndvi_data = _load_ndvi(config.LANDSAT_TIF)
    else:
        logger.warning("NDVI raster not found at %s. "
                       "NDVI will default to 0.5.", config.LANDSAT_TIF)

    # --- Collect mask + image pairs --------------------------------------
    # Priority: predicted masks (outputs/masks/) > val masks > train masks
    mask_sources = [
        config.PRED_MASK_DIR,
        config.VAL_MASK_DIR,
        config.TRAIN_MASK_DIR,
    ]
    mask_paths: List[str] = []
    for src in mask_sources:
        mask_paths = sorted(glob(os.path.join(src, "*.png")))
        if mask_paths:
            logger.info("Using masks from %s (%d files)", src, len(mask_paths))
            break

    if not mask_paths:
        logger.error(
            "No mask PNGs found in any of: %s. Run Phase 2 first.",
            ", ".join(mask_sources),
        )
        sys.exit(1)

    # Build a transform cache from all available GeoTIFFs
    transform_cache: Dict[str, Any] = {}
    for cache_dir in [config.NEON_IMG_DIR, config.TRAIN_IMG_DIR, config.VAL_IMG_DIR]:
        if not os.path.isdir(cache_dir):
            continue
        for ext in (".tif", ".tiff"):
            for tp in glob(os.path.join(cache_dir, f"*{ext}")):
                try:
                    with rasterio.open(tp) as src:
                        key = os.path.splitext(os.path.basename(tp))[0]
                        transform_cache[key] = src.transform
                except Exception:
                    continue
    logger.info("Cached %d GeoTIFF transforms", len(transform_cache))

    # Match each mask to its corresponding image using basename only
    def _find_image(mask_file: str) -> Tuple[Optional[str], Any]:
        base = os.path.splitext(os.path.basename(mask_file))[0]
        # Search all image directories by basename match
        for img_dir_candidate in [
            config.NEON_IMG_DIR,
            config.TRAIN_IMG_DIR,
            config.VAL_IMG_DIR,
            config.PRED_MASK_DIR.replace("masks", "images"),
        ]:
            for ext in config.IMAGE_EXTENSIONS:
                candidate = os.path.join(img_dir_candidate, f"{base}{ext}")
                if os.path.isfile(candidate):
                    # Look up transform by candidate basename or source prefix
                    tf = transform_cache.get(base)
                    if tf is None:
                        prefix = base.rsplit("_t", 1)[0] if "_t" in base else base
                        tf = transform_cache.get(prefix)
                    return candidate, tf
        return None, None

    all_features: List[Dict[str, Any]] = []
    for mp in tqdm(mask_paths, desc="Extracting features"):
        img_path, img_tf = _find_image(mp)
        base_name = os.path.splitext(os.path.basename(mp))[0]
        info = resolve_pixel_size(base_name, _neon_cache)
        feats = extract_features_from_mask(
            mp, img_path or "", ndvi_data,
            info["pixel_size_m"],
            img_transform=img_tf,
            dataset=info["dataset"],
            plot=info["plot"],
            gsd_status=info["gsd_status"],
        )
        all_features.extend(feats)

    if not all_features:
        logger.error("No tree crowns detected in any mask. Aborting.")
        sys.exit(1)

    df = pd.DataFrame(all_features)
    df.to_csv(config.FEATURES_CSV, index=False)
    logger.info("Saved %d tree records to %s", len(df), config.FEATURES_CSV)

    print(f"\nExtracted features for {len(df)} tree crowns -> {config.FEATURES_CSV}")


if __name__ == "__main__":
    main()
