"""
gsd_registry.py
Dataset/plot-specific GSD resolution and tile-geometry helpers.

The oxygen/allometric pipeline is driven by a GLOBAL pixel size in
phase3 (default 0.1 m/px, previously overridden only by the first NEON
GeoTIFF). That assumption is wrong for ReforesTree (~0.010-0.012 m/px).

This module answers, for any image/mask basename:
    * dataset        (neon | reforestree | synthetic | unknown)
    * plot           (ReforesTree plot or NEON site, else None)
    * pixel_size_m   (verified GSD in m/px, or None when unknown)
    * gsd_status     ("verified" | "unknown")

No values are invented: NEON GSD comes from the matching benchmark
GeoTIFF metadata when found (fell back to 0.1 only for the confirmed
uniform NEON RGB ~0.1 m/px); ReforesTree GSD comes from the verified per
plot orthomosaic GSD table in config.REF_PLOT_GSD. Everything else that
cannot be verified is reported as "unknown" (pixel_size_m=None).
"""

import math
import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

KNOWN_NEON_PREFIXES = (
    "2018", "ABBY", "BART", "BLAN", "BONA", "CLBJ", "DEJU", "DELA",
    "DSNY", "GRSM", "GUAN", "HARV", "HEAL", "JERC", "JORN", "KONZ",
    "LAJA", "LENO", "MLBS", "MOAB", "NIWO", "NOGP", "OSBS", "PUUM",
    "RMNP", "SCBI", "SERC", "SJER", "SOAP", "SRER", "STEI", "TALL",
    "TEAK", "UKFS", "UNDE", "WREF", "YELL",
)

NEON_FALLBACK_PIXEL_SIZE_M = 0.1

# Cache of NEON benchmark GeoTIFF metadata: stem -> (transform, crs, width, height)
_NEON_CACHE: Dict[str, Tuple[Any, Any, int, int]] = {}


def _load_neon_cache() -> Dict[str, Tuple[Any, Any, int, int]]:
    """Populate and return the NEON GeoTIFF metadata cache."""
    if _NEON_CACHE:
        return _NEON_CACHE
    for img_dir in (config.NEON_TRAIN_RGB_DIR, config.NEON_EVAL_RGB_DIR):
        if not os.path.isdir(img_dir):
            continue
        for tp in glob(os.path.join(img_dir, "*.tif")):
            stem = os.path.splitext(os.path.basename(tp))[0]
            try:
                with rasterio.open(tp) as src:
                    _NEON_CACHE[stem] = (src.transform, src.crs, src.width, src.height)
            except Exception:
                continue
    return _NEON_CACHE


# ---------------------------------------------------------------------------
# Dataset / plot classification
# ---------------------------------------------------------------------------

def classify_dataset(image_base: str) -> str:
    """Return one of: 'neon', 'reforestree', 'synthetic', 'unknown'."""
    if not image_base:
        return "unknown"
    if image_base.startswith("reforestree_"):
        return "reforestree"
    if image_base.startswith("synthetic_"):
        return "synthetic"
    if image_base.startswith(KNOWN_NEON_PREFIXES):
        return "neon"
    return "unknown"


def neon_site(image_base: str) -> Optional[str]:
    """Return the NEON site (e.g. 'OSBS', 'SJER') or None."""
    m = re.match(r"2018_([A-Z0-9]+)_", image_base)
    if m:
        return m.group(1)
    m = re.match(r"([A-Z0-9]+)", image_base)
    return m.group(1) if m else None


def refores_tree_plot(image_base: str) -> Optional[str]:
    """Return the ReforesTree plot name embedded in the tile basename.

    Pattern: reforestree_<First>_<Last>_<Plot Name> RGB_<idx>_<x0>_<y0>_<x1>_<y1>_t<k>
    e.g. reforestree_Flora_Pluas_Flora Pluas RGB_5_3800_0_7800_4000_t0
    """
    m = re.match(
        r"reforestree_(.+) RGB_\d+_\d+_\d+_\d+_\d+_t\d+$",
        image_base,
    )
    if not m:
        return None
    prefix = m.group(1)
    plot = prefix.rsplit("_", 1)[-1]
    if plot in config.REF_PLOT_GSD:
        return plot
    if "Manuel Macias" in prefix:
        return "Manuel Macias"
    return None


def ref_tile_window_t(image_base: str) -> Optional[Tuple[int, int, int]]:
    """Return (x0, y0, sub_tile_index) for a ReforesTree tile basename."""
    m = re.search(
        r"RGB_\d+_(\d+)_(\d+)_\d+_\d+_t(\d+)$",
        image_base,
    )
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# ---------------------------------------------------------------------------
# NEON source GeoTIFF resolution
# ---------------------------------------------------------------------------

def _neon_t_index(image_base: str) -> Optional[int]:
    m = re.search(r"_t(\d+)$", image_base)
    return int(m.group(1)) if m else None


def _neon_stem(image_base: str) -> str:
    """Strip the trailing tile index from a NEON basename."""
    m = re.match(r"^(.+)_t\d+$", image_base)
    return m.group(1) if m else image_base


def neon_source_geometadata(
    image_base: str,
    cache: Optional[Dict[str, Tuple[Any, Any, int, int]]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve (transform, crs, tile origin px) for a NEON tile, if possible.

    NEON tile naming: '<source_stem>_t<k>' where the source is a benchmark
    GeoTIFF and k indexes a non-overlapping 256x256 grid:
        cols = ceil(src_width / 256)
        origin = ((k % cols) * 256, (k // cols) * 256)
    Some names also carry a '_crop' suffix (e.g. 2018_BART_..._image_crop_t0)
    or a bare site tile (e.g. OSBS_029_t0 -> OSBS_029_2018.tif).
    """
    if cache is None:
        cache = _load_neon_cache()
    t_idx = _neon_t_index(image_base)
    stem = _neon_stem(image_base)

    candidates: List[str] = [stem]
    if os.path.splitext(stem)[1] == "":
        # Try benchmark suffix variants for standalone site tiles
        stripped = re.sub(r"_2019$|_2018$|_competition$", "", stem)
        candidates += [
            stripped + "_2018", stripped + "_2019", stripped + "_competition",
        ]
    for cand in candidates:
        if cand not in cache:
            continue
        transform, crs, width, height = cache[cand]
        cols = max(1, int(math.ceil(width / config.TILE_SIZE_SUB)))
        if t_idx is None:
            origin = (0, 0)
        else:
            origin = ((t_idx % cols) * config.TILE_SIZE_SUB,
                      (t_idx // cols) * config.TILE_SIZE_SUB)
        if origin[0] >= width or origin[1] >= height:
            return None
        return {
            "transform": transform,
            "crs": crs,
            "tile_origin": origin,
            "pixel_size_m": abs(transform.a) or abs(transform.e) or NEON_FALLBACK_PIXEL_SIZE_M,
            "source": cand + ".tif",
        }
    return None


# ---------------------------------------------------------------------------
# GSD resolution entry point
# ---------------------------------------------------------------------------

def resolve_pixel_size(
    image_base: str,
    neon_cache: Optional[Dict[str, Tuple[Any, Any, int, int]]] = None,
) -> Dict[str, Any]:
    """Return GSD info for a feature/mask basename.

    Returns a dict with keys:
        dataset, plot, pixel_size_m, gsd_status, source_raster
    pixel_size_m is None and gsd_status == 'unknown' when the GSD cannot be
    verified (synthetic images, Manuel Macias padded orthomosaic, etc.).
    """
    dataset = classify_dataset(image_base)
    plot = None
    pixel_size_m = None
    gsd_status = "unknown"
    source_raster = None

    if dataset == "neon":
        plot = neon_site(image_base)
        meta = neon_source_geometadata(image_base, neon_cache)
        if meta is not None:
            pixel_size_m = meta["pixel_size_m"]
            source_raster = os.path.join(
                "NeonTreeEvaluation", "...RGB", meta["source"]
            )
        else:
            # NEON benchmark RGB is uniform ~0.1 m/px across all verified
            # rasters; use the documented fallback when an exact GeoTIFF is
            # not found for this tile.
            pixel_size_m = NEON_FALLBACK_PIXEL_SIZE_M
        gsd_status = "verified"

    elif dataset == "reforestree":
        plot = refores_tree_plot(image_base)
        if plot in config.REF_PLOT_GSD:
            pixel_size_m = config.REF_PLOT_GSD[plot]
            gsd_status = "verified"
            source_raster = plot + " RGB.tif"
        else:
            plot = plot or "unknown"
            gsd_status = "unknown"
            pixel_size_m = None

    elif dataset == "synthetic":
        plot = None
        pixel_size_m = None
        gsd_status = "unknown"

    return {
        "dataset": dataset,
        "plot": plot,
        "pixel_size_m": pixel_size_m,
        "gsd_status": gsd_status,
        "source_raster": source_raster,
    }


# ---------------------------------------------------------------------------
# ReforesTree georeferencing geometry
# ---------------------------------------------------------------------------

def _ref_ortho_meta(plot: str) -> Optional[Tuple[Any, Any]]:
    """Return (transform, crs) for a validated ReforesTree orthomosaic."""
    if plot not in config.REF_PLOT_GSD:
        return None
    path = os.path.join(config.REFORESTREE_ORTHO_DIR, f"{plot} RGB.tif")
    if not os.path.isfile(path):
        return None
    with rasterio.open(path) as src:
        return src.transform, src.crs


def ref_tile_center_geo(image_base: str, cx_px: float, cy_px: float):
    """Georeference a ReforesTree 256 tile centroid in the plot orthomosaic.

    tile pixel (cx, cy) -> orthomosaic pixel
        (x0 + ((t % cols) * 256) + cx, y0 + ((t // cols) * 256) + cy)
    then affine transform -> (lon, lat) in EPSG:4326.
    """
    plot = refores_tree_plot(image_base)
    meta = _ref_ortho_meta(plot)
    if meta is None:
        return None
    if "Manuel Macias" in str(plot):
        return None
    transform, crs = meta
    win = ref_tile_window_t(image_base)
    if win is None:
        return None
    x0, y0, t = win
    cols = config.REF_TILE_COLS
    ox = x0 + (t % cols) * config.TILE_SIZE_SUB
    oy = y0 + (t // cols) * config.TILE_SIZE_SUB
    lon, lat = rasterio.transform.xy(
        transform, oy + cy_px, ox + cx_px, offset="center"
    )
    return lon, lat, crs


def neon_tile_center_geo(
    image_base: str,
    cx_px: float,
    cy_px: float,
    neon_cache: Optional[Dict[str, Tuple[Any, Any, int, int]]] = None,
):
    """Georeference a NEON 256 tile centroid in its source GeoTIFF."""
    meta = neon_source_geometadata(image_base, neon_cache)
    if meta is None:
        return None
    origin = meta["tile_origin"]
    transform, crs = meta["transform"], meta["crs"]
    lon, lat = rasterio.transform.xy(
        transform,
        origin[1] + cy_px,
        origin[0] + cx_px,
        offset="center",
    )
    return lon, lat, crs