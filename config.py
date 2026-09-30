"""
config.py
Central configuration for the Tree Oxygen Potential Estimation pipeline.
"""

import os
import logging
from typing import List

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join(ROOT, "dataset")
NEON_RGB_DIR = os.path.join(DATASET_DIR, "neon_rgb")
NEON_IMG_DIR = os.path.join(NEON_RGB_DIR, "images")
NEON_MASK_DIR = os.path.join(NEON_RGB_DIR, "masks")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, "images")
TRAIN_MASK_DIR = os.path.join(TRAIN_DIR, "masks")
VAL_IMG_DIR = os.path.join(VAL_DIR, "images")
VAL_MASK_DIR = os.path.join(VAL_DIR, "masks")
TEST_IMG_DIR = os.path.join(TEST_DIR, "images")
TEST_MASK_DIR = os.path.join(TEST_DIR, "masks")

LANDSAT_TIF = os.path.join(DATASET_DIR, "landsat8", "ndvi.tif")

# ---------------------------------------------------------------------------
# Dataset-specific GSD / georeference sources
# ---------------------------------------------------------------------------
# NEON benchmark RGB GeoTIFFs (training + evaluation partitions). All are
# ~0.1 m/px (res verified per raster, e.g. 0.10024 x 0.09975 or 0.1 x 0.1).
NEON_TRAIN_RGB_DIR = os.path.join(ROOT, "NeonTreeEvaluation", "training", "RGB")
NEON_EVAL_RGB_DIR = os.path.join(ROOT, "NeonTreeEvaluation", "evaluation", "evaluation", "RGB")

# ReforesTree source orthomosaics (EPSG:4326). Verified GSD per plot.
# 'Manuel Macias RGB_padded.tif' is intentionally excluded (no valid CRS).
REFORESTREE_ORTHO_DIR = os.path.join(
    ROOT, "reforestree_dataset", "wwf_ecuador", "RGB Orthomosaics"
)

# Measured source orthomosaic GSD (m/px) per validated ReforesTree plot.
# Sources:
#   0.0104  Carlos Vera Arteaga  (res 9.363e-08 deg x 9.424e-08 deg)
#   0.0118  Carlos Vera Guevara  (res 1.071e-07 deg x 1.078e-07 deg)
#   0.0097  Flora Pluas          (res 8.735e-08 deg x 8.788e-08 deg)
#   0.0105  Leonor Aspiazu       (res 9.484e-08 deg x 9.546e-08 deg)
#   0.0109  Nestor Macias        (res 9.895e-08 deg x 9.955e-08 deg)
REF_PLOT_GSD = {
    "Carlos Vera Arteaga": 0.0104,
    "Carlos Vera Guevara": 0.0118,
    "Flora Pluas": 0.0097,
    "Leonor Aspiazu": 0.0105,
    "Nestor Macias": 0.0109,
}
# 4000x4000 ReforesTree crop tiles are subdivided into non-overlapping
# 256x256 tiles on a 16-column grid (ceil(4000 / 256) == 16).
REF_TILE_COLS = 16
TILE_SIZE_SUB = 256

MODELS_DIR = os.path.join(ROOT, "models")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
PRED_MASK_DIR = os.path.join(OUTPUTS_DIR, "masks")
FEATURES_CSV = os.path.join(OUTPUTS_DIR, "tree_features.csv")
LOGS_DIR = os.path.join(ROOT, "logs")
GIS_DIR = os.path.join(ROOT, "gis")
MAP_DIR = os.path.join(GIS_DIR, "maps")
SHP_DIR = os.path.join(GIS_DIR, "shapefiles")

# ---------------------------------------------------------------------------
# DeepForest
# ---------------------------------------------------------------------------
DEEPFOREST_RELEASE_URL = (
    "https://github.com/weecology/DeepForest/releases/download/"
)
NEON_SAMPLE_URL = (
    "https://data.neonscience.org/api/v0/data/DP3.30010.001"
)

# ---------------------------------------------------------------------------
# Tiling / Dataset preparation
# ---------------------------------------------------------------------------
TILE_SIZE: int = 256
IMAGE_EXTENSIONS: List[str] = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
TRAIN_SPLIT: float = 0.8
VAL_SPLIT: float = 0.1
TEST_SPLIT: float = 0.1
RANDOM_SEED: int = 42


def binarize_mask(mask_array: np.ndarray, threshold: int = 0) -> np.ndarray:
    """Return a binary foreground mask using the shared threshold."""
    return (mask_array > threshold).astype(np.int64)

# ---------------------------------------------------------------------------
# SegFormer training
# ---------------------------------------------------------------------------
BATCH_SIZE: int = 8
GRAD_ACCUM_STEPS: int = 4
EPOCHS: int = 100
LEARNING_RATE: float = 6e-5
SEGFORMER_MODEL_NAME: str = "nvidia/mit-b0"
NUM_LABELS: int = 2  # background, tree
EARLY_STOPPING_PATIENCE: int = 10
EARLY_STOPPING_MIN_DELTA: float = 0.0001
LOSS_TYPE: str = "weighted_ce"  # "ce", "weighted_ce", or "dice_ce"
# Class weights computed from training set foreground ratio (~22.5% foreground)
CLASS_WEIGHTS: list[float] = None  # set to None for auto-compute, or [bg_w, fg_w]

# Maximum image dimension (width or height) allowed in training DataLoader.
# Images exceeding this are skipped at dataset init to prevent OOM
# from full-resolution NEON originals (10000x10000).
# Smaller TIFFs (e.g. 400x400 crops) pass through normally.
MAX_TRAIN_IMAGE_DIM: int = 5000

# ---------------------------------------------------------------------------
# Dataset enable flags — each source can be toggled independently
# ---------------------------------------------------------------------------
ENABLE_REFORESTREE: bool = True    # Primary — best annotations
ENABLE_NEON: bool = True           # Secondary — diversity & generalization
# ---------------------------------------------------------------------------
# Fine-tuning stages (after primary training completes)
# ---------------------------------------------------------------------------
ENABLE_NEON_FINETUNE: bool = False
FINETUNE_EPOCHS: int = 100
FINETUNE_LEARNING_RATE: float = 1e-5
FINETUNE_EARLY_STOPPING_PATIENCE: int = 5

# ---------------------------------------------------------------------------
# Augmentation hyperparameters (v3 improvements)
# ---------------------------------------------------------------------------
# Brightness/contrast simulates sunlight changes across flight campaigns
AUG_BRIGHTNESS_LIMIT: float = 0.2
AUG_CONTRAST_LIMIT: float = 0.2
# Hue/saturation/value shifts simulate seasonal foliage color variation
AUG_HUE_SHIFT: int = 10
AUG_SAT_SHIFT: int = 20
AUG_VALUE_SHIFT: int = 15
# Random gamma corrects for exposure differences between NEON sites
AUG_GAMMA_RANGE: tuple = (80, 120)  # percentage, 100 = identity
# Gaussian noise improves robustness to sensor noise at high ISO
AUG_GAUSS_NOISE_STD: tuple = (0.02, 0.1)
# Blur kernels simulate motion or defocus in aerial photography
AUG_BLUR_KERNEL: int = 3
AUG_MOTION_BLUR_KERNEL: int = 5

# ---------------------------------------------------------------------------
# Hybrid loss (v3 improvement)
# ---------------------------------------------------------------------------
USE_DICE_LOSS: bool = True
# Dice loss helps boundary prediction and small-object segmentation
# by optimizing per-pixel overlap directly (unlike CE which treats
# each pixel independently).

# ---------------------------------------------------------------------------
# LR Scheduler (v3 improvement)
# ---------------------------------------------------------------------------
# Ratio of total steps used for linear warmup (instead of fixed step count)
WARMUP_RATIO: float = 0.1
# Cosine decay provides smoother convergence than linear decay by
# reducing the LR gradually near the end of training, preventing
# the optimizer from bouncing around the minimum.

# ---------------------------------------------------------------------------
# Test-Time Augmentation (v3 improvement)
# ---------------------------------------------------------------------------
USE_TTA: bool = False
# When True, inference averages predictions over original + horizontal flip
# before argmax. This reduces variance without retraining.

# ---------------------------------------------------------------------------
# Allometric Equation Constants
# ---------------------------------------------------------------------------
# [Jucker2017] Crown diameter → DBH (pantropical)
#   DBH (cm) = (crown_diameter_m / a) ^ (1/b)
JUCKER_A: float = 0.5539
JUCKER_B: float = 0.6982

# [Feldpausch2012] DBH → Tree height (pantropical)
#   H (m) = 25.1 × (1 - exp(-0.04 × DBH))
FELDPAUSCH_A: float = 25.1
FELDPAUSCH_B: float = 0.04

# [Chave2014] DBH → AGB (pantropical)
#   AGB (kg) = 0.0673 × (ρ × DBH² × H) ^ 0.976
WOOD_DENSITY_G_CM3: float = 0.6  # tropical mixed species

# [IPCC2006] Carbon fraction of dry biomass
CARBON_FRACTION: float = 0.47

# [Stoichiometry] Photosynthesis: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂
#   O₂ produced = Carbon fixed × (32/12)
C_TO_O2_RATIO: float = 32.0 / 12.0

# Annual NPP fraction of total carbon stock
# Pan et al. (2011) — tropical forest annual carbon turnover
# Range: 0.05-0.12; using 0.05 as conservative estimate
# Reference: Pan Y. et al. (2011) A Large and Persistent Carbon
# Sink in the World's Forests. Science 333:988-993.
NPP_FRACTION: float = 0.05

# Output path for allometric results
TREE_OXYGEN_CSV: str = os.path.join(OUTPUTS_DIR, "tree_oxygen.csv")

# ---------------------------------------------------------------------------
# GIS
# ---------------------------------------------------------------------------
OUTPUT_EPSG: int = 4326

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(message)s"

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
MIN_CROWN_AREA_PX: int = 20
HEALTH_NDVI_THRESHOLD: float = 0.3
HEALTH_DENSITY_THRESHOLD: float = 0.5
