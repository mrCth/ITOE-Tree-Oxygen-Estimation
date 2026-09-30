# Phase 3 & 4 — Feature Extraction + Allometric Estimation (IEEE Paper Summary)

## Phase 3: Per-Tree Crown Feature Extraction (`phase3_feature_extraction.py`)

### Input
Segmentation prediction masks (binary {background, tree}) from Phase 2.
Source priority: `outputs/masks/` → val masks → train masks.
Sample GeoTIFF in `dataset/neon_rgb/images/` provides ground-sampling distance
(pixel_size_m, default 0.1 m/px = ~10 cm/px NEON RGB).

### Processing (OpenCV connected components)
1. Binarize mask → `cv2.findContours` (RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
2. Filter contours with `crown_area_px < 20` (config `MIN_CROWN_AREA_PX`)
3. Per crown compute:
   - **Crown Area**: `area_px × pixel_size²` (m²)
   - **Equivalent Crown Diameter**: `D = sqrt(4·area/π)` (m) — basis for allometry [Jucker2017]
   - **Perimeter** (px → m via pixel_size)
   - **Circularity**: `4π·area / perimeter²`
   - **Crown Density**: fraction of mask pixels inside bounding box
   - **Aspect Ratio**: bbox width/height
   - **NDVI**: queried at geographic centroid (pixel→geo via rasterio affine) from
     Landsat-8 NDVI raster (`dataset/landsat8/ndvi.tif`)
   - **Health Score**: `min(1, max(0, NDVI−0.3) × density/0.5)` — composite of NDVI & density
   - Centroid (px + lon/lat via affine transform)

### Output
`outputs/tree_features.csv` — 5,897 tree crowns extracted (7/28/2026 run).

| Feature | Mean | Median | Std |
|---------|-----:|-------:|----:|
| Crown area (m²) | 88.08 | 34.83 | 138.63 |
| Crown diameter (m) | 8.47 | 6.66 | 6.36 |
| Crown density | 0.968 | 1.00 | 0.095 |

### CAVEAT (important for paper)
NDVI column is **constant 0.5** in the current run — the Landsat NDVI geo-query
failed (CRS mismatch / transform lookup), so all NDVI values fell back to the neutral
0.5 default. Health score is therefore degenerate (~0.39 constant). Any analysis
using NDVI/health from this run is **not meaningful** — must be fixed before publication.

---

## Phase 4: Allometric Carbon & Oxygen Estimation (`phase4_allometric.py`)

### Pipeline (deterministic, no ML)
```
Equivalent Crown Diameter (m)
  → DBH (cm)   [Jucker et al. 2017]:      DBH = (D / 0.5539)^(1/0.6982)
  → Height (m) [Feldpausch et al. 2012]:  H = 25.1 × (1 − e^(−0.04·DBH))
  → AGB (kg)   [Chave et al. 2014]:       AGB = 0.0673 × (ρ·DBH²·H)^0.976
  → Carbon (kg)[IPCC 2006]:               C = AGB × 0.47
  → O₂ (kg)    [Photosynthesis stoichiometry]: O₂ = C × 32/12
```

Two O₂ outputs per tree:
- **O₂ stock** (kg): total O₂ equivalent of standing carbon stock
- **O₂ annual** (kg/yr): `C × NPP_fraction(0.05) × 32/12` — using NPP/AGB ratio from
  Pan et al. (2011, Science 333:988)

### Constants
| Constant | Value | Source |
|----------|-------|--------|
| Jucker a, b | 0.5539, 0.6982 | Jucker 2017, GCB |
| Feldpausch | 25.1, 0.04 | Feldpausch 2012, Biogeosciences |
| Wood density ρ | 0.6 g/cm³ | tropical mixed species |
| Carbon fraction | 0.47 | IPCC 2006 |
| C→O₂ ratio | 32/12 | stoichiometry |
| NPP fraction | 0.05 | Pan et al. 2011 |

### Output `outputs/tree_oxygen.csv` (5,897 trees)
| Estimate | Mean | Median | Min | Max |
|----------|-----:|-------:|----:|----:|
| DBH (cm) | 57.50 | 35.22 | 0.91 | 286.5 |
| Height (m) | 17.46 | 18.97 | 0.89 | 25.10 |
| AGB (kg) | 5,301 | 755 | 0.03 | 59,412 |
| Carbon (kg) | 2,492 | 355 | 0.01 | 27,924 |
| O₂ stock (kg) | 6,644 | 947 | 0.04 | 74,463 |
| O₂ annual (kg/yr) | 332 | 47 | 0.002 | 3,723 |
| **Σ O₂ annual (all trees)** | | | | **1,959,074 kg/yr** |

---

## Supplementary: ML Regression Attempt (XGBoost, `phase4_train_xgboost.py.bak`)
Supervised approach predicting per-tree O₂ from crown features (crown_area_m2,
perimeter_m, circularity, crown_density, ndvi, health_score) using ReforesTree
field data (AGB→carbon→O₂) as ground truth.

**Result is NOT usable**: only **4 training samples** matched (`n_train=4`),
MAE=26.17, RMSE=32.99, **R²=0.165**. Feature importance: crown_density (0.45),
crown_area_m2 (0.26), perimeter_m (0.17). Deprecated in favor of the deterministic
allometric chain. Recommend **excluding from the paper** or reporting as a failed
preliminary due to insufficient field-matched training data.

---

## Paper-Ready Key Points
1. Segmentation → connected-component analysis → per-crown geometric features.
2. Deterministic pantropical allometric chain (Jucker→Feldpausch→Chave→IPCC) gives
   per-tree DBH, height, AGB, carbon, and O₂ production from crown diameter alone.
3. Reproducible: constants centralized in `config.py`, no randomness.
4. **Must-fix before submission**: (a) NDVI geo-query (currently constant 0.5),
   (b) XGBoost model too weak (4 samples) — omit or replace with more field data.
