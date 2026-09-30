# 09 — Equations

> E9/E10 (FPAR/NPP-LUE) are **dead code** — constants removed from config, not in
> the submitted chain. Per IEEE discipline, exclude them and their citations from
> the paper, or replace with real allometric equations (E7–E8 now live in
> `phase4_allometric.py`, not the deprecated XGBoost script).

## E1: Circularity

- **Equation:** `C = (4 × π × A) / (P² + ε)`
- **Variables:** A = contour area (px²), P = contour perimeter (px), ε = 1e-6
- **Source file:** `phase3_feature_extraction.py:132`, `test_feature_extraction.py:96`
- **Purpose:** Shape regularity metric (1 = perfect circle)
- **Missing citation:** Standard geometric circularity index

## E2: Equivalent Crown Diameter

- **Equation:** `D = sqrt(4 × A / π)`
- **Variables:** A = contour area (px²)
- **Source file:** `test_feature_extraction.py:97`
- **Purpose:** Diameter of circle with same area as crown
- **Missing citation:** Standard geometric conversion

## E3: Crown Area (m²)

- **Equation:** `A_m2 = A_px × pixel_size_m²`
- **Variables:** A_px = area in pixels, pixel_size_m = ground sampling distance (m/px)
- **Source file:** `phase3_feature_extraction.py:142`
- **Purpose:** Convert pixel area to real-world area
- **Missing citation:** Standard unit conversion

## E4: Perimeter (m)

- **Equation:** `P_m = P_px × pixel_size_m`
- **Variables:** P_px = perimeter in pixels, pixel_size_m = GSD (m/px)
- **Source file:** `phase3_feature_extraction.py:184`
- **Purpose:** Convert pixel perimeter to real-world perimeter
- **Missing citation:** Standard unit conversion

## E5: Crown Density

- **Equation:** `D = Σ(mask_ij > 0) / (w × h + ε)` where i,j ∈ bounding box
- **Variables:** mask = binary prediction within bbox, w,h = bbox dimensions, ε = 1e-6
- **Source file:** `phase3_feature_extraction.py:138-139`
- **Purpose:** Fill ratio of bounding box
- **Missing citation:** Standard morphometric measure

## E6: Health Score

- **Equation:** `H = min(1.0, max(0.0, NDVI - T_NDVI) × D_crown / T_density)`
- **Variables:** NDVI = vegetation index, T_NDVI = 0.3, D_crown = crown density, T_density = 0.5
- **Source file:** `phase3_feature_extraction.py:172-176`
- **Purpose:** Composite vegetation health metric
- **Missing citation:** Custom composite index — no standard reference

## E7: Carbon from AGB (IPCC Default)

- **Equation:** `C = AGB × 0.47`
- **Variables:** AGB = above-ground biomass (kg), 0.47 = IPCC default carbon fraction
- **Source file:** `phase4_allometric.py` (`config.CARBON_FRACTION`)
- **Purpose:** Convert biomass to carbon stock
- **Missing citation:** IPCC 2006 Guidelines for National Greenhouse Gas Inventories (vol. 4, ch. 4)

## E8: Annual Oxygen Equivalent from Carbon

- **Equation:** `O2_annual = C × (32 / 12) × 0.05`
- **Variables:** C = carbon mass (kg), 32 = molecular weight O₂, 12 = molecular
  weight C, 0.05 = annual NPP share of standing carbon stock
- **Source file:** `phase4_allometric.py` (`config.C_TO_O2_RATIO`,
  `config.NPP_FRACTION`)
- **Purpose:** Stoichiometric conversion from carbon to O₂-equivalent, projected to
  net annual (above-ground) terms
- **Missing citation:** stoichiometry (photosynthesis) + **Pan 2011 anchor for 0.05
  (verify)**, running-equivalent already covered by title semantics

## E8b: Crown Diameter from Crown Area (used by chain)

- **Equation:** `CD = sqrt(4 × A_m2 / π)`
- **Variables:** A_m2 = crown area (m²)
- **Source file:** `phase4_allometric.py`
- **Purpose:** input crown metric for allometric chain

## E8c: DBH from Crown Diameter (Jucker 2017)

- **Equation:** `DBH = (CD / 0.5539)^(1/0.6982)`
- **Source file:** `phase4_allometric.py` (`config.JUCKER_A/B`)
- **⚠ verify constants vs source Table (manual check pending)**

## E8d: Height from DBH (Feldpausch 2012)

- **Equation:** `H = 25.1 × (1 − e^(−0.04 × DBH))`
- **Source file:** `phase4_allometric.py`

## E8e: AGB from DBH/H (Chave 2014)

- **Equation:** `AGB = 0.0673 × (ρ × DBH² × H)^0.976`, ρ = 0.6 g/cm³
- **Source file:** `phase4_allometric.py` (`config.WOOD_DENSITY_G_CM3`)

## E9: FPAR from NDVI — **REMOVED from config, do not include in paper**

- **Equation:** `FPAR = 1.24 × NDVI - 0.168`
- **Status:** Was defined but never used; constants now **deleted from `config.py`**
- **Note:** Included only to document dead-code removal — exclude from paper (E9)

## E10: NPP-Based O₂ Estimation — **REMOVED from config, do not include in paper**

- **Equation:** `NPP = FPAR × PAR × LUE`, `O2 = NPP × (32/12)`
- **Status:** Was defined but never used; constants now **deleted from `config.py`**
- **Note:** Excluded from paper (E10 dead code)

## E11: Pixel-to-Geographic Coordinate Conversion

- **Equation:** `col, row = T⁻¹ × (lon, lat)` (inverse affine transform)
- **Variables:** T = affine transform matrix from GeoTIFF
- **Source file:** `phase3_feature_extraction.py:68-74`, `phase5_gis_output.py:90-93`
- **Purpose:** Convert pixel centroids to geographic coordinates
- **Missing citation:** Standard GIS affine transformation (rasterio)

## E12: Class Weight Computation

- **Equation:** `w_c = (1 / freq_c) / Σ(1 / freq_c) × num_classes`
- **Variables:** freq_c = pixel frequency of class c, num_classes = 2
- **Source file:** `phase2_train_segformer.py:297-299`
- **Purpose:** Inverse-frequency class weighting for imbalanced segmentation
- **Missing citation:** Standard class-weighting technique

## E13: IoU (Intersection over Union)

- **Equation:** `IoU_c = TP_c / (TP_c + FP_c + FN_c + ε)`
- **Variables:** TP = true positives, FP = false positives, FN = false negatives, ε = 1e-8
- **Source file:** `phase2_train_segformer.py:170-177`, `diagnose_segformer.py:82-94`
- **Purpose:** Segmentation evaluation metric
- **Missing citation:** Standard metric (PASCAL VOC)

## E14: Dice Coefficient

- **Equation:** `Dice = 2TP / (2TP + FP + FN + ε)`
- **Source file:** `diagnose_segformer.py:94`
- **Purpose:** Segmentation evaluation metric
- **Missing citation:** Standard metric (Sørensen–Dice coefficient)

## E15: Aspect Ratio

- **Equation:** `AR = w / (h + ε)`
- **Variables:** w = bounding box width, h = bounding box height, ε = 1e-6
- **Source file:** `phase3_feature_extraction.py:135`
- **Purpose:** Crown elongation metric
- **Missing citation:** Standard morphometric measure
