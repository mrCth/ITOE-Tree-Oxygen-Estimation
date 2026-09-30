# Dataset Audit Report
Read-only inventory of `NeonTreeEvaluation/` and `reforestree_dataset/` for ground-truth availability (segmentation + DBH/AGB/oxygen validation). No files modified; no training run.

Date: 2026-09-28. Method: pandas/rasterio/geopandas/PIL inspection of every CSV, GeoTIFF, shapefile, H5, KML/TFW, XLSX.

---

## 1. Summary verdict

| Validation target | NEON | ReforesTree (field) | Paper implication |
|---|---|---|---|
| Crown delineation (box-level GT) | **Have**: 47 tiles RGB + 6,635 Pascal-VOC boxes + csv | **Do NOT have**: labels are DeepForest prediction outputs (score col), no hand-drawn boxes | "ReforesTree F1 0.86" measures model vs DeepForest-prediction agreement, not field truth. NEON box IoU is the only independent segmentation reference. |
| Crown → DBH (allometric input) | **None**: no per-tree DBH/AGB/species/height | **Subset**: 4,699 GPS points, DBH (circumference/π) on 2,329 trees; **only 171 distinct ground trees linked** to detections in `final_dataset.csv` (GPS radius ~4.3 m) | DBH validation restricted to ~171 matched trees carried by repo data; chain corr(X_d,X_g)=0.9455 ⇒ geolocation noise ~5%. |
| AGB | None | **Derived, not measured**: 99.8% of (species, DBH) pairs map to a unique AGB ⇒ AGB = pure function of DBH (+species) | AGB column cannot independently validate DBH→AGB; it re-uses a DBH allometry. carbon = AGB×0.39 (exactly). |
| Height | None | 230 valid (median 2.0 m; max 2003 = spurious); rest zero | Effectively unusable. |
| Oxygen / carbon flux | None | None (only via AGB→carbon lookup) | Cannot be validated; remains pure derivation. |

---

## 2. NEON — `NeonTreeEvaluation/`

- RGB GeoTIFFs: real CRS (EPSG:32617 / 32619 / 32611) + GeoTransform + **res 0.10 m** stored in raster metadata ⇒ **GSD 0.1 m is verified for NEON**.
- Box annotations: per-image Pascal-VOC XML (objects `tree`) + `benchmark_annotations.csv` (6,635 boxes).
- XML "height" hits = ROI pixel-height attributes (false positive), NOT tree height.
- No DBH / AGB / biomass / species / per-tree measurement anywhere in repo.
- Hyperspectral + LiDAR/CHM layers exist per tile but carry no tree-level GT.

**Useable GT**: detection/segmentation boxes for IoU/F1 on NEON evaluation sets.

## 3. ReforesTree — `reforestree_dataset/`

### 3.1 Imagery
- 6 plots (P2, P3, P6, P7, P8, P10), Ecuador (-2.18..-1.13 lat, -79.59..-79.54 lon).
- 7 orthomosaics GeoTIFF (EPSG:4326 = WGS84 geographic, 4-band RGBA uint8, 530–819 MB/file). Pixel size ≈ 8.7e-8–1.07e-7 deg ≈ **~0.9–1.1 cm/px source GSD** (from raster transform + TFW + KML bounds, mutually consistent).
- 105 tiles PNG 4000×4000 RGB (per plot 12–25); tile GSD implied ≈ 0.6–0.7 cm/px (`tiles/ortho_data.csv` ratio columns).
- **GSD conflict with paper**: manuscript assumes 0.1 m/px pixel size for ReforesTree crown-area→m². Source metadata implies ~1 cm/px ⇒ ~80× area-factor discrepancy (O2 estimates scale with GSD²). Direction/magnitude not re-verified (no re-run allowed); flagged for a pixel-size check before finalizing numbers.

### 3.2 Field inventory — `Final_Trees.shp` (4,699 points, EPSG:4326)
- `_Gps_1_*`: GPS lat/lon + **GPS precision ~4.3 m** ⇒ coordinates cannot link crown centroid to trunk at better than ±4 m.
- `Variedad_1` species (Cacao 1,808 + "Cacao de la muestra" 235 ≈ field_data cacao count; Musacea 1,510; Guaba 600; Mango 89; ...).
- `Diámetro` (>0: 2,329; med 37.4; max 663) vs `Diameter` (>0: 2,329; med 11.9; max 211): `Diameter = Diámetro / π` exactly ⇒ **`Diámetro` = circumference (cm), `Diameter` = DBH**.
- `Diámetr_1..3` / `Diamet_1..3`: separate measure logs (306/209/41 records >0), `Ave_Diamet` 2,633 >0 (med 11.2).
- `Altura_del`: 230 >0 (med 2.0, max 2003 = outlier) ⇒ height not usable.
- `Plot` (6), `Plant_Yr` (e.g. 2016).

### 3.3 Derived tables
- `field_data.csv` (4,663 rows) = Final_Trees + derived cols:
  - `diameter` zero-filled for cacao (2,042 rows =0, all have `updated diameter` >0); `updated diameter` 4,663 >0 (med 7.96, max 67.8) = usable DBH.
  - `AGB` 4,663 >0 (med 5.4 for cacao constant within identical DBH; global med 12.2) but **99.8% of (species, rounded DBH) pairs have a single AGB value ⇒ AGB is a computed lookup, not independent biomass**.
  - `carbon = AGB × 0.39` exactly.
  - `height` 230 >0 (med 2.0, max 2003) ⇒ unusable.
- `annotations/*.csv`: `all_annotations.csv` 8,520; per-plot 978–2,026; `cleaned/clean_annotations.csv` 7,969. Rows are **detection outputs** (cols `score`, `is_banana`, geographic `lon/lat`) ⇒ masks/training labels for ReforesTree are DeepForest predictions, not GT.
- `mapping/final_dataset.csv`: 4,663 detections joined to ground trees. **Only 171 distinct `tree_id` matched** (747 `ground_index`); avg 27.3 boxes/tree, one tree 2,991 boxes ⇒ massive detection over-segmentation; corr(X_d, X_g) = 0.9455.
- `Referencia.xlsx`: species common-name ↔ scientific-name lookup only (Theobroma cacao, ...). No AGB/BEF tables.
- `model/deepforest/final_model_4000_epochs_35.h5`: DeepForest (RetinaNet, resnet50-style keys) weights — the detector that produced the annotation CSVs.
- `archive.zip`: mirror of tiles (106) + wwf_ecuador (37) + annotations (8). No extra data.

---

## 4. Validation-feasibility matrix vs manuscript claims

| Paper element | Feasible validation from repo | Status |
|---|---|---|
| Segmentation F1/IoU on NEON_test / NEON_val | Yes, box-level IoU (independent) | ✅ verified numbers already reported |
| Segmentation "ReforesTree_test F1 0.8599" | Only vs DeepForest prediction masks (self-consistent, circular) | ⚠️ label prose needed |
| Crown area → DBH (Chave/Jucker) | ~171 GPS-linked trees subset (DBH present), GPS prec 4.3 m | ⚠️ limited, geometry-noise bound ~5–10% |
| DBH → AGB (Chave/Huang) | AGB = derived lookup, not measured; equations not independently verifiable in repo | ⚠️ state as computed, not validated |
| Height-based allometry (Jucker H) | Height GT unusable (230 valid) | ❌ must not claim |
| Oxygen equivalents | No GT anywhere | ❌ derivation-only |
| GSD 0.1 m for ReforesTree | Contradicted by source rasters (~1 cm/px) | ❌ must resolve before insertion |

## 5. Recommended honest wording (no file edits made)
- ReforesTree F1: "agreement with the DeepForest-derived reference masks", never "field-validated segmentation".
- DBH: "allometry applied to crowns matched to a subset of GPS-censused trees (n≈171, GPS error ±4.3 m)"; report as limited validation.
- AGB/oxygen: "computed via published allometries, no in-situ biomass reference available"; no independent accuracy claim.
- GSD: state ReforesTree orthomosaic native ~1 cm/px; audit could not confirm the 0.1 m assumption ⇒ keep missing-info item as-is (do not insert derived totals).

## 6. Reviewer-3 evidence recommendation
- NEON: cite benchmark corroboration (Weinstein et al. 2019) + local XML/CSV counts (6,635 boxes) as independent delineation GT.
- ReforesTree: present inverse plots + the n≈171 matched-tree DBH comparison (corr 0.9455 + per-plot counts) as the honest DBH evidence; do NOT promote AGB/oxygen accuracy.
- Flag openly that AGB is derived (not field biomass) and height GT is absent ⇒ allometric uncertainty (Reviewer 1) must be discussed, not asserted.

## 7. Blockers / not verified
- Re-run needed to quantify real-GSD effect on crown areas/O2 (0.1 m vs ~1 cm/px) — outside read-only scope.
- Jucker/Chave/Feldpausch coefficient parity vs primary tables still manual-verification pending — do not assert.
- Height max 2003 and zero-filled Diameter rows → exclude from any analysis claim.