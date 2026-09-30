# Georeferencing Audit — CamTreeO2 / ITOE Project

> Role: senior GIS + Remote Sensing engineer. Inspection-only. No code modified.
> Scope: `phase3_feature_extraction.py`, `phase4_allometric.py`,
> `phase5_gis_output.py`, `config.py`, `phase1_prepare_dataset.py`,
> `generate_dataset_split.py`, dataset contents, and GIS outputs.
> Date: 2026-08-06.

---

## 1. Current Coordinate Pipeline (as implemented)

```
Predicted mask PNG (tile, 256×256)
        │
        │ cv2.findContours → contour moments
        ▼
centroid_x_px / centroid_y_px        (tile-local pixel coords)
        │
        │ Phase 3: lon/lat computed via rasterio xy(img_transform)  ← transform lookup usually FAILS
        │          → lon/lat NOT saved to CSV (columns absent)
        ▼
tree_features.csv  (px coords only, no lon/lat)
        │
        │ Phase 4: allometric chain appends DBH/AGB/Carbon/O₂ columns
        ▼
tree_oxygen.csv    (px coords only, no lon/lat)
        │
        │ Phase 5: looks for lon/lat columns → NOT present
        │          looks up GeoTIFF transform cache → EMPTY (no .tif in dataset)
        │          FALLBACK fires
        ▼
lon = −78.4678 + cx_px × 1e-5        ← MOCK (Quito, Ecuador anchor)
lat = −0.1807 − cy_px × 1e-5         ← MOCK
        │
        ▼
GeoJSON / Shapefile / Folium map (EPSG:4326 labeled)
```

Every one of the 5,897 output points is the mock fallback, confirmed by data:
- Output lat ∈ [−0.183245, −0.180705], lon ∈ [−78.467795, −78.465255].
  Span ≈ 0.00254° ≈ 256 px × 1e-5° — exactly one tile's worth around the mock anchor.
- Verified against formula (T0000 t0): cx = (−78.4670681 + 78.4678)/1e-5 = 73.2 px,
  cy = (−0.1807 + 0.1828731)/1e-5 = 217.3 px — valid tile pixel coords. QED.
- Image prefixes in output include **BART** (NH), **OSBS** (FL), **SJER/TEAK** (CA),
  **JERC**, **reforestree_** (Ecuador), **synthetic_asia_** — **all** 26 sources placed
  in the same ~250 m patch near Quito, Ecuador.

---

## 2. Centroid Generation

- **File:** `phase3_feature_extraction.py`
- **Function:** `extract_features_from_mask()` (lines 151–157)
- **Type:** **contour centroid** (area centroid / center of mass of the contour's
  enclosed pixels), via image moments.

```python
# Centroid in pixel coordinates
M = cv2.moments(cnt)
if M["m00"] > 0:
    cx_px = M["m10"] / M["m00"]
    cy_px = M["m01"] / M["m00"]
else:
    cx_px, cy_px = x + w / 2, y + h / 2
```

Flow:
```
Predicted mask PNG ──(threshold >0, ×255)──► binary
      │
      └─► cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE) ──► per-contour
              │
              ├─► cv2.contourArea (px²)             → crown_area_px
              ├─► cv2.boundingRect                  → bbox, aspect_ratio, density
              ├─► cv2.arcLength                     → perimeter_px
              └─► cv2.moments                       → centroid_x_px, centroid_y_px
```

**The centroid itself is correct.** It is tile-local pixel space (col = x,
row = y), 0–256, which is consistent with the mask the model produced.

---

## 3. Pixel → Geographic Conversion

### Phase 3 (computed, then discarded)
`phase3_feature_extraction.py:162–166`:

```python
if img_transform is not None:
    try:
        lon, lat = xy(img_transform, cy_px, cx_px, offset="center")
    except Exception:
        pass
```

- `img_transform` comes from `_find_image()` which looks it up in a transform cache
  (line 255–268) keyed by basename. If `None`, **no conversion happens** — lon/lat stay
  `None`, NDVI falls back to 0.5 (line 174–176).
- Even on success, `xy()` returns coordinates in the **source CRS** (e.g., UTM
  easting/northing for NEON). No reprojection to WGS84 exists anywhere.
- **Critical: computed lon/lat are never written to the output dict** (lines
  185–200 contain only `centroid_x_px`/`centroid_y_px`, no `lon`/`lat` keys).
  `tree_features.csv` therefore has **no lon/lat columns**.

### Phase 5 (the step that actually produced the GeoJSON)
`phase5_gis_output.py:141–185`:

```python
has_geo_cols = "lon" in df.columns and "lat" in df.columns      # ALWAYS False

# Fallback reference point for mock coordinates (Quito, Ecuador — ReforesTree region)
MOCK_LAT_REF = -0.1807
MOCK_LON_REF = -78.4678
MOCK_SCALE = 0.00001  # ~1m per pixel at equator
...
if has_geo_cols:                      # never taken
    ...
if np.isnan(lon) or np.isnan(lat):    # always taken → GeoTIFF cache lookup
    tf_info = transform_cache.get(img_name)
    ...prefix match, startswith match...
    if tf_info is not None:
        lon, lat = _pixel_to_geo(cx_px, cy_px, transform)   # never reached (empty cache)
...
if np.isnan(lon) or np.isnan(lat):    # ALWAYS taken
    lon = MOCK_LON_REF + cx_px * MOCK_SCALE
    lat = MOCK_LAT_REF - cy_px * MOCK_SCALE
```

The intended chain (pixel → affine → UTM → WGS84 → GeoJSON) is **only 2 of 5 steps
present**:

| Step | Present? |
|------|----------|
| pixel → tile-local px | yes (centroid moments) |
| pixel → source-georef via affine | latent code only; never executes (empty cache) |
| tile-local → source offsets | **missing** — no tile-origin offset is added anywhere |
| UTM → WGS84 reprojection | **missing** — no pyproj / rasterio.warp in the codebase |
| → GeoJSON | yes, but fed mock lon/lat, then labeled EPSG:4326 |

---

## 4. CRS Handling

| Item | Value |
|------|-------|
| Source CRS | **unknown / never asserted** — no source GeoTIFF is read; `src.crs` is read in `_build_transform_cache` but the cache is empty. NEON DP3.30010.001 is UTM (NAD83) per product; BART tiles at UTM 19N ≈ EPSG:26919. |
| Destination CRS | WGS84 lat/lon (EPSG:4326) |
| Reprojection | **NOT performed.** `pyproj` appears **nowhere** in the codebase (grep: 0 hits). `rasterio.warp` not used. |
| `config.OUTPUT_EPSG = 4326` | only labels the output GeoDataFrame (`phase5_gis_output.py:222`); it never transforms any coordinate. |

Even in the hypothetical "success" path, the output of `rasterio.transform.xy()`
would be UTM easting/northing written into `latitude`/`longitude` columns and
labeled EPSG:4326 — a silent double error (no reprojection + wrong CRS tag).

---

## 5. Tile Metadata

**Nothing survives tiling.** `phase1_prepare_dataset.py`:

- `tile_image()` (lines 213–246) crops with PIL `Image.crop()` and stores only:
  `dataset`, `image_id`, `tile_id`, `foreground_ratio`. **No affine, no CRS, no
  pixel size, no source geotransform.**
- `save_tiles()` (lines 339–353) writes plain PNG files:
  `dataset/{split}/images/{tile_id}.png`. PNG has no georeferencing; **no sidecar
  (.pgw / .aux.xml / .json) is written.**
- `generate_split_json()` (lines 360–388) keeps `image`, `mask`, `foreground_ratio`,
  `image_id`, `tile_id` only.
- The pre-existing splits used for val/test (`source: pre_split_val`,
  `pre_split_test` in `dataset_split.json`) are user-placed PNG tiles — likewise
  carry no geo metadata.

Result: **tile-local pixel coordinates cannot be mapped back to any ground
location** from the artifacts on disk. The only latent recovery path is the
tile-id naming convention: `{image_id}_t{tile_index}` (tile origin =
tile_index → (col,row) = (index % cols, index // cols) × 256), provided the
original georeferenced image is obtainable.

Dataset inventory confirms the dead end:
- `dataset/neon_rgb/images/`: **1 file** — `OSBS_029.png` (DeepForest sample).
- Recursive search for `*.tif`/`*.tiff` under `dataset/`: **0 files** (so
  `LANDSAT_TIF = dataset/landsat8/ndvi.tif` also does not exist → NDVI = 0.5).
- `dataset/val/images` + `masks`: 3,720 PNG tiles (BART), georeferencing absent.

---

## 6. CSV Export

### `outputs/tree_features.csv` (Phase 3) — columns:
`image, tree_id, crown_area_px, crown_area_m2, equiv_crown_diameter_m, perimeter_px,
perimeter_m, circularity, aspect_ratio, crown_density, ndvi, health_score,
centroid_x_px, centroid_y_px`

- Contains **pixel coordinates** (`centroid_x_px`, `centroid_y_px`, tile-local).
- **No latitude/longitude, no UTM, no CRS.** The lon/lat computed in
  `extract_features_from_mask()` are never emitted.

### `outputs/tree_oxygen.csv` (Phase 4) — columns:
`image, tree_id, crown_area_px, crown_area_m2, equiv_crown_diameter_m, perimeter_px,
perimeter_m, circularity, aspect_ratio, crown_density, ndvi, health_score,
centroid_x_px, centroid_y_px, estimated_dbh_cm, estimated_height_m, estimated_agb_kg,
estimated_carbon_kg, estimated_o2_stock_kg, estimated_o2_annual_kg_yr`

- Same pixel coords, allometric estimates appended. **Still no lon/lat.**

### `gis/shapefiles/tree_oxygen.csv` (Phase 5) — columns:
`tree_id, latitude, longitude, crown_area_m2, equiv_crown_diameter_m, estimated_dbh_cm,
estimated_agb_kg, estimated_carbon_kg, estimated_o2_stock_kg, estimated_o2_annual_kg_yr,
o2_category, ndvi, health_score`

- **latitude/longitude come from the MOCK formula** (`MOCK_LAT_REF`,
  `MOCK_LON_REF`, `MOCK_SCALE`), i.e., derived from pixel coords via a hardcoded
  Quito, Ecuador anchor. The header comment in `phase5_gis_output.py` ("no random
  coordinates are generated") contradicts the actual code path.

---

## 7. GeoJSON Export

`phase5_gis_output.py:218–231`:

```python
geometry = [Point(xy) for xy in zip(out_df["longitude"], out_df["latitude"])]
gdf = gpd.GeoDataFrame(out_df, geometry=geometry, crs=f"EPSG:{config.OUTPUT_EPSG}")
gdf.to_file(geojson_path, driver="GeoJSON")
```

- Geometry: shapely `Point(longitude, latitude)` — order (x=lon, y=lat), correct
  convention.
- **Value traceback for one point:**

```
latitude  −0.1828731
  ← out_df["latitude"] = round(lat, 7)
      ← lat = MOCK_LAT_REF − cy_px × MOCK_SCALE        (line 185)
           cy_px = 217.3  (contour moment, tile-local row, Phase 3)
           MOCK_LAT_REF = −0.1807   (hardcoded, Quito, Ecuador)
           MOCK_SCALE   = 0.00001   (hardcoded ~1 m/px)
longitude −78.4670681
  ← lon = MOCK_LON_REF + cx_px × MOCK_SCALE            (line 184)
           cx_px = 73.2  (contour moment, tile-local col, Phase 3)
           MOCK_LON_REF = −78.4678  (hardcoded, Quito, Ecuador)
```

- The real source location of that image (`2018_BART_4_322000_4882000_image_crop`,
  Bartlett Experimental Forest, NH) is ≈ 44.06°N, −71.29°W — **~4,900 km away**
  from where the GeoJSON places it.

---

## 8. Root Cause Analysis

**Primary root cause: E — coordinates are hardcoded (mock fallback).**

`phase5_gis_output.py:144–147, 182–185` hardcodes a Quito, Ecuador anchor
(−0.1807, −78.4678) and fabricates every point from pixel centroids when no real
georeferencing resolves. The docstring's claim of "no random coordinates" is false
for the executed path.

**Contributing causes (all real, in order of severity):**

1. **D — Tile metadata lost.** Tiling to 256×256 PNG discards geotransform, CRS,
   pixel size, and source georef; no sidecar is preserved. There is no way to
   reconstruct ground coordinates from the artifacts on disk.
2. **B — GeoTransform belongs to the wrong (actually absent) dataset.** The
   transform cache scans `dataset/neon_rgb/images/` for `*.tif` — that directory
   holds a single `OSBS_029.png` (DeepForest sample). Zero GeoTIFFs exist in the
   project; `_build_transform_cache` therefore always returns an empty dict.
3. **C — CRS conversion is missing.** No `pyproj`/`rasterio.warp` anywhere. Even a
   successful transform lookup would emit UTM easting/northing mislabeled as
   lon/lat (EPSG:4326 tag is cosmetic only).
4. **A (latent) — Pixel→geo conversion is incomplete.** `_pixel_to_geo` applies the
   source image transform to *tile-local* pixel coordinates with no tile-origin
   offset; coordinates would be off by multiples of 256 px (25.6 m at 0.1 m GSD)
   even if the transform were found.
5. **Phase-3 gap:** lon/lat computed in `extract_features_from_mask` are discarded —
   `tree_features.csv` never carries them, forcing Phase 5 to re-derive everything
   from nothing.

**Most likely single root cause:** **E** (mock/hardcoded coordinates in
`phase5_gis_output.py`), enabled by **D** + **B** (no georeferenced source imagery
in the pipeline). The author appears to have intended a real path but the
georeferencing inputs never existed, so the fallback silently produced fabricated
locations for all 5,897 trees.

---

## 9. Required Fix (scope only — implementation NOT specified per audit brief)

Fix requires, in order:
1. Obtain true source georeferencing: original NEON GeoTIFFs (DP3.30010.001) and
   ReforesTree imagery, each with valid affine + CRS (NEON ≈ UTM/NAD83 zones;
   ReforesTree ≈ UTM 17S, WGS84). None are currently in the repository.
2. Persist per-tile georeferencing at tiling time (or a tile→source transform map
   keyed by `image_id` + tile index), OR reconstruct it from `{image_id}_t{idx}`
   naming + source transform.
3. Add source-CRS → WGS84 reprojection (pyproj / rasterio.warp).
4. Add the tile-origin offset before applying the source affine.
5. Remove the mock fallback or gate it behind an explicit `--mock` flag that never
   ships to paper outputs.
6. Re-derive `pixel_size_m` per source from real GSD metadata (currently default
   0.1 m/px because no GeoTIFF is found; if true GSD differs, **crown areas and all
   O₂ numbers change**).

---

## 10. Retraining Required? — **NO**

The SegFormer segmentation model is **not** involved in any georeferencing step.
Crown pixel locations and (at fixed GSD) crown areas are unaffected.

Pipeline stages that must be **re-run** (in order), using corrected georeferencing:

```
Phase 3  (feature extraction — emit lon/lat columns per tree)
   ↓
Phase 4  (allometric — carry lon/lat through to tree_oxygen.csv)
   ↓
Phase 5  (GIS export — real coordinates + reprojection)
```

Phase 5 alone would suffice **if and only if** lon/lat columns were added to
`tree_oxygen.csv` by an intermediate step; otherwise start at Phase 3.

**Caveat:** if true per-site GSD ≠ 0.1 m/px (the current default), then
`crown_area_m2` and therefore every allometric/O₂ figure change. In that case the
O₂ *results* change too — but still **no retraining**, only Phase 3 → 4 → 5 re-runs.

---

## 11. Affected Paper Sections

| Section | Change required |
|---------|-----------------|
| **Methodology** (georeferencing / feature extraction) | Rewrite. Remove claims of "real GeoTIFF georeferencing — no random coordinates". Describe true tile→source mapping + CRS reprojection, or state that the paper is **imagery-only** (no georeferenced output) if sources cannot be obtained. |
| **GIS Output / Maps** | All coordinates, GeoJSON, Shapefile, Folium maps, and heatmaps are wrong. Either regenerate from real georeferencing or **remove the GIS section entirely** from the paper. |
| **Results** | Per-site analysis (BART/OSBS/SJER/TEAK), geographic claims, and total O₂ numbers are affected **if real GSD differs from 0.1 m/px**; re-verify. NDVI/health values (all 0.5 / ≤0.4) must be dropped or fixed. |
| **Discussion / Limitations** | Add explicit limitations: no CRS reprojection, assumed 0.1 m GSD, pantropical allometry on temperate sites, no field validation of O₂. |

**Bottom line for the paper:** with the current repo, no geographic coordinate in
any output is trustworthy. Options: (a) acquire real source georeferencing and
re-run Phases 3–5 (no retraining), or (b) present the pipeline as
**imagery-based tree segmentation + allometric O₂ estimation without georeferenced
GIS output**, and delete all maps/coordinates/NDVI content from the submission.

---

## Appendix — Evidence Log

| Evidence | Finding |
|----------|---------|
| `gis/shapefiles/tree_oxygen.geojson` (5,897 features) | All points in lat [−0.1832, −0.1807], lon [−78.4678, −78.4653] ≈ 250 m patch at Quito anchor |
| `tree_id` prefixes | BART/OSBS/SJER/TEAK/JERC (US), reforestree_ (EC), synthetic_asia_ — all at same mock patch |
| Point–formula round-trip (T0000 t0) | cx=73.2, cy=217.3 px → matches mock formula exactly |
| `dataset/neon_rgb/images/` | only `OSBS_029.png` (DeepForest sample) |
| Recursive `*.tif*` under `dataset/` | 0 files (incl. `landsat8/ndvi.tif` → NDVI=0.5 confirmed) |
| `dataset/val|test` | PNG tiles only; `dataset_split.json` fields: image/mask/source — no geo metadata |
| Grep `pyproj` | 0 hits in project |
| Grep `rasterio.warp` | 0 hits |
| `tree_features.csv` / `tree_oxygen.csv` headers | no lon/lat columns |
| `phase5_gis_output.py:144-147,182-185` | hardcoded mock anchor + scale |
