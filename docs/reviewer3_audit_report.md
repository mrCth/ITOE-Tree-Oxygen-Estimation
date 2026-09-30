# Reviewer 3 Follow-up: Read-Only Technical Audit

Scope: `camptreeO2` repo. No files modified, no model run, no figure regeneration, no manuscript edit.
All statistics computed from existing CSV/annotation files; every methodological choice stated explicitly before its result.

---

# A. DBH Validation

## A.0 Representation of the ~171 matched ground trees
- Source: `reforestree_dataset/mapping/final_dataset.csv` (4,663 rows). This file is **precomputed**; no script in the repo produces it (closest is `merge_test.py`, a diagnostic that only writes `outputs/match_distance_histogram.png`).
- Each row = one DeepForest **detection box** joined to one field tree. `tree_id` = ground-tree identifier (documented as the ground inventory id), `ground_index` = another ground reference id.
- No pipeline script (phase1–phase5) reads `final_dataset.csv`. There is therefore **no existing per-tree DBH comparison and no existing aggregation method** in the repo.

Requested counts:
- Unique ground `tree_id`: **171** (distinct site+tree_id pairs: 174).
- Detection rows: **4,663** (`tree_id` non-null for all; `ground_index` non-null for all, 12 rows have `ground_index <= 0`).
- Detections per tree: min **2**, median **10**, mean **27.27**, max **2,991**.
- One tree dominates: **`tree_id = 0.0` absorbs 2,991 of 4,663 detections (64.1%)** → matching artifact, not plausible crown multiplicity.

## A.1 Pipeline DBH estimate
Methodological choice (stated explicitly): the repo's *crown-area-dependent* predicted DBH path is phase3+phase4 (mask → crown area → Jucker DBH). `final_dataset.csv` contains no crown area and no predicted DBH. To compute a DBH per matched detection, the only available quantity is the **detection bounding box**: `crown D = 2·sqrt(box_area_m2 / π)`, `DBH = (D/0.5539)^(1/0.6982)`. This is a **diagnostic proxy**, NOT the manuscript's method (the manuscript uses segmentation-mask crowns, not DeepForest boxes). It is used here only because it is the sole reproducible route through `final_dataset.csv`.
Second explicit choice: box to m² requires a GSD. Reported for **two** assumptions — `GSD_SOURCE 0.011 m/px` (ReforesTree orthomosaic native) and `GSD_DEFAULT 0.100 m/px` (the pipeline's global assumption). No aggregation exists → alternative B (median per tree) and C (mean per tree) computed and labelled as diagnostics; alternative A (primary match) does not exist (no primary-match flag in file).

Reference DBH: `diameter` (= field `Diámetro / π`, real measured DBH, median 11.9 cm) preferred; rows with `diameter == 0` are the 2,042 cacao records whose only value is `updated diameter` (~6.84 constant → **placeholder**, treated as not-a-measurement). Statistics below given for `measured_only` (primary) and `measured_or_updated` (all) separately.

## A.2 Results — per matched ground tree (aggregated)

### GSD_SOURCE = 0.011 m/px (native orthomosaic approx.)
| Aggregation | n | MAE | RMSE | Bias | MedAE | Pearson r | Spearman rho | R² | pred mean | ref mean | pred med | ref med | pred [min,max] | ref [min,max] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_median, measured_only | 170 | 4.246 | 5.141 | +2.650 | 3.992 | −0.089 | −0.156 | −1.39 | 9.77 | 12.42 | 9.50 | 12.48 | [5.5, 30.4] | [3.5, 20.4] |
| C_mean, measured_only | 170 | 4.103 | 5.295 | +1.655 | 3.566 | −0.040 | −0.086 | −1.54 | 10.77 | 12.42 | 9.78 | 12.48 | [5.5, 34.6] | [3.5, 20.4] |
| B_median, measured_or_updated | 171 | 4.026 | 4.954 | +2.548 | 3.585 | −0.041 | −0.107 | −1.21 | 9.80 | 12.35 | 9.44 | 12.41 | [6.1, 31.9] | [4.1, 20.4] |
| C_mean, measured_or_updated | 171 | 3.920 | 4.996 | +1.462 | 3.407 | −0.040 | −0.075 | −1.25 | 10.89 | 12.35 | 10.09 | 12.41 | [6.1, 33.5] | [4.1, 20.4] |

### GSD_DEFAULT = 0.100 m/px (pipeline assumption)
B_median measured_only: n=170, MAE **218.2**, RMSE **226.9**, bias −218.2, Pearson **−0.089**, R² **−4655**. Predicted DBH 129–718 cm (physically impossible). Result is fully GSD-dominated.

## A.3 Per-plot (B_median, measured_only, GSD 0.011)
| Plot | n | MAE | RMSE | Bias | ref median |
|---|---|---|---|---|---|
| Carlos Vera Arteaga RGB | 75 | 3.954 | 4.950 | +2.551 | 12.73 |
| Carlos Vera Guevara RGB | 94 | 4.497 | 5.308 | +2.784 | 12.41 |
| Flora Pluas RGB | 1 | 2.563 | 2.563 | −2.563 | 11.14 |

(GSD 0.1 per-plot: MAE 214–312 cm, all negative bias.)

## A.4 Interpretation (factual)
Predicted vs reference DBH shows **no monotonic/linear agreement** (Pearson ≈ −0.04…−0.16, R² ≤ 0 even before the GSD issue is considered — R² negative at both GSD settings means predictions are worse than a constant). Diagnostic MAE 3.9–4.5 cm at source-GSD is coincidental scale alignment of the box proxy, not model-validated DBH. **Do not report as validation evidence.**

---

# B. DBH Matching Reliability

1. **Matching rule**: not identifiable from the repo — `final_dataset.csv` is precomputed; the join code is absent. No radius, no nearest/plot/coordinate rule is recorded anywhere in repo code.
2. **Radius**: none stored. Distance (detection `X_d`,`Y_d` vs ground `X_g`,`Y_g`, same unit space): in px quantiles min 39.7 / median 2,213.8 / p95 3,865.3 / p99 4,281.6 / max 5,156.5. In metres **@0.011**: 0.44 / 24.4 / 42.5 / 47.1 / 56.7 m; **@0.0065**: 0.26 / 14.4 / 25.1 / 27.8 / 33.5 m. `X_d–X_g` corr 0.9455, `Y_d–Y_g` corr 0.9233.
3. **One detection → many ground trees**: yes structurally; not de-duplicated (747 distinct `ground_index` vs 171 distinct `tree_id`).
4. **Many detections → same ground tree**: yes — mean 27.3, max 2,991 per tree.
5. **One-to-one**: no.
6. **Rule**: undeterminable from repo (see 1).
7. **Ambiguous matches**: the tree-0 cluster (2,991 detections = 64% of all rows) is one large over-segmentation/overlap artifact; 12 rows have `ground_index ≤ 0`.
8. **Unmatched ground trees**: ≈ field inventory (4,663) − 171 matched unique `tree_id` (**≈ 4,492** not present; Final_Trees.shp has 4,699 points).
9. **Unmatched detections**: effectively none by construction (all rows carry ground ids), but 12 have `ground_index ≤ 0`.

**Limitation (required statement)**: median detection↔ground offset ≈ 14–24 m is **far beyond the 4.3 m GPS precision**; these cannot be claimed as precise tree-level matches. Only three plots participate (Vera Arteaga, Vera Guevara, Flora Pluas — no P2/P6/P7/P8/P10 trees matched).

---

# C. AGB Audit

- `reforestree_dataset/field_data.csv` AGB (4,663 records): **calculated, not measured**.
  - Unique (species, rounded-DBH) combinations: **961**; fraction mapping to a **single** AGB value: **1.0** (deterministic reconstruction from species + DBH possible).
  - `carbon = AGB × 0.39` exactly for all rows.
  - cacao records: `diameter = 0`, `updated diameter` constant ≈6.84, AGB constant ≈5.44 (placeholder-driven).
  - `corr(AGB, updated diameter)` = 0.40 (noise from majors placeholder).
- **No repo script computes this AGB** (only `_create_field_data.py`, which generates the *synthetic* OSBS `dataset/reforestree/field_data.csv` as `AGB = crown_m2 × U(5,15)` — a different, synthetic file; do not confuse the two).
- No independent biomass measurements exist in this repo (no destructive sampling, no independent AGB reference).

**Verdict: AGB independent validation = NO.** Any "AGB validation accuracy" obtained by comparing pipeline AGB (derived from crown→DBH) with field AGB (derived from the same species + DBH) is circular.

---

# D. ReforesTree F1 / IoU Audit (F1 0.8599, IoU 0.7543)

- **Evaluation script**: `evaluate_honest_sets.py`, model `models/segformer/stage2_v3`, set `Ref_test` (2,726 tiles from `dataset/dataset_split.json`), output `outputs/evaluation_all/honest_v3_final_summary.csv` (+ per-tile csv). Reproduced exactly from the per-tile file (P 0.8583, R 0.8616, F1 0.8599, IoU_tree 0.7543).
- **Definition**: micro **pixel-level** metrics, 256×256 upsampled argmax vs reference mask thresholded at 0. No IoU threshold, no object-level matching. F1 = `2·TP/(2·TP+FP+FN)` (line 125, equivalent to 2PR/(P+R)).
- **Reference mask source**: `dataset/test/masks/` (ReforesTree test tiles) derived from `dataset/reforestree/masks/` (105 original 4000×4000 PNG masks) → Phase 1 tiling. The 105 ReforesTree masks rasterize the **DeepForest box predictions** shipped as `reforestree_dataset/annotations/*.csv` / `cleaned/clean_annotations.csv` (7,969 boxes, every row carries a `score`, no species, no manual-label flag). Mask components are merged fulle-rect boxes (verified: sampled masks yield 5–18 components, near-rectangle fill; e.g. Flora tile `RGB_5_3800_0_7800_4000` = 110 boxes in CSV).
- **No manual/field crown polygon ground truth exists** for ReforesTree in this repo (field data = GPS points only).

**Required statement**: *The ReforesTree evaluation is agreement with DeepForest-derived reference annotations, not independent manual/field ground truth.*

---

# E. NEON F1 / IoU Audit (F1 0.3686, IoU 0.2260)

- **Evaluation script**: same `evaluate_honest_sets.py`, set `NEON_test` (305 tiles). Reproduced exactly (P 0.4836, R 0.2978, F1 0.3686, IoU_tree 0.2260).
- **Ground truth source**: NEON box annotations (Pascal-VOC XML → `benchmark_annotations.csv`, 6,635 boxes; `json_to_mask.py` rasterizes Supervisely JSON boxes for the neon_rgb splits). **Independent of the SegFormer predictions.** 
- **Type**: pixel-level micro on box-filled reference masks; no threshold/matching beyond `mask>0`; F1 = `2·TP/(2·TP+FP+FN)` (`evaluate_honest_sets.py:125`).
- **F1 = 2PR/(P+R)** formula verified identical to the implementation; implemented/documented at `evaluate_honest_sets.py:125` and re-derived from per-tile TP/FP/FN.

---

# F. All F1 Implementations

| file:line | function/context | formula | correct | used for manuscript |
|---|---|---|---|---|
| evaluate_honest_sets.py:125 | micro | 2TP/(2TP+FP+FN) | ✓ | **YES** (0.8599 / 0.3686) |
| evaluate_full_metrics.py:139 | micro | 2TP/(2TP+FP+FN) | ✓ | no |
| evaluate_all_models.py:86,95 | per-tile & micro | 2PR/(P+R) | ✓ | no |
| evaluate_reforestree_val.py:81,88 | per-tile & micro | 2PR/(P+R) | ✓ | no |
| evaluate_segformer.py:78 | per-tile | 2PR/(P+R) | ✓ | no |
| phase2_train_segformer_v2.py:299 | val loop | 2PR/(P+R) | ✓ | training monitor |
| phase2_train_segformer_v3.py:277 | val loop | 2PR/(P+R) | ✓ | training monitor |
| diagnose_nopng.py:39, diagnose_segformer.py:97, diagnose_v2.py:87 | diagnostics | 2PR/(P+R) | ✓ | no |
| visualize_v1.py:44 | viz | 2PR/(P+R), guarded | ✓ | no |

All binary-class F1 forms are mathematically identical (2PR/(P+R) ≡ 2TP/(2TP+FP+FN)). `evaluate_all_models.py` additionally produced the **pooled** combined-set `micro_f1 = 0.864202 / micro_iou = 0.760877` (`outputs/evaluation_all/all_models_metrics.csv`, v1-final) — the numbers earlier flagged as the pooled, not NEON-only, variant.

---

# G. Oxygen Pipeline (full chain, with files/lines/constants/units)

```
tile image (256×256 PNG; source: NEON crop / ReforesTree ortho tile / synthetic)
  → SegFormer stage2_v3 inference (phase2/eval scripts) → prediction mask (binary 256×256, argmax)
  → [masks previously stored in outputs/masks/ — currently EMPTY, features persist in outputs/tree_features.csv]
  → phase3_feature_extraction.py:
       connected components (cv2.findContours, RETR_EXTERNAL), drop area_px < 20 px (config.MIN_CROWN_AREA_PX:193)
       area_m2  = area_px × pixel_size_m²            (phase3:145)   ← GSD enters HERE ONLY
       D_eq (m) = √(4·area_m2/π)                     (phase3:149)
  → phase4_allometric.py:
       DBH(cm)  = (D_eq / JUCKER_A)^(1/JUCKER_B)     (phase4:79)    a=0.5539, b=0.6982 (config)
       H(m)     = 25.1·(1 − e^(−0.04·DBH))           (phase4:103)   Feldpausch
       AGB(kg)  = 0.0673·(ρ·DBH²·H)^0.976            (phase4:133)   ρ=0.6 g/cm³ (config)
       C(kg)    = AGB × 0.47                         (phase4:156)   IPCC
       O2_stock(kg)   = C × 32/12                    (phase4:184)
       O2_annual(kg/y)= C × 0.05 × 32/12             (phase4:185)   NPP_FRACTION 0.05
```

Units: crown m / DBH cm / H m / AGB kg / C kg / O₂ kg·yr⁻¹. `pixel_size_m` is the only place GSD enters (phase3:145 and perimeter), applied as a single global scalar. Outputs: `outputs/tree_oxygen.csv`, then phase5 GIS (no new math; mock-coordinate fallback `MOCK_SCALE=1e-5` in phase5:147 only when no geotransform).

---

# H. 5,897 Crowns Provenance

From `outputs/tree_oxygen.csv` (5,897 rows; crown features are `outputs/masks/`-inference of 2,943 unique tile stems; the masks themselves are now deleted, features persist):

| source | crowns | total crown_area_m2 | O₂ annual (kg/yr) |
|---|---|---|---|
| NEON | **4,774** | 125,605 (recomputed) | 471,368 |
| ReforesTree | **1,089** | 379,864 (recomputed) | 1,481,705 |
| synthetic (OSBS-029-style) | **34** | 13,924 (recomputed) | 6,001 |
| **Total** | **5,897** | 519,394 | **1,959,074.2** (reported 1,959,073 — rounding of per-row rounding) |

- **GSD used**: a single global 0.1 m/px applied to **all** datasets (phase3:215 reads only a NEON sample GeoTIFF res=0.10; ReforesTree never reads its source orthomosaics). Source raster GSD (Ref ~0.011 m/px) not used.
- **Source rasters for crowns**: 2943 tile stems = NEON crop tiles + ReforesTree 4000 px ortho tiles + 34 synthetic tiles; exact image list preserved in `tree_features.csv` `image` column.
- **Resize effect**: crown px are measured in 256×256 tiles that are 1:1 crops of the source crops (Ref 4000 px, NEON native), no resampling changes GSD.
- **Reproducible?** The 1,959,073 total is exactly reproducible from the *existing* `outputs/tree_oxygen.csv`; it is **not** reproducible end-to-end today because `outputs/masks/` is empty and NEON source dir now holds a single image (`OSBS_029.png`).

---

# I. GSD Conflict

Definitions found (all hard-coded or single-sample):
- `phase3_feature_extraction.py:215` `pixel_size_m = 0.1` default; `:220` overridden by `res` of the **first NEON GeoTIFF** found (=0.10). Global for all masks.
- `sensitivity_analysis.py:84` default `pixel_size_m: 0.1` (comment: "phase3 default (no GeoTIFF source available)"); candidates [0.05, 0.10, 0.15, 0.20, 0.30, 0.50] (`:119`).
- No per-dataset values, no config entry, no metadata-driven lookup for ReforesTree in any pipeline script.

Comparison:
- **NEON actual**: 0.10 m/px (raster metadata) — matches the assumption. VERIFIED.
- **ReforesTree actual**: ≈ 0.009–0.011 m/px (orthomosaic transform + TFW + KML; tile ratio implies ≈0.006 m/px). **Conflicts** with 0.1.
- **Oxygen pipeline used**: **B — assumed 0.1 m/px**, for NEON and ReforesTree alike.
- **Affected**: all 1,123 non-NEON crowns (ReforesTree 1,089 + synthetic 34) → their crown_area_m2 and all downstream O₂ inflated ≈ (0.1/GSD)² ≈ **×83** (0.011) to ≈ **×237** (0.0065) in this repo's current `tree_oxygen.csv`. Therefore **the current oxygen total is affected** — no correction made (out of scope for this audit).

---

# J. Allometric Constants (current implementation only)

| constant | value | source | used at | equation | units |
|---|---|---|---|---|---|
| JUCKER_A | 0.5539 | config.py:150 | phase4:79 | DBH=(D/a)^(1/b) | m→cm |
| JUCKER_B | 0.6982 | config.py:151 | phase4:79 | " | m→cm |
| FELDPAUSCH_A | 25.1 | config.py:155 | phase4:103 | H=A(1−e^(−B·DBH)) | m |
| FELDPAUSCH_B | 0.04 | config.py:156 | phase4:103 | " | cm⁻¹ |
| WOOD_DENSITY_G_CM3 | 0.6 | config.py:160 | phase4:133 | AGB=0.0673(ρ·DBH²·H)^0.976 | g/cm³ |
| (Chave coeff) 0.0673, 0.976 | hardcoded | — | phase4:133 | " | — |
| CARBON_FRACTION | 0.47 | config.py:163 | phase4:156 | C=AGB×0.47 | — |
| C_TO_O2_RATIO | 32/12 = 2.6667 | config.py:167 | phase4:184–185 | O₂=C×(32/12) | kg O₂/kg C |
| NPP_FRACTION | 0.05 | config.py:174 | phase4:185 | annual=stock×0.05 | yr⁻¹ |
| MIN_CROWN_AREA_PX | 20 | config.py:193 | phase3:131 | drop smaller | px |

Citations are encoded only as comments (Jucker2017, Chave2014, Feldpausch2012, IPCC2006, Pan2011). No literal coefficient check vs primary literature performed in this task.

---

# K. Reviewer 3 Readiness (factual status only)

| Reviewer issue | Evidence found | Status |
|---|---|---|
| Intermediate DBH validation | Matched-tree stats computed (n=170): Pearson −0.09, MAE 3.9–4.5 cm only under assumed 0.011 GSD; R² negative at both GSD settings; matching offset 14–24 m vs 4.3 m GPS; no aggregation code; no primary match | NOT POSSIBLE (as evidence of accuracy) |
| AGB validation | AGB deterministic from species+DBH (100% unique mapping), carbon=0.39×AGB; no independent biomass | NOT POSSIBLE (circular) |
| ReforesTree GT characterization | Reference masks = DeepForest box predictions (annotations CSVs with scores; masks = merged boxes) | NEEDS WORDING |
| NEON GT | Independent box annotations (XML/csv → masks), pixel-level eval reproduced exactly | READY |
| Crown merging | No merge code; cross-tile duplicates unhandled; single tree matched to 2,991 detections (64% of rows) | NOT AVAILABLE |
| F1 formula | 2TP/(2TP+FP+FN) = 2PR/(P+R); reproduced 0.8599/0.7543 and 0.3686/0.2260 from per-tile files | CORRECT |
| Oxygen validation | No independent oxygen/AGB reference; chain derived end-to-end | NOT AVAILABLE |
| GSD | NEON 0.10 verified; ReforesTree assumed 0.1 vs source ≈0.011 (×83–×237 area inflation); affects current 1,959,073 total | CONFLICT |
| Code/data reproducibility | Eval reproducible from stored tiles+manifests; totals reproducible from tree_oxygen.csv; but outputs/masks deleted, matching & Ref-mask creation scripts absent, NEON source reduced to 1 image | PARTIAL |

No manuscript changes recommended; this report is evidence and calculations only. No metric above demonstrates model accuracy because the reference data are not independent (ref: task rule).