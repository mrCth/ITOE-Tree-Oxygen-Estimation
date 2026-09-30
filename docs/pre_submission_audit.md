# Pre-Submission Technical Audit — ITOE Paper (CamTreeO2 Project)

> Audit role: senior IEEE reviewer (remote sensing + computer vision).
> Scope: full paper source material in `paper_export/`, actual pipeline code
> (`phase3_feature_extraction.py`, `phase4_allometric.py`, `config.py`), training
> logs, evaluation scripts, generated figures, and GIS outputs.
> Date: 2026-08-06.

---

## 1. Executive Summary

**Verdict: Major revision required. Do not submit in current state.**

The segmentation contribution (SegFormer-B0, native-256, two-stage) is coherent and
publishable as a *methods* paper — **if** the honest per-domain numbers are reported.
On ReforesTree (test/val) F1 is 0.84–0.87 and credible; on the NEON holdout (305
tiles) F1 collapses to 0.29–0.41 — the cross-domain generalization gap must be the
explicit narrative, not hidden behind a pooled "3,030 test" figure. Downstream, the
oxygen/allometric contribution carries a hard georeferencing bug (NEON trees located
in Ecuador), a constant NDVI=0.5 feature, pantropical equations applied to temperate
NEON sites without justification, and an unvalidated NPP fraction. The `paper_export`
source docs previously described the *deprecated* XGBoost/synthetic-data pipeline;
they have been corrected to the deterministic allometric chain (see files 02, 05, 07).
These issues would be caught by any competent reviewer and are fatal to acceptance if
left.

Five must-fix items (Section 2) block submission; the remaining items are
quality/clarity fixes.

---

## 2. Critical Issues (Must-Fix)

### 🔴 C1. Georeferencing bug: NEON trees reported at Ecuadorian coordinates
`gis/shapefiles/tree_oxygen.geojson` and `outputs/tree_oxygen.csv` place NEON BART
(Bartlett Experimental Forest, New Hampshire, USA) tiles — e.g.
`T0000_2018_BART_4_322000_4882000_image_crop_t0` — at latitude ≈ −0.18°, longitude
≈ −78.46° (Ecuador, the ReforesTree domain). BART UTM 19N (322000 E, 4882000 N)
should decode to ≈ 44.06°N, 71.3°W.
- Root cause suspected: pixel→geo affine transform from the wrong source GeoTIFF
  (ReforesTree transform applied to NEON tiles) or wrong CRS assumption. **No CRS
  reprojection is implemented anywhere** (`12_missing_information.md`, item 10).
- Impact: every map/GIS figure and every per-tree coordinate claim is wrong. A
  reviewer with any GIS experience will run one point through UTM and reject the paper.
- Fix: reconstruct transforms per source image from correct GeoTIFF metadata; verify
  by spot-checking ≥ 10 UTM→WGS84 conversions against site names; state CRS in the
  paper; if NDVI/geography is not actually used in the allometric chain, consider
  removing coordinates entirely and framing as *imagery-only* per-tree estimates.

### 🔴 C2. NDVI is constant 0.5 in every output record
All 5,897 records carry `ndvi = 0.5` and `health_score ≤ 0.4` — the Landsat 8 raster
was never matched to the tiles (fallback path in `phase3_feature_extraction.py`).
- If NDVI/health are claimed as features or discussed in the paper, this is a
  fabricated-signal problem. Either (a) properly georeference and query NDVI (blocked
  by C1), or (b) drop NDVI/health from the paper entirely and state that the pipeline
  is RGB-only. Option (b) is strongly recommended for a 6-page IEEE paper.

### 🔴 C3. Source documents contradict the claimed methodology
`paper_export/05_allometric_pipeline.md` and `07_experiments.md` describe the
deprecated pipeline: XGBoost regression trained on **4 synthetic samples**
(R² = 0.165, MAE = 26.17), synthetic field data (`AGB = crown_m2 × random(5,15)`),
and a **missing DBH step**. The actual ITOE methodology
(`phase4_allometric.py`, `v2_methodology.md`) is deterministic: crown diameter →
DBH (Jucker 2017) → height (Feldpausch 2012) → AGB (Chave 2014) → carbon (IPCC 0.47)
→ O₂ (×32/12, NPP fraction 0.05).
- Fix: purge all XGBoost/synthetic content from the new paper. Do **not** reference
  R²=0.165 or the 4-sample training anywhere. The `pipeline_diagram.png` (Section 9)
  must also be regenerated.

### 🔴 C4. Honest per-domain metrics (replacing old pooled "NEON test" numbers)
- Earlier numbers "NEON test (3,030 tiles): v1-final F1=0.8642 vs v2-final F1=0.8560"
  (this audit's old §6) were computed over **all** `dataset/test/images` tiles mixed
  together = ReforesTree-dominant (2,726/3,031). They must NOT be labelled "NEON".
- Honest per-domain eval (`evaluate_honest_sets.py`, `outputs/evaluation_all/honest_*.csv`):
  - NEON_test (305 tiles): v1-final F1=0.3128, v2-final F1=0.2934, v3-final F1=0.3686
    (tree IoU 0.1854 / 0.1719 / 0.2260). **v1 does NOT beat v2/v3 on honest NEON.**
  - ReforesTree_test (2,726 tiles): v1-final F1=0.8698, v2-final F1=0.8622,
    v3-final F1=0.8599 (IoU 0.7696 / 0.7578 / 0.7543).
  - ReforesTree val (770): v1 F1=0.8460 (IoU 0.7331) > v2 F1=0.8405 (0.7248) >
    v3 F1=0.8350 (0.7168).
  - NEON_val (2,950): v1 F1=0.3575, v2 F1=0.4578, v3 F1=0.5351.
- **The dominant finding is cross-domain collapse, not v1-vs-v2:** NEON holdout F1
  is 0.29–0.41 (vs 0.84–0.87 on ReforesTree). v2/v3's native-256 design does improve
  NEON_val monotonically (0.36→0.46→0.54) but NEON_test stays near-chance.
- Fix: report per-domain splits in a table (NEON_test / ReforesTree_test / NEON_val /
  ReforesTree_val, NOT a pooled 3,030 figure); reframe contribution as "two-stage
  transfer improves the target-domain (ReforesTree) operating point at negligible
  cost, with a large, honest NEON-test generalization gap" + significance analysis
  (paired test on per-tile IoU).

### 🔴 C5. Deployed model ≠ best checkpoint
`load_best_model_at_end = False` (documented in `v2_methodology.md`, §7) means the
saved/deployed weights (v3 `segformer_tree_crown_v3` = `stage2_v3` final, last step
~18,000) are not the training-time best. v3 best tree IoU 0.7180/F1~0.8358 @ step
2,000 vs re-eval final (deployed) ReforesTree val IoU 0.7168/F1 0.8350. Phase 3/4
inference and all downstream oxygen numbers use the deployed (last-step) weights.
- Fix: state explicitly that downstream numbers derive from last-step weights, or
  reload best before Phase 3. Pick one number set and use it everywhere — honest
  per-domain eval (`evaluate_honest_sets.py`) loads deployed weights.

---

## 3. Methodology Audit

### 3.1 Segmentation
- SegFormer-B0, MiT-B0 backbone, ADE20K-pretrained, 2-class head. ~3.7M params. Sound
  and cheap. State parameter count and inference throughput (ms/tile) — reviewers like it.
- **Native-256 design decision is well-motivated** (avoids 256→512→256 bilinear
  smoothing). Good ablation material — but see C4, the ablation must be reported both
  on ReforesTree val and NEON test.
- Two-stage training: Stage 1 all-data, Stage 2 ReforesTree fine-tune. Coherent with
  domain-shift narrative.
- **Epoch counts — resolved from logs/config**: `config.py` now has `EPOCHS=100`,
  `FINETUNE_EPOCHS=100`. v1 Stage 1 actually ran 50 epochs (26,500 steps,
  `train.log`); v2/v3 **skipped** Stage 1 (reused `stage1_v2`/`stage1_v3`,
  `train_v2.log`/`train_v3.log`). Stage 2 = 20 epochs (v1/v2) or 100 with early-stop
  ~89 (v3). Paper must state ONE set per variant actually run.
- Stage 2 = 20 epochs (v1/v2) / 100 (v3), LR 1e-5 vs Stage 1 LR 6e-5. State cosine
  schedule + warmup (v2: 500 steps; v3: warmup_ratio 0.1).

### 3.2 Loss and training details
- Weighted CE, inverse-frequency weights, sum = 2. Stage 1 bg=0.973/tree=1.027;
  Stage 2 bg=1.191/tree=0.809. Consistent (verified arithmetic).
- FP16, effective batch 32 (8 × 4 grad accum), AdamW, weight decay 0.01, seed 42.
  Early stopping patience 8 eval rounds on tree IoU. Fine — report the eval interval
  (2,000 steps) and that stopping fired (state how early, e.g. "stopped at round 3/8").
- **v3 ablation conflates three changes** (Dice+CE hybrid loss, stronger
  augmentation, 100-ep fine-tune). `phase2_train_segformer_v3.py` gates loss on
  `USE_DICE_LOSS` (config `True`) — state which single change v3 isolates, or split
  the ablation.

### 3.3 Allometric chain (`phase4_allometric.py`)
- Chain: CD → DBH (Jucker 2017, `DBH = (CD/0.5539)^(1/0.6982)`) → H (Feldpausch 2012,
  `25.1·(1−e^(−0.04·DBH))`) → AGB (Chave 2014, `0.0673·(ρ·DBH²·H)^0.976`, ρ=0.6) →
  C (×0.47) → O₂ (×32/12), annual = stock × NPP fraction 0.05.
- **Verify the Jucker a/b constants against the source table.** Jucker et al. (2017)
  reports several crown-diameter models; the values 0.5539/0.6982 must match the
  exact table row cited (pantropical, "all trees" model). Mismatched parameters are a
  common reviewer trap. Cite page/table.
- **Tropical-equation transferability**: Chave 2014 and Jucker 2017 are pantropical;
  NEON imagery includes temperate US sites (BART, etc.). Applying tropical AGB
  equations to temperate forests requires a defensible claim or sensitivity analysis.
  At minimum, state it as a limitation and quantify sensitivity (e.g., ±20% ρ).
- **NPP fraction 0.05 (Pan 2011) needs an exact citation anchor.** Pan et al. (2011)
  Science reports forest carbon pools/fluxes; 0.05 = annual NPP/standing AGB-carbon
  must be derived and cited precisely, or reviewers will reject the annual O₂ numbers
  as arbitrary.
- **Annual O₂ is NET-equivalent, not gross production.** O₂ stock = carbon × 32/12 is
  the O₂ released to sequester that carbon over the tree's lifetime; annual = 5% of
  that. Title/claims must say "net annual O₂ equivalent (above-ground)", not "oxygen
  production", or the photosynthesis-vs-respiration confusion will be flagged (also
  in `12_missing_information.md` item 7).
- Below-ground biomass excluded (Chave is AGB-only). State explicitly.
- Crown-diameter from area assumes circular crowns and treats the segmented blob
  (incl. internal gaps) as solid canopy. State as simplification.

### 3.4 Phase 3 features
- Contour extraction (RETR_EXTERNAL), min area 20 px, area/diameter/perimeter/circularity/
  aspect-ratio/density/NDVI/health. Fine as morphological features.
- **pixel_size assumed 0.1 m/px** (default) when GeoTIFF resolution unavailable
  (`12_missing_information.md` item 8). NEON GSD varies by flight line. Since crown
  area feeds the entire allometric chain, a wrong GSD propagates nonlinearly through
  DBH→AGB→O₂. If the true GSD is known per source, use it; otherwise do a GSD
  sensitivity sweep (0.05/0.1/0.2 m) and report the resulting O₂ spread.
- Health score is capped at 0.4 with NDVI=0.5 (see C2). Drop or fix.

---

## 4. Dataset Audit

- **Train 16,932 / val 3,720 / test 3,031 tiles** (256×256 per manifest), image-level
  80/10/10, seed 42; per-domain test = NEON 305 + ReforesTree 2,726. Disk test dir has
  3,030 (orphan `OSBS_029` in `neon_rgb` listed as test, no tile saved) — reconcile
  and report exact number used. Class distribution bg 51.4% / tree 48.6%.
- **Smart sampling** (0%/0.5%/100% thresholds at 20%/50%/100% keep rates) is a
  defensible negative-mining strategy; document it in one sentence.
- **ReforesTree field data is SYNTHETIC** (`_create_field_data.py`: `AGB =
  crown_m2 × random(5,15)`). The new deterministic chain does NOT use field data, which
  makes this moot for the O₂ estimates — but the paper must **not** cite any
  field-data validation. Do not mention R²=0.165.
- **No ground truth for O₂**: the entire oxygen number is model-derived. The paper must
  be framed as *estimation framework + worked example*, not validated prediction. A
  reviewer will ask "what is the O₂ error?" — answer: unquantified, must be stated.
- **ReforesTree tree class 59.5%** (Stage 2) vs 48.6% global. Fine, report both.
- Multi-site, multi-date (NEON years/sites), mixed GSD. State the site list explicitly
  (BART, OSBS, ...) and per-site tile counts; reviewers check for domain description.

---

## 5. Equation Audit

| Eq | Status | Note |
|----|--------|------|
| E1 Circularity `4πA/(P²+ε)` | OK | cite standard morphometry |
| E2 Equiv. crown diameter `√(4A/π)` | OK | used as chain input — make this an explicit equation |
| E3 Area `A_px·p²` | OK | cite GSD source |
| E4 Perimeter `P_px·p` | OK | — |
| E5 Crown density | OK | define bbox fill ratio |
| E6 Health score | **drop** | depends on constant NDVI (C2) |
| E7 C = AGB × 0.47 | OK | IPCC 2006 (vol 4, ch 4) |
| E8 O₂ = C × 32/12 | OK | stoichiometry; state it is mass-equivalent |
| E9 FPAR = 1.24·NDVI − 0.168 | **remove** | defined but unused in code (`12_missing_information.md`) |
| E10 NPP = FPAR·PAR·LUE | **remove** | unused in code; including dead equations invites review attack |
| E11 pixel→geo | **blocked** | wrong in practice (C1) |
| E12 class weights | OK | document formula |
| E13 IoU / E14 Dice | OK | cite VOC / Sørensen-Dice |
| E15 aspect ratio | OK | — |
| NEW: DBH from CD (Jucker) | **add** | make it an explicit numbered equation; verify constants |
| NEW: H from DBH (Feldpausch) | **add** | — |
| NEW: AGB (Chave) | **add** | — |
| NEW: annual O₂ = stock × 0.05 | **add** | justify 0.05 with Pan 2011 |

Rule for IEEE: if an equation appears in the paper it must be used and cited; if it is
not used, delete it. E9/E10 are dead code and must not appear.

---

## 6. Results Audit

- **ReforesTree val (770)**: v1 P=0.7855/R=0.9165/F1=0.8460/IoU=0.7331;
  v2 P=0.8376/R=0.8433/F1=0.8405/IoU=0.7248; v3 P=0.8359/R=0.8342/F1=0.8350/IoU=0.7168.
- **ReforesTree test (2,726)**: v1-final F1=0.8698/IoU=0.7696; v2-final
  F1=0.8622/IoU=0.7578; v3-final F1=0.8599/IoU=0.7543.
- **NEON test (305, per-domain)**: v1-final F1=0.3128/IoU=0.1854; v2-final
  F1=0.2934/IoU=0.1719; v3-final F1=0.3686/IoU=0.2260. (Old pooled "3,030 mixed-set"
  numbers — v1-final 0.8642, v2-final 0.8560 — were ReforesTree-dominated and are
  withdrawn; do not cite.)
- **NEON val (2,950)**: v1-final F1=0.3575, v2-final F1=0.4578, v3-final F1=0.5351.
- **Metric convention**: micro-averaged over pooled pixels, computed per domain via
  `evaluate_honest_sets.py`. State this explicitly and distinguish from the
  *bounding-box detection F1* used by some ITC benchmarks (e.g., ReforesTree's own
  F1 score). Mixing pixel-F1 and box-F1 in one table is a common reviewer trap —
  define once, use everywhere.
- **Training-time best vs re-eval**: v2 best tree IoU=0.7308, mean IoU=0.6553,
  pixel acc=0.8037/0.8020 (two sources differ slightly), F1=0.8445 vs re-eval
  0.8405. Pick one reproducible set and cite `evaluate_all_models.py` +
  `evaluate_reforestree_val.py` outputs.
- **v2/v3 differences are small** (ΔIoU 0.008) with no significance testing. If the
  paper claims v2 superiority, run a paired per-tile test (e.g., Wilcoxon) and report
  p-values or drop the claim.
- **Oxygen results**: 5,897 crowns; ΣO₂ annual = 1,959,074 kg/yr; median 47 kg/tree/yr
  vs mean 332 kg/yr — distribution is heavily right-skewed (large crowns dominate).
  Report median, IQR, and the top-1% share, not just the total. This is scientifically
  the most defensible framing.

---

## 7. Figure Audit

All six figures regenerated from **honest per-domain metrics** via
`generate_itoe_figures.py` (PNG + SVG + PDF, 300 DPI) in `outputs/paper_figures/`.
Verified. `figure_metrics_loader.py` sources numbers from
`outputs/evaluation_all/`.

| Fig | Content | Issues |
|-----|---------|--------|
| fig1_model_comparison | v1/v2/v3 bar chart (ReforesTree val/test) | driven by honest loader; verify star/marker still matches chosen framing |
| fig2_neon_test | **grouped NEON_test vs ReforesTree_test** per variant | now honest: NEON_test F1 ≈0.29–0.41 vs Ref≈0.86 — headline gap, must be the narrative (C4) |
| fig3_comparison | per-metric comparison | gap arrow/annotation fine after tweaks (y=1.62 / 2.32) |
| fig4_pr_scatter | precision-recall scatter on NEON-only | verify annotation coords against honest P/R |
| fig5_dataset | dataset illustration from live manifest | shows 16,932/3,720/3,031 + per-domain counts; fine |
| fig6_ablation | native-256 vs 512 / loss / epochs | honest data; must state which split each bar uses (Ref_val dominated) |

General: add (a) error bars / per-tile IQR if space allows; (b) a prediction-overlay
example (true/false positives/negatives) — reviewers always ask; (c) consistent color
scheme for v1/v2/v3 across all figures (they currently differ in what "best" means).

---

## 8. Table Audit

Required tables and their status:
- **Dataset table** — sites, #images, #tiles, GSD, split counts. Missing site-level
  breakdown (BART/OSBS/...). Add.
- **Architecture/hyperparameter table** — epochs/cores/LR/batch/aug. Per-variant epoch
  numbers now resolved (Section 3.1); include Stage 1 skip note for v2/v3.
- **Results table** — v1/v2/v3 × {NEON_test, ReforesTree_test, NEON_val,
  ReforesTree_val} × {P,R,F1,IoU}. Include all four splits for all variants
  (NEON rows now exist for v2/v3: honest CSV).
- **Ablation table** — must isolate single changes (native-256 vs 512 with same
  data/loss; loss swap with same data/resolution). Current v3 conflates three changes.
- **Oxygen table** — per-tree O₂ stats (median/IQR/mean/total) + sensitivity
  (pixel_size/GSD 0.05→0.15 m is dominant: total O₂ −87%→+217%; ρ 0.4→0.8:
  −33%→+32%). `outputs/sensitivity/sensitivity_oaat_summary.csv` now exists.
- **No O₂-validation table** exists (no ground truth). State "framework + worked
  example" instead of implying validation.

---

## 9. Diagram Audit

`paper_export/pipeline_diagram.png/.svg` (from `generate_diagram.py`) was **wrong for
the ITOE paper** — **FIXED (regenerated)**:
- Phase 4 box said "XGBoost Regression" → now "Phase 4 — Allometric O₂ chain
  (deterministic): CD→DBH (Jucker) → H (Feldpausch) → AGB (Chave) → C(×0.47) →
  O₂ (×32/12 × NPP 0.05) → kg O₂/yr".
- `field_data.csv (AGB, Carbon) SYNTHETIC` label removed (kept on data source box as
  "synthetic, unused" — decision: keep the note or drop it, camera-ready to choose).
- "FPAR = 1.24×NDVI − 0.168 (unused in code)" annotation → replaced with
  "no gross daily O₂, no XGBoost/SHAP".
- Stage labels now match actual runs: stage1_v3 (reused), Stage 2 = 100 ep early-stop
  ~89 (v3); split boxes carry train 16,932 / val 3,720 / test 3,031. Deployed model =
  `segformer_tree_crown_v3/` (last step).
- Two competing generators exist (`paper_export/generate_diagram.py` and root
  `generate_pipeline_diagram.py`, which reads real `tree_oxygen` output). One of them
  must be declared the canonical source and deleted.

---

## 10. IEEE Compliance Checklist

- [ ] Page limit (typ. 6 pp. for IEEE conference): trim to fit — current scope is
  large.
- [ ] IEEE two-column format, Times, 10 pt.
- [ ] Abstract ≤ 250 words, self-contained (define ITOE, NEON, ReforesTree).
- [ ] IEEE reference style `[1]` numbered; cite: Xie et al. (SegFormer, NeurIPS 2021),
      Jucker 2017, Feldpausch 2012, Chave 2014, IPCC 2006, Pan 2011, Albumentations,
      ADE20K, HF Transformers, OpenCV.
- [ ] All figures 300 DPI, embedded fonts, no U+2605 glyph issue (use mathtext star as
      already fixed in fig1/fig3).
- [ ] SI units throughout (kg/yr, m², g/cm³).
- [ ] Equations numbered and referenced in text; unused equations deleted.
- [ ] Acronyms defined at first use (ITC, GSD, NPP, AGB, DBH, F1, IoU).
- [ ] Funding / dataset licenses (NEON data usage, ReforesTree license, Roboflow).
      Roboflow API key is hardcoded in `phase0_download_dataset.py` — remove before
      any code release; do not mention in paper.
- [ ] Author list, affiliation, ORCID placeholders.

---

## 11. Reviewer #2 Simulation (hostile but fair)

**R1 (Segmentation):** "Your comparison table pools the mixed test set (3,030/3,031
tiles, 90% ReforesTree) and calls it 'NEON test'. On the true NEON holdout (305
tiles) your F1 is 0.29–0.41 for every variant — near-chance — while ReforesTree test
is 0.84–0.87. The gap is never shown. Report the four-split table (NEON_test,
ReforesTree_test, NEON_val, ReforesTree_val) with per-domain metrics, or a reader
will reasonably accuse you of mislabelling the test set."

**R2 (Oxygen/allometry):** "The entire O₂ estimate is a chain of five unvalidated
assumptions: pantropical Jucker/Chave/Feldpausch equations applied to temperate US
NEON sites, a constant 0.6 g/cm³ wood density, a 0.47 carbon fraction, an NPP
fraction of 0.05 sourced only to a global synthesis, and a default 0.1 m GSD. You now
show a pixel-size/GSD sweep (dominant: −87%→+217% total O₂) — good — but there is
still no ground truth. As presented the oxygen numbers are a sensitivity-bounded
framework, not a validated estimate. State that framing explicitly."

**R3 (Data integrity):** "Your Figure/Table place NEON BART imagery in Ecuador
(coordinates ≈ −0.18°N, −78.46°W). If the geographic transform is wrong here, every
map product and every NDVI query is suspect. The paper reports an NDVI feature but
every crown shows NDVI = 0.5. Please clarify what was actually measured."

**R4 (Reproducibility):** "Training-time best (tree IoU 0.7308) differs from the
reported/used checkpoint (0.7248). Which model produced the 5,897 crowns? Stage 1
epochs are given as 50, 100, and ~20 in different sections. Please state exact
hyperparameters and the exact weights used for inference." *This remains live —
per-variant epoch numbers are now resolved in Section 3.1 but must be propagated
into the manuscript, one table, consistent everywhere.*

**R5 (Scope):** "This is binary crown *delineation*, not individual tree crown
*delineation with counts*. Binary masks cannot separate touching crowns, yet your
pipeline reports 5,897 discrete trees and per-tree oxygen. How many crowns are
merged? What is the crown-count error vs. a manual count on a subset?"

**R6 (Novelty):** "The individual components — SegFormer, allometric equations, GSD
conversion — are each standard. The contribution must be the integration and the
two-stage transfer. Either benchmark against a baseline (UNet, DeepForest, Mask
R-CNN) or sharpen the claimed novelty."

---

## 12. Final Checklist

### 🔴 Critical — fix before submission
- [ ] C1: correct georeferencing or drop coordinates from the paper
- [ ] C2: drop NDVI/health feature or fix its data source
- [x] C3: purge all XGBoost/synthetic/4-sample/R²=0.165 content
- [ ] C4: report the four honest splits (NEON_test/ReforesTree_test/NEON_val/
      ReforesTree_val) for all variants; add significance or soften the native-256 claim
- [ ] C5: choose one checkpoint and use it consistently (or reload best)
- [x] Regenerate pipeline diagram without XGBoost/synthetic/dead-equation labels
- [ ] Propagate resolved per-variant epoch numbers (Stage 1: v1=50 / v2,v3=SKIP;
      Stage 2: v1,v2=20 / v3=100 early-stop ~89) into the manuscript, one place
- [ ] Remove hardcoded Roboflow API key before any code release

### 🟡 Should fix — before submission
- [ ] Add per-tile statistics (median/IQR) instead of only pooled micro metrics
- [ ] Add prediction-example figure (TP/FP/FN) and under-segmentation analysis (R5)
- [ ] Verify Jucker 2017 constants against the cited table row; cite table
- [ ] Justify NPP fraction 0.05 with an exact Pan 2011 anchor
- [ ] Report crown-count error on a manual subset (5–10 tiles)
- [ ] Baseline comparison (UNet or DeepForest) or explicit novelty framing (R6)
- [ ] Add site-level dataset table (BART/OSBS/... per-site tile counts, GSD)
- [ ] Reconcile best-vs-reeval metric pairs (pixel acc 0.8037 vs 0.8020; F1 0.8445 vs 0.8405)
- [ ] State below-ground exclusion and net-vs-gross O₂ semantics in limitations

### 🟢 OK — verified
- [ ] Class weights arithmetic (Stage 1 0.973/1.027; Stage 2 1.191/0.809) consistent
- [ ] Allometric chain arithmetic (CD→DBH→H→AGB→C→O₂) internally consistent
  (verified against tree_oxygen.csv outputs; baseline reproduces 1,959,073 kg/yr)
- [ ] O₂ sensitivity sweep done (`outputs/sensitivity/`): pixel_size dominant
  (0.05→0.15 m: −87%→+217% total); ρ 0.4→0.8: −33%→+32%; carbon 0.45→0.50:
  −4%→+6%; NPP 0.025→0.10: −50%→+100%
- [ ] All 6 figures at 300 DPI (PNG+SVG+PDF) from honest per-domain loader
- [ ] Per-domain NEON/ReforesTree metric sets internally consistent
  (`outputs/evaluation_all/honest_*.csv`)
- [ ] Smart sampling and image-level split described (once, in final text)

---

## 13. Recommended Next Step

Rewrite the paper from `v2_methodology.md` + `phase34_methodology.md` (the
*current* methodology), using `paper_export/` only as citation/equation reference —
not as pipeline truth. Address C1–C5 in the writing. Sensitivity sweep already done
(Table 4 content); regenerate the diagram and one prediction-example figure. After
that, re-run this audit.
