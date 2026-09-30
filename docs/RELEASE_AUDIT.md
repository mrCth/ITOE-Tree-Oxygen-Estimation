# RELEASE_AUDIT.md

Camera-ready / reproducibility release audit for the ITOE paper.

## Scope

This package publishes source code and generated outputs for verification and
reproduction. **Datasets are external / pre-existing and are NOT included.**
The experiments did not download the datasets through an in-code API
workflow; therefore no dataset-download mechanism is part of this release and
`phase0_download_dataset.py` is **excluded** from the package. The main
project keeps `phase0_download_dataset.py` untouched; only the release
package omits it.

## Reproducibility pipeline (from prepared data)

```text
Prepared Dataset
      ↓
phase1_prepare_dataset.py
      ↓
phase2_train_segformer.py
      ↓
phase3_feature_extraction.py
      ↓
phase4_allometric.py
      ↓
phase5_gis_output.py

outputs/tree_oxygen.csv
      ↓
scripts/sensitivity_analysis.py
      ↓
outputs/sensitivity/
```

## Files included

| Group | Files |
|---|---|
| Code (root) | `config.py`, `gsd_registry.py`, `requirements.txt`, `phase1_prepare_dataset.py`, `phase2_train_segformer.py`, `phase3_feature_extraction.py`, `phase4_allometric.py`, `phase5_gis_output.py` |
| Sensitivity code | `scripts/sensitivity_analysis.py` |
| GIS | `gis/assets/leaflet.{js,css}`, `gis/maps/tree_oxygen_map.html`, `gis/maps/tree_oxygen_heatmap.html`, `gis/shapefiles/tree_oxygen.{csv,geojson,shp,shx,dbf,prj,cpg}` |
| Outputs | `outputs/tree_features.csv`, `outputs/tree_oxygen.csv`, `outputs/sensitivity/{oxygen_sensitivity.csv, oxygen_sensitivity.json, oxygen_sensitivity.png, README.md, SENSITIVITY_AUDIT.md}` |
| Docs | `reviewer3_audit_report.md`, `georeferencing_audit.md`, `pre_submission_audit.md`, `dataset_audit_report.md`, `phase34_methodology.md`, `v2_methodology.md`, `09_equations.md` |
| Meta | `README.md`, `LICENSE` (placeholder), `.gitignore` |

## Files excluded (deliberately)

| Excluded | Reason |
|---|---|
| `phase0_download_dataset.py` | dataset is external/pre-existing; download API is not the real workflow; avoids exposing API key |
| `NeonTreeEvaluation/`, `reforestree_dataset/`, `dataset/` | external raw data, large, license-bound |
| `models/`, `lightning_logs/` | trained weights, regenerable, large |
| logs / `.bat` / `.bak` / `__pycache__` / temp files | run artifacts with local paths |
| `sensitivity_analysis.py` (project root) | superseded by `scripts/sensitivity_analysis.py` |
| `outputs/sensitivity/{baseline_summary.csv, sensitivity_oaat_summary.csv}` | obsolete pre-GSD-fix run |
| large figure PNGs, `outputs/visualizations_*`, `outputs/paper_figures/*` | regenerable / noise |

## Secret audit

Scan applied to every `.py`, `.md`, `.txt`, `.json`, `.yaml`, `.yml`, `.csv`
and `.html` file in the release package (patterns: API key, `sk-...`, `AKIA`,
`secret`, `bearer`, `password`, `token`; plus the known Roboflow key literal
previously hard-coded in the excluded `phase0_download_dataset.py`). Result:
**0 real secrets found** — the only pattern hit was a self-reference to that
literal in this audit line; the literal is not present anywhere else and the
source file that contained it is not part of the package. No push was
performed.

## Baseline values (verified from shipped outputs)

- `outputs/tree_oxygen.csv`: 5,863 valid O₂ rows (34 synthetic NaN excluded),
  total **472,202.59 kg/yr**
- `outputs/sensitivity/oxygen_sensitivity.json`: same baseline; AGB explicit
  recompute mean rel. diff 0.046 %
- Sensitivity result parity: wood density −32.68 % / +32.42 %, carbon
  fraction −4.26 % / +6.38 %, NPP −50 % / +100 %.