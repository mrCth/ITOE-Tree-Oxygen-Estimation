# ITOE: Cross-Domain Tree Crown Delineation for Tree-Level Annual Oxygen Estimation

Repository for the camera-ready / reproducibility release of the paper
**"ITOE: Cross-Domain Tree Crown Delineation for Tree-Level Annual Oxygen
Estimation"**.

## What is in this repository

This repository publishes the **source code** and **generated outputs** needed
to verify and reproduce the results of the paper. It is **not** a full data
package.

> The datasets used in the experiments are **external / pre-existing data**
> and are **NOT included** in this repository. The experiments in this paper
> did **not** download them through an in-code API workflow; the data were
> obtained and prepared separately before training/inference.

Users must obtain and prepare the datasets themselves before running the
training / inference pipeline. See the individual data-source documentation
and the audit documents under `docs/` for what each source expects.

## Reproduction pipeline

The reproducible pipeline starts from a **prepared dataset**:

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
```

Sensitivity analysis runs on the corrected allometric output:

```text
outputs/tree_oxygen.csv
      ↓
scripts/sensitivity_analysis.py
      ↓
outputs/sensitivity/
```

There is **no `phase0_download_dataset.py`** in this release and it is not a
reproduction step of this package.

## Getting started

```bash
pip install -r requirements.txt
```

Then follow the pipeline scripts in the order listed above. Configuration is
centralized in `config.py` and `gsd_registry.py` (per-plot GSD / CRS
sources).

## Repository layout

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── config.py
├── gsd_registry.py
├── phase1_*.py  phase2_*.py  phase3_*.py  phase4_*.py  phase5_gis_output.py
├── scripts/sensitivity_analysis.py
├── gis/{assets,maps,shapefiles}/
├── outputs/tree_features.csv, tree_oxygen.csv, sensitivity/
└── docs/
```

## Key outputs (from the corrected pipeline)

* Trees with valid annual O₂ estimate: **5,863** (34 synthetic rows excluded)
* Total annual oxygen-equivalent: **472,202.59 kg/yr**
* Deterministic one-at-a-time parameter-sensitivity analysis (baseline ρ=0.6, cf=0.47, NPP=0.05):

  * wood density 0.4–0.8 → −32.68% / +32.42%
  * carbon fraction 0.45–0.50 → −4.26% / +6.38%
  * NPP fraction 0.025–0.10 → −50.0% / +100.0%

These ranges represent **deterministic parameter-sensitivity results under the tested scenarios**. They are not statistical confidence intervals or prediction intervals. See `outputs/sensitivity/SENSITIVITY_AUDIT.md` for the full source trace, validation checks, and reproducibility details.


See `outputs/sensitivity/SENSITIVITY_AUDIT.md` for the full trace.

## Documentation

`docs/` contains the reproducibility and audit documents:
`reviewer3_audit_report.md`, `georeferencing_audit.md`,
`pre_submission_audit.md`, `dataset_audit_report.md`,
`phase34_methodology.md`, `v2_methodology.md`, `09_equations.md`,
`RELEASE_AUDIT.md`.

## License

See `LICENSE`. Citation details are provided in the paper.
