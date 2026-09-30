# Sensitivity analysis — outputs

Deterministic one-at-a-time sensitivity of the **annual oxygen-equivalent**
estimate, run from the corrected `outputs/tree_oxygen.csv` baseline
(`python scripts/sensitivity_analysis.py`).

## Files

| File | Contents |
|---|---|
| `oxygen_sensitivity.csv` | scenario table (parameter, value, baseline, total/mean/median O₂, % change, n) |
| `oxygen_sensitivity.json` | machine-readable baseline + validation + scenarios |
| `oxygen_sensitivity.png` | plot of total O₂ vs each parameter (5,863 trees) |
| `SENSITIVITY_AUDIT.md` | full trace, results, paper-comparison, verdict |
| `baseline_summary.csv` ⚠️ | **OBSOLETE** — produced by the old root-level `sensitivity_analysis.py` (uniform 0.1 m/px GSD). Do not cite. |
| `sensitivity_oaat_summary.csv` ⚠️ | **OBSOLETE** — same old run. Do not cite. |

## Key results (baseline = 472,202.59 kg/yr over 5,863 trees)

- Wood density 0.4–0.8 g/cm³ → **−32.68 % to +32.42 %**
- Carbon fraction 0.45–0.50 (base 0.47) → **−4.26 % to +6.38 %**
- NPP fraction 0.025–0.10 (base 0.05) → **−50.00 % to +100.00 %**

These match the ranges currently in the paper ("approximately" wording).