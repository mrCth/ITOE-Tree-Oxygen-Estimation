"""
sensitivity_analysis.py
Deterministic one-at-a-time (OAAT) sensitivity of the annual oxygen-equivalent
estimate to three uncertain parameters.

Baseline source (CORRECTED pipeline output — do not recompute from scratch):
    outputs/tree_oxygen.csv          (produced by src/phase4_allometric.py)
    Baseline chain (src/config.py + phase3/phase4):
        crown_area_px -> m2          area_px * pixel_size_m^2        [phase3, per-plot GSD]
        crown diameter D (m)         D = sqrt(4*area/pi)             [phase3]
        DBH (cm)                     (D/JUCKER_A)^(1/JUCKER_B)       [Jucker2017]
        Height (m)                   25.1*(1 - exp(-0.04*DBH))       [Feldpausch2012]
        AGB (kg)                     0.0673*(rho*DBH^2*H)^0.976      [Chave2014]  rho=0.6
        Carbon (kg)                  AGB * 0.47                      [IPCC2006]
        O2 stock (kg)                Carbon * (32/12)                [stoichiometry]
        O2 annual (kg/yr)            Carbon * NPP(0.05) * (32/12)    [Pan2011]

One-at-a-time rule: every scenario varies EXACTLY ONE parameter; all others
stay at the baseline values above. Because O2_annual is
    agb(rho) * cf * npp * (32/12)   with  agb = k * rho^0.976,
a scenario is computed by exact per-tree scaling of the recorded baseline row:

    wood_density :  o2' = o2_base * (rho/0.6)^0.976
    carbon_frac  :  o2' = o2_base * (cf/0.47)
    npp_frac     :  o2' = o2_base * (npp/0.05)

A full explicit recomputation of the Chave2014 AGB at rho=0.6 is also carried
out (from the stored DBH and Height columns) and compared to the recorded
baseline AGB as a validation step; this verifies the exponent and constant
without allowing the sensitivity itself to drift from the corrected baseline.

Details of the paper's currently written ranges (wood density 0.4-0.8 ->
about -32.7/+32.4 %, carbon fraction 0.45-0.50 -> about -4.3/+6.4 %, NPP
fraction 0.025-0.10 -> about -50/+100 %) are intentionally NOT assumed here;
they are compared against the actual outputs below.

Usage:
    python scripts/sensitivity_analysis.py

Writes (relative to repository root):
    outputs/sensitivity/oxygen_sensitivity.csv
    outputs/sensitivity/oxygen_sensitivity.json
    outputs/sensitivity/oxygen_sensitivity.png
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if (_p / "config.py").is_file():
        sys.path.insert(0, str(_p))
        break
else:  # pragma: no cover
    sys.exit("config.py not found; run from the repository root")

import config  # noqa: E402   (constants used below)

OUT_DIR = Path(config.OUTPUTS_DIR) / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Baseline parameters (project constants — do not edit without a source)
RHO_BASE = float(config.WOOD_DENSITY_G_CM3)   # 0.6 g/cm3 [Chave2014]
CF_BASE = float(config.CARBON_FRACTION)       # 0.47     [IPCC2006]
NPP_BASE = float(config.NPP_FRACTION)         # 0.05     [Pan2011]
C2O = float(config.C_TO_O2_RATIO)             # 32/12

CHAVE_EXP = 0.976   # phase4_allometric.dbh_to_agb exponent [Chave2014]
CHAVE_K = 0.0673    # phase4_allometric.dbh_to_agb constant  [Chave2014]

# Candidate values (deterministic, from the task spec)
RHO_CANDIDATES = [0.4, 0.5, 0.6, 0.7, 0.8]
CF_CANDIDATES = [0.45, 0.47, 0.50]
NPP_CANDIDATES = [0.025, 0.05, 0.075, 0.10]

COLS = ["parameter", "parameter_value", "baseline_value",
        "total_o2_kg_year", "mean_o2_kg_tree_year", "median_o2_kg_tree_year",
        "percent_change_from_baseline", "n_trees"]


def load_valid(df):
    """Return rows with a valid annual O2 (synthetic NaN rows are excluded)."""
    mask = df["estimated_o2_annual_kg_yr"].notna()
    return df.loc[mask].copy()


def metrics(o2):
    return {
        "total_o2_kg_year": float(np.sum(o2)),
        "mean_o2_kg_tree_year": float(np.mean(o2)),
        "median_o2_kg_tree_year": float(np.median(o2)),
        "n_trees": int(len(o2)),
    }


def pct(base, value):
    return (value - base) / base * 100.0


def validate_against_explicit_chain(df):
    """Re-derive AGB at rho=0.6 from stored DBH/Height and compare to the
    recorded baseline AGB. Verifies exponent/constant and drift."""
    dbh = df["estimated_dbh_cm"].to_numpy(float)
    h = df["estimated_height_m"].to_numpy(float)
    agb_re = CHAVE_K * (RHO_BASE * (dbh ** 2) * h) ** CHAVE_EXP
    agb_stored = df["estimated_agb_kg"].to_numpy(float)
    valid = agb_stored > 0
    rel = np.abs(agb_re[valid] - agb_stored[valid]) / agb_stored[valid]
    return {
        "max_rel_diff": float(rel.max() if len(rel) else float("nan")),
        "mean_rel_diff": float(rel.mean() if len(rel) else float("nan")),
        "frac_rows_rel_gt_1pct": float((rel > 1.0).mean() if len(rel) else float("nan")),
        "total_stored_agb_kg": float(agb_stored[valid].sum()),
        "total_recompute_agb_kg": float(agb_re[valid].sum()),
        "n_valid_agb": int(valid.sum()),
    }


def run(csv_path):
    df = pd.read_csv(csv_path)
    valid = load_valid(df)
    if "synthetic" in pd.unique(valid["dataset"]):
        print("FATAL: synthetic crowns present in valid set")
        sys.exit(1)

    o2_base = valid["estimated_o2_annual_kg_yr"].to_numpy(float)
    base_metrics = metrics(o2_base)
    validation = validate_against_explicit_chain(valid)

    print(f"Baseline from {csv_path}")
    print(f"  rows={base_metrics['n_trees']}  "
          f"total={base_metrics['total_o2_kg_year']:,.2f} kg/yr  "
          f"mean={base_metrics['mean_o2_kg_tree_year']:.3f}  "
          f"median={base_metrics['median_o2_kg_tree_year']:.3f}")
    print(f"AGB(rho=0.6) explicit recompute vs stored: max rel diff "
          f"{validation['max_rel_diff']*100:.3f}%")

    base_value = {"wood_density": RHO_BASE, "carbon_fraction": CF_BASE,
                  "npp_fraction": NPP_BASE}
    scenarios = []

    def add_scenario(name, values, scale_fn):
        for v in values:
            o2_sc = scale_fn(o2_base, v)
            m = metrics(o2_sc)
            pc = pct(base_metrics["total_o2_kg_year"], m["total_o2_kg_year"])
            scenarios.append({
                "parameter": name,
                "parameter_value": v,
                "baseline_value": base_value[name],
                **{k: round(m[k], 3) if isinstance(m[k], float) else m[k]
                   for k in ("total_o2_kg_year", "mean_o2_kg_tree_year",
                             "median_o2_kg_tree_year")},
                "percent_change_from_baseline": round(pc, 4),
                "n_trees": m["n_trees"],
            })
            print(f"  {name:16s} = {v:<6g}  total {m['total_o2_kg_year']:>13,.2f}"
                  f"  ({pc:+.2f}%)")

    print("\nWood density sweep (carbon_fraction and npp_fraction held fixed):")
    add_scenario("wood_density", RHO_CANDIDATES,
                 lambda o, v: o * (v / RHO_BASE) ** CHAVE_EXP)
    print("Carbon fraction sweep (wood_density=0.6, npp=0.05 held fixed):")
    add_scenario("carbon_fraction", CF_CANDIDATES,
                 lambda o, v: o * (v / CF_BASE))
    print("NPP fraction sweep (wood_density=0.6, carbon=0.47 held fixed):")
    add_scenario("npp_fraction", NPP_CANDIDATES,
                 lambda o, v: o * (v / NPP_BASE))

    df_out = pd.DataFrame(scenarios, columns=COLS)
    df_out.to_csv(OUT_DIR / "oxygen_sensitivity.csv", index=False)

    # baseline rows inside the scenario table for machine-readable parity
    baseline_rows = {
        "wood_density": scenarios[0]["baseline_value"],
        "carbon_fraction": next(s["baseline_value"]
                                for s in scenarios
                                if s["parameter"] == "carbon_fraction"),
        "npp_fraction": next(s["baseline_value"]
                             for s in scenarios
                             if s["parameter"] == "npp_fraction"),
    }

    payload = {
        "baseline": {
            "source": str(csv_path),
            "rows": base_metrics["n_trees"],
            "total_o2_kg_year": round(base_metrics["total_o2_kg_year"], 4),
            "mean_o2_kg_tree_year": round(base_metrics["mean_o2_kg_tree_year"], 6),
            "median_o2_kg_tree_year": round(base_metrics["median_o2_kg_tree_year"], 6),
            "parameters": {
                "wood_density_g_cm3": RHO_BASE,
                "carbon_fraction": CF_BASE,
                "npp_fraction": NPP_BASE,
                "carbon_to_o2_ratio": C2O,
                "chave_exp": CHAVE_EXP,
            },
        },
        "validation": {
            "synthetic_excluded": int((df["dataset"] == "synthetic").sum()),
            "agb_explicit_recompute": {
                "max_rel_diff_pct": round(validation["max_rel_diff"] * 100, 4),
                "mean_rel_diff_pct": round(validation["mean_rel_diff"] * 100, 4),
                "frac_rows_rel_gt_1pct": round(validation["frac_rows_rel_gt_1pct"] * 100, 4),
                "total_stored_agb_kg": round(validation["total_stored_agb_kg"], 2),
                "total_recompute_agb_kg": round(validation["total_recompute_agb_kg"], 2),
            },
        },
        "scenarios": scenarios,
    }
    with open(OUT_DIR / "oxygen_sensitivity.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nSaved -> outputs/sensitivity/oxygen_sensitivity.csv + .json")


def make_plot():
    """Plot the CSV produced by run() — never dummy data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(OUT_DIR / "oxygen_sensitivity.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    titles = [("wood_density", r"Wood density $\rho$ (g/cm$^3$)"),
              ("carbon_fraction", "Carbon fraction"),
              ("npp_fraction", "Annual NPP fraction")]
    for ax, (param, label) in zip(axes, titles):
        sub = df[df["parameter"] == param]
        total = sub["total_o2_kg_year"]
        pctv = sub["percent_change_from_baseline"]
        ax.plot(sub["parameter_value"], total, "o-", color="#1f77b4")
        ax.axhline(total.iloc[0], color="gray", ls=":", lw=1,
                   label="baseline")
        ax.set_xlabel(label)
        ax.set_ylabel("total annual O₂ (kg/yr)")
        ax.grid(alpha=0.3)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(6, 6))
        for x, y, p in zip(sub["parameter_value"], total, pctv):
            ax.annotate(f"{p:+.1f}%", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8)
        ax.legend(fontsize=8)
    fig.suptitle("One-at-a-time sensitivity of annual oxygen-equivalent "
                 "(baseline = corrected outputs/tree_oxygen.csv, 5,863 trees)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "oxygen_sensitivity.png", dpi=150)
    print("Saved -> outputs/sensitivity/oxygen_sensitivity.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.TREE_OXYGEN_CSV))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    run(args.csv)
    if not args.no_plot:
        make_plot()