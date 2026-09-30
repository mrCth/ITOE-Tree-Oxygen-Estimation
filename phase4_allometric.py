"""
phase4_allometric.py
Estimate per-tree oxygen production using pantropical allometric equations.

Pipeline:
    Crown Diameter (m)
        → DBH (cm)         [Jucker et al. 2017]
        → Height (m)        [Feldpausch et al. 2012]
        → AGB (kg)          [Chave et al. 2014]
        → Carbon (kg)       [IPCC 2006]
        → O₂ (kg/year)      [Stoichiometry of photosynthesis]

Reads from:
    outputs/tree_features.csv  (Phase 3 output)

Outputs:
    outputs/tree_oxygen.csv

This script is deterministic — no random seeds, no train/test splits.

References:
    [Jucker2017]    Jucker et al. (2017) Allometric equations for integrating
                    remote sensing imagery into forest monitoring programmes.
                    Global Change Biology. DOI: 10.1111/gcb.13388
    [Chave2014]     Chave et al. (2014) Improved allometric models to estimate
                    the aboveground biomass of tropical trees.
                    Global Change Biology. DOI: 10.1111/gcb.12629
    [Feldpausch2012] Feldpausch et al. (2012) Tree height integrated into
                    pantropical forest biomass estimates.
                    Biogeosciences. DOI: 10.5194/bg-9-3381-2012
    [IPCC2006]      IPCC (2006) Guidelines for National Greenhouse Gas
                    Inventories. Volume 4, Chapter 4.
    [Stoichiometry] 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂
                    Carbon MW=12, Oxygen MW=32
                    O₂ produced = Carbon fixed × (32/12)
"""

import logging
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


# ============================================================================
# Allometric functions
# ============================================================================

def crown_diameter_to_dbh(crown_diameter_m: float) -> float:
    """Convert equivalent crown diameter to DBH using pantropical relationship.

    DBH (cm) = (D / a) ^ (1/b)

    Parameters
    ----------
    crown_diameter_m : float
        Equivalent crown diameter in metres.

    Returns
    -------
    float
        Estimated DBH in cm.

    References
    ----------
    [Jucker2017] Jucker et al. (2017) Global Change Biology.
    """
    if crown_diameter_m <= 0:
        return 0.0
    return (crown_diameter_m / config.JUCKER_A) ** (1.0 / config.JUCKER_B)


def dbh_to_height(dbh_cm: float) -> float:
    """Estimate tree height from DBH using pantropical height-diameter model.

    H (m) = 25.1 × (1 - exp(-0.04 × DBH))

    Parameters
    ----------
    dbh_cm : float
        Diameter at breast height in cm.

    Returns
    -------
    float
        Estimated tree height in metres.

    References
    ----------
    [Feldpausch2012] Feldpausch et al. (2012) Biogeosciences.
    """
    if dbh_cm <= 0:
        return 0.0
    return config.FELDPAUSCH_A * (1.0 - np.exp(-config.FELDPAUSCH_B * dbh_cm))


def dbh_to_agb(dbh_cm: float, wood_density: float = None, height_m: float = None) -> float:
    """Estimate aboveground biomass from DBH using pantropical allometric equation.

    AGB (kg) = 0.0673 × (ρ × DBH² × H) ^ 0.976

    Parameters
    ----------
    dbh_cm : float
        Diameter at breast height in cm.
    wood_density : float, optional
        Wood density in g/cm³. Defaults to config.WOOD_DENSITY_G_CM3.
    height_m : float, optional
        Tree height in metres. If None, estimated from DBH via Feldpausch.

    Returns
    -------
    float
        Estimated aboveground biomass in kg.

    References
    ----------
    [Chave2014] Chave et al. (2014) Global Change Biology.
    """
    if dbh_cm <= 0:
        return 0.0
    rho = wood_density if wood_density is not None else config.WOOD_DENSITY_G_CM3
    h = height_m if height_m is not None else dbh_to_height(dbh_cm)
    agb = 0.0673 * (rho * (dbh_cm ** 2) * h) ** 0.976
    return float(agb)


def agb_to_carbon(agb_kg: float) -> float:
    """Convert aboveground biomass to carbon stock.

    Carbon (kg) = AGB × 0.47

    Parameters
    ----------
    agb_kg : float
        Aboveground biomass in kg.

    Returns
    -------
    float
        Carbon stock in kg.

    References
    ----------
    [IPCC2006] IPCC (2006) Guidelines for National Greenhouse Gas Inventories.
    """
    return agb_kg * config.CARBON_FRACTION


def carbon_to_o2(carbon_kg: float) -> dict:
    """Convert carbon stock to O₂ estimates.

    Returns two values:
      o2_stock_kg     : total O₂ equivalent of standing carbon stock
                        (Chave 2014 total AGB × IPCC carbon fraction)
      o2_annual_kg_yr : estimated annual O₂ production
                        = carbon_stock × NPP_FRACTION × C_TO_O2_RATIO
                        Reference: Pan et al. (2011) NPP/AGB ratio

    Parameters
    ----------
    carbon_kg : float
        Carbon stock in kg.

    Returns
    -------
    dict
        Keys: o2_stock_kg, o2_annual_kg_yr.

    References
    ----------
    [Stoichiometry] Photosynthesis: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂
    [Pan2011]        Pan Y. et al. (2011) Science 333:988-993.
    """
    o2_stock = carbon_kg * config.C_TO_O2_RATIO
    o2_annual = carbon_kg * config.NPP_FRACTION * config.C_TO_O2_RATIO
    return {
        "o2_stock_kg":     round(o2_stock, 4),
        "o2_annual_kg_yr": round(o2_annual, 4),
    }


def estimate_o2_from_crown(crown_diameter_m: float) -> Dict[str, float]:
    """Full allometric chain: crown diameter → DBH → AGB → Carbon → O₂.

    Parameters
    ----------
    crown_diameter_m : float
        Equivalent crown diameter in metres.

    Returns
    -------
    dict
        All intermediate and final estimates:
        estimated_dbh_cm, estimated_height_m, estimated_agb_kg,
        estimated_carbon_kg, estimated_o2_kg_year.
    """
    dbh_cm = crown_diameter_to_dbh(crown_diameter_m)
    height_m = dbh_to_height(dbh_cm)
    agb_kg = dbh_to_agb(dbh_cm, height_m=height_m)
    carbon_kg = agb_to_carbon(agb_kg)
    o2_estimates = carbon_to_o2(carbon_kg)

    return {
        "estimated_dbh_cm":           round(dbh_cm, 4),
        "estimated_height_m":         round(height_m, 4),
        "estimated_agb_kg":           round(agb_kg, 4),
        "estimated_carbon_kg":        round(carbon_kg, 4),
        "estimated_o2_stock_kg":      o2_estimates["o2_stock_kg"],
        "estimated_o2_annual_kg_yr":  o2_estimates["o2_annual_kg_yr"],
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    # --- Load Phase 3 features ---
    if not os.path.isfile(config.FEATURES_CSV):
        logger.error("Features CSV not found at %s. Run Phase 3 first.", config.FEATURES_CSV)
        sys.exit(1)

    df = pd.read_csv(config.FEATURES_CSV)
    logger.info("Loaded %d tree records from %s", len(df), config.FEATURES_CSV)

    if "equiv_crown_diameter_m" not in df.columns:
        logger.error("Column 'equiv_crown_diameter_m' not found. Re-run Phase 3.")
        sys.exit(1)

    # --- Apply allometric chain to each tree ---
    allometric_cols = ["estimated_dbh_cm", "estimated_height_m",
                       "estimated_agb_kg", "estimated_carbon_kg",
                       "estimated_o2_stock_kg", "estimated_o2_annual_kg_yr"]

    records = df.to_dict("records")
    results = []
    for rec in records:
        crown_d = float(rec.get("equiv_crown_diameter_m", 0))
        estimates = estimate_o2_from_crown(crown_d)
        row = dict(rec)
        row.update(estimates)
        results.append(row)

    out_df = pd.DataFrame(results)

    # --- Save output ---
    out_path = config.TREE_OXYGEN_CSV
    out_df.to_csv(out_path, index=False)
    logger.info("Saved %d tree records to %s", len(out_df), out_path)

    # --- Validation summary ---
    o2_stock_vals = out_df["estimated_o2_stock_kg"]
    o2_annual_vals = out_df["estimated_o2_annual_kg_yr"]
    dbh_vals = out_df["estimated_dbh_cm"]
    agb_vals = out_df["estimated_agb_kg"]

    positive_o2 = (o2_annual_vals > 0).sum()

    print(f"\n{'='*52}")
    print(f"  Phase 4 - Allometric O2 Estimation Summary")
    print(f"{'='*52}")
    print(f"  Trees processed     : {len(out_df)}")
    print(f"  Trees with O2 > 0   : {positive_o2}")
    print(f"  {'-'*48}")
    print(f"  DBH (cm)       - mean: {dbh_vals.mean():.2f}  "
          f"median: {dbh_vals.median():.2f}  "
          f"min: {dbh_vals.min():.2f}  max: {dbh_vals.max():.2f}")
    print(f"  AGB (kg)       - mean: {agb_vals.mean():.2f}  "
          f"median: {agb_vals.median():.2f}  "
          f"min: {agb_vals.min():.2f}  max: {agb_vals.max():.2f}")
    print(f"  O2 stock (kg)  - mean: {o2_stock_vals.mean():.2f}  "
          f"median: {o2_stock_vals.median():.2f}  "
          f"min: {o2_stock_vals.min():.2f}  max: {o2_stock_vals.max():.2f}")
    print(f"  O2 annual(kg/y)- mean: {o2_annual_vals.mean():.2f}  "
          f"median: {o2_annual_vals.median():.2f}  "
          f"min: {o2_annual_vals.min():.2f}  max: {o2_annual_vals.max():.2f}")
    print(f"  {'-'*48}")
    print(f"  Constants used:")
    print(f"    Wood density   : {config.WOOD_DENSITY_G_CM3} g/cm3  [Chave2014]")
    print(f"    Carbon fraction: {config.CARBON_FRACTION}  [IPCC2006]")
    print(f"    C->O2 ratio    : {config.C_TO_O2_RATIO:.4f}  [Stoichiometry]")
    print(f"    NPP fraction   : {config.NPP_FRACTION}  [Pan2011]")
    print(f"    Jucker a,b     : {config.JUCKER_A}, {config.JUCKER_B}  [Jucker2017]")
    print(f"    Feldpausch     : {config.FELDPAUSCH_A}, {config.FELDPAUSCH_B}  [Feldpausch2012]")
    print(f"{'='*52}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
