"""
phase5_gis_output.py
Generate GIS-ready outputs from allometric O₂ estimation results.

Uses real GeoTIFF georeferencing to convert pixel coordinates to
latitude / longitude — no random coordinates are generated.

Reads from:
    outputs/tree_oxygen.csv  (Phase 4 output)

Outputs:
    gis/shapefiles/tree_oxygen.geojson
    gis/shapefiles/tree_oxygen.shp
    gis/shapefiles/tree_oxygen.csv
    gis/maps/tree_oxygen_map.html
    gis/maps/tree_oxygen_heatmap.html

References:
    [Jucker2017]    Jucker et al. (2017) Global Change Biology
    [Chave2014]     Chave et al. (2014) Global Change Biology
    [IPCC2006]      IPCC (2006) Guidelines for NGGI
    [Stoichiometry] 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂
"""

import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import folium
import folium.plugins
import numpy as np
import pandas as pd
from rasterio.warp import transform as warp_transform
from shapely.geometry import Point
import geopandas as gpd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsd_registry import (
    _load_neon_cache,
    classify_dataset,
    neon_tile_center_geo,
    ref_tile_center_geo,
)

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

OXYGEN_COL = "estimated_o2_annual_kg_yr"


# ---------------------------------------------------------------------------
# Basemap tile layers
# OpenStreetMap (tile.openstreetmap.org) is reliable in most regions but can be
# blocked/redirected in some networks. Use CartoDB basemaps (public CDN, no key)
# as the default and offer alternates; tree markers/heatmap are vector overlays
# and stay visible even when tile layers cannot load.
# ---------------------------------------------------------------------------

CARTO_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
    'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
)
BASEMAPS = [
    dict(name="CartoDB Positron (default)",
         tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
         subdomains="abc"),
    dict(name="CartoDB Voyager",
         tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
         subdomains="abc"),
    dict(name="ESRI World Imagery",
         tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Imagery/MapServer/tile/{z}/{y}/{x}",
         subdomains=""),
]


def add_basemaps(m: folium.Map) -> None:
    """Attach the CartoDB default + alternate tile layers.

    NOTE: LayerControl (L.control.layers) is intentionally NOT used — it
    crashes internally on the generated HTML (t.setZIndex on undefined),
    halting every subsequent layer script. Basemap switching is handled by
    add_basemap_switcher() with plain DOM buttons instead.
    """
    for cfg in BASEMAPS:
        folium.TileLayer(
            tiles=cfg["tiles"],
            attr=CARTO_ATTRIBUTION,
            name=cfg["name"],
            subdomains=cfg["subdomains"] if cfg["subdomains"] else None,
            overlay=False,
        ).add_to(m)


_BASEMAP_KEYS = {
    "Positron": "cartocdn.com/light_all",
    "Voyager": "rastertiles/voyager",
    "Satelite": "World_Imagery",
}


def add_basemap_switcher(m: folium.Map) -> None:
    """Floating basemap toggle buttons (replaces broken LayerControl)."""
    mvar = m.get_name()
    labels = list(_BASEMAP_KEYS.keys())
    btns = "".join(
        f'<button class="rj bm" data-bm="{_BASEMAP_KEYS[k]}" '
        f"{('style=' + chr(39) + 'outline:2px solid #3388ff' + chr(39)) if i == 0 else ''}>"
        f"{k}</button>"
        for i, k in enumerate(labels)
    )
    html = (
        f'<div style="position:absolute;z-index:1000;top:10px;right:10px;'
        f'display:flex;flex-direction:column;gap:4px;'
        f'font:13px Segoe UI,Arial;">{btns}</div>'
        f"<script>"
        f"window._bmMap_{len(labels)} = {mvar};"
        f"document.addEventListener('DOMContentLoaded',function(){{"
        f"var map={mvar};"
        f"document.querySelectorAll('.bm').forEach(function(b){{"
        f"b.addEventListener('click',function(){{"
        f"var key=b.getAttribute('data-bm');"
        f"map.eachLayer(function(l){{"
        f"if(l instanceof L.TileLayer){{"
        f"var u=l._url||'';"
        f"var on=u.indexOf(key)>=0;"
        f"l.setOpacity(on?1:0);"
        f"}}"
        f"}});"
        f"document.querySelectorAll('.bm').forEach(function(x){{"
        f"x.style.outline=x===b?'2px solid #3388ff':'none';"
        f"}});"
        f"}});"
        f"}});"
        f"}});"
        f"</script>"
    )
    m.get_root().html.add_child(folium.Element(html))


def _inline_heat_lib(m: folium.Map) -> bool:
    """Inline the leaflet.heat plugin into the map so the heat layer does NOT
    depend on an external CDN (some networks block jsdelivr/unpkg)."""
    try:
        import folium as _f
        lib = os.path.join(os.path.dirname(_f.__file__),
                           "templates", "leaflet_heat.min.js")
        if not os.path.isfile(lib):
            return False
        js = open(lib, encoding="utf-8").read()
        m.get_root().html.add_child(folium.Element(f"<script>{js}</script>"))
        logger.info("Inlined leaflet.heat plugin (%d bytes)", len(js))
        return True
    except Exception as e:  # pragma: no cover
        logger.warning("Could not inline leaflet.heat: %s", e)
    return False


def _inline_assets(html: str) -> str:
    """Inline vendored leaflet.js/css so the map works with no CDN at all.
    Also strips CDN script tags for the heat plugin (already inlined)."""
    asset_dir = os.path.join(config.GIS_DIR, "assets")
    try:
        leaflet_js = open(os.path.join(asset_dir, "leaflet.js"),
                          encoding="utf-8").read()
        leaflet_css = open(os.path.join(asset_dir, "leaflet.css"),
                           encoding="utf-8").read()
    except OSError as e:  # pragma: no cover
        logger.warning("Vendored leaflet assets not found (%s); "
                       "map keeps CDN refs", e)
        return html
    html = re.sub(
        r'<script src="[^"]*leaflet[^"]*leaflet\.js"[^>]*></script>',
        lambda m: f"<script>{leaflet_js}</script>", html)
    html = re.sub(
        r'<link[^>]*leaflet[^>]*\.css[^>]*>',
        lambda m: f"<style>{leaflet_css}</style>", html)
    html = re.sub(
        r'<script src="[^"]*leaflet_heat[^"]*"[^>]*></script>', "", html)
    return html


def _write_map(m: folium.Map, path: str) -> None:
    """Render the map, inline all vendored assets, then save a single HTML."""
    html = m.get_root().render()
    html = _inline_assets(html)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.info("Map saved (self-contained) to %s", path)


def add_region_jump(m: folium.Map, geo_df: pd.DataFrame) -> None:
    """Add a floating button panel to fly to each tree cluster.

    Display aid only — derived entirely from the georeferenced records.
    """
    presets = [("Zoom to all",
                float(geo_df["latitude"].min()), float(geo_df["longitude"].min()),
                float(geo_df["latitude"].max()), float(geo_df["longitude"].max()))]
    zoom_map = {"SJER": 12, "OSBS": 12, "BART": 12, "TEAK": 12, "JERC": 14}
    for site in ["SJER", "OSBS", "BART", "TEAK", "JERC"]:
        sub = geo_df[(geo_df["dataset"] == "neon") & (geo_df["plot"] == site)]
        if not sub.empty:
            presets.append((f"NEON {site}",
                            float(sub["latitude"].mean()),
                            float(sub["longitude"].mean()),
                            zoom_map[site]))
    sub = geo_df[geo_df["dataset"] == "reforestree"]
    if not sub.empty:
        presets.append(("Ecuador (ReforesTree)",
                        float(sub["latitude"].mean()),
                        float(sub["longitude"].mean()), 10))

    mvar = m.get_name()
    btns = []
    for p in presets:
        if len(p) == 5:
            btns.append(
                f'<button class="rj" '
                f'onclick="{mvar}.fitBounds([[{p[1]:.5f},{p[2]:.5f}],'
                f"[{p[3]:.5f},{p[4]:.5f}]])"
                f'">Zoom to all</button>')
        else:
            btns.append(f'<button class="rj" '
                        f'onclick="{mvar}.flyTo([{p[1]:.5f},{p[2]:.5f}],{p[3]})">'
                        f"{p[0]}</button>")
    el = (
        '<div id="region-jump" style="position:absolute;top:70px;right:10px;'
        'z-index:1000;display:flex;flex-direction:column;gap:3px;'
        'font:12px Arial,sans-serif;background:rgba(255,255,255,.95);'
        'border:1px solid #bbb;border-radius:6px;padding:6px;'
        'box-shadow:0 1px 4px rgba(0,0,0,.25)">'
        '<b style="margin-bottom:2px">Jump to cluster</b>'
        + "".join(btns)
        + '</div>'
        + '<style>.rj{cursor:pointer;border:1px solid #999;border-radius:4px;'
          'background:#f6f6f6;padding:2px 6px;text-align:left}'
          '.rj:hover{background:#ddd}</style>'
    )
    m.get_root().html.add_child(folium.Element(el))


# ---------------------------------------------------------------------------
# Colour coding
# ---------------------------------------------------------------------------

def _assign_color(o2_val: float, p33: float, p66: float) -> Tuple[str, str]:
    """Assign colour and label based on O₂ percentile thresholds.

    Returns (hex colour, label).
    """
    if o2_val > p66:
        return "#1a7a2e", "High"
    elif o2_val >= p33:
        return "#f0a500", "Medium"
    else:
        return "#cc2200", "Low"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(config.MAP_DIR, exist_ok=True)
    os.makedirs(config.SHP_DIR, exist_ok=True)

    if not os.path.isfile(config.TREE_OXYGEN_CSV):
        logger.error("Tree oxygen CSV not found at %s. Run Phase 4 first.",
                     config.TREE_OXYGEN_CSV)
        sys.exit(1)

    df = pd.read_csv(config.TREE_OXYGEN_CSV)
    logger.info("Loaded %d tree records from %s", len(df), config.TREE_OXYGEN_CSV)

    if df.empty:
        logger.error("Tree oxygen CSV is empty.")
        sys.exit(1)

    if OXYGEN_COL not in df.columns:
        logger.error("Column '%s' not found. Re-run Phase 4.", OXYGEN_COL)
        sys.exit(1)

    # --- Build pixel→geo cache from source GeoTIFFs ----------------------
    _neon_cache = _load_neon_cache()
    logger.info("Cached metadata for %d NEON GeoTIFFs", len(_neon_cache))

    # --- Compute percentile thresholds for colour coding -----------------
    o2_vals = df[OXYGEN_COL].values.astype(float)
    o2_vals = o2_vals[np.isfinite(o2_vals)]
    p33 = float(np.percentile(o2_vals, 33))
    p66 = float(np.percentile(o2_vals, 67))
    logger.info("O2 colour thresholds -- p33=%.2f, p66=%.2f", p33, p66)

    # --- Build per-tree GIS records --------------------------------------
    trees: List[Dict[str, Any]] = []

    trees_with_coords = 0
    for _, row in df.iterrows():
        img_name = str(row.get("image", ""))
        dataset = classify_dataset(img_name)

        cx_px = float(row.get("centroid_x_px", 0))
        cy_px = float(row.get("centroid_y_px", 0))

        lon = np.nan
        lat = np.nan
        georef_status = "unknown"

        # Real georeferencing only — no synthetic coordinates.
        try:
            if dataset == "neon":
                res = neon_tile_center_geo(img_name, cx_px, cy_px, _neon_cache)
                if res is not None:
                    lon_utm, lat_utm, crs = res
                    if crs is not None and crs.to_epsg() not in (4326, None):
                        lon, lat = warp_transform(
                            crs, "EPSG:4326", [lon_utm], [lat_utm]
                        )
                        lon, lat = lon[0], lat[0]
                    else:
                        lon, lat = lon_utm, lat_utm
                    georef_status = "verified"
            elif dataset == "reforestree":
                res = ref_tile_center_geo(img_name, cx_px, cy_px)
                if res is not None:
                    lon, lat, _ = res
                    georef_status = "verified"
        except Exception as e:
            logger.debug("Georeference failed for %s: %s", img_name, e)

        if np.isfinite(lon) and np.isfinite(lat):
            trees_with_coords += 1
        else:
            lon, lat = np.nan, np.nan

        o2_val = float(row.get(OXYGEN_COL, 0))
        hex_color, label = _assign_color(o2_val, p33, p66)

        trees.append({
            "tree_id": f"T{row.get('tree_id', 0):04d}_{img_name}",
            "latitude": round(float(lat), 7),
            "longitude": round(float(lon), 7),
            "georef_status": georef_status,
            "dataset": dataset,
            "plot": row.get("plot", ""),
            "pixel_size_m": row.get("pixel_size_m", np.nan),
            "gsd_status": row.get("gsd_status", ""),
            "crown_area_m2": row.get("crown_area_m2", np.nan),
            "equiv_crown_diameter_m": row.get("equiv_crown_diameter_m", np.nan),
            "estimated_dbh_cm": row.get("estimated_dbh_cm", np.nan),
            "estimated_agb_kg": row.get("estimated_agb_kg", np.nan),
            "estimated_carbon_kg": row.get("estimated_carbon_kg", np.nan),
            "estimated_o2_stock_kg": row.get("estimated_o2_stock_kg", np.nan),
            OXYGEN_COL: round(o2_val, 4),
            "o2_category": label,
            "ndvi": row.get("ndvi", np.nan),
            "health_score": row.get("health_score", np.nan),
        })

    if not trees:
        logger.error("No trees could be georeferenced. Aborting.")
        sys.exit(1)

    out_df = pd.DataFrame(trees)
    logger.info("GIS records: %d trees (%d with real coordinates)",
                len(out_df), trees_with_coords)

    # --- Export CSV (all trees, NaN coords for non-georeferenced) --------
    csv_path = os.path.join(config.SHP_DIR, "tree_oxygen.csv")
    out_df.to_csv(csv_path, index=False)
    logger.info("CSV saved to %s", csv_path)

    # --- Keep only georeferenced trees for vector/map outputs ------------
    geo_df = out_df[out_df["longitude"].notna() & out_df["latitude"].notna()].copy()
    if geo_df.empty:
        logger.error("No trees with real coordinates. Skipping vector/map outputs.")
        sys.exit(1)
    logger.info("Georeferenced records for maps/vectors: %d", len(geo_df))

    # --- Export GeoJSON --------------------------------------------------
    geometry = [
        Point(xy) for xy in zip(geo_df["longitude"], geo_df["latitude"])
    ]
    gdf = gpd.GeoDataFrame(geo_df, geometry=geometry,
                           crs=f"EPSG:{config.OUTPUT_EPSG}")

    geojson_path = os.path.join(config.SHP_DIR, "tree_oxygen.geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")
    logger.info("GeoJSON saved to %s", geojson_path)

    # --- Export Shapefile -------------------------------------------------
    shp_path = os.path.join(config.SHP_DIR, "tree_oxygen.shp")
    gdf.to_file(shp_path, driver="ESRI Shapefile")
    logger.info("Shapefile saved to %s", shp_path)

    # --- Interactive Folium map ------------------------------------------
    center_lat = geo_df["latitude"].mean()
    center_lon = geo_df["longitude"].mean()
    bounds = [
        [geo_df["latitude"].min(), geo_df["longitude"].min()],
        [geo_df["latitude"].max(), geo_df["longitude"].max()],
    ]
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles=None)
    add_basemaps(m)

    for _, t in geo_df.iterrows():
        hex_color, label = _assign_color(t[OXYGEN_COL], p33, p66)
        popup_html = (
            f"<b>{t['tree_id']}</b><br>"
            f"<b>Annual O₂ production:</b> {t[OXYGEN_COL]:.2f} kg/year<br>"
            f"<b>Total O₂ equivalent:</b> {t['estimated_o2_stock_kg']:.2f} kg<br>"
            f"<b>Crown diameter:</b> {t['equiv_crown_diameter_m']:.2f} m<br>"
            f"<b>Estimated DBH:</b> {t['estimated_dbh_cm']:.2f} cm<br>"
            f"<b>Estimated AGB:</b> {t['estimated_agb_kg']:.2f} kg<br>"
            f"<b>Carbon stock:</b> {t['estimated_carbon_kg']:.2f} kg<br>"
            f"<b>NDVI:</b> {t['ndvi']:.3f}<br>"
            f"<b>Health:</b> {t['health_score']:.2f}"
        )
        folium.CircleMarker(
            location=[t["latitude"], t["longitude"]],
            radius=6,
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(m)

    html_path = os.path.join(config.MAP_DIR, "tree_oxygen_map.html")
    add_region_jump(m, geo_df)
    add_basemap_switcher(m)
    m.fit_bounds(bounds, padding=(25, 25))
    _write_map(m, html_path)

    # --- Oxygen heatmap ---------------------------------------------------
    hm = folium.Map(tiles=None)
    add_basemaps(hm)
    _inline_heat_lib(hm)
    heat_data = [
        [t["latitude"], t["longitude"], t[OXYGEN_COL]]
        for _, t in geo_df.iterrows()
    ]
    folium.plugins.HeatMap(
        heat_data,
        min_opacity=0.4,
        max_zoom=18,
        radius=18,
        blur=12,
        gradient={0.4: "blue", 0.6: "lime", 0.8: "orange", 1.0: "red"},
    ).add_to(hm)
    add_region_jump(hm, geo_df)
    add_basemap_switcher(hm)
    hm.fit_bounds(bounds, padding=(25, 25))
    _write_map(hm, os.path.join(config.MAP_DIR, "tree_oxygen_heatmap.html"))

    # --- Summary ----------------------------------------------------------
    total_o2_annual = (
        geo_df[OXYGEN_COL].sum() if not geo_df.empty else float("nan")
    )
    n_high = (geo_df["o2_category"] == "High").sum()
    n_med = (geo_df["o2_category"] == "Medium").sum()
    n_low = (geo_df["o2_category"] == "Low").sum()
    n_unknown_geo = (out_df["georef_status"] == "unknown").sum()

    print(f"\n{'='*48}")
    print(f"  Phase 5 - GIS Output Summary")
    print(f"{'='*48}")
    print(f"  Trees in CSV         : {len(out_df)}")
    print(f"  Trees georeferenced  : {len(geo_df)}  ({trees_with_coords})")
    print(f"  Trees no coordinates : {n_unknown_geo}")
    print(f"  Total annual O2 (georeferenced) : {total_o2_annual:.2f} kg/year")
    print(f"  High (>p66)          : {n_high}")
    print(f"  Medium (p33-p66)     : {n_med}")
    print(f"  Low (<p33)           : {n_low}")
    print(f"  {'-'*44}")
    print(f"  CSV      : {csv_path}")
    print(f"  GeoJSON  : {geojson_path}")
    print(f"  Shapefile: {shp_path}")
    print(f"  Map      : {html_path}")
    print(f"  Heatmap  : {os.path.join(config.MAP_DIR, 'tree_oxygen_heatmap.html')}")
    print(f"{'='*48}")


if __name__ == "__main__":
    main()
