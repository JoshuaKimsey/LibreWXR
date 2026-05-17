#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Generate the LibreWXR coverage maps.

Writes two PNGs to ``docs/``:

  * ``coverage-map-radar.png``  — radar composites
  * ``coverage-map-models.png`` — regional NWP grids

ECMWF IFS provides global coverage and is not drawn.

One-off authoring tool — not a runtime dependency.  Regenerate after
adding or removing a radar source or regional NWP grid.

Each polygon is rendered by walking its grid's perimeter in the
source's native projection (LCC, polar stereographic, LAEA, rotated
lat/lon, or regular lat/lon) and inverse-projecting each edge sample
to WGS84 lat/lon, so the resulting outline follows the actual curved
domain shape rather than a misleading lat/lon bounding box.

Basemap: Natural Earth Vector 1:110m country polygons (CC0).

# Regenerate with:
#   python3 -m venv /tmp/coverage-map-venv
#   /tmp/coverage-map-venv/bin/pip install matplotlib pyproj shapely
#   curl -L https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson -o /tmp/ne_countries.geojson
#   /tmp/coverage-map-venv/bin/python scripts/generate_coverage_map.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from math import asin, atan2, cos, degrees, radians, sin
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from pyproj import CRS, Transformer
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parent.parent
RADAR_OUTPUT = REPO_ROOT / "docs" / "coverage-map-radar.png"
MODEL_OUTPUT = REPO_ROOT / "docs" / "coverage-map-models.png"
BASEMAP_PATH = Path("/tmp/ne_countries.geojson")

# Pull radar station lists from the project source so adding a new
# station updates the map without a second edit.
sys.path.insert(0, str(REPO_ROOT / "src"))
from librewxr.data.radar_stations import (  # noqa: E402
    NEXRAD_CONUS,
    NEXRAD_ALASKA,
    NEXRAD_HAWAII,
    NEXRAD_PUERTO_RICO,
    NEXRAD_GUAM,
    CANADA_STATIONS,
    CWA_STATIONS,
    MMD_PENINSULAR_STATIONS,
    MMD_EAST_STATIONS,
    SNET_STATIONS,
    OPERA_STATIONS,
    RADAR_RANGE_KM,
    REGION_RADAR_RANGE,
)


# ── Coverage source definitions ────────────────────────────────────────


@dataclass(frozen=True)
class Source:
    """A coverage polygon — radar composite or regional NWP grid."""
    label: str
    color: str
    polygon: np.ndarray  # shape (N, 2), [lon, lat]


def project_grid_perimeter(
    crs_grid: CRS,
    x_min: float, x_max: float, y_min: float, y_max: float,
    samples_per_edge: int = 80,
) -> np.ndarray:
    """Walk a rectangular grid's perimeter in projected (x, y) space and
    inverse-project to (lon, lat).  Smooth enough to capture curvature
    of LCC / polar stereographic / LAEA projections in geographic
    coordinates."""
    transformer = Transformer.from_crs(crs_grid, "EPSG:4326", always_xy=True)
    edge_x = np.linspace(x_min, x_max, samples_per_edge)
    edge_y = np.linspace(y_min, y_max, samples_per_edge)
    xs = np.concatenate([
        edge_x,                              # top
        np.full(samples_per_edge, x_max),    # right
        edge_x[::-1],                        # bottom
        np.full(samples_per_edge, x_min),    # left
    ])
    ys = np.concatenate([
        np.full(samples_per_edge, y_max),
        edge_y[::-1],
        np.full(samples_per_edge, y_min),
        edge_y,
    ])
    lon, lat = transformer.transform(xs, ys)
    return np.column_stack([lon, lat])


def latlon_box(west: float, east: float, south: float, north: float) -> np.ndarray:
    """Simple geographic bounding-box polygon."""
    return np.array([
        [west, north], [east, north], [east, south], [west, south], [west, north],
    ])


def union_of_radar_circles(
    stations: list[tuple[float, float]],
    radius_km: float,
    samples_per_circle: int = 72,
) -> list[np.ndarray]:
    """Build the union of radar coverage circles as one or more polygons.

    Each station gets a circle of ``radius_km`` km approximated by
    ``samples_per_circle`` points using the local flat-earth scale
    (1° lat ≈ 111 km, 1° lon ≈ 111·cos(lat) km — same convention the
    project's ``coverage.py`` mask uses).  All circles are unioned via
    shapely and returned as a list of (N, 2) lon/lat arrays — one per
    disjoint piece of coverage (so e.g. Iceland's stations stay
    separated from continental Europe's where the gap exceeds station
    range).
    """
    circles: list[ShapelyPolygon] = []
    angles = np.linspace(0, 2 * np.pi, samples_per_circle, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    for st_lat, st_lon in stations:
        dlat = (radius_km / 111.0) * cos_a
        dlon = (radius_km / (111.0 * cos(radians(st_lat)))) * sin_a
        ring = np.column_stack([st_lon + dlon, st_lat + dlat])
        # Shapely tolerates open rings (closes them implicitly).
        circles.append(ShapelyPolygon(ring))

    merged = unary_union(circles)
    polygons: list[np.ndarray] = []
    if merged.geom_type == "Polygon":
        polygons.append(np.asarray(merged.exterior.coords))
    elif merged.geom_type == "MultiPolygon":
        for poly in merged.geoms:
            polygons.append(np.asarray(poly.exterior.coords))
    else:
        raise ValueError(f"unexpected merge result: {merged.geom_type}")
    return polygons


# Explicit rotated-pole inverse via Cartesian rotation on the unit
# sphere.  pyproj's ob_tran proved fiddly with the ECCC encoding —
# rolling our own keeps the convention obvious.
def _rotated_inv_xy(
    rlat: np.ndarray, rlon: np.ndarray,
    pole_lat: float, pole_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotated lat/lon (deg) → geographic lat/lon (deg).

    Inverse of the COSMO / WMO north-pole rotated_ll forward used by
    the HRDPS grid module: the rotated north pole sits at geographic
    (pole_lat, pole_lon), and rotated coordinates measure latitude
    from that pole.
    """
    phi_r = np.radians(rlat)
    lam_r = np.radians(rlon)
    # Cartesian point on unit sphere in the rotated frame.
    x = np.cos(phi_r) * np.cos(lam_r)
    y = np.cos(phi_r) * np.sin(lam_r)
    z = np.sin(phi_r)
    # Reverse step: rotate by +(90 - pole_lat) about the y-axis to
    # restore the rotated pole back to its geographic latitude.
    cb = cos(radians(90 - pole_lat))
    sb = sin(radians(90 - pole_lat))
    x1 = cb * x + sb * z
    y1 = y
    z1 = -sb * x + cb * z
    # Reverse step: rotate by +pole_lon about the z-axis to restore
    # the rotated pole back to its geographic longitude.
    ca = cos(radians(pole_lon))
    sa = sin(radians(pole_lon))
    x2 = ca * x1 - sa * y1
    y2 = sa * x1 + ca * y1
    z2 = z1
    lat = np.degrees(np.arcsin(np.clip(z2, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y2, x2))
    return lat, lon


def _rotated_fwd_xy(
    lat: float, lon: float, pole_lat: float, pole_lon: float,
) -> tuple[float, float]:
    """Scalar forward rotation, used to compute rotated bounds from a
    known geographic corner."""
    phi = radians(lat); lam = radians(lon)
    x = cos(phi) * cos(lam); y = cos(phi) * sin(lam); z = sin(phi)
    # Step 1: rotate by pole_lon about z.
    ca = cos(radians(pole_lon)); sa = sin(radians(pole_lon))
    x1 = ca * x + sa * y; y1 = -sa * x + ca * y; z1 = z
    # Step 2: rotate by (90 - pole_lat) about y.
    cb = cos(radians(90 - pole_lat)); sb = sin(radians(90 - pole_lat))
    x2 = cb * x1 - sb * z1; y2 = y1; z2 = sb * x1 + cb * z1
    return degrees(asin(z2)), degrees(atan2(y2, x2))


def rotated_pole_polygon(
    pole_lat: float, pole_lon: float,
    rlat_min: float, rlat_max: float,
    rlon_min: float, rlon_max: float,
    samples_per_edge: int = 80,
) -> np.ndarray:
    """Perimeter polygon for a rotated-pole grid, in geographic coords."""
    edge_rlon = np.linspace(rlon_min, rlon_max, samples_per_edge)
    edge_rlat = np.linspace(rlat_min, rlat_max, samples_per_edge)
    rlons = np.concatenate([
        edge_rlon,
        np.full(samples_per_edge, rlon_max),
        edge_rlon[::-1],
        np.full(samples_per_edge, rlon_min),
    ])
    rlats = np.concatenate([
        np.full(samples_per_edge, rlat_max),
        edge_rlat[::-1],
        np.full(samples_per_edge, rlat_min),
        edge_rlat,
    ])
    lat, lon = _rotated_inv_xy(rlats, rlons, pole_lat, pole_lon)
    return np.column_stack([lon, lat])


# ── Source definitions ─────────────────────────────────────────────────


def build_radar_sources() -> list[Source]:
    """Build per-station union-of-circles polygons for every radar source.

    Reads the station lists straight from
    ``librewxr.data.radar_stations`` so adding or removing a station
    propagates through to the map automatically.  Range is
    ``RADAR_RANGE_KM`` (240 km) by default, overridden per-region via
    ``REGION_RADAR_RANGE`` (300 km for OPERA's C-band network) — same
    numbers the runtime coverage mask uses.
    """
    radar: list[Source] = []

    def range_for(region: str) -> float:
        return REGION_RADAR_RANGE.get(region, RADAR_RANGE_KM)

    # MRMS — five composites sharing one upstream operator (NOAA) and
    # one legend swatch.  Each composite gets its own union polygon
    # built from that region's NEXRAD stations.
    mrms_color = "#1f77b4"
    mrms_groups = [
        ("MRMS — CONUS",       NEXRAD_CONUS,        range_for("USCOMP")),
        ("MRMS — Alaska",      NEXRAD_ALASKA,       range_for("AKCOMP")),
        ("MRMS — Hawaii",      NEXRAD_HAWAII,       range_for("HICOMP")),
        ("MRMS — Puerto Rico", NEXRAD_PUERTO_RICO,  range_for("PRCOMP")),
        ("MRMS — Guam",        NEXRAD_GUAM,         range_for("GUCOMP")),
    ]
    for label, stations, radius_km in mrms_groups:
        for poly in union_of_radar_circles(stations, radius_km):
            radar.append(Source(label, mrms_color, poly))

    # MSC Canada — ECCC's 32 S-band dual-pol stations at 240 km each.
    for poly in union_of_radar_circles(CANADA_STATIONS, range_for("CACOMP")):
        radar.append(Source("MSC Canada radar", "#17becf", poly))

    # MARN/SNET (El Salvador) — single S-band radar at San Andrés volcano
    # publishing a 120 km range product.  Smallest radar footprint in the
    # ensemble; fills the Central American gap between MRMS Caribbean and
    # the South American Cone NWP coverage.
    for poly in union_of_radar_circles(SNET_STATIONS, range_for("SVCOMP")):
        radar.append(Source("MARN/SNET (El Salvador)", "#bcbd22", poly))

    # OPERA — ~155 European stations at 300 km each (C-band).  The
    # union naturally splits into a continental piece, Iceland,
    # Ireland-and-Britain, etc. where station gaps exceed range.
    for poly in union_of_radar_circles(OPERA_STATIONS, range_for("OPERA")):
        radar.append(Source("OPERA (Europe)", "#9467bd", poly))

    # CWA / QPESUMS Taiwan — 7 S-band radars covering Taiwan + a
    # substantial W. Pacific buffer for typhoon tracking.
    for poly in union_of_radar_circles(CWA_STATIONS, range_for("TWCOMP")):
        radar.append(Source("CWA / QPESUMS (Taiwan)", "#e377c2", poly))

    # MET Malaysia — 12-radar S-band network split across Peninsular
    # Malaysia (7 stations) and East Malaysia / Borneo (5 stations),
    # both feeding a single combined composite GIF.  Stations are
    # presented per-region so the legend matches the two LibreWXR
    # regions (MYPENINSULAR + MYEAST), but they share one upstream
    # operator and one swatch.
    mmd_color = "#2ca02c"
    for poly in union_of_radar_circles(
        MMD_PENINSULAR_STATIONS, range_for("MYPENINSULAR"),
    ):
        radar.append(Source("MET Malaysia (Peninsular)", mmd_color, poly))
    for poly in union_of_radar_circles(
        MMD_EAST_STATIONS, range_for("MYEAST"),
    ):
        radar.append(Source("MET Malaysia (East / Borneo)", mmd_color, poly))

    return radar


def build_model_sources() -> list[Source]:
    models: list[Source] = []

    # HRRR-CONUS (LCC)
    hrrr_crs = CRS.from_proj4(
        "+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 "
        "+R=6371229 +no_defs"
    )
    hrrr_x_max = -2699020.143 + 1798 * 3000
    hrrr_y_min = 1588193.847 - 1058 * 3000
    models.append(Source(
        "HRRR-CONUS", "#d62728",
        project_grid_perimeter(
            hrrr_crs, -2699020.143, hrrr_x_max, hrrr_y_min, 1588193.847,
        ),
    ))

    # HRRR-Alaska (polar stereographic)
    hrrr_ak_crs = CRS.from_proj4(
        "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-135 +R=6371229 +no_defs"
    )
    ak_x_max = -3425051.0 + 1298 * 3000
    ak_y_min = -1344804.1 - 918 * 3000
    models.append(Source(
        "HRRR-Alaska", "#ff9896",
        project_grid_perimeter(
            hrrr_ak_crs, -3425051.0, ak_x_max, ak_y_min, -1344804.1,
        ),
    ))

    # HRDPS (rotated lat/lon).  Pole at (36.08852°N, 65.30514°E); grid
    # 2540 x 1290 at 0.0225° rotated spacing.  Compute rotated bounds
    # by forward-projecting two known geographic corners (SW and SE)
    # and walking the perimeter between them — this script's Cartesian
    # rotation convention differs by 180° from the closed-form COSMO
    # formulas the HRDPS module uses, so spans look "backwards" relative
    # to the module's documented orientation but the resulting polygon
    # is identical in geographic coordinates.
    pole_lat, pole_lon = 36.08852, 65.30514
    rlat_sw, rlon_sw = _rotated_fwd_xy(39.6260, -133.6295, pole_lat, pole_lon)
    rlat_se, rlon_se = _rotated_fwd_xy(27.285, -66.966, pole_lat, pole_lon)
    # SW → SE crosses the ±180° rlon line; unwrap the SE rlon so
    # linspace traces the correct (short) arc rather than the long way
    # around the rotated sphere.
    if rlon_se < rlon_sw:
        rlon_se += 360.0
    rlat_min = rlat_sw
    rlat_max = rlat_sw + 1289 * 0.0225
    models.append(Source(
        "HRDPS (Canada)", "#e377c2",
        rotated_pole_polygon(
            pole_lat=pole_lat, pole_lon=pole_lon,
            rlat_min=rlat_min, rlat_max=rlat_max,
            rlon_min=rlon_sw, rlon_max=rlon_se,
        ),
    ))

    # DMI HARMONIE DINI (LCC over Europe)
    dini_crs = CRS.from_proj4(
        "+proj=lcc +lat_0=55.5 +lon_0=-8.0 +lat_1=55.5 +lat_2=55.5 "
        "+R=6371229 +no_defs"
    )
    dini_fwd = Transformer.from_crs("EPSG:4326", dini_crs, always_xy=True)
    dini_x0, dini_y0 = dini_fwd.transform(-25.422, 39.671)
    dini_x_max = dini_x0 + 1905 * 2000
    dini_y_max = dini_y0 + 1605 * 2000
    models.append(Source(
        "DMI HARMONIE DINI", "#2ca02c",
        project_grid_perimeter(dini_crs, dini_x0, dini_x_max, dini_y0, dini_y_max),
    ))

    # DWD ICON-EU (regular lat/lon)
    models.append(Source(
        "DWD ICON-EU", "#bcbd22",
        latlon_box(-23.5, 62.5, 29.5, 70.5),
    ))

    # AROME Antilles (regular lat/lon, longitudes shifted from 0..360E)
    models.append(Source(
        "AROME Antilles", "#ff7f0e",
        latlon_box(-75.3, -51.7, 9.7, 22.9),
    ))

    # WRF-SMN Argentina (LCC, southern hemisphere, 6,370 km sphere)
    smn_crs = CRS.from_proj4(
        "+proj=lcc +lat_0=-35 +lon_0=-65 +lat_1=-35 +lat_2=-35 "
        "+R=6370000 +no_defs"
    )
    smn_fwd = Transformer.from_crs("EPSG:4326", smn_crs, always_xy=True)
    smn_x0, smn_y0 = smn_fwd.transform(-94.3308, -54.3868)
    smn_x_max = smn_x0 + 998 * 4000
    smn_y_max = smn_y0 + 1248 * 4000
    models.append(Source(
        "WRF-SMN Argentina", "#8c564b",
        project_grid_perimeter(smn_crs, smn_x0, smn_x_max, smn_y0, smn_y_max),
    ))

    return models


# ── Rendering ─────────────────────────────────────────────────────────


def load_countries_polygons(path: Path) -> list[list[np.ndarray]]:
    """Return list of polygons; each is a list of rings (outer + holes)."""
    with path.open() as f:
        gj = json.load(f)
    polygons: list[list[np.ndarray]] = []
    for feature in gj["features"]:
        geom = feature["geometry"]
        if geom is None:
            continue
        gtype = geom["type"]
        if gtype == "Polygon":
            polygons.append([np.asarray(r) for r in geom["coordinates"]])
        elif gtype == "MultiPolygon":
            for poly in geom["coordinates"]:
                polygons.append([np.asarray(r) for r in poly])
    return polygons


def _draw_basemap(ax) -> None:
    for rings in load_countries_polygons(BASEMAP_PATH):
        outer = rings[0]
        if outer.shape[0] < 3:
            continue
        ax.add_patch(PathPatch(
            MplPath(outer), facecolor="#f7f3eb",
            edgecolor="#8b8b8b", linewidth=0.4, zorder=1,
        ))


def _unwrap_longitudes(lon: np.ndarray) -> np.ndarray:
    """Make a polygon's longitude trace continuous across the ±180° line.

    Each consecutive Δlon > 180° gets compensated by subtracting 360°
    from every subsequent point; Δlon < −180° adds 360°.  The result
    may exit the [−180, 180] window — that's intentional, the renderer
    plots three offset copies so axis clipping at ±180° lands the
    correct piece on each side of the map.
    """
    if len(lon) < 2:
        return lon.copy()
    out = lon.astype(float).copy()
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        if d > 180:
            out[i:] -= 360
        elif d < -180:
            out[i:] += 360
    return out


def _draw_polygon(ax, src: Source, alpha_fill: float, hatch: str | None) -> None:
    """Render a coverage polygon with clean handling of antimeridian wrap.

    The polygon's longitudes are first unwrapped into one continuous
    trace (which may extend past ±180°), then the same closed polygon
    is plotted three times — at offsets −360°, 0°, +360° — so whichever
    copy lands inside the [−180°, 180°] axis window is drawn cleanly
    and the others are clipped away by matplotlib.  This avoids the
    half-closed sliver artefact you get from splitting the polygon at
    the wrap and rendering each segment separately.
    """
    poly = src.polygon
    if poly.shape[0] < 3:
        return
    lon_unwrapped = _unwrap_longitudes(poly[:, 0])
    lat = poly[:, 1]
    for offset in (-360.0, 0.0, 360.0):
        shifted_lon = lon_unwrapped + offset
        # Skip copies that can't possibly overlap the visible window.
        if shifted_lon.max() < -180 or shifted_lon.min() > 180:
            continue
        ax.fill(
            shifted_lon, lat,
            facecolor=src.color, edgecolor=src.color,
            alpha=alpha_fill, linewidth=1.2,
            hatch=hatch, zorder=2,
        )


def render(
    sources: list[Source],
    output_path: Path,
    title: str,
    subtitle: str,
    legend_title: str,
    alpha_fill: float = 0.45,
    hatch: str | None = None,
    dedupe_label_prefix: str | None = None,
) -> None:
    """Render one map with the given polygon set.

    ``dedupe_label_prefix`` collapses multiple entries that share a
    common prefix (e.g. "MRMS — CONUS", "MRMS — Alaska") into one
    legend entry titled with the prefix.
    """
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-65, 85)
    ax.set_aspect(1.3)
    ax.set_facecolor("#e8f1f8")

    _draw_basemap(ax)
    for src in sources:
        _draw_polygon(ax, src, alpha_fill=alpha_fill, hatch=hatch)

    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-60, 81, 30))
    ax.grid(True, which="major", color="#c0c0c0", linewidth=0.4, zorder=0)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlabel("Longitude (°)", fontsize=9)
    ax.set_ylabel("Latitude (°)", fontsize=9)
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, pad=12)

    # Legend
    seen: set[str] = set()
    handles: list[Patch] = []
    for s in sources:
        key = s.label
        if dedupe_label_prefix and s.label.startswith(dedupe_label_prefix):
            key = dedupe_label_prefix.rstrip(" —")
        if key in seen:
            continue
        seen.add(key)
        label = key
        if dedupe_label_prefix and key == dedupe_label_prefix.rstrip(" —"):
            # Count distinct labels (not Source rows — each composite may
            # produce several disjoint polygon pieces from the station
            # union, which would otherwise inflate the count).
            distinct = {
                x.label for x in sources
                if x.label.startswith(dedupe_label_prefix)
            }
            label = f"{key} ({len(distinct)} composites)"
        handles.append(Patch(
            facecolor=s.color, edgecolor=s.color,
            alpha=min(0.95, alpha_fill + 0.10), hatch=hatch,
            label=label,
        ))
    ax.legend(
        handles=handles, loc="lower left",
        title=legend_title, fontsize=9, title_fontsize=10,
        framealpha=0.95,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=120, facecolor="white")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    if not BASEMAP_PATH.exists():
        raise SystemExit(
            f"Basemap not found at {BASEMAP_PATH}. Download it first:\n"
            "  curl -L https://raw.githubusercontent.com/nvkelso/"
            "natural-earth-vector/master/geojson/"
            "ne_110m_admin_0_countries.geojson -o /tmp/ne_countries.geojson"
        )

    RADAR_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    render(
        sources=build_radar_sources(),
        output_path=RADAR_OUTPUT,
        title="LibreWXR — Radar Composite Coverage",
        subtitle="NOAA MRMS · MSC Canada · MARN/SNET · OPERA Europe · CWA / QPESUMS Taiwan · MET Malaysia",
        legend_title="Radar composites",
        alpha_fill=0.40,
        hatch="//",
        dedupe_label_prefix="MRMS — ",
    )
    render(
        sources=build_model_sources(),
        output_path=MODEL_OUTPUT,
        title="LibreWXR — Regional NWP Model Coverage",
        subtitle="Chain-dispatched per pixel; ECMWF IFS provides global coverage everywhere else",
        legend_title="Regional NWP models",
        alpha_fill=0.50,
        hatch=None,
    )
