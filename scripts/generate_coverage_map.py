#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Generate the LibreWXR coverage maps.

Writes six PNGs to ``docs/``:

  * ``coverage-map-radar.png``                  — global radar composites
  * ``coverage-map-models.png``                 — global regional NWP grids
  * ``coverage-map-europe-radar.png``           — Europe zoom (radar)
  * ``coverage-map-europe-models.png``          — Europe zoom (models)
  * ``coverage-map-north-america-radar.png``    — N. America zoom (radar)
  * ``coverage-map-north-america-models.png``   — N. America zoom (models)

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

import importlib.util
import json
from dataclasses import dataclass
from math import asin, atan2, cos, degrees, radians, sin
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from pyproj import CRS, Transformer
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
RADAR_OUTPUT = DOCS_DIR / "coverage-map-radar.png"
MODEL_OUTPUT = DOCS_DIR / "coverage-map-models.png"
EUROPE_RADAR_OUTPUT = DOCS_DIR / "coverage-map-europe-radar.png"
EUROPE_MODEL_OUTPUT = DOCS_DIR / "coverage-map-europe-models.png"
NA_RADAR_OUTPUT = DOCS_DIR / "coverage-map-north-america-radar.png"
NA_MODEL_OUTPUT = DOCS_DIR / "coverage-map-north-america-models.png"
BASEMAP_PATH = Path("/tmp/ne_countries.geojson")

# Default radar range — matches ``librewxr.data.coverage.DEFAULT_RADAR_RANGE_KM``.
# Inlined here so this script doesn't need to import the project's runtime
# stack (which pulls in opencv, httpx, pydantic, …) just to read one constant.
RADAR_RANGE_KM = 240.0


def _load_data_module(path: Path):
    """Load a Python module by file path, bypassing package ``__init__.py``.

    Each ``stations.py`` we read is a pure-data file (lists of lat/lon
    tuples + small dicts), but its enclosing package's ``__init__.py``
    typically imports the source's runtime code (httpx, fsspec, h5py,
    pydantic via config, …).  Going through ``importlib.util`` skips
    all of that — this script then runs with only matplotlib + pyproj
    + shapely as its header recipe claims.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SRC = REPO_ROOT / "src" / "librewxr" / "sources" / "regional"
_marn  = _load_data_module(_SRC / "central_america/el_salvador/radar/marn/stations.py")
_cwa   = _load_data_module(_SRC / "east_asia/taiwan/radar/cwa/stations.py")
_opera = _load_data_module(_SRC / "europe/radar/opera/stations.py")
_msc   = _load_data_module(_SRC / "north_america/canada/radar/msc_canada/stations.py")
_usa   = _load_data_module(_SRC / "north_america/usa/radar/stations.py")
_mmd   = _load_data_module(_SRC / "southeast_asia/malaysia/radar/mmd/stations.py")

MARN_RANGES = _marn.RANGE_OVERRIDES
SNET_STATIONS = _marn.STATIONS
CWA_RANGES = _cwa.RANGE_OVERRIDES
CWA_STATIONS = _cwa.STATIONS
OPERA_RANGES = _opera.RANGE_OVERRIDES
OPERA_STATIONS = _opera.STATIONS
CANADA_STATIONS = _msc.STATIONS
NEXRAD_ALASKA = _usa.NEXRAD_ALASKA
NEXRAD_CONUS = _usa.NEXRAD_CONUS
NEXRAD_GUAM = _usa.NEXRAD_GUAM
NEXRAD_HAWAII = _usa.NEXRAD_HAWAII
NEXRAD_PUERTO_RICO = _usa.NEXRAD_PUERTO_RICO
MMD_EAST_STATIONS = _mmd.EAST_STATIONS
MMD_PENINSULAR_STATIONS = _mmd.PENINSULAR_STATIONS
MMD_RANGES = _mmd.RANGE_OVERRIDES

# Combined per-region range map for the map renderer.  Mirrors what
# ``data.coverage`` builds at runtime — provider-supplied range overrides
# fall back to ``RADAR_RANGE_KM`` (240 km Doppler) for anything else.
REGION_RADAR_RANGE: dict[str, float] = {
    **MARN_RANGES,
    **CWA_RANGES,
    **OPERA_RANGES,
    **MMD_RANGES,
}


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

    Reads the station lists straight from each
    ``librewxr.sources.regional.../radar/<source>/stations.py`` so adding
    or removing a station propagates through to the map automatically.
    Range is ``RADAR_RANGE_KM`` (240 km) by default, overridden per-region
    via each source's ``RANGE_OVERRIDES`` (300 km for OPERA's C-band
    network, 120 km for the SNET overlay, 450 km for the CWA typhoon
    extent) — same numbers the runtime coverage mask uses.
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

    # AROME Guyane — French Guiana + neighbours
    models.append(Source(
        "AROME Guyane", "#9467bd",
        latlon_box(-56.75, -46.30, 1.05, 8.95),
    ))

    # AROME Indien — Réunion/Mayotte/Madagascar/Comoros (largest AROME-OM grid)
    models.append(Source(
        "AROME Indien", "#17becf",
        latlon_box(32.75, 67.60, -25.90, -3.45),
    ))

    # AROME Nouvelle-Calédonie
    models.append(Source(
        "AROME Nouvelle-Calédonie", "#1f77b4",
        latlon_box(158.50, 171.50, -26.00, -13.75),
    ))

    # AROME Polynésie (Society + Tuamotu archipelagoes)
    models.append(Source(
        "AROME Polynésie", "#c5b0d5",
        latlon_box(-157.50, -144.50, -25.25, -12.60),
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


def _draw_basemap(
    ax,
    bounds: tuple[float, float, float, float] | None = None,
    linewidth: float = 0.4,
) -> None:
    """Draw country polygons.  When ``bounds`` is given, skip countries
    whose extent doesn't overlap the (lon_min, lon_max, lat_min, lat_max)
    window — keeps regional figures from spending most of their draw
    time on Antarctica."""
    for rings in load_countries_polygons(BASEMAP_PATH):
        outer = rings[0]
        if outer.shape[0] < 3:
            continue
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            if outer[:, 0].max() < xmin or outer[:, 0].min() > xmax:
                continue
            if outer[:, 1].max() < ymin or outer[:, 1].min() > ymax:
                continue
        ax.add_patch(PathPatch(
            MplPath(outer), facecolor="#f7f3eb",
            edgecolor="#8b8b8b", linewidth=linewidth, zorder=1,
        ))


def _filter_sources_to_bounds(
    sources: list[Source],
    bounds: tuple[float, float, float, float],
) -> list[Source]:
    """Return only the sources whose polygon intersects the bounds box.

    Used to keep regional legends honest — a CONUS view should advertise
    only the radars that actually fall in CONUS, not every MRMS composite
    that exists globally.
    """
    xmin, xmax, ymin, ymax = bounds
    window = shapely_box(xmin, ymin, xmax, ymax)
    out: list[Source] = []
    for src in sources:
        if src.polygon.shape[0] < 3:
            continue
        try:
            if ShapelyPolygon(src.polygon).intersects(window):
                out.append(src)
        except Exception:
            # Polygons that wrap the antimeridian etc. — never relevant
            # for the European or North-American windows we use here.
            pass
    return out


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
    bounds: tuple[float, float, float, float] = (-180, 180, -65, 85),
    aspect: float = 1.3,
    figsize: tuple[float, float] = (16, 9),
    xtick_step: int = 30,
    ytick_step: int = 30,
) -> None:
    """Render one map with the given polygon set.

    ``bounds`` is (lon_min, lon_max, lat_min, lat_max); defaults to a
    world view.  Regional callers pass a tighter window plus a matching
    ``aspect`` (1/cos(mid_latitude)) and tick steps.

    ``dedupe_label_prefix`` collapses multiple entries that share a
    common prefix (e.g. "MRMS — CONUS", "MRMS — Alaska") into one
    legend entry titled with the prefix.
    """
    xmin, xmax, ymin, ymax = bounds
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect(aspect)
    ax.set_facecolor("#e8f1f8")

    _draw_basemap(ax, bounds=bounds)
    for src in sources:
        _draw_polygon(ax, src, alpha_fill=alpha_fill, hatch=hatch)

    # Round ticks to the nearest multiple of the step so axes look tidy
    # at arbitrary bounds.
    xstart = int(np.ceil(xmin / xtick_step) * xtick_step)
    xstop = int(np.floor(xmax / xtick_step) * xtick_step) + 1
    ystart = int(np.ceil(ymin / ytick_step) * ytick_step)
    ystop = int(np.floor(ymax / ytick_step) * ytick_step) + 1
    ax.set_xticks(np.arange(xstart, xstop, xtick_step))
    ax.set_yticks(np.arange(ystart, ystop, ytick_step))
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

    # ── Regional zoom: Europe ────────────────────────────────────────
    # Window roughly Iceland → Caucasus, N. Africa → Svalbard.  Aspect
    # 1/cos(52°) ≈ 1.62 keeps shapes square at the mid-latitude.
    europe_bounds = (-25.0, 45.0, 30.0, 75.0)
    europe_aspect = 1.0 / cos(radians(52.0))
    render(
        sources=_filter_sources_to_bounds(build_radar_sources(), europe_bounds),
        output_path=EUROPE_RADAR_OUTPUT,
        title="LibreWXR — Radar Composite Coverage (Europe)",
        subtitle="OPERA C-band composite — ~155 national radars across the EU + neighbours",
        legend_title="Radar composite",
        alpha_fill=0.40,
        hatch="//",
        bounds=europe_bounds,
        aspect=europe_aspect,
        figsize=(13, 11),
        xtick_step=10, ytick_step=5,
    )
    render(
        sources=_filter_sources_to_bounds(build_model_sources(), europe_bounds),
        output_path=EUROPE_MODEL_OUTPUT,
        title="LibreWXR — Regional NWP Coverage (Europe)",
        subtitle="DMI HARMONIE DINI + DWD ICON-EU; ECMWF IFS provides global coverage everywhere else",
        legend_title="Regional NWP models",
        alpha_fill=0.45,
        bounds=europe_bounds,
        aspect=europe_aspect,
        figsize=(13, 11),
        xtick_step=10, ytick_step=5,
    )

    # ── Regional zoom: North America ─────────────────────────────────
    # Wide enough to catch a future Caribbean tier (Cayman, Bermuda,
    # PRCOMP) without crowding the CONUS+Canada main story.  Alaska is
    # off-frame to the west; it gets the global view.  Aspect at ~40°N.
    na_bounds = (-141.0, -52.0, 8.0, 72.0)
    na_aspect = 1.0 / cos(radians(40.0))
    render(
        sources=_filter_sources_to_bounds(build_radar_sources(), na_bounds),
        output_path=NA_RADAR_OUTPUT,
        title="LibreWXR — Radar Composite Coverage (North America)",
        subtitle="NOAA MRMS (CONUS / Puerto Rico) + MSC Canada — Caribbean radars to follow",
        legend_title="Radar composites",
        alpha_fill=0.40,
        hatch="//",
        dedupe_label_prefix="MRMS — ",
        bounds=na_bounds,
        aspect=na_aspect,
        figsize=(13, 11),
        xtick_step=10, ytick_step=10,
    )
    render(
        sources=_filter_sources_to_bounds(build_model_sources(), na_bounds),
        output_path=NA_MODEL_OUTPUT,
        title="LibreWXR — Regional NWP Coverage (North America)",
        subtitle="NOAA HRRR (CONUS+Alaska) + ECCC HRDPS + Météo-France AROME-Antilles/Guyane",
        legend_title="Regional NWP models",
        alpha_fill=0.45,
        bounds=na_bounds,
        aspect=na_aspect,
        figsize=(13, 11),
        xtick_step=10, ytick_step=10,
    )
