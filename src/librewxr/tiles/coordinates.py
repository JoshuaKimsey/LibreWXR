# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import math
from functools import lru_cache

import numpy as np

from librewxr.config import settings
from librewxr.data.regions import REGIONS, RegionDef

# Legacy constants for USCOMP (kept for backward compatibility)
_USCOMP = REGIONS["USCOMP"]
WEST = _USCOMP.west
EAST = _USCOMP.east
NORTH = _USCOMP.north
SOUTH = _USCOMP.south
PIXEL_SIZE = _USCOMP.pixel_size
COMPOSITE_WIDTH = _USCOMP.width
COMPOSITE_HEIGHT = _USCOMP.height


# ── WGS84 ellipsoidal constants ────────────────────────────────────

_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_WGS84_E2 = 2 * _WGS84_F - _WGS84_F ** 2
_WGS84_E = math.sqrt(_WGS84_E2)

# ── Lambert Azimuthal Equal Area (LAEA) projection ────────────────────


def _laea_forward(
    lon: np.ndarray, lat: np.ndarray, region: RegionDef
) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 ellipsoidal Lambert Azimuthal Equal Area forward projection.

    Implements the oblique case per Snyder (1987) §24 / EPSG guidance
    note 7-2.  The projection parameters are taken from the RegionDef's
    ``laea_*`` fields (lat_0, lon_0, x_0, y_0).
    """
    phi_0 = math.radians(region.laea_lat0)
    lam_0 = math.radians(region.laea_lon0)

    # Eccentricity-derived constants at the origin latitude
    sin_phi0 = math.sin(phi_0)
    cos_phi0 = math.cos(phi_0)
    q_p = (1 - _WGS84_E2) * (
        1 / (1 - _WGS84_E2) - (1 / (2 * _WGS84_E)) * math.log((1 - _WGS84_E) / (1 + _WGS84_E))
    )
    q_0 = _laea_q(sin_phi0)
    beta_0 = math.asin(q_0 / q_p)
    R_q = _WGS84_A * math.sqrt(q_p / 2)
    D = _WGS84_A * cos_phi0 / (
        math.sqrt(1 - _WGS84_E2 * sin_phi0 ** 2) * R_q * math.cos(beta_0)
    )

    # Per-point computations (vectorized)
    phi = np.radians(lat)
    lam = np.radians(lon)
    sin_phi = np.sin(phi)
    q = _laea_q_vec(sin_phi)
    beta = np.arcsin(np.clip(q / q_p, -1.0, 1.0))

    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    lam_diff = lam - lam_0

    B = R_q * np.sqrt(
        2.0 / (
            1
            + math.sin(beta_0) * sin_beta
            + math.cos(beta_0) * cos_beta * np.cos(lam_diff)
        )
    )

    x = B * D * cos_beta * np.sin(lam_diff) + region.laea_x0
    y = (B / D) * (
        math.cos(beta_0) * sin_beta
        - math.sin(beta_0) * cos_beta * np.cos(lam_diff)
    ) + region.laea_y0

    return x, y


def _laea_q(sin_phi: float) -> float:
    """Authalic latitude helper q (scalar)."""
    return (1 - _WGS84_E2) * (
        sin_phi / (1 - _WGS84_E2 * sin_phi ** 2)
        - (1 / (2 * _WGS84_E)) * math.log(
            (1 - _WGS84_E * sin_phi) / (1 + _WGS84_E * sin_phi)
        )
    )


def _laea_q_vec(sin_phi: np.ndarray) -> np.ndarray:
    """Authalic latitude helper q (vectorized)."""
    return (1 - _WGS84_E2) * (
        sin_phi / (1 - _WGS84_E2 * sin_phi ** 2)
        - (1 / (2 * _WGS84_E)) * np.log(
            (1 - _WGS84_E * sin_phi) / (1 + _WGS84_E * sin_phi)
        )
    )


def _laea_pixel_coords(
    lon: np.ndarray, lat: np.ndarray, region: RegionDef
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lon/lat 1D arrays to 2D grid of (col_f, row_f) for a LAEA region."""
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y = _laea_forward(lon_grid, lat_grid, region)
    col_grid = (x - region.grid_x_min) / region.grid_scale
    row_grid = (region.grid_y_max - y) / region.grid_scale
    return col_grid, row_grid


# ── Region-aware coordinate functions ────────────────────────────────


@lru_cache(maxsize=settings.coord_cache_size)
def region_pixel_indices(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute composite pixel indices for a tile within a specific region.

    Returns (row_indices, col_indices) arrays of shape (tile_size, tile_size).
    Values of -1 indicate pixels outside the region's coverage.
    """
    n = 2**z
    cx = np.arange(tile_size, dtype=np.float64) + 0.5
    cy = np.arange(tile_size, dtype=np.float64) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    if region.proj == "laea":
        col_grid, row_grid = _laea_pixel_coords(lon, lat, region)
    else:
        col_f = (lon - region.west) / region.pixel_size
        row_f = (region.north - lat) / region._ps_y
        col_grid, row_grid = np.meshgrid(col_f, row_f)

    col_idx = np.rint(col_grid).astype(np.int32)
    row_idx = np.rint(row_grid).astype(np.int32)

    oob = (
        (col_idx < 0)
        | (col_idx >= region.width)
        | (row_idx < 0)
        | (row_idx >= region.height)
    )
    col_idx[oob] = -1
    row_idx[oob] = -1

    col_idx.flags.writeable = False
    row_idx.flags.writeable = False
    return row_idx, col_idx


@lru_cache(maxsize=settings.coord_cache_size)
def region_pixel_indices_padded(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256, pad: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Compute composite pixel indices for a tile with padding within a region."""
    n = 2**z
    cx = np.arange(-pad, tile_size + pad, dtype=np.float64) + 0.5
    cy = np.arange(-pad, tile_size + pad, dtype=np.float64) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    if region.proj == "laea":
        col_grid, row_grid = _laea_pixel_coords(lon, lat, region)
    else:
        col_f = (lon - region.west) / region.pixel_size
        row_f = (region.north - lat) / region._ps_y
        col_grid, row_grid = np.meshgrid(col_f, row_f)

    col_idx = np.rint(col_grid).astype(np.int32)
    row_idx = np.rint(row_grid).astype(np.int32)

    oob = (
        (col_idx < 0)
        | (col_idx >= region.width)
        | (row_idx < 0)
        | (row_idx >= region.height)
    )
    col_idx[oob] = -1
    row_idx[oob] = -1

    col_idx.flags.writeable = False
    row_idx.flags.writeable = False
    return row_idx, col_idx


@lru_cache(maxsize=settings.coord_cache_size)
def region_pixel_indices_fractional(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fractional composite pixel coordinates for bilinear interpolation."""
    n = 2**z
    cx = np.arange(tile_size, dtype=np.float64) + 0.5
    cy = np.arange(tile_size, dtype=np.float64) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    if region.proj == "laea":
        col_grid, row_grid = _laea_pixel_coords(lon, lat, region)
    else:
        col_f = (lon - region.west) / region.pixel_size
        row_f = (region.north - lat) / region._ps_y
        col_grid, row_grid = np.meshgrid(col_f, row_f)

    row_grid = np.clip(row_grid, 0, region.height - 1).astype(np.float32)
    col_grid = np.clip(col_grid, 0, region.width - 1).astype(np.float32)

    row_grid.flags.writeable = False
    col_grid.flags.writeable = False
    return row_grid, col_grid


def tile_overlaps_region(region: RegionDef, z: int, x: int, y: int) -> bool:
    """Check if a tile has any overlap with a region's coverage area."""
    tw, ts, te, tn = tile_bounds(z, x, y)
    return not (
        te < region.west or tw > region.east
        or tn < region.south or ts > region.north
    )


def overlapping_regions(
    z: int, x: int, y: int, enabled: list[str] | None = None
) -> list[RegionDef]:
    """Return list of regions that overlap a given tile.

    Sorted by pixel_size ascending (finest resolution first).
    """
    if enabled is None:
        enabled = list(REGIONS.keys())

    result = []
    for name in enabled:
        region = REGIONS.get(name)
        if region and tile_overlaps_region(region, z, x, y):
            result.append(region)

    # Finest resolution first (smallest pixel_size)
    result.sort(key=lambda r: r.pixel_size)
    return result


@lru_cache(maxsize=settings.coord_cache_size)
def tile_pixel_latlons(
    z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lat/lon for each pixel in a Web Mercator tile.

    Returns (lat_grid, lon_grid) float64 arrays of shape (tile_size, tile_size).
    Used for temperature lookups that need geographic coordinates.
    """
    n = 2**z
    cx = np.arange(tile_size, dtype=np.float64) + 0.5
    cy = np.arange(tile_size, dtype=np.float64) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_grid.flags.writeable = False
    lat_grid.flags.writeable = False
    return lat_grid, lon_grid


@lru_cache(maxsize=settings.coord_cache_size)
def tile_pixel_latlons_padded(
    z: int, x: int, y: int, tile_size: int = 256, pad: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lat/lon for a tile with padding."""
    n = 2**z
    cx = np.arange(-pad, tile_size + pad, dtype=np.float64) + 0.5
    cy = np.arange(-pad, tile_size + pad, dtype=np.float64) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_grid.flags.writeable = False
    lat_grid.flags.writeable = False
    return lat_grid, lon_grid


# ── Legacy USCOMP-only functions (kept for backward compatibility) ───


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in EPSG:4326 for a tile."""
    n = 2**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


@lru_cache(maxsize=settings.coord_cache_size)
def tile_pixel_indices(
    z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute USCOMP pixel indices for a tile (legacy wrapper)."""
    return region_pixel_indices(_USCOMP, z, x, y, tile_size)


@lru_cache(maxsize=settings.coord_cache_size)
def tile_pixel_indices_padded(
    z: int, x: int, y: int, tile_size: int = 256, pad: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Compute USCOMP pixel indices with padding (legacy wrapper)."""
    return region_pixel_indices_padded(_USCOMP, z, x, y, tile_size, pad)


@lru_cache(maxsize=settings.coord_cache_size)
def tile_pixel_indices_fractional(
    z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute USCOMP fractional indices (legacy wrapper)."""
    return region_pixel_indices_fractional(_USCOMP, z, x, y, tile_size)


def tile_overlaps_composite(z: int, x: int, y: int) -> bool:
    """Check if a tile overlaps USCOMP (legacy wrapper)."""
    return tile_overlaps_region(_USCOMP, z, x, y)
