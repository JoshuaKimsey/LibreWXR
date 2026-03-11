# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import math
from functools import lru_cache

import numpy as np

from librewrx.data.regions import REGIONS, RegionDef

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

# ── LCC projection ─────────────────────────────────────────────────


def _lcc_forward(
    lon: np.ndarray, lat: np.ndarray, region: RegionDef
) -> tuple[np.ndarray, np.ndarray]:
    """Lambert Conformal Conic forward projection (vectorized).

    Converts lon/lat (degrees) to projected x/y (meters) using the
    LCC parameters stored in the RegionDef.
    """
    lat_r = np.radians(lat)
    lat_0_r = math.radians(region.lcc_lat0)
    lat_1_r = math.radians(region.lcc_lat1)
    lon_0_r = math.radians(region.lcc_lon0)
    R = region.lcc_R

    n = math.sin(lat_1_r)
    F = math.cos(lat_1_r) * math.tan(math.pi / 4 + lat_1_r / 2) ** n / n
    rho = R * F / np.tan(np.pi / 4 + lat_r / 2) ** n
    rho_0 = R * F / math.tan(math.pi / 4 + lat_0_r / 2) ** n
    theta = n * (np.radians(lon) - lon_0_r)

    x = rho * np.sin(theta)
    y = rho_0 - rho * np.cos(theta)
    return x, y


def _lcc_pixel_coords(
    lon: np.ndarray, lat: np.ndarray, region: RegionDef
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lon/lat 1D arrays to 2D grid of (col_f, row_f) for an LCC region."""
    # Build 2D grids from 1D lon/lat (LCC is non-separable)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y = _lcc_forward(lon_grid, lat_grid, region)
    col_grid = (x - region.grid_x_min) / region.grid_scale
    row_grid = (region.grid_y_max - y) / region.grid_scale
    return col_grid, row_grid


# ── Polar stereographic projection ────────────────────────────────


def _stere_forward(
    lon: np.ndarray, lat: np.ndarray, region: RegionDef
) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 ellipsoidal north polar stereographic forward projection (vectorized).

    Converts lon/lat (degrees) to projected x/y (meters) using the
    polar stereographic parameters stored in the RegionDef.
    """
    lat_ts_r = math.radians(region.stere_lat_ts)
    lon_0_r = math.radians(region.stere_lon0)

    # Conformal latitude at true-scale parallel
    sp_ts = math.sin(lat_ts_r)
    t_c = math.tan(math.pi / 4 - lat_ts_r / 2) * (
        (1 + _WGS84_E * sp_ts) / (1 - _WGS84_E * sp_ts)
    ) ** (_WGS84_E / 2)
    m_c = math.cos(lat_ts_r) / math.sqrt(1 - _WGS84_E2 * sp_ts ** 2)

    lat_r = np.radians(lat)
    sp = np.sin(lat_r)
    t = np.tan(np.pi / 4 - lat_r / 2) * (
        (1 + _WGS84_E * sp) / (1 - _WGS84_E * sp)
    ) ** (_WGS84_E / 2)
    rho = _WGS84_A * m_c * t / t_c

    lam_diff = np.radians(lon) - lon_0_r
    x = rho * np.sin(lam_diff) + region.stere_x0
    y = -rho * np.cos(lam_diff) + region.stere_y0
    return x, y


def _stere_pixel_coords(
    lon: np.ndarray, lat: np.ndarray, region: RegionDef
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lon/lat 1D arrays to 2D grid of (col_f, row_f) for a stere region."""
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y = _stere_forward(lon_grid, lat_grid, region)
    col_grid = (x - region.grid_x_min) / region.grid_scale
    row_grid = (region.grid_y_max - y) / region.grid_scale
    return col_grid, row_grid


# ── Region-aware coordinate functions ────────────────────────────────


@lru_cache(maxsize=8192)
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

    if region.proj == "lcc":
        col_grid, row_grid = _lcc_pixel_coords(lon, lat, region)
    elif region.proj == "stere":
        col_grid, row_grid = _stere_pixel_coords(lon, lat, region)
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


@lru_cache(maxsize=8192)
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

    if region.proj == "lcc":
        col_grid, row_grid = _lcc_pixel_coords(lon, lat, region)
    elif region.proj == "stere":
        col_grid, row_grid = _stere_pixel_coords(lon, lat, region)
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


@lru_cache(maxsize=8192)
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

    if region.proj == "lcc":
        col_grid, row_grid = _lcc_pixel_coords(lon, lat, region)
    elif region.proj == "stere":
        col_grid, row_grid = _stere_pixel_coords(lon, lat, region)
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


@lru_cache(maxsize=8192)
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


@lru_cache(maxsize=8192)
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


@lru_cache(maxsize=4096)
def tile_pixel_indices(
    z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute USCOMP pixel indices for a tile (legacy wrapper)."""
    return region_pixel_indices(_USCOMP, z, x, y, tile_size)


@lru_cache(maxsize=4096)
def tile_pixel_indices_padded(
    z: int, x: int, y: int, tile_size: int = 256, pad: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Compute USCOMP pixel indices with padding (legacy wrapper)."""
    return region_pixel_indices_padded(_USCOMP, z, x, y, tile_size, pad)


@lru_cache(maxsize=4096)
def tile_pixel_indices_fractional(
    z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute USCOMP fractional indices (legacy wrapper)."""
    return region_pixel_indices_fractional(_USCOMP, z, x, y, tile_size)


def tile_overlaps_composite(z: int, x: int, y: int) -> bool:
    """Check if a tile overlaps USCOMP (legacy wrapper)."""
    return tile_overlaps_region(_USCOMP, z, x, y)
