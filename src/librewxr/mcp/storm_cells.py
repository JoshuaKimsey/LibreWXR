# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Inverse-projection helpers for the MCP get_storm_cells tool.

Converts storm-cell centroids from region pixel coordinates (row, col)
back to geographic (lat, lon).  For latlon regions the math is direct
and exact; for laea and tmerc regions we build a cached 100x100 inverse
grid (lat/lon -> pixel coords via forward projection) and look up the
nearest grid point.
"""

import math
import logging

import numpy as np
from functools import lru_cache

from librewxr.data.regions import REGIONS, RegionDef
from librewxr.tiles.coordinates import _laea_forward, _tmerc_forward

logger = logging.getLogger(__name__)

# Coarse grid resolution for the inverse-projection lookup.  100x100
# means each grid cell covers ~1% of the region extent -- for a 1000x1000
# pixel USCOMP region spanning ~20 degrees, that's ~0.2 deg approx 22 km
# resolution, more than enough for a storm-cell centroid (the cell itself
# is typically 5-50 km across).  Higher values trade memory for accuracy;
# 100 is a good default.
_INVERSE_GRID_N = 100


@lru_cache(maxsize=64)
def _build_inverse_grid(region: RegionDef) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a coarse lat/lon -> pixel-coord lookup grid for a region.

    For latlon regions, the inverse is direct math and this grid is not
    needed (but building it is harmless -- the direct path is just faster).
    For laea/tmerc regions, this grid lets us convert pixel coords back to
    lat/lon by nearest-neighbor lookup: build a 100x100 grid of lat/lon
    points over the region's bbox, project each FORWARD to get its pixel
    coords, then for a cell at (row, col) find the nearest grid point and
    read back its lat/lon.

    Returns (lat_grid, lon_grid, row_grid, col_grid) -- all shape
    (N, N) float arrays.  ``lat_grid[i,j]`` and ``lon_grid[i,j]`` are the
    geographic coords of the (i,j) grid point; ``row_grid[i,j]`` and
    ``col_grid[i,j]`` are that point's region-pixel coords (fractional).
    """
    n = _INVERSE_GRID_N
    lats = np.linspace(region.south, region.north, n, dtype=np.float64)
    lons = np.linspace(region.west, region.east, n, dtype=np.float64)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    if region.proj == "latlon":
        # Direct math: row = (north - lat) / _ps_y, col = (lon - west) / pixel_size
        row_grid = (region.north - lat_grid) / region._ps_y
        col_grid = (lon_grid - region.west) / region.pixel_size
    elif region.proj == "laea":
        x, y = _laea_forward(lon_grid.ravel(), lat_grid.ravel(), region)
        col_flat = (x.ravel() - region.grid_x_min) / region.grid_scale
        row_flat = (region.grid_y_max - y.ravel()) / region.grid_scale
        row_grid = row_flat.reshape(n, n)
        col_grid = col_flat.reshape(n, n)
    elif region.proj == "tmerc":
        x, y = _tmerc_forward(lon_grid.ravel(), lat_grid.ravel(), region)
        col_flat = (x.ravel() - region.grid_x_min) / region.grid_scale
        row_flat = (region.grid_y_max - y.ravel()) / region.grid_scale
        row_grid = row_flat.reshape(n, n)
        col_grid = col_flat.reshape(n, n)
    else:
        raise ValueError(f"Unknown projection '{region.proj}' for region '{region.name}'")

    return lat_grid, lon_grid, row_grid, col_grid


def cell_pixel_to_latlon(region: RegionDef, row: float, col: float) -> tuple[float, float]:
    """Convert a region-pixel (row, col) to geographic (lat, lon).

    For latlon regions, uses direct arithmetic (exact).  For laea/tmerc
    regions, uses a nearest-neighbor lookup on the cached 100x100 inverse
    grid built by ``_build_inverse_grid``.  Accuracy is ~1% of the region
    extent (typically 1-2 km for a midlatitude region) -- more than enough
    for a storm-cell centroid.
    """
    if region.proj == "latlon":
        lat = region.north - row * region._ps_y
        lon = region.west + col * region.pixel_size
        return float(lat), float(lon)

    # Projected region: nearest-neighbor on the cached inverse grid.
    lat_grid, lon_grid, row_grid, col_grid = _build_inverse_grid(region)
    d2 = (row_grid - row) ** 2 + (col_grid - col) ** 2
    flat_idx = int(d2.argmin())
    i, j = divmod(flat_idx, _INVERSE_GRID_N)
    return float(lat_grid[i, j]), float(lon_grid[i, j])
