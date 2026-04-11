# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Precomputed radar coverage masks.

At startup, build a boolean mask per region marking which pixels (in a
coarse lat/lon grid) lie within range of any radar station. The tile
renderer consults this to decide whether an empty pixel should receive
ECMWF fallback — previously we relied on ``values == 0``, but IEM N0Q
encodes "clear sky within radar range" and "outside radar range"
identically, causing either bleed-through or cutouts at the coverage
boundary.
"""
from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from librewxr.data.radar_stations import (
    RADAR_RANGE_KM,
    REGION_RADAR_RANGE,
    REGION_STATIONS,
)
from librewxr.data.regions import REGIONS, RegionDef

logger = logging.getLogger(__name__)

# Coarse grid resolution for coverage masks. 0.05° ≈ 5.5 km at the equator,
# much finer than the ~240 km radar range so blob edges are smooth.
MASK_RESOLUTION_DEG = 0.05

# Station coverage mask cache: region name -> (mask, west, south, dx, dy)
_COVERAGE_MASKS: dict[str, tuple[np.ndarray, float, float, float, float]] = {}


def _build_region_mask(region: RegionDef, stations: list[tuple[float, float]]) -> None:
    """Build a boolean coverage mask for one region and store it.

    Uses an equirectangular approximation (valid for regional bboxes):
    distance ≈ sqrt((Δlat·111)² + (Δlon·111·cos(lat))²) in km.
    """
    west, east = region.west, region.east
    south, north = region.south, region.north

    # Mask grid covering the region's bbox at MASK_RESOLUTION_DEG.
    dx = MASK_RESOLUTION_DEG
    dy = MASK_RESOLUTION_DEG
    nx = max(1, int(math.ceil((east - west) / dx)))
    ny = max(1, int(math.ceil((north - south) / dy)))

    # Pixel centers
    lon_axis = west + (np.arange(nx) + 0.5) * dx
    lat_axis = south + (np.arange(ny) + 0.5) * dy

    lat_grid, lon_grid = np.meshgrid(lat_axis, lon_axis, indexing="ij")

    mask = np.zeros((ny, nx), dtype=bool)
    range_km = REGION_RADAR_RANGE.get(region.name, RADAR_RANGE_KM)
    range_km_sq = range_km * range_km

    for st_lat, st_lon in stations:
        dlat_km = (lat_grid - st_lat) * 111.0
        # Use station's own latitude for cos factor (good enough within 240 km).
        dlon_km = (lon_grid - st_lon) * 111.0 * math.cos(math.radians(st_lat))
        d2 = dlat_km * dlat_km + dlon_km * dlon_km
        mask |= d2 <= range_km_sq

    _COVERAGE_MASKS[region.name] = (mask, west, south, dx, dy)
    logger.info(
        "coverage mask %s: %dx%d @ %.2f° (%d stations, %.1f%% covered)",
        region.name, ny, nx, MASK_RESOLUTION_DEG, len(stations),
        100.0 * mask.mean(),
    )


def build_coverage_masks() -> None:
    """Build coverage masks for every region with known stations."""
    for region_name, stations in REGION_STATIONS.items():
        region = REGIONS.get(region_name)
        if region is None:
            continue
        _build_region_mask(region, stations)


def sample_coverage(
    region_name: str, lat_grid: np.ndarray, lon_grid: np.ndarray,
) -> np.ndarray:
    """Return a boolean array: True where the point is within radar range.

    ``lat_grid`` and ``lon_grid`` have matching shape. If no mask exists
    for the region (e.g. GERMANY, whose composite has a proper footprint),
    returns an all-True array — meaning "assume the whole region is covered".
    """
    entry = _COVERAGE_MASKS.get(region_name)
    if entry is None:
        return np.ones(lat_grid.shape, dtype=bool)

    mask, west, south, dx, dy = entry
    ny, nx = mask.shape

    col = np.floor((lon_grid - west) / dx).astype(np.int32)
    row = np.floor((lat_grid - south) / dy).astype(np.int32)

    in_bounds = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    # Clamp for safe indexing, then mask out-of-bounds to False.
    col_c = np.clip(col, 0, nx - 1)
    row_c = np.clip(row, 0, ny - 1)
    result = mask[row_c, col_c]
    return result & in_bounds


# ---------------------------------------------------------------------------
# Feather masks: distance-transform gradient at coverage boundaries
# ---------------------------------------------------------------------------
# Used by nowcast blending to smoothly transition between extrapolated radar
# and IFS forecast at the edge of radar coverage, preventing hard seams.

# Distance (in coverage mask pixels) over which the feather ramps from 0 to 1.
# At MASK_RESOLUTION_DEG=0.05°, 15 pixels ≈ 0.75° ≈ 80 km at mid-latitudes.
FEATHER_DISTANCE_PX = 15

# Feather mask cache: region name -> (feather, west, south, dx, dy)
_FEATHER_MASKS: dict[str, tuple[np.ndarray, float, float, float, float]] = {}


def build_feather_masks() -> None:
    """Build feather masks from existing coverage masks.

    Must be called after ``build_coverage_masks()``.  For each region
    with a coverage mask, computes a distance transform from the mask
    boundary inward and normalizes to 0–1 over ``FEATHER_DISTANCE_PX``.
    """
    for region_name, (mask, west, south, dx, dy) in _COVERAGE_MASKS.items():
        # cv2.distanceTransform needs uint8: 255 inside coverage, 0 outside
        mask_uint8 = mask.astype(np.uint8) * 255
        dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        feather = np.clip(dist / FEATHER_DISTANCE_PX, 0.0, 1.0).astype(np.float32)
        _FEATHER_MASKS[region_name] = (feather, west, south, dx, dy)
        logger.info(
            "feather mask %s: %dx%d, feather_px=%d",
            region_name, feather.shape[0], feather.shape[1], FEATHER_DISTANCE_PX,
        )


def sample_feather(
    region_name: str, lat_grid: np.ndarray, lon_grid: np.ndarray,
) -> np.ndarray:
    """Return a float array 0–1: how far inside radar coverage each point is.

    0.0 = at the coverage boundary or outside,
    1.0 = well inside coverage (≥ ``FEATHER_DISTANCE_PX`` mask pixels from edge).

    If no feather mask exists for the region, returns all-ones (no feathering).
    """
    entry = _FEATHER_MASKS.get(region_name)
    if entry is None:
        return np.ones(lat_grid.shape, dtype=np.float32)

    feather, west, south, dx, dy = entry
    ny, nx = feather.shape

    col = np.floor((lon_grid - west) / dx).astype(np.int32)
    row = np.floor((lat_grid - south) / dy).astype(np.int32)

    in_bounds = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    col_c = np.clip(col, 0, nx - 1)
    row_c = np.clip(row, 0, ny - 1)
    result = feather[row_c, col_c]
    result[~in_bounds] = 0.0
    return result
