# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import logging
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from librewxr.config import settings
from librewxr.data.coord_store import (
    CoordStore,
    KIND_FRACTIONAL,
    KIND_FRACTIONAL_PAD,
    KIND_INDICES,
    KIND_INDICES_PAD,
    KIND_LATLON,
    KIND_LATLON_PAD,
)
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

logger = logging.getLogger(__name__)


# ── Shared on-disk coordinate store ────────────────────────────────────────
# Best-effort shared store (see librewxr.data.coord_store).  The six cached
# coordinate functions below publish / read their computed arrays here so
# every render worker maps the same read-only memmap pages instead of
# computing identical arrays per process.  ANY failure degrades to the pure
# in-process compute path — the store is never a single point of failure.

_STORE: CoordStore | None = None
_STORE_DISABLED: bool = False


def _get_store() -> CoordStore | None:
    """Return the lazily-constructed shared store, or None when disabled.

    Call-time gate: bypassed entirely unless ``settings.coord_store_enabled``
    is set AND ``settings.cache_dir`` is non-empty.  A construction failure
    is logged once and disables the store for this process (degrading to
    in-process compute).
    """
    global _STORE, _STORE_DISABLED
    if _STORE_DISABLED:
        return None
    if not settings.coord_store_enabled or not settings.cache_dir:
        return None
    if _STORE is not None:
        return _STORE
    try:
        _STORE = CoordStore(
            Path(settings.cache_dir), settings.get_enabled_regions(),
        )
    except Exception:
        logger.warning(
            "coord_store: failed to initialize; falling back to in-process compute",
            exc_info=True,
        )
        _STORE_DISABLED = True
        return None
    return _STORE


def _reset_coord_store() -> None:
    """Test hook: reset the store singleton and the disabled flag."""
    global _STORE, _STORE_DISABLED
    _STORE = None
    _STORE_DISABLED = False


def prune_shared_coord_store() -> tuple[int, int] | None:
    """Prune the shared coord store to its byte budget. Returns
    (removed_bytes, removed_entries), or None when the store is disabled.
    Best-effort: any failure is logged and returns None. Intended to be
    called once per fetch cycle by the process that owns store maintenance
    (pipeline in multi mode, main process in single mode) - never by
    render workers."""
    store = _get_store()
    if store is None:
        return None
    try:
        removed_bytes, removed_entries = store.prune(
            settings.coord_store_mb * 1024 * 1024
        )
    except Exception:
        logger.warning(
            "coord_store: prune failed; skipping this cycle", exc_info=True,
        )
        return None
    if removed_entries > 0:
        logger.info(
            "coord_store: pruned %d entries (%.1f MB)",
            removed_entries, removed_bytes / (1024 * 1024),
        )
    return removed_bytes, removed_entries


def coord_store_cold() -> bool:
    """True when the shared coord store has no entry for a canonical
    always-warmed key under the CURRENT signature - i.e. a warm pass would
    compute + publish rather than mmap.  False when the store is disabled
    (jitter buys nothing without cross-worker dedup) or the probe key is
    present.  Best-effort: any error returns False.

    The probe is the z=0 whole-earth lat/lon entry, which every
    ``warm_coordinate_caches`` pass publishes (tile 0/0 overlaps every
    region, and the plain latlon grid is warmed unconditionally).  Because
    ``entry_path`` folds the current store signature into the key, a code
    or region change (new signature) correctly reports cold even while
    old-generation files remain on disk.
    """
    store = _get_store()
    if store is None:
        return False
    try:
        probe = store.entry_path(KIND_LATLON, None, 0, 0, 0, 256, 0)
        return not probe.exists()
    except Exception:
        logger.debug(
            "coord_store: cold-probe failed; assuming warm", exc_info=True,
        )
        return False


def _try_open(
    store: CoordStore, kind: str, region_name: str | None,
    z: int, x: int, y: int, tile_size: int, pad: int,
    expected_shape: tuple[int, ...], dtype: np.dtype,
) -> np.ndarray | None:
    """Store read wrapped in try/except so a store bug can't break rendering."""
    try:
        return store.open(
            kind, region_name, z, x, y, tile_size, pad, expected_shape, dtype,
        )
    except Exception:
        logger.warning(
            "coord_store: open(%s, %s, %d, %d, %d, %d, %d) failed; "
            "falling back to compute",
            kind, region_name, z, x, y, tile_size, pad,
            exc_info=True,
        )
        return None


def _try_publish(
    store: CoordStore, kind: str, region_name: str | None,
    z: int, x: int, y: int, tile_size: int, pad: int,
    data: np.ndarray,
) -> None:
    """Store write wrapped in try/except (best-effort, never raises)."""
    try:
        store.publish(kind, region_name, z, x, y, tile_size, pad, data)
    except Exception:
        logger.warning(
            "coord_store: publish(%s, %s, %d, %d, %d, %d, %d) failed",
            kind, region_name, z, x, y, tile_size, pad,
            exc_info=True,
        )


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


# ── Transverse Mercator projection (spherical) ────────────────────────
# Snyder (1987) §8 — used by composites that specify their grid on a
# sphere rather than an ellipsoid (DPC Italy: R=6371229, lat_0=42°,
# lon_0=12.5°).  For met composites that stay well inside ±1000 km of
# the central meridian, the spherical form is accurate to a few metres
# vs. the WGS84 ellipsoidal form — well below 1 km/pixel resolution.


def _tmerc_forward(
    lon: np.ndarray, lat: np.ndarray, region: RegionDef
) -> tuple[np.ndarray, np.ndarray]:
    """Spherical Transverse Mercator forward projection.

    Reads parameters from the RegionDef's ``tmerc_*`` fields.  The
    formulas have singularities at λ - λ₀ = ±90°; those are far outside
    every meteorological domain we care about — Italy's λ range stays
    within ±8° of λ₀=12.5°.
    """
    phi_0 = math.radians(region.tmerc_lat0)
    lam_0 = math.radians(region.tmerc_lon0)
    R = region.tmerc_radius
    k0 = region.tmerc_k0

    phi = np.radians(lat)
    lam = np.radians(lon)
    lam_diff = lam - lam_0

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_dlam = np.cos(lam_diff)
    sin_dlam = np.sin(lam_diff)

    # x = R · k0 · arctanh(cos(φ) · sin(λ - λ₀))
    # Clip just inside ±1 to keep arctanh finite at the antipodal cusp.
    B = cos_phi * sin_dlam
    B = np.clip(B, -0.9999999, 0.9999999)
    x = R * k0 * np.arctanh(B)

    # y = R · k0 · (atan2(sin(φ), cos(φ) · cos(λ-λ₀)) - φ₀) — quadrant-safe
    # equivalent of the textbook arctan(tan(φ) / cos(λ-λ₀)) form.
    y = R * k0 * (np.arctan2(sin_phi, cos_phi * cos_dlam) - phi_0)

    return x, y


def _tmerc_pixel_coords(
    lon: np.ndarray, lat: np.ndarray, region: RegionDef
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lon/lat 1D arrays to 2D grid of (col_f, row_f) for a tmerc region."""
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x, y = _tmerc_forward(lon_grid, lat_grid, region)
    col_grid = (x - region.grid_x_min) / region.grid_scale
    row_grid = (region.grid_y_max - y) / region.grid_scale
    return col_grid, row_grid


# ── Region-aware coordinate functions ────────────────────────────────


def _compute_region_pixel_indices(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Uncached compute body of ``region_pixel_indices``."""
    n = 2**z
    cx = np.arange(tile_size, dtype=np.float64) + 0.5
    cy = np.arange(tile_size, dtype=np.float64) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    if region.proj == "laea":
        col_grid, row_grid = _laea_pixel_coords(lon, lat, region)
    elif region.proj == "tmerc":
        col_grid, row_grid = _tmerc_pixel_coords(lon, lat, region)
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
def region_pixel_indices(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute composite pixel indices for a tile within a specific region.

    Returns (row_indices, col_indices) arrays of shape (tile_size, tile_size).
    Values of -1 indicate pixels outside the region's coverage.
    """
    store = _get_store()
    if store is not None:
        arr = _try_open(
            store, KIND_INDICES, region.name, z, x, y, tile_size, 0,
            expected_shape=(2, tile_size, tile_size), dtype=np.int32,
        )
        if arr is not None:
            return arr[0], arr[1]  # views onto the shared memmap
        row_idx, col_idx = _compute_region_pixel_indices(region, z, x, y, tile_size)
        _try_publish(
            store, KIND_INDICES, region.name, z, x, y, tile_size, 0,
            np.stack((row_idx, col_idx)),
        )
        # Re-open so the lru pins shared file-backed pages, not these heap
        # arrays.  Identical bytes even if a racing worker won the publish
        # (pure function), so returning the store views is always correct.
        arr = _try_open(
            store, KIND_INDICES, region.name, z, x, y, tile_size, 0,
            expected_shape=(2, tile_size, tile_size), dtype=np.int32,
        )
        if arr is not None:
            return arr[0], arr[1]
        return row_idx, col_idx  # publish or re-open failed: heap fallback
    return _compute_region_pixel_indices(region, z, x, y, tile_size)


def _compute_region_pixel_indices_padded(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256, pad: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Uncached compute body of ``region_pixel_indices_padded``."""
    n = 2**z
    cx = np.arange(-pad, tile_size + pad, dtype=np.float64) + 0.5
    cy = np.arange(-pad, tile_size + pad, dtype=np.float64) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    if region.proj == "laea":
        col_grid, row_grid = _laea_pixel_coords(lon, lat, region)
    elif region.proj == "tmerc":
        col_grid, row_grid = _tmerc_pixel_coords(lon, lat, region)
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
    store = _get_store()
    if store is not None:
        arr = _try_open(
            store, KIND_INDICES_PAD, region.name, z, x, y, tile_size, pad,
            expected_shape=(2, tile_size + 2 * pad, tile_size + 2 * pad),
            dtype=np.int32,
        )
        if arr is not None:
            return arr[0], arr[1]  # views onto the shared memmap
        row_idx, col_idx = _compute_region_pixel_indices_padded(
            region, z, x, y, tile_size, pad,
        )
        _try_publish(
            store, KIND_INDICES_PAD, region.name, z, x, y, tile_size, pad,
            np.stack((row_idx, col_idx)),
        )
        # Re-open so the lru pins shared file-backed pages, not these heap
        # arrays.  Identical bytes even if a racing worker won the publish
        # (pure function), so returning the store views is always correct.
        arr = _try_open(
            store, KIND_INDICES_PAD, region.name, z, x, y, tile_size, pad,
            expected_shape=(2, tile_size + 2 * pad, tile_size + 2 * pad),
            dtype=np.int32,
        )
        if arr is not None:
            return arr[0], arr[1]
        return row_idx, col_idx  # publish or re-open failed: heap fallback
    return _compute_region_pixel_indices_padded(region, z, x, y, tile_size, pad)


def _compute_region_pixel_indices_fractional(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Uncached compute body of ``region_pixel_indices_fractional``."""
    n = 2**z
    cx = np.arange(tile_size, dtype=np.float64) + 0.5
    cy = np.arange(tile_size, dtype=np.float64) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    if region.proj == "laea":
        col_grid, row_grid = _laea_pixel_coords(lon, lat, region)
    elif region.proj == "tmerc":
        col_grid, row_grid = _tmerc_pixel_coords(lon, lat, region)
    else:
        col_f = (lon - region.west) / region.pixel_size
        row_f = (region.north - lat) / region._ps_y
        col_grid, row_grid = np.meshgrid(col_f, row_f)

    row_grid = np.clip(row_grid, 0, region.height - 1).astype(np.float32)
    col_grid = np.clip(col_grid, 0, region.width - 1).astype(np.float32)

    row_grid.flags.writeable = False
    col_grid.flags.writeable = False
    return row_grid, col_grid


@lru_cache(maxsize=settings.coord_cache_size)
def region_pixel_indices_fractional(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fractional composite pixel coordinates for bilinear interpolation."""
    store = _get_store()
    if store is not None:
        arr = _try_open(
            store, KIND_FRACTIONAL, region.name, z, x, y, tile_size, 0,
            expected_shape=(2, tile_size, tile_size), dtype=np.float32,
        )
        if arr is not None:
            return arr[0], arr[1]  # views onto the shared memmap
        row_grid, col_grid = _compute_region_pixel_indices_fractional(
            region, z, x, y, tile_size,
        )
        _try_publish(
            store, KIND_FRACTIONAL, region.name, z, x, y, tile_size, 0,
            np.stack((row_grid, col_grid)),
        )
        # Re-open so the lru pins shared file-backed pages, not these heap
        # arrays.  Identical bytes even if a racing worker won the publish
        # (pure function), so returning the store views is always correct.
        arr = _try_open(
            store, KIND_FRACTIONAL, region.name, z, x, y, tile_size, 0,
            expected_shape=(2, tile_size, tile_size), dtype=np.float32,
        )
        if arr is not None:
            return arr[0], arr[1]
        return row_grid, col_grid  # publish or re-open failed: heap fallback
    return _compute_region_pixel_indices_fractional(region, z, x, y, tile_size)


def _compute_region_pixel_indices_fractional_padded(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256, pad: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Uncached compute body of ``region_pixel_indices_fractional_padded``."""
    n = 2**z
    cx = np.arange(-pad, tile_size + pad, dtype=np.float64) + 0.5
    cy = np.arange(-pad, tile_size + pad, dtype=np.float64) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(math.pi * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    if region.proj == "laea":
        col_grid, row_grid = _laea_pixel_coords(lon, lat, region)
    elif region.proj == "tmerc":
        col_grid, row_grid = _tmerc_pixel_coords(lon, lat, region)
    else:
        col_f = (lon - region.west) / region.pixel_size
        row_f = (region.north - lat) / region._ps_y
        col_grid, row_grid = np.meshgrid(col_f, row_f)

    row_grid = np.clip(row_grid, 0, region.height - 1).astype(np.float32)
    col_grid = np.clip(col_grid, 0, region.width - 1).astype(np.float32)

    row_grid.flags.writeable = False
    col_grid.flags.writeable = False
    return row_grid, col_grid


@lru_cache(maxsize=settings.coord_cache_size)
def region_pixel_indices_fractional_padded(
    region: RegionDef, z: int, x: int, y: int, tile_size: int = 256, pad: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Fractional pixel coords for a tile with padding (bilinear + blur path)."""
    store = _get_store()
    if store is not None:
        arr = _try_open(
            store, KIND_FRACTIONAL_PAD, region.name, z, x, y, tile_size, pad,
            expected_shape=(2, tile_size + 2 * pad, tile_size + 2 * pad),
            dtype=np.float32,
        )
        if arr is not None:
            return arr[0], arr[1]  # views onto the shared memmap
        row_grid, col_grid = _compute_region_pixel_indices_fractional_padded(
            region, z, x, y, tile_size, pad,
        )
        _try_publish(
            store, KIND_FRACTIONAL_PAD, region.name, z, x, y, tile_size, pad,
            np.stack((row_grid, col_grid)),
        )
        # Re-open so the lru pins shared file-backed pages, not these heap
        # arrays.  Identical bytes even if a racing worker won the publish
        # (pure function), so returning the store views is always correct.
        arr = _try_open(
            store, KIND_FRACTIONAL_PAD, region.name, z, x, y, tile_size, pad,
            expected_shape=(2, tile_size + 2 * pad, tile_size + 2 * pad),
            dtype=np.float32,
        )
        if arr is not None:
            return arr[0], arr[1]
        return row_grid, col_grid  # publish or re-open failed: heap fallback
    return _compute_region_pixel_indices_fractional_padded(
        region, z, x, y, tile_size, pad,
    )


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

    Calls the LRU-cached ``_overlapping_regions_cached`` with the enabled
    list normalized to a hashable tuple; a fresh list is returned each
    time so callers can't mutate the cached entry.
    """
    if enabled is None:
        enabled_names = tuple(REGIONS.keys())
    else:
        enabled_names = tuple(enabled)
    return list(_overlapping_regions_cached(z, x, y, enabled_names))


@lru_cache(maxsize=settings.coord_cache_size)
def _overlapping_regions_cached(
    z: int, x: int, y: int, enabled: tuple[str, ...]
) -> list[RegionDef]:
    """Cached body of ``overlapping_regions`` (see its docstring)."""
    result = []
    for name in enabled:
        region = REGIONS.get(name)
        if region and tile_overlaps_region(region, z, x, y):
            result.append(region)

    # Finest resolution first (smallest pixel_size)
    result.sort(key=lambda r: r.pixel_size)
    return result


def _compute_tile_pixel_latlons(
    z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Uncached compute body of ``tile_pixel_latlons``."""
    n = 2**z
    cx = np.arange(tile_size, dtype=np.float32) + 0.5
    cy = np.arange(tile_size, dtype=np.float32) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.float32(math.pi) * (1 - 2 * (y + cy / tile_size) / n)))
    lat = np.degrees(lat_rad)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lon_grid.flags.writeable = False
    lat_grid.flags.writeable = False
    return lat_grid, lon_grid


@lru_cache(maxsize=settings.coord_cache_size)
def tile_pixel_latlons(
    z: int, x: int, y: int, tile_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lat/lon for each pixel in a Web Mercator tile.

    Returns (lat_grid, lon_grid) float32 arrays of shape (tile_size, tile_size).
    Used for temperature lookups that need geographic coordinates.
    float32 provides ~7 decimal digits (~0.00001° ≈ 1 m precision),
    far exceeding any radar data resolution.
    """
    store = _get_store()
    if store is not None:
        arr = _try_open(
            store, KIND_LATLON, None, z, x, y, tile_size, 0,
            expected_shape=(2, tile_size, tile_size), dtype=np.float32,
        )
        if arr is not None:
            return arr[0], arr[1]  # views onto the shared memmap
        lat_grid, lon_grid = _compute_tile_pixel_latlons(z, x, y, tile_size)
        _try_publish(
            store, KIND_LATLON, None, z, x, y, tile_size, 0,
            np.stack((lat_grid, lon_grid)),
        )
        # Re-open so the lru pins shared file-backed pages, not these heap
        # arrays.  Identical bytes even if a racing worker won the publish
        # (pure function), so returning the store views is always correct.
        arr = _try_open(
            store, KIND_LATLON, None, z, x, y, tile_size, 0,
            expected_shape=(2, tile_size, tile_size), dtype=np.float32,
        )
        if arr is not None:
            return arr[0], arr[1]
        return lat_grid, lon_grid  # publish or re-open failed: heap fallback
    return _compute_tile_pixel_latlons(z, x, y, tile_size)


def _compute_tile_pixel_latlons_padded(
    z: int, x: int, y: int, tile_size: int = 256, pad: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Uncached compute body of ``tile_pixel_latlons_padded``."""
    n = 2**z
    cx = np.arange(-pad, tile_size + pad, dtype=np.float32) + 0.5
    cy = np.arange(-pad, tile_size + pad, dtype=np.float32) + 0.5

    lon = (x + cx / tile_size) / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.float32(math.pi) * (1 - 2 * (y + cy / tile_size) / n)))
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
    store = _get_store()
    if store is not None:
        arr = _try_open(
            store, KIND_LATLON_PAD, None, z, x, y, tile_size, pad,
            expected_shape=(2, tile_size + 2 * pad, tile_size + 2 * pad),
            dtype=np.float32,
        )
        if arr is not None:
            return arr[0], arr[1]  # views onto the shared memmap
        lat_grid, lon_grid = _compute_tile_pixel_latlons_padded(
            z, x, y, tile_size, pad,
        )
        _try_publish(
            store, KIND_LATLON_PAD, None, z, x, y, tile_size, pad,
            np.stack((lat_grid, lon_grid)),
        )
        # Re-open so the lru pins shared file-backed pages, not these heap
        # arrays.  Identical bytes even if a racing worker won the publish
        # (pure function), so returning the store views is always correct.
        arr = _try_open(
            store, KIND_LATLON_PAD, None, z, x, y, tile_size, pad,
            expected_shape=(2, tile_size + 2 * pad, tile_size + 2 * pad),
            dtype=np.float32,
        )
        if arr is not None:
            return arr[0], arr[1]
        return lat_grid, lon_grid  # publish or re-open failed: heap fallback
    return _compute_tile_pixel_latlons_padded(z, x, y, tile_size, pad)


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


# ── Web Mercator lat/lon-centered window helpers ─────────────────────
# Forward Web Mercator + window-origin helpers for lat/lon-centered
# tiles.  Cheap pure-scalar stdlib math only (no numpy); no lru_cache
# either — float keys would grow the cache unboundedly.

MERCATOR_MAX_LAT = 85.05112878


def latlon_to_global_pixel(
    lat: float, lon: float, z: int, tile_size: int
) -> tuple[float, float]:
    """Forward Web Mercator: returns fractional global pixel coords (px, py) at zoom z.

    Clamp lat to +-MERCATOR_MAX_LAT.  Normalize lon to [-180, 180) via
    ((lon + 180.0) % 360.0) - 180.0.

    n = 2**z; world = n * tile_size
    px = (lon + 180.0) / 360.0 * world
    py = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * world

    Reference for the same math:
    sources/regional/east_asia/japan/radar/jma/decoder.py:116-122.
    """
    n = 2**z
    world = n * tile_size
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat = min(max(lat, -MERCATOR_MAX_LAT), MERCATOR_MAX_LAT)
    px = (lon + 180.0) / 360.0 * world
    py = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * world
    return px, py


def window_origin(
    lat: float, lon: float, z: int, tile_size: int
) -> tuple[int, int]:
    """Return (px0, py0): the global-pixel origin of the tile_size x tile_size
    window centered (to the nearest integer pixel) on lat/lon.

    raw_x = round(px - tile_size / 2); px0 = raw_x % world (wrap-aware,
    lands in [0, world)).  raw_y = round(py - tile_size / 2);
    py0 = min(max(raw_y, 0), world - tile_size) (clamped; world >=
    tile_size always since z >= 0).
    """
    px, py = latlon_to_global_pixel(lat, lon, z, tile_size)
    n = 2**z
    world = n * tile_size
    raw_x = round(px - tile_size / 2)
    px0 = raw_x % world
    raw_y = round(py - tile_size / 2)
    py0 = min(max(raw_y, 0), world - tile_size)
    return px0, py0


# ---------------------------------------------------------------------------
# Cache pre-warming
# ---------------------------------------------------------------------------


def compute_blur_radius(
    region: RegionDef, z: int, x: int, y: int, tile_size: int
) -> float:
    """Pick a Gaussian blur radius matched to the visible region pixel size.

    Reads the local Jacobian of ``region_pixel_indices_fractional`` at the
    tile centre to find how many tile pixels a single region pixel covers
    (``tile_per_region``).  Blur radius scales as a quarter of that span,
    which is the σ that rounds a single region-pixel "block" at its
    edges without merging it with its neighbours (the visible Gaussian
    width is ~3σ, so a quarter-block σ touches half a block on each side).
    At low zoom the ratio is < 1 and the radius collapses to
    ``smooth_radius`` (baseline); at high zoom on a very coarse source
    growth is capped at ``tile_size / 32`` to keep the kernel from
    smearing unrelated cells together.
    """
    base = settings.smooth_radius
    if base <= 0:
        return 0.0
    row_f, col_f = region_pixel_indices_fractional(region, z, x, y, tile_size)
    cy = cx = tile_size // 2
    drow = abs(float(row_f[cy + 1, cx] - row_f[cy - 1, cx])) / 2.0
    dcol = abs(float(col_f[cy, cx + 1] - col_f[cy, cx - 1])) / 2.0
    if drow < 1e-6 or dcol < 1e-6:
        return base
    tile_per_region = max(1.0 / drow, 1.0 / dcol)
    raw = base * max(1.0, tile_per_region * 0.25)
    return min(raw, tile_size / 32.0)


def warm_coordinate_caches(
    enabled_regions: list[str] | None, max_zoom: int, tile_size: int = 256
) -> int:
    """Pre-populate all coordinate LRU caches up to ``max_zoom``.

    Iterates every tile coordinate at zooms 0 through ``max_zoom``,
    computes overlapping regions, and calls each cached coordinate
    function so that real tile requests never pay the cold-start cost
    of trigonometric projections and array allocations.

    Warms exactly the keys the request path uses: the plain indices /
    fractional / latlon grids unconditionally (coverage and overlay paths),
    and the padded variants only when the render path's derived pad
    (``int(compute_blur_radius(...) * 3)`` when the sigma >= 0.5) is > 0.
    Because the wrappers are store-backed, warming publishes to the shared
    on-disk store and later workers' warm passes become store hits.

    Returns the number of unique (region, z, x, y, tile_size) cache
    entries warmed.
    """
    if max_zoom <= 0:
        return 0
    warmed = 0
    for z in range(max_zoom + 1):
        n = 2**z
        for y in range(n):
            for x in range(n):
                regions = overlapping_regions(z, x, y, enabled_regions)
                if not regions:
                    continue
                # Tile-level lat/lon grids (used by ECMWF fallback, arrows).
                tile_pixel_latlons(z, x, y, tile_size)
                for region in regions:
                    region_pixel_indices(region, z, x, y, tile_size)
                    region_pixel_indices_fractional(region, z, x, y, tile_size)
                    warmed += 1
                # Derive the pad exactly like the render path (smooth=True
                # default) from the finest overlapping region, then warm the
                # padded variants with THAT pad so warm and request keys agree.
                sigma = compute_blur_radius(regions[0], z, x, y, tile_size)
                pad = int(sigma * 3) if sigma >= 0.5 else 0
                if pad > 0:
                    tile_pixel_latlons_padded(z, x, y, tile_size, pad)
                    for region in regions:
                        region_pixel_indices_padded(region, z, x, y, tile_size, pad)
                        region_pixel_indices_fractional_padded(region, z, x, y, tile_size, pad)
    return warmed


# All decorated coordinate cache functions (for bulk clear / size queries).
# Legacy wrappers (tile_pixel_indices, etc.) are excluded because they
# delegate to the corresponding region_pixel_* function and thus share
# the same underlying numpy arrays — counting them would double-count.
ALL_CACHES = [
    region_pixel_indices,
    region_pixel_indices_padded,
    region_pixel_indices_fractional,
    region_pixel_indices_fractional_padded,
    tile_pixel_latlons,
    tile_pixel_latlons_padded,
]

# Per-cache estimate of how many bytes each cached result tuple consumes.
# Calculated as: 2 arrays × dtype_size × rows × cols.
# Defaults assume tile_size=256, pad=8 (padded: 272).
_CACHE_ENTRY_BYTES = {
    # region_pixel_indices: 2 × int32 × 256 × 256
    region_pixel_indices: 2 * 4 * 256 * 256,
    # region_pixel_indices_padded: 2 × int32 × 272 × 272
    region_pixel_indices_padded: 2 * 4 * 272 * 272,
    # region_pixel_indices_fractional: 2 × float32 × 256 × 256
    region_pixel_indices_fractional: 2 * 4 * 256 * 256,
    # region_pixel_indices_fractional_padded: 2 × float32 × 272 × 272
    region_pixel_indices_fractional_padded: 2 * 4 * 272 * 272,
    # tile_pixel_latlons: 2 × float32 × 256 × 256
    tile_pixel_latlons: 2 * 4 * 256 * 256,
    # tile_pixel_latlons_padded: 2 × float32 × 272 × 272
    tile_pixel_latlons_padded: 2 * 4 * 272 * 272,
}


def coord_cache_stats() -> dict:
    """Per-cache hit/miss/fill stats for the /health endpoint.

    Hit ratio + fill ratio are what you want when tuning
    ``LIBREWXR_COORD_CACHE_SIZE``: low hit ratio with full caches means
    the cap is too small; full caches with high hit ratio means it's
    well-sized; partial fills mean the cap has headroom.
    """
    caches: dict[str, dict] = {}
    max_size = 0
    for fn in ALL_CACHES:
        info = fn.cache_info()
        max_size = info.maxsize or 0
        total = info.hits + info.misses
        caches[fn.__name__] = {
            "entries": info.currsize,
            "hits": info.hits,
            "misses": info.misses,
            "hit_ratio": round(info.hits / total, 3) if total else None,
        }
    return {
        "max_size": max_size,
        "caches": caches,
        # Shared on-disk store stats (hits/misses/publishes/entries/bytes);
        # None when the store is disabled or failed to initialize (degraded
        # to the in-process compute path).
        "store": _STORE.stats() if _STORE is not None else None,
    }


def coord_cache_bytes() -> int:
    """Estimate total memory consumed by all coordinate LRU caches.

    Uses ``lru_cache.cache_info().currsize`` (number of populated entries)
    multiplied by the per-entry byte cost of each cache's return value.

    This is an approximation — entries called with non-default tile_size
    or pad values will have different sizes, but the vast majority of
    calls use the defaults (256 / 8).
    """
    total = 0
    for fn in ALL_CACHES:
        info = fn.cache_info()
        total += info.currsize * _CACHE_ENTRY_BYTES[fn]
    return total
