# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Sampling utilities for the MCP server.

Pure synchronous functions for decoding dBZ pixels, converting dBZ to
rain rate (Marshall-Palmer), resolving the finest radar region covering
a lat/lon point, sampling a radar frame at a single point, and sampling
a nowcast frame at a point.
"""

import math

import numpy as np

from librewxr.data.regions import REGIONS, RegionDef
from librewxr.data.coverage import sample_coverage
from librewxr.tiles.coordinates import _laea_forward, _tmerc_forward

# ---------------------------------------------------------------------------
# Marshall-Palmer Z-R relation constants (rain + snow).
# Mirrors src/librewxr/sources/world/ifs/grid.py:31-35 -- copied locally
# so this module doesn't pull the IFS grid + its dep tree.
# ---------------------------------------------------------------------------
ZR_A_RAIN = 200.0
ZR_B_RAIN = 1.6
ZR_A_SNOW = 2000.0
ZR_B_SNOW = 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decode_dbz(pixel: int | np.uint8) -> float | None:
    """Inverse of the IEM uint8 dBZ encoder.

    The encoder formula (``_dbz_float_to_uint8`` in
    ``librewxr.sources._helpers``) is::

        pixel = clamp((dBZ + 32) * 2, 0, 255)

    Everything <= -32.0 dBZ maps to pixel 0 (NODATA / transparent in
    all colour schemes).  This function reverses the mapping:

        pixel == 0   -->  ``None``
        pixel  > 0   -->  ``pixel / 2.0 - 32.0``

    Examples:
        ``decode_dbz(64)``   → 0.0 dBZ
        ``decode_dbz(255)``  → 95.5 dBZ
        ``decode_dbz(1)``    → -31.5 dBZ
    """
    p = int(pixel)
    if p == 0:
        return None
    return float(p) / 2.0 - 32.0


def dbz_to_rate_mmh(dbz: float | None, is_snow: bool = False) -> float:
    """Marshall-Palmer Z-R relation: convert reflectivity to rain rate.

    ``Z = A * R^B``   →   ``R = (Z / A)^(1 / B)``

    with ``Z = 10^(dBZ / 10)``.  The constants are taken from
    ``librewxr.sources.world.ifs.grid.py`` (Marshall-Palmer):

        Rain:  A=200, B=1.6
        Snow:  A=2000, B=2.0

    Returns 0.0 for ``dbz is None`` or ``dbz <= 0`` (no measurable
    precipitation, matching the renderer's noise-floor convention).
    """
    if dbz is None or dbz <= 0.0:
        return 0.0

    A = ZR_A_SNOW if is_snow else ZR_A_RAIN
    B = ZR_B_SNOW if is_snow else ZR_B_RAIN
    Z = 10.0 ** (dbz / 10.0)
    R = (Z / A) ** (1.0 / B)
    return float(R)


def resolve_region_for_point(
    lat: float, lon: float, enabled_regions: list[str],
) -> RegionDef | None:
    """Find the finest-resolution enabled radar region covering *lat/lon*.

    Uses an equirectangular cos(lat) station-range mask
    (``librewxr.data.coverage.sample_coverage``) to confirm the point
    is within radar range of operational stations.  Among all candidates
    that cover the point, the one with the smallest ``pixel_size``
    (finest resolution) is returned, matching the renderer's
    ``overlapping_regions`` sort at ``tiles/coordinates.py:343``.

    Returns ``None`` if no enabled region covers the point.
    """
    best: RegionDef | None = None
    for name in enabled_regions:
        region = REGIONS.get(name)
        if region is None:
            continue

        # Bounding-box prefilter.
        if not (region.west <= lon <= region.east and region.south <= lat <= region.north):
            continue

        # Radar-coverage mask check (1-element 1D arrays).
        mask = sample_coverage(name, np.array([lat]), np.array([lon]))
        if not mask[0]:
            continue

        # Pick the finest resolution.
        if best is None or region.pixel_size < best.pixel_size:
            best = region

    return best


def sample_region_at_point(
    region: RegionDef, lat: float, lon: float, frame_array: np.ndarray,
) -> tuple[float | None, str]:
    """Sample one pixel from a radar region's uint8 frame at *lat/lon*.

    Projects the geographic coordinate into the region's grid, rounds
    to the nearest integer pixel, and returns the decoded dBZ value.

    Returns a ``(dbz, status)`` tuple where *status* is ``"in_range"``
    for a valid sample inside the array bounds, or ``"out_of_range"``
    when the point maps outside the grid dimensions.
    """
    if region.proj == "latlon":
        col_f = (lon - region.west) / region.pixel_size
        row_f = (region.north - lat) / region._ps_y
    elif region.proj == "laea":
        x, y = _laea_forward(
            np.asarray([lon], dtype=np.float64),
            np.asarray([lat], dtype=np.float64),
            region,
        )
        col_f = (float(x[0]) - region.grid_x_min) / region.grid_scale
        row_f = (region.grid_y_max - float(y[0])) / region.grid_scale
    elif region.proj == "tmerc":
        x, y = _tmerc_forward(
            np.asarray([lon], dtype=np.float64),
            np.asarray([lat], dtype=np.float64),
            region,
        )
        col_f = (float(x[0]) - region.grid_x_min) / region.grid_scale
        row_f = (region.grid_y_max - float(y[0])) / region.grid_scale
    else:
        raise ValueError(f"Unknown projection '{region.proj}' for region '{region.name}'")

    col = int(round(col_f))
    row = int(round(row_f))

    if col < 0 or col >= region.width or row < 0 or row >= region.height:
        return (None, "out_of_range")

    pixel = int(frame_array[row, col])
    dbz = decode_dbz(pixel)
    return (dbz, "in_range")


def sample_nowcast_at_point(
    region_name: str,
    lat: float,
    lon: float,
    nowcast_frame,
) -> tuple[float | None, float, str]:
    """Sample a nowcast frame's region array at *lat/lon*.

    *nowcast_frame* is a ``NowcastFrame`` (has ``.regions`` and
    ``.blend_weight``).  Delegates the per-pixel sampling to
    :func:`sample_region_at_point`.

    Returns ``(dbz, blend_weight, status)``.  When the region is
    missing from either ``REGIONS`` or the frame's ``.regions`` dict,
    returns ``(None, blend_weight, "out_of_range")``.
    """
    if region_name not in REGIONS:
        return (None, float(nowcast_frame.blend_weight), "out_of_range")

    arr = nowcast_frame.regions.get(region_name)
    if arr is None:
        return (None, float(nowcast_frame.blend_weight), "out_of_range")

    dbz, coverage = sample_region_at_point(REGIONS[region_name], lat, lon, arr)
    return (dbz, float(nowcast_frame.blend_weight), coverage)
