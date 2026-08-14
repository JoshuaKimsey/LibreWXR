# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Region definition for the NOAA Enterprise Rain Rate (RRQPE) radar layer.

The GLB-5 blend is a global observed-precip grid (native 0.02° plate
carrée over lat +70..-60, lon -180..180 — see ``grid.py`` for the data
layout).  As a radar contribution it becomes one coarse global region
that sorts LAST in the multi-region compositor (``overlapping_regions``
orders by ``pixel_size`` ascending, and 0.04° at the default downsample
of 2 is coarser than every Doppler-derived composite), so it fills only
pixels no finer radar region claims — exactly the desired bottom tier.

Bounds mirror the native product.  The explicit ``grid_width`` /
``grid_height`` pin the decoded block-averaged grid's dimensions so the
renderer's index math never derives them from the bbox (which would
round to the same values here, but the pin keeps the two in lockstep).
The downsample factor is evaluated at module import — region definitions
are static per process and settings are fixed at startup, so the factor
can never drift mid-run.

No ``REGION_GROUP`` is exposed (only the group label on the ``RegionDef``
itself), so RRQPE lands in the ALL group but in no narrow alias.
"""
from __future__ import annotations

from librewxr.config import settings
from librewxr.data.regions import RegionDef


def _downsample_factor() -> int:
    """The configured block-averaging factor for the native 0.02° grid."""
    return max(1, int(getattr(settings, "rrqpe_downsample", 2)))


F = _downsample_factor()
RRQPE_PIXEL = 0.02 * F

RRQPE = RegionDef(
    name="RRQPE",
    west=-180.0, east=180.0, south=-60.0, north=70.0,
    pixel_size=RRQPE_PIXEL,
    pixel_size_y=RRQPE_PIXEL,
    group="GLOBAL",
    # 18000 native columns are already a multiple of any factor;
    # the 6501 native rows are cropped to the largest multiple of the
    # factor before block-averaging ((6501 // F) * F rows), so the
    # stored grid is ((6501 // F) * F // F, 18000 // F).
    grid_width=18000 // F,
    grid_height=(6501 // F) * F // F,
    # Coarse global fill layer — no meaningful convective cells at the
    # 25 km² storm-cell minimum (see ``RegionDef.storm_cells``).
    storm_cells=False,
)


# Coverage polygon: the full lon -180..180 × lat -58..68 band — a 2°
# inset from the grid edges so the LZA-degraded fringe (edge-of-scan /
# heavily distorted rows) stays with IFS instead of being claimed as
# radar.  A single ring rasterises correctly in ``data/coverage.py``:
# the mask build maps lon linearly onto a column index (no antimeridian
# unwrap), and ``cv2.fillPoly`` clips the two ±180° vertices cleanly —
# the resulting mask is identical to the two-half-world split the
# coverage-map script uses for its Web-Mercator canvas.
RRQPE_COVERAGE_POLYGON: list[tuple[float, float]] = [
    (-58.0, -180.0),
    (-58.0, 180.0),
    (68.0, 180.0),
    (68.0, -180.0),
]


REGIONS: list[RegionDef] = [RRQPE]
