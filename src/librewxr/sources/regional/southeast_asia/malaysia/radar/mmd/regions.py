# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Region definitions for the MET Malaysia radar composite.

Both regions are sub-rectangles of a single animated GIF served under
CC-BY-4.0 from ``api.met.gov.my/static/images/radar-latest.gif``
(6 frames at 10-min cadence, ~60 min of backfill per fetch).

Two upstream canvas layouts are accepted: the legacy 1352x570 and the
current 1352x769.  As of 2026-09 upstream extended the canvas by 199
rows below the map area (pure white filler in the map column, a text
panel in the chrome column).  The radar map occupies rows 0-569 in both
layouts with identical pixel scale, so the sub-rectangle crops below are
layout-independent; any other canvas size fails loudly.

The combined GIF renders both coverage zones in a shared equirectangular
grid over the union bounding box, with a vertical band of pure sea
(South China Sea) between the peninsular and east coverage.  Pixels are
non-square (45.32 px/° lon, 53.47 px/° lat) — ``pixel_size_y`` captures
the latitude axis separately.

Bounds match RainViewer's published ``MYCOMP71`` / ``MYCOMP72`` metadata
(data.rainviewer.com/images/MYCOMP71/0_products.json,
MYCOMP72/0_products.json), which mirrors the same MET Malaysia products.
The internal Rainbow 5 / LEONARDO processing software's native Albers
Equal Area projection is treated as equirect for serving purposes — same
simplification RainViewer makes for the other regional sources (e.g.
MARN El Salvador).
"""
from __future__ import annotations

from librewxr.data.regions import RegionDef


MYPENINSULAR = RegionDef(
    name="MYPENINSULAR",
    west=96.92, east=106.28, south=-1.33, north=8.97,
    pixel_size=1.0 / 45.323,           # 0.02207°/px (lon axis)
    pixel_size_y=1.0 / 53.471,         # 0.01870°/px (lat axis)
    group="SOUTHEAST_ASIA",
    grid_width=424, grid_height=551,
)

MYEAST = RegionDef(
    name="MYEAST",
    west=107.08, east=121.19, south=-1.48, north=9.18,
    pixel_size=1.0 / 45.323,           # 0.02207°/px (lon axis)
    pixel_size_y=1.0 / 53.471,         # 0.01870°/px (lat axis)
    group="SOUTHEAST_ASIA",
    # grid_height 570 = map-area rows y in [0,570), shared by both the
    # legacy 1352x570 and current 1352x769 GIF canvases.  The 199 extra
    # rows of the current canvas sit below the map and are not part of
    # any region crop.
    grid_width=640, grid_height=570,
)

REGIONS: list[RegionDef] = [MYPENINSULAR, MYEAST]
