# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Météo-France AROME Guyane regional precipitation source.

Thin subclass of ``AROMEOverseasGrid`` for the French Guiana domain
(~1156 km E-W × 877 km N-S at 0.025°).  All fetch / decode / cache
machinery lives in ``librewxr.sources._shared.arome``.

Data attribution: Météo-France, Etalab Open Licence v2.0.
"""
from __future__ import annotations

from typing import ClassVar

from librewxr.sources._shared.arome import AROMEOverseasGrid


class AROMEGuyaneGrid(AROMEOverseasGrid):
    """AROME Guyane grid — French Guiana + Suriname + Amapá borders.

    Grid corners back-decoded from GRIB Section 3 of a representative
    ``arome-om-GUYANE__0025__SP1__006H`` file on 2026-05-21.  The
    domain is small (~1156 km E-W, ~877 km N-S), so the feather
    tightens to ~38 km (15 cells × 0.025° × ~110 km/° at the equator).
    """

    name: ClassVar[str] = "arome_guyane"
    friendly_name: ClassVar[str] = "AROME Guyane"
    settings_prefix: ClassVar[str] = "arome_guyane"
    memmap_subdir: ClassVar[str] = "arome_guyane"

    url_token: ClassVar[str] = "GUYANE"

    LAT_NORTH: ClassVar[float] = 8.95
    LAT_SOUTH: ClassVar[float] = 1.05
    LON_WEST_DEG_E: ClassVar[float] = 303.25
    LON_EAST_DEG_E: ClassVar[float] = 313.70
    GRID_DLAT: ClassVar[float] = 0.025
    GRID_DLON: ClassVar[float] = 0.025
    GRID_WIDTH: ClassVar[int] = 419
    GRID_HEIGHT: ClassVar[int] = 317
    FEATHER_DISTANCE_CELLS: ClassVar[int] = 15
