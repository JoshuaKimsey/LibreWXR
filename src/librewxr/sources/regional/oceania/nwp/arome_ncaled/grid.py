# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Météo-France AROME Nouvelle-Calédonie regional precipitation source.

Thin subclass of ``AROMEOverseasGrid`` for the SW Pacific domain
(~1357 km E-W × 1360 km N-S at 0.025°).  All fetch / decode / cache
machinery lives in ``librewxr.sources._shared.arome``.

Data attribution: Météo-France, Etalab Open Licence v2.0.
"""
from __future__ import annotations

from typing import ClassVar

from librewxr.sources._shared.arome import AROMEOverseasGrid


class AROMENCaledGrid(AROMEOverseasGrid):
    """AROME Nouvelle-Calédonie grid — Grande Terre, Loyalty Islands, Vanuatu side.

    Grid corners back-decoded from GRIB Section 3 of a representative
    ``arome-om-NCALED__0025__SP1__006H`` file on 2026-05-21.  Mid-size
    domain (~1357 × 1360 km), feather ~45 km (18 cells).
    """

    name: ClassVar[str] = "arome_ncaled"
    friendly_name: ClassVar[str] = "AROME Nouvelle-Calédonie"
    settings_prefix: ClassVar[str] = "arome_ncaled"
    memmap_subdir: ClassVar[str] = "arome_ncaled"

    url_token: ClassVar[str] = "NCALED"

    LAT_NORTH: ClassVar[float] = -13.75
    LAT_SOUTH: ClassVar[float] = -26.00
    LON_WEST_DEG_E: ClassVar[float] = 158.50
    LON_EAST_DEG_E: ClassVar[float] = 171.50
    GRID_DLAT: ClassVar[float] = 0.025
    GRID_DLON: ClassVar[float] = 0.025
    GRID_WIDTH: ClassVar[int] = 521
    GRID_HEIGHT: ClassVar[int] = 491
    FEATHER_DISTANCE_CELLS: ClassVar[int] = 18
