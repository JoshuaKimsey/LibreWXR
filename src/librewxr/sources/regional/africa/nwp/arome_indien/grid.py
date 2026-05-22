# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Météo-France AROME Indien regional precipitation source.

Thin subclass of ``AROMEOverseasGrid`` for the SW Indian Ocean
domain (~3742 km E-W × 2492 km N-S at 0.025°) — the largest of the
AROME overseas grids.  Covers Réunion, Mayotte, the Comoros, almost
all of Madagascar, and adjacent waters.  All fetch / decode / cache
machinery lives in ``librewxr.sources._shared.arome``.

Data attribution: Météo-France, Etalab Open Licence v2.0.
"""
from __future__ import annotations

from typing import ClassVar

from librewxr.sources._shared.arome import AROMEOverseasGrid


class AROMEIndienGrid(AROMEOverseasGrid):
    """AROME Indien grid — SW Indian Ocean.

    Grid corners back-decoded from GRIB Section 3 of a representative
    ``arome-om-INDIEN__0025__SP1__006H`` file on 2026-05-21.  The
    domain is large (~3742 km E-W, ~2492 km N-S), so the feather
    widens to ~63 km (25 cells × 0.025° × ~110 km/° at low latitude),
    matching the HRRR/DMI ~75 km range more closely than the smaller
    AROME overseas grids.
    """

    name: ClassVar[str] = "arome_indien"
    friendly_name: ClassVar[str] = "AROME Indien"
    settings_prefix: ClassVar[str] = "arome_indien"
    memmap_subdir: ClassVar[str] = "arome_indien"

    url_token: ClassVar[str] = "INDIEN"

    LAT_NORTH: ClassVar[float] = -3.45
    LAT_SOUTH: ClassVar[float] = -25.90
    LON_WEST_DEG_E: ClassVar[float] = 32.75
    LON_EAST_DEG_E: ClassVar[float] = 67.60
    GRID_DLAT: ClassVar[float] = 0.025
    GRID_DLON: ClassVar[float] = 0.025
    GRID_WIDTH: ClassVar[int] = 1395
    GRID_HEIGHT: ClassVar[int] = 899
    FEATHER_DISTANCE_CELLS: ClassVar[int] = 25
