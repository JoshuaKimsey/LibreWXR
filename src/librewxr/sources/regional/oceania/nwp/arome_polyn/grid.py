# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Météo-France AROME Polynésie regional precipitation source.

Thin subclass of ``AROMEOverseasGrid`` for the French Polynesia
domain (~1365 km E-W × 1404 km N-S at 0.025°).  All fetch / decode /
cache machinery lives in ``librewxr.sources._shared.arome``.

Data attribution: Météo-France, Etalab Open Licence v2.0.
"""
from __future__ import annotations

from typing import ClassVar

from librewxr.sources._shared.arome import AROMEOverseasGrid


class AROMEPolynGrid(AROMEOverseasGrid):
    """AROME Polynésie grid — Society + Tuamotu + Marquesas + surrounding S Pacific.

    Grid corners back-decoded from GRIB Section 3 of a representative
    ``arome-om-POLYN__0025__SP1__006H`` file on 2026-05-21.  Mid-size
    domain (~1365 × 1404 km), feather ~45 km (18 cells).

    Note: an earlier survey claimed Polynésie ran 12-hourly, but a
    fresh bucket probe on 2026-05-21 confirmed 6-hourly cycles like
    every other AROME-OM variant.
    """

    name: ClassVar[str] = "arome_polyn"
    friendly_name: ClassVar[str] = "AROME Polynésie"
    settings_prefix: ClassVar[str] = "arome_polyn"
    memmap_subdir: ClassVar[str] = "arome_polyn"

    url_token: ClassVar[str] = "POLYN"

    LAT_NORTH: ClassVar[float] = -12.60
    LAT_SOUTH: ClassVar[float] = -25.25
    LON_WEST_DEG_E: ClassVar[float] = 202.50
    LON_EAST_DEG_E: ClassVar[float] = 215.50
    GRID_DLAT: ClassVar[float] = 0.025
    GRID_DLON: ClassVar[float] = 0.025
    GRID_WIDTH: ClassVar[int] = 521
    GRID_HEIGHT: ClassVar[int] = 507
    FEATHER_DISTANCE_CELLS: ClassVar[int] = 18
