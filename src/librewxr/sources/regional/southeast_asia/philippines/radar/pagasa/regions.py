# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Region definition for the PAGASA PANAHON Philippines radar composite.

``PHCOMP`` is the full Philippine national mosaic footprint served by the
PANAHON web app's 2048×2048 PNG endpoint — Luzon + Visayas + Mindanao +
W. Palawan, with edges of Sabah and N. Sulawesi.  Bounds are the
``leftBottom`` / ``rightTop`` of the web app's composite view, taken from
the PANAHON JS bundle.

The 2048×2048 PNG covers a wider lat span (18.66°) than lon span
(14.10°), so pixels are non-square in geographic units —
``pixel_size_y`` captures the lat axis separately.
"""
from __future__ import annotations

from librewxr.data.regions import RegionDef


PHCOMP = RegionDef(
    name="PHCOMP",
    west=115.4154914129329, east=129.51727937415484,
    south=3.8016540706290445, north=22.458510294136033,
    pixel_size=(129.51727937415484 - 115.4154914129329) / 2048,    # 0.006886°/px (lon)
    pixel_size_y=(22.458510294136033 - 3.8016540706290445) / 2048, # 0.009110°/px (lat)
    group="SOUTHEAST_ASIA",
    grid_width=2048, grid_height=2048,
)

REGIONS: list[RegionDef] = [PHCOMP]