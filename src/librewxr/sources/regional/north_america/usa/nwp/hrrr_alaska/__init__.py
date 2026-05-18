# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NOAA HRRR-Alaska — self-contained NWP source package.

3 km polar-stereographic grid over Alaska + adjoining Canadian
territory, 6-hr cadence, T+48 forecast horizon.  Paired with HRRR
(CONUS) via the shared ``na_nwp_source == "hrrr"`` setting.
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import HRRRAlaskaGrid

__all__ = ["HRRRAlaskaGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return an HRRR-Alaska contribution when ``na_nwp_source == "hrrr"``."""
    if getattr(settings, "na_nwp_source", "hrrr") != "hrrr":
        return None
    return NWPContribution(
        instance=HRRRAlaskaGrid(cache_dir=cache_dir),
        priority=11,
        name="HRRR-Alaska",
    )
