# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""SMN Argentina WRF — self-contained NWP source package.

4 km regional WRF run covering the South American Southern Cone
(Argentina + Chile + Uruguay + Paraguay + Bolivia + S. Brazil).
Disjoint from every other regional source, so position only matters
relative to IFS.
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import WRFSMNGrid

__all__ = ["WRFSMNGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return a WRF-SMN contribution when enabled."""
    if not getattr(settings, "wrf_smn_enabled", True):
        return None
    return NWPContribution(
        instance=WRFSMNGrid(cache_dir=cache_dir),
        priority=40,
        name="WRF-SMN",
    )
