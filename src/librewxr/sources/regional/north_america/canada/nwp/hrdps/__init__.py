# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""ECCC HRDPS — self-contained NWP source package.

2.5 km Canadian deterministic prediction system.  Independent of
``na_nwp_source`` because the HRDPS / HRRR domains barely overlap —
running both together is the common case (HRRR fills CONUS, HRDPS
fills Canada and the northern fringe).
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import HRDPSGrid

__all__ = ["HRDPSGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return an HRDPS contribution when ``settings.hrdps_enabled``."""
    if not getattr(settings, "hrdps_enabled", False):
        return None
    return NWPContribution(
        instance=HRDPSGrid(cache_dir=cache_dir),
        priority=20,
        name="HRDPS",
    )
