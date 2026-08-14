# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NOAA Enterprise Rain Rate (RRQPE) GLB-5 blend — self-contained NWP source package.

Satellite-derived *observed* precipitation (10-minute global rain rate
from the geostationary constellation) consumed from NOAA's anonymous
Open Data S3 bucket ``noaa-enterprise-rainrate-pds``.

Highest-priority member of the NWP chain (priority 5, ahead of every
model) for PAST-frame fill only — observations outrank model output for
observed frames.  No future frame is ever stored, so nowcast/forecast
timestamps automatically fall through to the models behind it.
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import RRQPEGrid

__all__ = ["RRQPEGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return an RRQPE contribution when ``settings.rrqpe_enabled`` is set."""
    if not getattr(settings, "rrqpe_enabled", True):
        return None
    return NWPContribution(
        instance=RRQPEGrid(cache_dir=cache_dir),
        priority=5,
        name="NOAA RRQPE",
        slug="rrqpe_grid",
        # Global observed layer, not part of the regional model chain —
        # the ``regional_nwp_enabled`` master switch never drops it.
        regional=False,
    )
