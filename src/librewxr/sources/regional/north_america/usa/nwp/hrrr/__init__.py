# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NOAA HRRR — self-contained NWP source package.

3 km LCC grid over the contiguous United States, 15-min cadence,
T+18 forecast horizon.  Paired with HRRR-Alaska via the shared
``na_nwp_source == "hrrr"`` setting — enabling one enables both,
since they're the same NCEP model on disjoint domains served from
the same anonymous S3 bucket.

Also exports ``compute_snow_mask`` for use by other regional NWP
sources (DMI DINI, ICON-EU, WRF-SMN, HRRR-Alaska) that share the
"T2m < threshold => snow" classifier.
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import HRRRGrid

__all__ = ["HRRRGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return an HRRR contribution when ``na_nwp_source == "hrrr"``."""
    if getattr(settings, "na_nwp_source", "hrrr") != "hrrr":
        return None
    return NWPContribution(
        instance=HRRRGrid(cache_dir=cache_dir),
        priority=10,
        name="HRRR",
    )
