# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""ECMWF IFS — self-contained NWP source package.

Global 9 km precipitation grid from Open-Meteo's S3 mirror.  Lowest
priority in the NWP chain (highest numeric ``priority``) so it always
runs after every regional source — its job is to catch everywhere the
regionals don't cover.  Also feeds the cloud-cover layer and the
snow/rain classification fallback when no regional source supports
snow at a given pixel.
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import ECMWFGrid

__all__ = ["ECMWFGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return an IFS contribution when ``settings.ecmwf_enabled`` is set."""
    if not getattr(settings, "ecmwf_enabled", True):
        return None
    return NWPContribution(
        instance=ECMWFGrid(cache_dir=cache_dir),
        priority=1000,
        name="ECMWF IFS",
    )
