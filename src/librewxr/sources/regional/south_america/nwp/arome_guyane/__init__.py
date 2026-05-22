# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Météo-France AROME Guyane — self-contained NWP source package.

2.5 km regional model over French Guiana and adjacent Suriname /
Amapá Brazil borderlands.  Disjoint from every other regional source
in the chain — sits before IFS to win inside its domain.
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import AROMEGuyaneGrid

__all__ = ["AROMEGuyaneGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return an AROME Guyane contribution when enabled."""
    if not getattr(settings, "arome_guyane_enabled", True):
        return None
    return NWPContribution(
        instance=AROMEGuyaneGrid(cache_dir=cache_dir),
        priority=26,
        name="AROME Guyane",
    )
