# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Météo-France AROME Indien — self-contained NWP source package.

2.5 km regional model over the SW Indian Ocean: Réunion, Mayotte,
Comoros, most of Madagascar, plus adjacent waters and southern
fringe of mainland East Africa.  The largest of the AROME-OM grids
(~3700 km E-W × 2500 km N-S).  Disjoint from every other regional
source in the chain — sits before IFS to win inside its domain.
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import AROMEIndienGrid

__all__ = ["AROMEIndienGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return an AROME Indien contribution when enabled."""
    if not getattr(settings, "arome_indien_enabled", True):
        return None
    return NWPContribution(
        instance=AROMEIndienGrid(cache_dir=cache_dir),
        priority=27,
        name="AROME Indien",
    )
