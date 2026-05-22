# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Météo-France AROME Nouvelle-Calédonie — self-contained NWP source.

2.5 km regional model over Nouvelle-Calédonie and adjacent SW
Pacific (Vanuatu side).  Disjoint from every other regional source
in the chain — sits before IFS to win inside its domain.
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import AROMENCaledGrid

__all__ = ["AROMENCaledGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return an AROME Nouvelle-Calédonie contribution when enabled."""
    if not getattr(settings, "arome_ncaled_enabled", True):
        return None
    return NWPContribution(
        instance=AROMENCaledGrid(cache_dir=cache_dir),
        priority=28,
        name="AROME Nouvelle-Calédonie",
    )
