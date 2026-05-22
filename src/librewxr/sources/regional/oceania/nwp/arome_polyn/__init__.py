# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Météo-France AROME Polynésie française — self-contained NWP source.

2.5 km regional model over French Polynesia (Society, Tuamotu, and
Marquesas archipelagoes) plus surrounding waters.  Disjoint from
every other regional source in the chain — sits before IFS to win
inside its domain.
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import AROMEPolynGrid

__all__ = ["AROMEPolynGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return an AROME Polynésie contribution when enabled."""
    if not getattr(settings, "arome_polyn_enabled", True):
        return None
    return NWPContribution(
        instance=AROMEPolynGrid(cache_dir=cache_dir),
        priority=29,
        name="AROME Polynésie",
    )
