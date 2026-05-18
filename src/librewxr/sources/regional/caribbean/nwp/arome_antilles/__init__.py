# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Météo-France AROME Antilles — self-contained NWP source package.

2.5 km regional model over the eastern Caribbean covering Guadeloupe,
Martinique, and French Guiana.  Disjoint from every other regional
source in the chain, so only its position relative to IFS matters —
it sits before IFS to win inside its domain.
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import AROMEAntillesGrid

__all__ = ["AROMEAntillesGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return an AROME-Antilles contribution when enabled."""
    if not getattr(settings, "arome_antilles_enabled", True):
        return None
    return NWPContribution(
        instance=AROMEAntillesGrid(cache_dir=cache_dir),
        priority=25,
        name="AROME Antilles",
    )
