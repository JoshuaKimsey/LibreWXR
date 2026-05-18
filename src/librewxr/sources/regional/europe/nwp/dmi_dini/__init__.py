# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""DMI DINI (HARMONIE) — self-contained NWP source package.

2 km Nordic regional model — covers more of populated Europe than the
name suggests.  Active only under the ``dini_with_icon_eu`` profile;
takes precedence over ICON-EU inside its footprint.
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import DMIDiniGrid

__all__ = ["DMIDiniGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return a DMI-DINI contribution under ``dini_with_icon_eu``."""
    if getattr(settings, "eu_nwp_profile", "icon_eu_only") != "dini_with_icon_eu":
        return None
    return NWPContribution(
        instance=DMIDiniGrid(cache_dir=cache_dir),
        priority=30,
        name="DMI DINI",
    )
