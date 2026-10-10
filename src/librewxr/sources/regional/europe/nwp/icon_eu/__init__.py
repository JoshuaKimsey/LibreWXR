# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""DWD ICON-EU — self-contained NWP source package.

7 km European regional model.  Gated by
``settings.eu_nwp_profile``: on under ``icon_eu_only`` and
``dini_with_icon_eu``, off under ``ifs`` and ``dini_only``.  When
paired with DMI DINI (``dini_with_icon_eu``) DINI wins inside its
smaller 2 km domain and ICON-EU catches the rest of Europe.
"""
from __future__ import annotations

from librewxr.sources._base import NWPContribution

from .grid import ICONEUGrid

__all__ = ["ICONEUGrid", "nwp_provider"]


def nwp_provider(settings, cache_dir) -> NWPContribution | None:
    """Return an ICON-EU contribution under ``icon_eu_only`` or ``dini_with_icon_eu``."""
    profile = getattr(settings, "eu_nwp_profile", "icon_eu_only")
    if profile not in ("icon_eu_only", "dini_with_icon_eu"):
        return None
    return NWPContribution(
        instance=ICONEUGrid(cache_dir=cache_dir),
        priority=35,
        name="ICON-EU",
    )
