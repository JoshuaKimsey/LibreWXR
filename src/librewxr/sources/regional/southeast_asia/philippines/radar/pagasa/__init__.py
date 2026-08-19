# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""PAGASA PANAHON radar composite — self-contained source package.

Covers the entire Philippines (Luzon + Visayas + Mindanao + W. Palawan,
plus edges of Sabah and N. Sulawesi) via a single 2048×2048 national
mosaic served anonymously from ``cdn.panahon.gov.ph``.  Public domain per
Philippine IP code RA 8293 §176 (government-works exception); attribution
to PAGASA / DOST recorded in the package README.  One region rides on one
API pair per cycle: ``PHCOMP`` in the ``SOUTHEAST_ASIA`` group, peer to
MET Malaysia.

This package is auto-discovered by ``librewxr.sources`` at import time;
the ``radar_provider`` function below is what wires the source into the
fetcher.  ``REGIONS`` and ``REGION_GROUP`` are picked up by
``librewxr.data.regions`` to populate the global region map.
"""
from __future__ import annotations

from librewxr.sources._base import RadarSourceContribution

from .regions import PHCOMP, REGIONS
from .source import PAGASASource
from .stations import PHCOMP_STATIONS, RANGE_OVERRIDES, STATION_MAP

# Discovery hooks — see librewxr.data.regions._merge_discovered_regions.
REGION_GROUP = "SOUTHEAST_ASIA"

__all__ = [
    "PAGASASource",
    "PHCOMP",
    "PHCOMP_STATIONS",
    "RANGE_OVERRIDES",
    "REGIONS",
    "REGION_GROUP",
    "STATION_MAP",
    "radar_provider",
]


def radar_provider(settings) -> RadarSourceContribution | None:
    """Return a PAGASA contribution, or ``None`` when disabled.

    Honours ``settings.pagasa_enabled`` (default ``True``).  When disabled,
    the fetcher sees no source for ``PHCOMP`` and drops the region from its
    working set even if a user's region spec is a group alias (e.g.
    ``SOUTHEAST_ASIA``, ``ALL``) that would otherwise pull it in.
    """
    if not getattr(settings, "pagasa_enabled", True):
        return None
    instance = PAGASASource(settings.pagasa_base_url)
    return RadarSourceContribution(
        regions=list(REGIONS),
        instance=instance,
        group=REGION_GROUP,
        station_map={k: list(v) for k, v in STATION_MAP.items()},
        range_overrides=dict(RANGE_OVERRIDES),
    )