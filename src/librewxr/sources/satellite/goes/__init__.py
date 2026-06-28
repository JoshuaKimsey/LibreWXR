# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""GOES-18 / GOES-19 ABI satellite source package.

High-resolution (2 km IR, 2 km VIS) geostationary imagery for the
Americas at 5-minute cadence.  Auto-selects GOES-18 (West, 137°W) or
GOES-19 (East, 75.2°W) based on the operator's BBOX center longitude.

When the operator's location is not in the Americas (or cannot be
determined), returns ``[]`` so GMGSI takes over as the global fallback.
"""
from __future__ import annotations

from librewxr.sources._base import SatelliteContribution

from .source import (
    GOES18IRSource,
    GOES18VISSource,
    GOES19IRSource,
    GOES19VISSource,
)

__all__ = [
    "GOES18IRSource",
    "GOES18VISSource",
    "GOES19IRSource",
    "GOES19VISSource",
    "satellite_provider",
]


def _center_longitude(settings) -> float | None:
    """Determine the operator's center longitude from BBOX or station_lon."""
    bbox = getattr(settings, "get_bbox", lambda: None)()
    if bbox is not None:
        _, west, _, east = bbox
        return (west + east) / 2.0

    station_lon = getattr(settings, "station_lon", None)
    if station_lon is not None:
        return float(station_lon)

    return None


def satellite_provider(settings, cache_dir) -> list[SatelliteContribution]:
    """Return GOES IR + VIS contributions when the station is in the Americas.

    Selection logic:
    - Longitude between -170° and -30°: GOES coverage
    - West of -100°: GOES-18 (West)
    - East of -100°: GOES-19 (East)
    - Otherwise: return [] (fall through to Himawari or GMGSI)
    """
    if not getattr(settings, "goes_enabled", True):
        return []

    center_lon = _center_longitude(settings)
    if center_lon is None:
        return []

    if not (-170.0 <= center_lon <= -30.0):
        return []

    retention = getattr(settings, "satellite_max_frames", 36)
    contributions: list[SatelliteContribution] = []

    if center_lon < -100.0:
        # GOES-18 (West)
        if getattr(settings, "goes_ir_enabled", True):
            contributions.append(
                SatelliteContribution(
                    instance=GOES18IRSource(
                        cache_dir=cache_dir, max_frames=retention,
                    ),
                    priority=5,
                    name="GOES-18 IR",
                    slug="goes18_ir_grid",
                ),
            )
        if getattr(settings, "goes_vis_enabled", True):
            contributions.append(
                SatelliteContribution(
                    instance=GOES18VISSource(
                        cache_dir=cache_dir, max_frames=retention,
                    ),
                    priority=6,
                    name="GOES-18 VIS",
                    slug="goes18_vis_grid",
                ),
            )
    else:
        # GOES-19 (East)
        if getattr(settings, "goes_ir_enabled", True):
            contributions.append(
                SatelliteContribution(
                    instance=GOES19IRSource(
                        cache_dir=cache_dir, max_frames=retention,
                    ),
                    priority=5,
                    name="GOES-19 IR",
                    slug="goes19_ir_grid",
                ),
            )
        if getattr(settings, "goes_vis_enabled", True):
            contributions.append(
                SatelliteContribution(
                    instance=GOES19VISSource(
                        cache_dir=cache_dir, max_frames=retention,
                    ),
                    priority=6,
                    name="GOES-19 VIS",
                    slug="goes19_vis_grid",
                ),
            )

    return contributions
