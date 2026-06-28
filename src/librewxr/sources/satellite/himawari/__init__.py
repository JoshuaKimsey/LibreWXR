# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Himawari-9 AHI satellite source package.

High-resolution (2 km IR, 1 km VIS) geostationary imagery for
Asia-Pacific at 10-minute cadence from ``s3://noaa-himawari9/``.

Auto-selected when the operator's BBOX center is between 60°E and 180°E.
Returns ``[]`` for stations outside Asia-Pacific, falling through to
GMGSI as the global fallback.
"""
from __future__ import annotations

from librewxr.sources._base import SatelliteContribution

from .source import HimawariIRSource, HimawariVISSource

__all__ = ["HimawariIRSource", "HimawariVISSource", "satellite_provider"]


def _center_longitude(settings) -> float | None:
    bbox = getattr(settings, "get_bbox", lambda: None)()
    if bbox is not None:
        _, west, _, east = bbox
        return (west + east) / 2.0

    station_lon = getattr(settings, "station_lon", None)
    if station_lon is not None:
        return float(station_lon)

    return None


def satellite_provider(settings, cache_dir) -> list[SatelliteContribution]:
    """Return Himawari IR + VIS contributions for Asia-Pacific stations.

    Coverage: longitude 60°E to 180°E (Japan, Korea, SE Asia, Oceania,
    eastern Africa coast, western Pacific).
    """
    if not getattr(settings, "himawari_enabled", True):
        return []

    center_lon = _center_longitude(settings)
    if center_lon is None:
        return []

    if not (60.0 <= center_lon <= 180.0):
        return []

    retention = getattr(settings, "satellite_max_frames", 36)
    contributions: list[SatelliteContribution] = []

    if getattr(settings, "himawari_ir_enabled", True):
        contributions.append(
            SatelliteContribution(
                instance=HimawariIRSource(
                    cache_dir=cache_dir, max_frames=retention,
                ),
                priority=5,
                name="Himawari-9 IR",
                slug="himawari9_ir_grid",
            ),
        )

    if getattr(settings, "himawari_vis_enabled", True):
        contributions.append(
            SatelliteContribution(
                instance=HimawariVISSource(
                    cache_dir=cache_dir, max_frames=retention,
                ),
                priority=6,
                name="Himawari-9 VIS",
                slug="himawari9_vis_grid",
            ),
        )

    return contributions
