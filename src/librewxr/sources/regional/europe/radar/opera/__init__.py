# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""OPERA pan-European composite — self-contained source package.

Single region (``OPERA``) in the ``EUROPE`` group, fetched as ODIM HDF5
from the MeteoGate S3 buckets (Cloudferro ``openradar-24h``, with the
ECMWF ``openradar-24h-backup`` as fallback).  5-min cadence, 24-hour
rolling archive, LAEA projection.

Discovered automatically by ``librewxr.sources``.  Cross-country source
— no per-country directory because the OPERA composite genuinely covers
30+ countries.
"""
from __future__ import annotations

from librewxr.sources._base import RadarSourceContribution

from .regions import OPERA, REGIONS
from .source import OperaSource, _parse_opera_hdf5
from .stations import RANGE_OVERRIDES, STATION_MAP, STATIONS

REGION_GROUP = "EUROPE"

__all__ = [
    "OPERA",
    "OperaSource",
    "RANGE_OVERRIDES",
    "REGIONS",
    "REGION_GROUP",
    "STATIONS",
    "STATION_MAP",
    "_parse_opera_hdf5",
    "radar_provider",
]


def radar_provider(settings) -> RadarSourceContribution | None:
    """Return an OPERA contribution.

    OPERA has no enable toggle — user controls inclusion via region
    selection (``LIBREWXR_ENABLED_REGIONS=EUROPE`` or similar).  The
    fetcher only assigns the source to ``OPERA`` when the user actually
    enabled it, so eager instantiation here is cheap (HTTP client opens
    lazily on first fetch).
    """
    instance = OperaSource(settings.opera_bucket_urls)
    return RadarSourceContribution(
        regions=list(REGIONS),
        instance=instance,
        group=REGION_GROUP,
        station_map={k: list(v) for k, v in STATION_MAP.items()},
        range_overrides=dict(RANGE_OVERRIDES),
    )
