# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NOAA Enterprise Rain Rate (RRQPE) GLB-5 blend — self-contained radar source package.

Satellite-derived *observed* precipitation (10-minute global rain rate
from the geostationary constellation) consumed from NOAA's anonymous
Open Data S3 bucket ``noaa-enterprise-rainrate-pds``.

A single coarse global radar region (``RRQPE``) that sorts LAST in the
multi-region compositor — it fills only pixels no finer radar region
claims, so it never overwrites a Doppler composite's authoritative "no
echo" zeros.  Always-on: the region is fetched and rendered even when
``LIBREWXR_ENABLED_REGIONS`` is a narrow group, so the global observed
tier is always present under the regional radars.

The package exposes ``REGIONS`` at import time (like the USA radar
package) and deliberately defers the ``_base`` import into
``radar_provider`` — the discovery walker's merge pass imports source
packages while ``librewxr.sources._base`` may still be initialising, so
any module that imports ``_base`` at the top would fail that pass and
its REGIONS contribution would be lost.
"""
from __future__ import annotations

from pathlib import Path

from .grid import RRQPEGrid
from .regions import RRQPE, RRQPE_COVERAGE_POLYGON, REGIONS
from .source import RRQPESource

__all__ = ["RRQPEGrid", "RRQPESource", "radar_provider"]


def radar_provider(settings):
    """Return an RRQPE radar contribution when ``settings.rrqpe_enabled`` is set."""
    from librewxr.sources._base import RadarSourceContribution

    if not getattr(settings, "rrqpe_enabled", True):
        return None
    # Radar providers get no ``cache_dir`` argument; the scan cache is
    # fetch-side working state only -- nothing reads it back across
    # restarts (no loader in any __init__).  The persistent copy that
    # survives restarts is the FrameStore's ``radar/<ts>_RRQPE.dat``,
    # restored via RadarFrameCache.  Route this per-cycle write traffic to
    # the volatile (RAM-backed) directory when one is configured
    # (LIBREWXR_VOLATILE_CACHE_DIR), else fall back to the durable cache
    # dir, else a temp dir.
    volatile = getattr(settings, "volatile_cache_dir", "") or None
    durable = getattr(settings, "cache_dir", "") or None
    scan_cache_dir = Path(volatile) if volatile else (Path(durable) if durable else None)
    return RadarSourceContribution(
        regions=REGIONS,
        instance=RRQPESource(
            RRQPEGrid(cache_dir=scan_cache_dir),
        ),
        group=RRQPE.group,
        # The published extent is the full 2°-inset global band (see
        # regions.py) — no Doppler station circles apply.
        coverage_polygons={RRQPE.name: RRQPE_COVERAGE_POLYGON},
        # Global observed tier: stays in the effective enabled set
        # regardless of the region spec.
        always_enabled=True,
    )
