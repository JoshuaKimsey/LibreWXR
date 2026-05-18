# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NOAA MRMS — self-contained source package.

Serves USCOMP / AKCOMP / HICOMP / PRCOMP / GUCOMP from the NCEP MRMS
GRIB2 endpoint when ``settings.na_source`` is ``mrms`` or
``mrms_fallback``, and additionally CACOMP (via the CONUS product) when
``settings.ca_source`` is ``mrms`` or ``mrms_with_msc_blend``.  The two
settings are independent: e.g. ``na_source=iem`` with
``ca_source=mrms`` activates MRMS *only* for CACOMP.

US regions and the IEM-flavored station inventory live one level up at
``sources/regional/north_america/usa/radar/`` because they're shared
with IEM.  MRMS contributes its own multi-product source, the
``MRMS_PRODUCTS`` / ``MRMS_EXTENTS`` tables, and the NEXRAD+Canadian
station combination it actually ingests.
"""
from __future__ import annotations

from librewxr.sources._base import RadarSourceContribution
from librewxr.sources.regional.north_america.canada.radar.msc_canada.regions import (
    CACOMP,
)

from ..regions import REGIONS as USA_REGIONS
from .products import MRMS_EXTENTS, MRMS_PRODUCTS
from .source import (
    MRMSCompositeSource,
    MRMSSource,
    _parse_mrms_grib2,
    _resample_mrms_to_region,
    _suppress_eccodes_stderr,
)
from .stations import STATION_MAP

__all__ = [
    "MRMSCompositeSource",
    "MRMSSource",
    "MRMS_EXTENTS",
    "MRMS_PRODUCTS",
    "STATION_MAP",
    "_parse_mrms_grib2",
    "_resample_mrms_to_region",
    "_suppress_eccodes_stderr",
    "radar_provider",
]


def radar_provider(settings) -> RadarSourceContribution | None:
    """Return an MRMS contribution when MRMS is the active US or CA source.

    Activation is the OR of two settings:
      - ``na_source in {"mrms", "mrms_fallback"}`` → MRMS serves the 5 US
        regions (USCOMP, AKCOMP, HICOMP, PRCOMP, GUCOMP)
      - ``ca_source in {"mrms", "mrms_with_msc_blend"}`` → MRMS serves
        CACOMP via the same CONUS product

    Either or both may be active.  The contribution's ``regions`` list is
    constructed dynamically so the fetcher only wires MRMS to the slots
    it should own; the cross-source ``_iem_fallback`` (US side) and
    ``_cacomp_msc_source`` (CA blend partner) machinery in
    ``data/fetcher.py`` handles the rest.

    A single ``MRMSCompositeSource`` covers everything; inside that
    wrapper, per-product ``MRMSSource`` instances are created lazily on
    first request (USCOMP and CACOMP share the CONUS product instance).
    """
    na_source = getattr(settings, "na_source", "mrms_fallback")
    ca_source = getattr(settings, "ca_source", "mrms_with_msc_blend")
    us_active = na_source in ("mrms", "mrms_fallback")
    ca_active = ca_source in ("mrms", "mrms_with_msc_blend")
    if not us_active and not ca_active:
        return None

    regions: list = []
    if us_active:
        regions.extend(USA_REGIONS)
    if ca_active:
        regions.append(CACOMP)

    instance = MRMSCompositeSource(settings.mrms_base_url)
    return RadarSourceContribution(
        regions=regions,
        instance=instance,
        group="US",
        station_map={k: list(v) for k, v in STATION_MAP.items()},
    )
