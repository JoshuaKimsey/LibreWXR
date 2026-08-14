# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Source registry — auto-discovers radar and NWP source packages.

Each source lives as a self-contained subpackage under one of:

    librewxr.sources.world.<source>            # global sources (e.g. IFS)
    librewxr.sources.regional.<region>.<...>.<source>

A source package's ``__init__.py`` exposes one of these provider
functions:

    def radar_provider(settings) -> RadarSourceContribution | None: ...
    def nwp_provider(settings, cache_dir) -> NWPContribution | None: ...

Returning ``None`` means "disabled / not configured" — the source is
skipped silently.

At import time we walk every subpackage and collect the provider
functions into ``RADAR_PROVIDERS`` and ``NWP_PROVIDERS``.  ``fetcher.py``
and ``main.py`` call those at startup to build the actual source map
and NWP chain.

Phase 0 (2026-05-17): scaffolding only.  No source packages have been
migrated yet, so both lists are empty in practice — the wiring below
is exercised by the registry tests but produces no runtime behavior.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import re
from collections.abc import Iterator
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from librewxr.sources._base import NWPContribution, SatelliteContribution

logger = logging.getLogger(__name__)


_SLUG_NONWORD = re.compile(r"\W+", re.UNICODE)


def nwp_grid_slug(contribution: "NWPContribution") -> str:
    """Return the snapshot / ``/health`` key for an NWP contribution.

    Honors ``contribution.slug`` when set; otherwise derives a key from
    ``contribution.name`` (lowercase, non-word characters collapsed to
    underscores, ``_grid`` suffix).  The result is what ``stores`` dicts
    in ``data_pipeline.py`` and ``main.py`` use as their per-grid key,
    and what ``api/routes.py`` emits in the ``/health`` payload.
    """
    if contribution.slug:
        return contribution.slug
    base = _SLUG_NONWORD.sub("_", contribution.name.lower()).strip("_")
    return f"{base}_grid"


def satellite_source_slug(contribution: "SatelliteContribution") -> str:
    """Return the snapshot / ``/health`` key for a satellite contribution.

    Mirrors ``nwp_grid_slug``: honors ``contribution.slug`` when set,
    otherwise lowercases ``contribution.name``, collapses non-word
    characters to underscores, and suffixes ``_grid``.  Used by
    ``data_pipeline.py`` and ``main.py`` as the satellite stores' key
    in the cross-worker snapshot dict.
    """
    if contribution.slug:
        return contribution.slug
    base = _SLUG_NONWORD.sub("_", contribution.name.lower()).strip("_")
    return f"{base}_grid"


def iter_source_packages() -> Iterator[ModuleType]:
    """Yield every importable subpackage under ``librewxr.sources``.

    Walks the package tree once; modules that fail to import are
    logged and skipped (don't take the whole server down if one
    contributor's package is broken).
    """
    pkg = importlib.import_module(__name__)
    for module_info in pkgutil.walk_packages(pkg.__path__, prefix=f"{__name__}."):
        if not module_info.ispkg:
            continue
        try:
            yield importlib.import_module(module_info.name)
        except Exception:
            logger.exception("Failed to import source package %s", module_info.name)


def _collect_providers() -> tuple[list, list, list, list]:
    radar, nwp, sat, nowcast = [], [], [], []
    for mod in iter_source_packages():
        rp = getattr(mod, "radar_provider", None)
        if callable(rp):
            radar.append(rp)
        np_ = getattr(mod, "nwp_provider", None)
        if callable(np_):
            nwp.append(np_)
        sp = getattr(mod, "satellite_provider", None)
        if callable(sp):
            sat.append(sp)
        ncp = getattr(mod, "nowcast_provider", None)
        if callable(ncp):
            nowcast.append(ncp)
    return radar, nwp, sat, nowcast


(
    RADAR_PROVIDERS,
    NWP_PROVIDERS,
    SATELLITE_PROVIDERS,
    NOWCAST_PROVIDERS,
) = _collect_providers()


def collect_radar_contributions(settings) -> list:
    """Walk active radar providers; return their contributions.

    Returns ``[]`` when ``settings.radar_enabled`` is False, short-
    circuiting every provider call so no S3 / HTTP / cache machinery
    gets stood up.  Used by the fetcher and the coverage-metadata
    helper so both honour the global radar toggle from one place.
    """
    if not getattr(settings, "radar_enabled", True):
        return []
    contributions = []
    for provider in RADAR_PROVIDERS:
        try:
            contribution = provider(settings)
        except Exception:
            logger.exception("Radar source provider %r raised", provider)
            continue
        if contribution is None:
            continue
        contributions.append(contribution)
    return contributions


def collect_nwp_contributions(settings, cache_dir) -> list:
    """Walk active NWP providers; return contributions sorted by priority.

    Returns a list of ``NWPContribution`` objects (each carrying the
    instantiated grid, its name, and its priority) — already sorted so
    callers can feed it straight into ``NWPChain``.  Providers that
    return ``None`` (e.g. ``hrdps_enabled=False`` or
    ``eu_nwp_profile != "dini_with_icon_eu"``) are filtered out.

    Honours the ``regional_nwp_enabled`` master switch: when False,
    every contribution flagged ``regional=True`` is dropped, leaving
    the global IFS base layer alone.  Useful during satellite-only
    or nowcast-only development where the regional download volume
    isn't worth waiting on.
    """
    regional_enabled = getattr(settings, "regional_nwp_enabled", True)
    contributions = []
    for provider in NWP_PROVIDERS:
        try:
            contribution = provider(settings, cache_dir)
        except Exception:
            logger.exception("NWP source provider %r raised", provider)
            continue
        if contribution is None:
            continue
        if contribution.regional and not regional_enabled:
            continue
        contributions.append(contribution)
    contributions.sort(key=lambda c: c.priority)
    return contributions


def collect_nowcast_contributions(settings) -> list:
    """Walk active nowcast providers; return their contributions.

    Returns ``[]`` when ``settings.nowcast_enabled`` is False or
    ``settings.radar_enabled`` is False (no radar implies no analysis
    leg to seed any external nowcast against).  Each contribution
    pins exactly one region by ``region_name``; the consuming
    ``NowcastGenerator`` keys its dispatch by that name and falls back
    to internal optical-flow extrapolation for any region without a
    contribution.
    """
    if not getattr(settings, "radar_enabled", True):
        return []
    if not getattr(settings, "nowcast_enabled", True):
        return []
    contributions = []
    for provider in NOWCAST_PROVIDERS:
        try:
            contribution = provider(settings)
        except Exception:
            logger.exception("Nowcast source provider %r raised", provider)
            continue
        if contribution is None:
            continue
        contributions.append(contribution)
    return contributions


def collect_satellite_contributions(settings, cache_dir) -> list:
    """Walk active satellite providers; return contributions sorted by priority.

    A provider may return *one or more* ``SatelliteContribution`` objects
    (one per channel; e.g. GMGSI returns LW + VIS together).  Each
    contribution carries its own instance, name, slug, and priority.

    Returns ``[]`` when ``settings.satellite_enabled`` is False,
    short-circuiting every provider call so no S3 / HTTP / cache
    machinery gets stood up.  Mirrors the radar-toggle pattern from
    ``collect_radar_contributions``; future satellite-master toggles
    fold in here rather than at each provider.
    """
    if not getattr(settings, "satellite_enabled", True):
        return []
    contributions = []
    for provider in SATELLITE_PROVIDERS:
        try:
            result = provider(settings, cache_dir)
        except Exception:
            logger.exception("Satellite source provider %r raised", provider)
            continue
        if result is None:
            continue
        # Providers may return a single contribution or a list — normalize.
        if isinstance(result, list):
            contributions.extend(c for c in result if c is not None)
        else:
            contributions.append(result)
    contributions.sort(key=lambda c: c.priority)
    return contributions


def always_enabled_region_names(settings) -> list[str]:
    """Region names contributed by providers flagged ``always_enabled``.

    These regions are fetched and rendered regardless of
    ``LIBREWXR_ENABLED_REGIONS`` — the coarse global observed-precip
    layer (RRQPE) must always be present as the bottom compositing tier,
    even in a CONUS-only deployment.  The per-source ``*_enabled``
    toggles and the global ``radar_enabled`` master switch still gate
    the contribution (via ``collect_radar_contributions``), so a
    disabled source contributes nothing here.
    """
    names: list[str] = []
    for contribution in collect_radar_contributions(settings):
        if not contribution.always_enabled:
            continue
        for region in contribution.regions:
            if region.name not in names:
                names.append(region.name)
    return names


def enabled_regions_with_always_on(settings) -> list[str]:
    """The resolved region spec plus every always-on contribution region.

    Union helper for boot-time callers that need the full effective
    enabled set (fetcher, lifespans, pipeline).  The base list comes
    from ``settings.get_enabled_regions()``; always-on regions are
    appended once each.
    """
    names = list(settings.get_enabled_regions())
    for name in always_enabled_region_names(settings):
        if name not in names:
            names.append(name)
    return names


def collect_radar_coverage_metadata(
    settings,
) -> tuple[
    dict[str, list[tuple[float, float]]],
    dict[str, float],
    dict[str, list[tuple[float, float]] | list[list[tuple[float, float]]]],
]:
    """Walk active radar providers; return merged station + range + polygon maps.

    Returns ``(station_map, range_overrides, coverage_polygons)``
    aggregated across every provider that didn't return ``None`` for
    the given ``settings``.  Used by ``data.coverage.build_coverage_masks``
    to size the coverage masks directly from per-source data, with no
    central station table.

    The walk respects the same provider gating as the fetcher (e.g.
    MRMS vs. IEM via ``na_source``), so the coverage masks always reflect
    whichever source is actually fetching frames.  If two providers
    contribute the same region key (shouldn't happen in practice — the
    fetcher itself would also fight over it), later providers win for
    that key.
    """
    station_map: dict[str, list[tuple[float, float]]] = {}
    range_overrides: dict[str, float] = {}
    coverage_polygons: dict[
        str, list[tuple[float, float]] | list[list[tuple[float, float]]],
    ] = {}
    for contribution in collect_radar_contributions(settings):
        for region_name, stations in contribution.station_map.items():
            station_map[region_name] = list(stations)
        for region_name, range_km in contribution.range_overrides.items():
            range_overrides[region_name] = range_km
        for region_name, polygon in contribution.coverage_polygons.items():
            coverage_polygons[region_name] = list(polygon)
    return station_map, range_overrides, coverage_polygons
