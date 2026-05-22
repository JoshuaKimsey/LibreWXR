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
    from librewxr.sources._base import NWPContribution

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


def _collect_providers() -> tuple[list, list]:
    radar, nwp = [], []
    for mod in iter_source_packages():
        rp = getattr(mod, "radar_provider", None)
        if callable(rp):
            radar.append(rp)
        np_ = getattr(mod, "nwp_provider", None)
        if callable(np_):
            nwp.append(np_)
    return radar, nwp


RADAR_PROVIDERS, NWP_PROVIDERS = _collect_providers()


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
    """
    contributions = []
    for provider in NWP_PROVIDERS:
        try:
            contribution = provider(settings, cache_dir)
        except Exception:
            logger.exception("NWP source provider %r raised", provider)
            continue
        if contribution is None:
            continue
        contributions.append(contribution)
    contributions.sort(key=lambda c: c.priority)
    return contributions


def collect_radar_coverage_metadata(
    settings,
) -> tuple[
    dict[str, list[tuple[float, float]]],
    dict[str, float],
]:
    """Walk active radar providers; return merged station + range maps.

    Returns ``(station_map, range_overrides)`` aggregated across every
    provider that didn't return ``None`` for the given ``settings``.
    Used by ``data.coverage.build_coverage_masks`` to size station-circle
    masks directly from per-source data, with no central station table.

    The walk respects the same provider gating as the fetcher (e.g.
    MRMS vs. IEM via ``na_source``), so the coverage masks always reflect
    whichever source is actually fetching frames.  If two providers
    contribute the same region key (shouldn't happen in practice — the
    fetcher itself would also fight over it), later providers win for
    that key.
    """
    station_map: dict[str, list[tuple[float, float]]] = {}
    range_overrides: dict[str, float] = {}
    for contribution in collect_radar_contributions(settings):
        for region_name, stations in contribution.station_map.items():
            station_map[region_name] = list(stations)
        for region_name, range_km in contribution.range_overrides.items():
            range_overrides[region_name] = range_km
    return station_map, range_overrides
