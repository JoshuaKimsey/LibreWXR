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
from collections.abc import Iterator
from types import ModuleType

logger = logging.getLogger(__name__)


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
