# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Protocols and contribution dataclasses for the sources registry.

Each source package under ``librewxr.sources`` exports a provider
function (``radar_provider`` or ``nwp_provider``) returning one of the
contribution dataclasses below.  The discovery walker in
``librewxr.sources.__init__`` collects those providers; ``fetcher.py``
and ``main.py`` call them to assemble the actual radar source map and
NWP chain at startup.

Source-specific config still lives in ``librewxr.config.settings`` and
is passed into each provider — Phase 0 of the refactor (2026-05-17)
deliberately kept config centralized to avoid Pydantic gymnastics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np

from librewxr.data.regions import RegionDef


@runtime_checkable
class RadarSource(Protocol):
    """Shape every radar source class must satisfy.

    Concrete implementations are duck-typed against this — they don't
    need to inherit from it.  ``runtime_checkable`` is included so
    tests can ``isinstance``-check without imports.
    """

    async def fetch_frame(
        self, region: RegionDef
    ) -> tuple[datetime, np.ndarray] | None: ...

    async def fetch_archive_frame(
        self, region: RegionDef, when: datetime
    ) -> tuple[datetime, np.ndarray] | None: ...

    async def close(self) -> None: ...


@runtime_checkable
class NWPGrid(Protocol):
    """Shape every NWP grid class must satisfy.

    ``fetch()`` signature varies by grid (different return types, kwargs,
    timestep semantics) so it's intentionally not declared here — the
    NWPChain in ``data/fetcher.py`` knows what to call.  This protocol
    only pins down the universally-required parts.
    """

    name: str

    async def close(self) -> None: ...


@dataclass
class RadarSourceContribution:
    """Return value from a source package's ``radar_provider(settings)``.

    A single contribution may cover multiple regions (e.g. MMD's one
    instance serves both MYPENINSULAR and MYEAST from one HTTP fetch).
    """

    regions: list[RegionDef]
    instance: RadarSource
    group: str
    preempts: tuple[str, ...] = ()
    stations: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class NWPContribution:
    """Return value from a source package's ``nwp_provider(settings, cache_dir)``.

    ``priority`` controls position in the NWPChain: lower runs earlier,
    so narrower / higher-resolution domains should use lower numbers.
    """

    instance: NWPGrid
    priority: int
    name: str
