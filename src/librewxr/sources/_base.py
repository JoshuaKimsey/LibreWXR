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
        self, region: RegionDef, minutes_ago: int
    ) -> np.ndarray | None: ...

    async def fetch_archive_frame(
        self, region: RegionDef, when: datetime
    ) -> np.ndarray | None: ...

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


@runtime_checkable
class NowcastSource(Protocol):
    """Shape every external nowcast source class must satisfy.

    External nowcast sources publish forecast-leg frames for a specific
    region — their own model-extrapolated nowcasts that we ingest
    directly rather than computing via internal optical-flow.  Use this
    when an upstream agency publishes a higher-quality nowcast than our
    generic extrapolation could produce (e.g. JMA HRPN which fuses
    XRAIN, models convective cell growth, and maintains gauge mass
    balance through the forecast horizon).

    The internal ``NowcastGenerator`` still owns:
      - boundary feathering at the contribution's coverage edge,
        where the external nowcast hands off to optical-flow extrapolation
      - blend weight against the radar analysis at T=0
      - snow-mask overlay if applicable

    Each call returns a list of ``(target_time, frame_data)`` tuples
    covering one update cycle's forecast frames (typically 5-min steps
    out to the horizon).  Returning ``None`` signals "skip this cycle"
    (e.g. transient upstream error); the internal extrapolation fills
    in for that cycle.
    """

    async def fetch_forecast(
        self, region: RegionDef
    ) -> list[tuple[int, np.ndarray]] | None: ...

    async def close(self) -> None: ...


@runtime_checkable
class SatelliteSource(Protocol):
    """Shape every satellite source class must satisfy.

    A satellite source ingests one channel of imagery (e.g. GMGSI LW or
    GMGSI VIS) on its own cadence, stores frames internally, and lets
    the renderer sample lat/lon grids out of any stored timestamp.

    Same ``fetch`` ergonomics as NWPGrid: signature varies per source
    (channel-specific kwargs, native cadence), so it's declared by the
    concrete class.  The pipeline introspects the signature via
    ``inspect.signature`` when dispatching, the same way it does for
    NWPGrid implementations.
    """

    name: str
    timestamps: list[int]

    async def close(self) -> None: ...


@dataclass
class RadarSourceContribution:
    """Return value from a source package's ``radar_provider(settings)``.

    A single contribution may cover multiple regions (e.g. MMD's one
    instance serves both MYPENINSULAR and MYEAST from one HTTP fetch).

    ``station_map`` is keyed by region name and feeds the coverage-mask
    builder in ``data/coverage.py``.  Regions without a station list and
    without a coverage polygon get no explicit mask (full-region bbox
    coverage is assumed by ``sample_coverage``).  ``range_overrides``
    likewise feeds the mask builder — any region missing here uses the
    240 km default Doppler reach.

    ``coverage_polygons`` is the alternative for gauge-corrected QPE
    composites whose published extent doesn't match individual Doppler
    ranges (JMA HRPN's tile pyramid traces a tilted polygon along the
    archipelago; OPERA, MRMS would qualify too).  When a region appears
    here, its mask is built from the polygon and the station-circle
    path is skipped.  Vertices are (latitude, longitude) tuples in
    polygon order — winding direction doesn't matter (the fill is
    rasterised either way).  Each region's value is either a single
    ring (``list[(lat, lon)]``) or a list of disjoint rings
    (``list[list[(lat, lon)]]``) for multi-island / open-ocean-tendril
    shapes (DPC Italy uses the latter for mainland + Sicily + Sardinia
    + the deep-Tyrrhenian / Ionian patches where no OPERA neighbour
    reaches).

    ``always_enabled`` marks the contribution's regions as part of the
    effective enabled set regardless of ``LIBREWXR_ENABLED_REGIONS`` —
    the coarse global observed-precip layer (RRQPE) must keep fetching
    and rendering as the bottom compositing tier even in a CONUS-only
    deployment.  The ``*_enabled`` per-source toggles and the global
    ``radar_enabled`` master switch still gate the contribution itself.
    """

    regions: list[RegionDef]
    instance: RadarSource
    group: str
    preempts: tuple[str, ...] = ()
    station_map: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    range_overrides: dict[str, float] = field(default_factory=dict)
    coverage_polygons: dict[
        str, list[tuple[float, float]] | list[list[tuple[float, float]]],
    ] = field(default_factory=dict)
    always_enabled: bool = False


@dataclass
class NWPContribution:
    """Return value from a source package's ``nwp_provider(settings, cache_dir)``.

    ``priority`` controls position in the NWPChain: lower runs earlier,
    so narrower / higher-resolution domains should use lower numbers.

    ``slug`` overrides the auto-generated snapshot / ``/health`` key.
    Leave it ``None`` to derive the key from ``name`` (lowercase,
    non-alphanumerics replaced with underscores, ``_grid`` suffixed).
    Three sources opt out: IFS (legacy key drops the "_ifs"), and the
    two AROME-OM variants whose display names include non-ASCII
    characters that wouldn't round-trip through the slugger.

    ``regional`` flags the contribution as part of the regional NWP
    chain — when ``regional_nwp_enabled`` is False, the central
    collector drops every regional contribution and lets IFS carry
    the chain alone.  Defaults to True so new sources don't have to
    opt in to the gate.  IFS itself sets ``regional=False``.
    """

    instance: NWPGrid
    priority: int
    name: str
    slug: str | None = None
    regional: bool = True


@dataclass
class NowcastContribution:
    """Return value from a source package's ``nowcast_provider(settings)``.

    A nowcast contribution covers exactly one region (the region where
    the upstream agency's nowcast is valid).  The internal pipeline
    handles boundary feathering at the region's edge, where the
    contribution's coverage hands off to optical-flow extrapolation.

    ``horizon_minutes`` is the maximum forecast horizon the contribution
    publishes.  The internal extrapolation runs beyond that horizon if
    needed (e.g. JMA publishes 60 min; if user requests 80 min,
    minutes 60-80 fall back to optical-flow extrapolation seeded by
    JMA's T+60 frame).
    """

    region_name: str
    instance: NowcastSource
    horizon_minutes: int = 60


@dataclass
class SatelliteContribution:
    """Return value from a source package's ``satellite_provider(settings, cache_dir)``.

    One contribution per channel: a multi-channel source family (like
    GMGSI with LW + VIS) returns multiple contributions, each carrying
    its own instance and slug.  The renderer references them by slug
    when assembling composites.

    ``priority`` is reserved for the day a second satellite-mosaic
    source appears (none planned).  Lower number wins for any pixel
    both cover.  Single-source today, so the field is mostly
    ceremonial.

    ``slug`` overrides the auto-generated snapshot / ``/health`` key.
    Leave it ``None`` to derive the key from ``name`` (lowercase,
    non-alphanumerics replaced with underscores, ``_grid`` suffixed).
    """

    instance: SatelliteSource
    priority: int
    name: str
    slug: str | None = None
