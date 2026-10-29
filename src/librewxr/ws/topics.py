# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""WebSocket topic registry and payload builders.

The HTTP JSON API stays canonical: every topic payload here is produced
by the same builder the equivalent HTTP route uses, so a subscriber sees
exactly the bytes ``/public/weather-maps.json``, ``/v2/alerts``,
``/v2/storm-cells``, and ``/v2/lightning`` would return.

Builders read the ``routes`` module globals at call time (the codebase
has no dependency injection).  Everything expensive -- the shapely union
in the alerts builder, the per-strike conversion loop, Pydantic
serialization -- runs through ``asyncio.to_thread`` so the render loop
that serves tiles never stalls.
"""

import asyncio
import time

from librewxr.api import routes
from librewxr.config import settings
from librewxr.mcp import tools as mcp_tools

TOPIC_WEATHER_MAPS = "weather-maps"
TOPIC_ALERTS = "alerts"
TOPIC_STORM_CELLS = "storm-cells"
TOPIC_LIGHTNING = "lightning"
ALL_TOPICS: tuple[str, ...] = (
    TOPIC_WEATHER_MAPS,
    TOPIC_ALERTS,
    TOPIC_STORM_CELLS,
    TOPIC_LIGHTNING,
)

# Topics whose payload is rebuilt from a ``state.json`` snapshot refresh.
# ``lightning`` is deliberately absent: its store reloads from its own
# on-disk artifact on the poll tick instead.
STATE_TOPICS: tuple[str, ...] = (
    TOPIC_WEATHER_MAPS,
    TOPIC_ALERTS,
    TOPIC_STORM_CELLS,
)


def topic_available(topic: str) -> bool:
    """Whether the underlying data for ``topic`` is wired up right now.

    Mirrors the HTTP routes' 503 gates so a WS subscriber can never be
    promised data the REST API would refuse to serve.
    """
    if topic == TOPIC_WEATHER_MAPS:
        return routes.frame_store is not None
    if topic == TOPIC_ALERTS:
        return routes.alerts_enabled and routes.alerts_store is not None
    if topic == TOPIC_STORM_CELLS:
        return routes.storm_cell_store is not None
    if topic == TOPIC_LIGHTNING:
        return settings.lightning_enabled and routes.lightning_store is not None
    return False


async def topic_cache_key(topic: str):
    """Cheap content key for ``topic``'s current data.

    Lets the hub cache a serialized snapshot and skip rebuilding it while
    the underlying data is unchanged.  Gathers the same inputs the full
    builder would read, without building the payload.
    """
    if topic == TOPIC_WEATHER_MAPS:
        timestamps = (
            await routes.frame_store.get_timestamps()
            if routes.frame_store is not None
            else []
        )
        nowcast_timestamps: tuple[int, ...] = ()
        if routes.nowcast_store is not None:
            nowcast_timestamps = tuple(await routes.nowcast_store.get_timestamps())
        infrared_timestamps: tuple[int, ...] = ()
        gmgsi_lw = (
            routes.satellite_grids.get("gmgsi_lw_grid")
            if routes.satellite_grids
            else None
        )
        if gmgsi_lw is not None and gmgsi_lw.timestamps:
            infrared_timestamps = tuple(gmgsi_lw.timestamps)
        return (tuple(sorted(timestamps)), nowcast_timestamps, infrared_timestamps)
    if topic == TOPIC_ALERTS:
        return routes.alerts_store.last_updated if routes.alerts_store is not None else None
    if topic == TOPIC_STORM_CELLS:
        return (
            routes.storm_cell_store.cells_version
            if routes.storm_cell_store is not None
            else None
        )
    if topic == TOPIC_LIGHTNING:
        return (
            routes.lightning_store.version
            if routes.lightning_store is not None
            else None
        )
    return None


async def build_topic_snapshot(topic: str) -> tuple[str, str]:
    """Build ``(payload_json, fingerprint_json)`` for a state topic.

    ``fingerprint_json`` is the payload with volatile fields (the
    weather-maps ``generated`` timestamp) excluded, so a pure clock tick
    never looks like a data change to the hub's push diff.
    """
    if topic == TOPIC_WEATHER_MAPS:
        model = await routes.build_weather_maps_response()

        def _render() -> tuple[str, str]:
            payload_json = model.model_dump_json()
            fingerprint_json = model.model_dump_json(exclude={"generated"})
            return payload_json, fingerprint_json

        return await asyncio.to_thread(_render)

    if topic == TOPIC_ALERTS:
        # Sync property; a shallow copy of the entry list.
        alerts = routes.alerts_store.alerts

        def _render() -> tuple[str, str]:
            model = routes.build_alerts_response(alerts)
            payload_json = model.model_dump_json()
            return payload_json, payload_json

        return await asyncio.to_thread(_render)

    if topic == TOPIC_STORM_CELLS:
        cells = await mcp_tools.get_storm_cells(routes.storm_cell_store)

        def _render() -> tuple[str, str]:
            model = routes.build_storm_cells_geojson(cells)
            payload_json = model.model_dump_json()
            return payload_json, payload_json

        return await asyncio.to_thread(_render)

    raise ValueError(f"unknown topic: {topic}")


async def build_lightning_snapshot() -> tuple[str, int]:
    """Build the lightning snapshot as ``(payload_json, watermark_s)``.

    Strikes are in ascending chronological order (the store's native
    order) -- that ordering is a documented protocol guarantee, so do not
    reverse it to match the REST API's newest-first default.  The
    watermark is the newest strike time in the payload; the hub uses it to
    request strictly-newer strikes on subsequent ticks.
    """
    store = routes.lightning_store
    await store.maybe_reload()

    def _render() -> tuple[str, int]:
        points = store.points_in(
            -90, 90, -180, 180,
            since_s=int(time.time()) - settings.lightning_max_age,
        )
        if points.shape[0] == 0:
            watermark_s = int(store.window_end_s or 0)
        else:
            watermark_s = int(points["time_s"].max())
        model = routes.build_lightning_geojson(
            mcp_tools.lightning_strike_dicts(points)
        )
        return model.model_dump_json(), watermark_s

    return await asyncio.to_thread(_render)


async def build_lightning_delta(since_s: int) -> tuple[str, int] | None:
    """Build the strikes strictly newer than ``since_s``.

    Returns ``None`` when nothing new arrived (the hub then pushes no
    message at all rather than an empty update).
    """
    store = routes.lightning_store

    def _render() -> tuple[str, int] | None:
        # ``points_in`` bounds are inclusive, so ``since_s + 1`` makes the
        # lower edge strict and no strike is ever delivered twice.
        points = store.points_in(-90, 90, -180, 180, since_s=since_s + 1)
        if points.shape[0] == 0:
            return None
        model = routes.build_lightning_geojson(
            mcp_tools.lightning_strike_dicts(points)
        )
        return model.model_dump_json(), int(points["time_s"].max())

    return await asyncio.to_thread(_render)
