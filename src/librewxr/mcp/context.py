# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Stdio-mode lifespan that builds stores from a state.json snapshot.

Mirrors ``_render_only_lifespan`` from ``main.py`` but drops all
rendering-only singletons (TileCache, TileWarmer, TileRequestTracker,
MemoryMonitor, satellite grids).  Only the data-store + snapshot +
poller + coverage-mask pieces are kept.
"""

import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from librewxr.config import settings
from librewxr.data.coverage import build_coverage_masks, build_feather_masks
from librewxr.data.master_state import (
    _load_and_apply_state,
    apply_state,
    load_state,
    state_mtime,
)
from librewxr.data.nowcast import NowcastStore
from librewxr.data.nwp_source import NWPChain
from librewxr.data.store import FrameStore
from librewxr.data.alerts_store import AlertsStore
from librewxr.api import routes
from librewxr.sources import (
    collect_nwp_contributions,
    collect_radar_coverage_metadata,
    nwp_grid_slug,
)

logger = logging.getLogger(__name__)


@dataclass
class McpContext:
    """Holds references the lifespan manages for cleanup."""

    frame_store: FrameStore | None = None
    nwp_grids_by_slug: dict = field(default_factory=dict)
    nwp_chain: NWPChain | None = None
    nowcast_store: NowcastStore | None = None
    alerts_store: AlertsStore | None = None
    enabled_regions: list[str] = field(default_factory=list)
    alerts_enabled: bool = False
    poller_task: asyncio.Task | None = None
    poller_stop: asyncio.Event | None = None


async def _wait_for_state(cache_dir: Path, timeout: float) -> None:
    """Block until state.json exists under cache_dir, or fail loudly.

    Polling rather than inotify keeps the implementation portable to Docker
    bind-mounts and shared NFS volumes.  During cold start the pipeline takes
    minutes to complete its first fetch cycle, so we poll lazily (every 2 s)
    and only log on entry + every 30 s.

    This is a copy of ``librewxr.main._wait_for_state``, inlined here to
    avoid importing ``main.py`` and its heavy dependency tree at module
    load time.
    """
    deadline = time.time() + timeout if timeout > 0 else None
    poll = max(settings.state_poll_interval, 2.0)
    log_every = 30.0
    started = time.time()
    last_logged = 0.0
    if state_mtime(cache_dir) is not None:
        return
    logger.info("Waiting for pipeline state.json under %s ...", cache_dir)
    while True:
        await asyncio.sleep(poll)
        if state_mtime(cache_dir) is not None:
            logger.info(
                "Pipeline state.json appeared after %.0fs",
                time.time() - started,
            )
            return
        if deadline is not None and time.time() > deadline:
            raise RuntimeError(
                f"Timed out after {timeout:.0f}s waiting for state.json "
                f"under {cache_dir}.  Is the data pipeline running?"
            )
        elapsed = time.time() - started
        if elapsed - last_logged >= log_every:
            logger.info(
                "Still waiting for state.json (%.0fs elapsed) ...", elapsed,
            )
            last_logged = elapsed


@asynccontextmanager
async def build_stdio_lifespan(mcp_instance):
    """FastMCP lifespan that builds stores from the pipeline's state.json.

    Mirrors ``_render_only_lifespan`` from ``main.py`` but drops all
    rendering-only singletons (``TileCache``, ``TileWarmer``,
    ``TileRequestTracker``, ``MemoryMonitor``, satellite grids).
    Only the data-store + snapshot + poller + coverage-mask pieces
    are kept.
    """
    # ---- cache_dir requirement --------------------------------------------
    if not settings.cache_dir:
        raise RuntimeError(
            "LIBREWXR_MCP stdio transport requires LIBREWXR_CACHE_DIR to be "
            "set (it's the shared volume the pipeline writes state.json into)."
        )
    cache_dir = Path(settings.cache_dir)

    # ---- Wait for state.json to exist ------------------------------------
    await _wait_for_state(cache_dir, settings.state_wait_timeout)

    # ---- Build empty stores ----------------------------------------------
    store = FrameStore(max_frames=settings.max_frames, cache_dir=cache_dir)
    nwp_contribs = collect_nwp_contributions(settings, cache_dir)
    nwp_grids_by_slug: dict[str, object] = {
        nwp_grid_slug(c): c.instance for c in nwp_contribs
    }
    nowcast_store = (
        NowcastStore(cache_dir=cache_dir)
        if (settings.nowcast_enabled or settings.arrow_flow_enabled)
        else None
    )
    alerts_store = AlertsStore() if settings.alerts_enabled else None

    # ---- Build the stores dict for apply_state ---------------------------
    stores: dict[str, object | None] = {
        "frame_store": store,
        **nwp_grids_by_slug,
        "nowcast_store": nowcast_store,
        "alerts_store": alerts_store,
    }

    # ---- Load snapshot ----------------------------------------------------
    payload = load_state(cache_dir)
    if payload is None:
        raise RuntimeError(
            f"state.json disappeared between mtime check and load -- "
            f"is something else writing to {cache_dir}?"
        )
    refreshed = apply_state(payload, stores)
    logger.info(
        "MCP stdio loaded snapshot: %s",
        ", ".join(refreshed) if refreshed else "(empty)",
    )

    # ---- Drop sources not in snapshot ------------------------------------
    for name in list(stores.keys()):
        if name not in refreshed and name != "frame_store":
            stores[name] = None
    nwp_grids_by_slug = {
        slug: stores[slug] for slug in nwp_grids_by_slug if stores[slug] is not None
    }
    nowcast_store = stores["nowcast_store"]
    alerts_store = stores["alerts_store"]

    # ---- Coverage masks --------------------------------------------------
    station_map, range_overrides, coverage_polygons = collect_radar_coverage_metadata(
        settings,
    )
    build_coverage_masks(
        station_map,
        range_overrides=range_overrides,
        coverage_polygons=coverage_polygons,
    )
    build_feather_masks()

    # ---- NWP chain -------------------------------------------------------
    chain_sources = [
        c.instance
        for c in nwp_contribs
        if nwp_grid_slug(c) in nwp_grids_by_slug
    ]
    nwp_chain = NWPChain(chain_sources)
    logger.info(
        "MCP stdio NWP chain: [%s]",
        ", ".join(s.name for s in nwp_chain.sources),
    )

    # ---- Enabled regions -------------------------------------------------
    enabled = settings.get_enabled_regions()

    # ---- Write routes.* --------------------------------------------------
    routes.frame_store = store
    routes.tile_cache = None
    routes.nwp_grids = nwp_grids_by_slug
    routes.ecmwf_grid = nwp_grids_by_slug.get("ecmwf_grid")
    routes.nwp_chain = nwp_chain
    routes.satellite_grids = {}
    routes.tile_warmer = None
    routes.nowcast_store = nowcast_store
    routes.tile_request_tracker = None
    routes.start_time = time.time()
    routes.enabled_regions = enabled
    routes.radar_cache = None
    routes.radar_fetcher = None
    routes.alerts_store = alerts_store
    routes.alerts_fetcher = None
    routes.alerts_enabled = alerts_store is not None

    # ---- State poller ----------------------------------------------------
    last_mtime = state_mtime(cache_dir)
    # Seed the diff from the boot payload so the first poll skips unchanged stores.
    last_payload: dict | None = payload
    poller_stop = asyncio.Event()

    async def _poll_state() -> None:
        nonlocal last_mtime, last_payload
        while not poller_stop.is_set():
            try:
                # Jitter the poll cadence (uniform over [0.5, 1.5] x
                # interval) so concurrent readers don't hit state.json in
                # lockstep on every pipeline cycle.
                await asyncio.wait_for(
                    poller_stop.wait(),
                    timeout=settings.state_poll_interval * (0.5 + random.random()),
                )
                return
            except asyncio.TimeoutError:
                pass
            mtime = state_mtime(cache_dir)
            if mtime is None or mtime == last_mtime:
                continue
            try:
                # load_state (json.loads) + apply_state (memmap re-opens)
                # run in a worker thread so the event loop never blocks;
                # the previous payload is passed for per-store diffing.
                payload, refreshed = await asyncio.to_thread(
                    _load_and_apply_state, cache_dir, stores, last_payload,
                )
                if payload is None:
                    continue
                last_mtime = mtime
                last_payload = payload
                logger.debug(
                    "MCP stdio refreshed: %s",
                    ", ".join(refreshed),
                )
            except Exception:
                logger.exception(
                    "Failed to refresh state from %s", cache_dir,
                )

    poller_task = asyncio.create_task(_poll_state())

    # ---- Build McpContext for cleanup tracking ---------------------------
    ctx = McpContext(
        frame_store=store,
        nwp_grids_by_slug=nwp_grids_by_slug,
        nwp_chain=nwp_chain,
        nowcast_store=nowcast_store,
        alerts_store=alerts_store,
        enabled_regions=enabled,
        alerts_enabled=alerts_store is not None,
        poller_task=poller_task,
        poller_stop=poller_stop,
    )

    logger.info(
        "MCP stdio context ready (cache_dir=%s, regions=%s)",
        cache_dir,
        ", ".join(enabled),
    )

    try:
        yield {"mcp_context": ctx}
    finally:
        # ---- Teardown ----------------------------------------------------
        poller_stop.set()
        try:
            await poller_task
        except Exception:
            logger.exception("Poller shutdown error")
        if nowcast_store is not None:
            nowcast_store.cleanup()
        store.cleanup()
        logger.info("MCP stdio context shutdown complete")
