# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rich.logging import RichHandler
from starlette.exceptions import HTTPException as StarletteHTTPException

from librewxr.api import routes
from librewxr.config import settings
from librewxr.data.coverage import build_coverage_masks, build_feather_masks
from librewxr.data.fetcher import RadarFetcher
from librewxr.data.master_state import apply_state, load_state, state_mtime
from librewxr.data.nowcast import NowcastGenerator, NowcastStore
from librewxr.data.nwp_source import NWPChain
from librewxr.data.store import FrameStore
from librewxr.sources import (
    collect_nowcast_contributions,
    collect_nwp_contributions,
    collect_radar_coverage_metadata,
    collect_satellite_contributions,
    nwp_grid_slug,
    satellite_source_slug,
)
from librewxr.data.alerts_store import AlertsStore
from librewxr.data.alerts_fetcher import WMOAlertsFetcher
from librewxr.memory import MemoryMonitor, detect_memory_limit_mb
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import (
    ALL_CACHES,
    warm_coordinate_caches,
)
from librewxr.tiles.request_tracker import TileRequestTracker
from librewxr.tiles.warmer import TileWarmer

# Map dotted logger names to short subsystem tags so concurrent startup
# (radar / IFS / NWP / GMGSI all firing in parallel) reads cleanly in the log.
# Anything not in the map falls back to the last segment of the module
# path (e.g. an unmapped third-party logger keeps its own short name).
_LOG_TAGS = {
    "librewxr.main": "main",
    "librewxr.config": "config",
    "librewxr.memory": "memory",
    "librewxr.api.routes": "api",
    "librewxr.data.sources": "radar",
    "librewxr.data.fetcher": "fetcher",
    "librewxr.data.store": "store",
    "librewxr.data.regions": "regions",
    "librewxr.data.coverage": "coverage",
    "librewxr.sources.world.ifs.grid": "ifs",
    "librewxr.sources.world.ifs.interpolation": "ifs",
    "librewxr.sources.regional.north_america.usa.nwp.hrrr.grid": "hrrr",
    "librewxr.sources.regional.north_america.usa.nwp.hrrr_alaska.grid": "hrrr-ak",
    "librewxr.sources.regional.europe.nwp.icon_eu.grid": "icon-eu",
    "librewxr.sources.regional.europe.nwp.dmi_dini.grid": "dmi-dini",
    "librewxr.sources.regional.north_america.canada.nwp.hrdps.grid": "hrdps",
    "librewxr.sources.regional.caribbean.nwp.arome_antilles.grid": "arome-ant",
    "librewxr.sources.regional.south_america.nwp.wrf_smn.grid": "wrf-smn",
    "librewxr.data.nowcast": "nowcast",
    "librewxr.tiles.warmer": "warmer",
    "librewxr.tiles.cache": "tiles",
    "librewxr.tiles.renderer": "tiles",
    "librewxr.tiles.satellite_renderer": "tiles",
    "librewxr.tiles.coordinates": "tiles",
    "librewxr.data.alerts_fetcher": "alerts",
    "librewxr.data.alerts_store": "alerts",
}


class _TagFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.tag = _LOG_TAGS.get(record.name, record.name.rsplit(".", 1)[-1])
        return super().format(record)


_handler = RichHandler(rich_tracebacks=True, show_path=False)
_handler.setFormatter(_TagFormatter("[%(tag)s] %(message)s"))
logging.basicConfig(
    level=logging.INFO,
    handlers=[_handler],
    force=True,
)
# Suppress noisy per-request INFO logs from httpx/httpcore — we already log
# fetch results ourselves in sources.py / fetcher.py.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _clear_coord_caches() -> None:
    """Clear all coordinate LRU caches."""
    for fn in ALL_CACHES:
        fn.cache_clear()
    logger.info("Coordinate caches cleared by memory monitor")


async def _wait_for_state(cache_dir, timeout: float) -> None:
    """Block until state.json exists under cache_dir, or fail loudly.

    A render-only worker is useless without a snapshot to read.  Polling
    rather than inotify keeps the implementation portable to Docker
    bind-mounts and shared NFS volumes.

    During cold start the pipeline takes minutes to complete its first
    fetch cycle, so we poll lazily (every 2 s) and only log on entry +
    every 30 s — N workers each polling at 1 Hz produces real log spam.
    """
    deadline = time.time() + timeout if timeout > 0 else None
    poll = max(settings.state_poll_interval, 2.0)
    log_every = 30.0
    started = time.time()
    last_logged = 0.0
    if state_mtime(cache_dir) is not None:
        return
    logger.info("Waiting for pipeline state.json under %s …", cache_dir)
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
                "Still waiting for state.json (%.0fs elapsed) …", elapsed,
            )
            last_logged = elapsed


@asynccontextmanager
async def _render_only_lifespan(app: FastAPI):
    """Lifespan for tile-server workers in the multi-worker split.

    Pulls all radar / NWP / satellite data from the snapshot the data
    pipeline writes, and refreshes it in place every time
    ``state.json``'s mtime advances.  No fetcher, no NWP HTTP clients,
    no nowcast computation — just rendering.
    """
    if not settings.cache_dir:
        raise RuntimeError(
            "LIBREWXR_RENDER_ONLY=1 requires LIBREWXR_CACHE_DIR to be set "
            "(it's the shared volume the pipeline writes state.json into)."
        )
    from pathlib import Path
    cache_dir = Path(settings.cache_dir)

    await _wait_for_state(cache_dir, settings.state_wait_timeout)

    # Empty stores; __setstate__ wires up cache_dir and reopens memmaps.
    # We construct the same superset the pipeline can dump (via the
    # provider walk) so apply_state picks up whichever entries are
    # present in the snapshot.  Sources disabled by config produce no
    # contribution and so won't be loaded even if the snapshot includes
    # them — keep settings in sync between pipeline and render workers.
    store = FrameStore(max_frames=settings.max_frames, cache_dir=cache_dir)
    cache = TileCache(max_mb=settings.tile_cache_mb)
    nwp_contribs = collect_nwp_contributions(settings, cache_dir)
    nwp_grids_by_slug: dict[str, object] = {
        nwp_grid_slug(c): c.instance for c in nwp_contribs
    }
    satellite_contribs = collect_satellite_contributions(settings, cache_dir)
    satellite_grids_by_slug: dict[str, object] = {
        satellite_source_slug(c): c.instance for c in satellite_contribs
    }
    nowcast_store = NowcastStore(cache_dir=cache_dir) if settings.nowcast_enabled else None
    alerts_store = AlertsStore() if settings.alerts_enabled else None

    stores: dict[str, object | None] = {
        "frame_store": store,
        **nwp_grids_by_slug,
        **satellite_grids_by_slug,
        "nowcast_store": nowcast_store,
        "alerts_store": alerts_store,
    }

    payload = load_state(cache_dir)
    if payload is None:
        raise RuntimeError(
            f"state.json disappeared between mtime check and load — "
            f"is something else writing to {cache_dir}?"
        )
    refreshed = apply_state(payload, stores)
    logger.info(
        "Render-only worker loaded snapshot: %s",
        ", ".join(refreshed) if refreshed else "(empty)",
    )

    # Stores that didn't appear in the snapshot are useless (e.g. ICON-EU
    # in a CONUS-only deployment) — drop the references so the NWP chain
    # and routes don't dispatch to empty grids.
    for name in list(stores.keys()):
        if name not in refreshed and name != "frame_store":
            stores[name] = None
    # Rebuild the slug → grid dict from the (post-drop) stores so the
    # routes / chain only see grids that actually loaded from disk.
    nwp_grids_by_slug = {
        slug: stores[slug] for slug in nwp_grids_by_slug if stores[slug] is not None
    }
    satellite_grids_by_slug = {
        slug: stores[slug]
        for slug in satellite_grids_by_slug
        if stores[slug] is not None
    }
    ecmwf_grid = nwp_grids_by_slug.get("ecmwf_grid")
    nowcast_store = stores["nowcast_store"]
    alerts_store = stores["alerts_store"]

    enabled = settings.get_enabled_regions()
    station_map, range_overrides, coverage_polygons = collect_radar_coverage_metadata(settings)
    build_coverage_masks(
        station_map,
        range_overrides=range_overrides,
        coverage_polygons=coverage_polygons,
    )
    build_feather_masks()

    # Chain order mirrors ``collect_nwp_contributions`` (sorted by
    # priority) — only include grids that survived the snapshot drop
    # above.  Built from the sorted ``nwp_contribs`` rather than the
    # dict so chain order stays stable independent of dict iteration.
    chain_sources = [
        c.instance for c in nwp_contribs
        if nwp_grid_slug(c) in nwp_grids_by_slug
    ]
    nwp_chain = NWPChain(chain_sources)
    logger.info(
        "Render-only NWP chain: [%s]",
        ", ".join(s.name for s in nwp_chain.sources),
    )

    pool_size = settings.warmer_threads or max((os.cpu_count() or 4) - 1, 1)
    request_executor = ThreadPoolExecutor(max_workers=pool_size)
    asyncio.get_running_loop().set_default_executor(request_executor)

    mem_limit = detect_memory_limit_mb(settings.memory_limit_mb)
    monitor = MemoryMonitor(
        tile_cache=cache,
        coord_cache_clear_fn=_clear_coord_caches,
        memory_limit_mb=mem_limit,
        check_interval=settings.memory_pressure_check_interval,
    )

    tile_request_tracker = (
        TileRequestTracker(
            min_zoom=settings.tile_tracking_min_zoom,
            max_entries=settings.tile_tracking_max_entries,
        )
        if settings.tile_tracking_enabled
        else None
    )

    routes.frame_store = store
    routes.tile_cache = cache
    routes.nwp_grids = nwp_grids_by_slug
    routes.ecmwf_grid = ecmwf_grid
    routes.nwp_chain = nwp_chain
    routes.satellite_grids = satellite_grids_by_slug
    routes.tile_warmer = None
    routes.nowcast_store = nowcast_store
    routes.tile_request_tracker = tile_request_tracker
    routes.start_time = time.time()
    routes.enabled_regions = enabled
    routes.radar_cache = None
    routes.radar_fetcher = None
    # Alerts ride the master_state snapshot — pipeline owns the WMO ingest,
    # render workers just read alerts_store via apply_state.  alerts_fetcher
    # stays None here (no duplicate fetching), and alerts_enabled tracks
    # whether the snapshot actually included an alerts_store entry.
    routes.alerts_store = alerts_store
    routes.alerts_fetcher = None
    routes.alerts_enabled = alerts_store is not None

    last_mtime = state_mtime(cache_dir)
    poller_stop = asyncio.Event()

    async def _poll_state() -> None:
        nonlocal last_mtime
        while not poller_stop.is_set():
            try:
                await asyncio.wait_for(
                    poller_stop.wait(), timeout=settings.state_poll_interval,
                )
                return
            except asyncio.TimeoutError:
                pass
            mtime = state_mtime(cache_dir)
            if mtime is None or mtime == last_mtime:
                continue
            try:
                payload = load_state(cache_dir)
                if payload is None:
                    continue
                refreshed = apply_state(payload, stores)
                last_mtime = mtime
                logger.debug(
                    "Render worker refreshed: %s", ", ".join(refreshed),
                )
                # Newly-loaded frames may have invalidated cached tiles —
                # safest to clear, the worker will repopulate on demand.
                cache.clear()
            except Exception:
                logger.exception("Failed to refresh state from %s", cache_dir)

    poller_task = asyncio.create_task(_poll_state())
    await monitor.start()
    logger.info(
        "Render-only worker ready (cache_dir=%s, regions=%s, tile_cache=%d MB)",
        cache_dir, ", ".join(enabled), settings.tile_cache_mb,
    )

    try:
        yield
    finally:
        poller_stop.set()
        try:
            await poller_task
        except Exception:
            logger.exception("Poller shutdown error")
        await monitor.stop()
        request_executor.shutdown(wait=False)
        cache.clear()
        store.cleanup()
        if nowcast_store is not None:
            nowcast_store.cleanup()
        logger.info("Render-only worker shutdown complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.render_only:
        async with _render_only_lifespan(app):
            yield
        return

    store = FrameStore(max_frames=settings.max_frames)
    cache = TileCache(max_mb=settings.tile_cache_mb)
    from pathlib import Path
    nwp_cache_dir = Path(settings.cache_dir) if settings.cache_dir else None
    # Walk the auto-discovered NWP providers under ``librewxr.sources``;
    # each returns a contribution (or ``None`` when its config flag is
    # off).  Chain order is set by ``NWPContribution.priority``: HRRR
    # (10) → HRRR-Alaska (11) → HRDPS (20) → AROME Antilles (25) → DMI
    # DINI (30) → ICON-EU (35) → WRF-SMN (40) → IFS (1000 — catch-all).
    nwp_contribs = collect_nwp_contributions(settings, nwp_cache_dir)
    nwp_grids_by_slug: dict[str, object] = {
        nwp_grid_slug(c): c.instance for c in nwp_contribs
    }
    # IFS is still special-cased by the radar tile arrow path and the
    # tile warmer; pull it out by slug for those consumers.  Everything
    # else flows through ``nwp_chain`` / ``nwp_grids_by_slug``.
    ecmwf_grid = nwp_grids_by_slug.get("ecmwf_grid")
    nwp_chain = NWPChain([c.instance for c in nwp_contribs])
    logger.info("NWP chain: [%s]", ", ".join(s.name for s in nwp_chain.sources))
    satellite_contribs = collect_satellite_contributions(settings, nwp_cache_dir)
    satellite_grids_by_slug: dict[str, object] = {
        satellite_source_slug(c): c.instance for c in satellite_contribs
    }
    if satellite_contribs:
        logger.info(
            "Satellite chain: [%s]",
            ", ".join(c.name for c in satellite_contribs),
        )
    enabled = settings.get_enabled_regions()

    # Precompute radar station coverage masks used by the ECMWF fallback
    # to distinguish "outside radar range" from "clear sky within range".
    # Each radar provider contributes its own per-region station list +
    # range override; the registry walk merges them based on the active
    # settings (e.g. NA source = MRMS pulls in NEXRAD + Canadian; NA
    # source = IEM pulls NEXRAD only).
    station_map, range_overrides, coverage_polygons = collect_radar_coverage_metadata(settings)
    build_coverage_masks(
        station_map,
        range_overrides=range_overrides,
        coverage_polygons=coverage_polygons,
    )
    build_feather_masks()

    # Nowcast store and generator
    nowcast_store = None
    nowcast_generator = None
    if settings.nowcast_enabled:
        nowcast_store = NowcastStore()
        nowcast_contribs = collect_nowcast_contributions(settings)
        nowcast_generator = NowcastGenerator(
            store, nowcast_store, cache=cache,
            nowcast_contributions=nowcast_contribs,
        )
        external_names = [c.region_name for c in nowcast_contribs]
        if external_names:
            logger.info(
                "Nowcast enabled: %d frames (external sources: %s)",
                settings.nowcast_frames, ", ".join(external_names),
            )
        else:
            logger.info("Nowcast enabled: %d frames", settings.nowcast_frames)

    # Separate thread pools for direct requests and background warming.
    # Direct requests get their own pool so they are never queued behind
    # warming tasks.  The warmer gets an equal-sized pool so it can use
    # all cores when no requests are active.  Brief over-subscription
    # when both are active is handled well by the OS scheduler.
    pool_size = settings.warmer_threads or max((os.cpu_count() or 4) - 1, 1)
    request_executor = ThreadPoolExecutor(max_workers=pool_size)
    warmer_executor = ThreadPoolExecutor(max_workers=pool_size)
    asyncio.get_running_loop().set_default_executor(request_executor)

    warmer = TileWarmer(
        store, cache,
        executor=warmer_executor,
        enabled_regions=enabled,
        nowcast_store=nowcast_store,
        ecmwf_grid=ecmwf_grid,
        nwp_chain=nwp_chain,
    )

    # Memory pressure monitor
    mem_limit = detect_memory_limit_mb(settings.memory_limit_mb)
    monitor = MemoryMonitor(
        tile_cache=cache,
        coord_cache_clear_fn=_clear_coord_caches,
        memory_limit_mb=mem_limit,
        check_interval=settings.memory_pressure_check_interval,
    )

    tile_request_tracker = (
        TileRequestTracker(
            min_zoom=settings.tile_tracking_min_zoom,
            max_entries=settings.tile_tracking_max_entries,
        )
        if settings.tile_tracking_enabled
        else None
    )

    # --- WMO Alerts subsystem ---
    alerts_store = None
    alerts_fetcher = None
    if settings.alerts_enabled:
        alerts_cache = Path(settings.cache_dir) if settings.cache_dir else None
        if alerts_cache is None and settings.alerts_cache_dir:
            alerts_cache = Path(settings.alerts_cache_dir)

        alerts_store = AlertsStore()
        alerts_fetcher = WMOAlertsFetcher(
            store=alerts_store,
            cache_dir=str(alerts_cache) if alerts_cache else None,
            interval=settings.alerts_fetch_interval,
            concurrency=settings.alerts_concurrency,
        )
        routes.alerts_store = alerts_store
        routes.alerts_fetcher = alerts_fetcher
        routes.alerts_enabled = True
        await alerts_fetcher.start()
        logger.info(
            "Alerts: WMO ingest started (interval=%ds)",
            settings.alerts_fetch_interval,
        )
    else:
        routes.alerts_enabled = False
        logger.info("Alerts: disabled (LIBREWXR_ALERTS_ENABLED=false)")

    # Wire up the shared state
    routes.frame_store = store
    routes.tile_cache = cache
    routes.nwp_grids = nwp_grids_by_slug
    routes.ecmwf_grid = ecmwf_grid
    routes.nwp_chain = nwp_chain
    routes.satellite_grids = satellite_grids_by_slug
    routes.tile_warmer = warmer
    routes.nowcast_store = nowcast_store
    routes.tile_request_tracker = tile_request_tracker
    routes.start_time = time.time()
    routes.enabled_regions = enabled

    radar_cache = None
    if settings.cache_dir:
        from pathlib import Path

        from librewxr.data.radar_cache import RadarFrameCache
        from librewxr.data.regions import REGIONS

        radar_cache = RadarFrameCache(Path(settings.cache_dir))
        regions_by_name = {name: REGIONS[name] for name in enabled}
        restored = radar_cache.load_frames(regions_by_name)
        if restored:
            for frame in restored:
                await store.add_frame(frame)
            logger.info(
                "Restored %d radar frame(s) from disk cache (%d → %d)",
                len(restored),
                restored[0].timestamp,
                restored[-1].timestamp,
            )

    fetcher = RadarFetcher(
        store, cache,
        nwp_contributions=nwp_contribs,
        satellite_contributions=satellite_contribs,
        nowcast_generator=nowcast_generator,
        warmer=warmer,
        radar_cache=radar_cache,
    )
    routes.radar_cache = radar_cache
    routes.radar_fetcher = fetcher
    logger.info(
        "Starting LibreWXR (public_url=%s, max_zoom=%d, regions=%s, "
        "tile_cache=%d MB, memory_limit=%d MB, nowcast=%s, "
        "alerts=%s, cache_dir=%s)",
        settings.public_url,
        settings.max_zoom,
        ", ".join(enabled),
        settings.tile_cache_mb,
        mem_limit,
        f"{settings.nowcast_frames} frames" if settings.nowcast_enabled else "off",
        "enabled" if settings.alerts_enabled else "off",
        settings.cache_dir or "(none)",
    )
    await fetcher.start()
    await monitor.start()

    # Pre-warm coordinate caches so the first tile requests at each zoom
    # don't pay the cost of trigonometric projections and array allocations.
    if settings.warm_coord_zoom > 0:
        start = time.time()
        loop = asyncio.get_running_loop()
        warmed = await loop.run_in_executor(
            warmer_executor,
            warm_coordinate_caches,
            enabled,
            settings.warm_coord_zoom,
        )
        logger.info(
            "Coordinate caches warmed: %d entries up to zoom %d (%.2fs)",
            warmed, settings.warm_coord_zoom, time.time() - start,
        )

    yield

    await monitor.stop()
    await fetcher.stop()
    if alerts_fetcher is not None:
        await alerts_fetcher.close()
    warmer.shutdown()
    warmer_executor.shutdown(wait=False)
    request_executor.shutdown(wait=False)
    if nowcast_store is not None:
        nowcast_store.cleanup()
    cache.clear()
    store.cleanup()
    logger.info("LibreWXR shutdown complete")


app = FastAPI(title="LibreWXR", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(routes.router)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Log requests to non-existent endpoints (path only, no client info)."""
    if exc.status_code == 404 and exc.detail == "Not Found":
        logger.warning("404 Not Found: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


def main():
    import uvicorn
    # Optional direct TLS: only enabled when both cert and key are set.
    # Otherwise serve plain HTTP (TLS handled by a reverse proxy / tunnel).
    ssl_kwargs = {}
    if settings.ssl_certfile and settings.ssl_keyfile:
        ssl_kwargs = {
            "ssl_certfile": settings.ssl_certfile,
            "ssl_keyfile": settings.ssl_keyfile,
        }
    uvicorn.run(
        "librewxr.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info",
        access_log=False,
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
