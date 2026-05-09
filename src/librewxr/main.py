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
from librewxr.data.arome_antilles_grid import AROMEAntillesGrid
from librewxr.data.wrf_smn_grid import WRFSMNGrid
from librewxr.data.cloud_grid import CloudGrid
from librewxr.data.coverage import build_coverage_masks, build_feather_masks
from librewxr.data.dmi_dini_grid import DMIDiniGrid
from librewxr.data.ecmwf_grid import ECMWFGrid
from librewxr.data.fetcher import RadarFetcher
from librewxr.data.hrdps_grid import HRDPSGrid
from librewxr.data.hrrr_alaska_grid import HRRRAlaskaGrid
from librewxr.data.hrrr_grid import HRRRGrid
from librewxr.data.icon_eu_grid import ICONEUGrid
from librewxr.data.master_state import apply_state, load_state, state_mtime
from librewxr.data.nowcast import NowcastGenerator, NowcastStore
from librewxr.data.nwp_source import NWPChain
from librewxr.data.radar_stations import MRMS_STATIONS
from librewxr.data.store import FrameStore
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
# (radar / IFS / cloud all firing in parallel) reads cleanly in the log.
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
    "librewxr.data.ecmwf_grid": "ifs",
    "librewxr.data.ecmwf_interpolation": "ifs",
    "librewxr.data.hrrr_grid": "hrrr",
    "librewxr.data.hrrr_alaska_grid": "hrrr-ak",
    "librewxr.data.icon_eu_grid": "icon-eu",
    "librewxr.data.dmi_dini_grid": "dmi-dini",
    "librewxr.data.hrdps_grid": "hrdps",
    "librewxr.data.arome_antilles_grid": "arome-ant",
    "librewxr.data.wrf_smn_grid": "wrf-smn",
    "librewxr.data.cloud_grid": "cloud",
    "librewxr.data.cloud_cache": "cloud",
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
    """
    deadline = time.time() + timeout if timeout > 0 else None
    poll = max(settings.state_poll_interval, 0.25)
    while True:
        if state_mtime(cache_dir) is not None:
            return
        if deadline is not None and time.time() > deadline:
            raise RuntimeError(
                f"Timed out after {timeout:.0f}s waiting for state.json "
                f"under {cache_dir}.  Is the data pipeline running?"
            )
        logger.info("Waiting for pipeline state.json under %s …", cache_dir)
        await asyncio.sleep(poll)


@asynccontextmanager
async def _render_only_lifespan(app: FastAPI):
    """Lifespan for tile-server workers in the multi-worker split.

    Pulls all radar / NWP / cloud data from the snapshot the data
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
    # We construct the same superset the pipeline can dump so apply_state
    # picks up whichever entries are present in the snapshot.
    store = FrameStore(max_frames=settings.max_frames, cache_dir=cache_dir)
    cache = TileCache(max_mb=settings.tile_cache_mb)
    ecmwf_grid = ECMWFGrid(cache_dir=cache_dir)
    hrrr_grid = HRRRGrid(cache_dir=cache_dir)
    hrrr_alaska_grid = HRRRAlaskaGrid(cache_dir=cache_dir)
    hrdps_grid = HRDPSGrid(cache_dir=cache_dir)
    arome_antilles_grid = AROMEAntillesGrid(cache_dir=cache_dir)
    wrf_smn_grid = WRFSMNGrid(cache_dir=cache_dir)
    icon_eu_grid = ICONEUGrid(cache_dir=cache_dir)
    dmi_dini_grid = DMIDiniGrid(cache_dir=cache_dir)
    cloud_grid = CloudGrid(cache_dir=cache_dir) if settings.satellite_enabled else None
    nowcast_store = NowcastStore(cache_dir=cache_dir) if settings.nowcast_enabled else None

    stores = {
        "frame_store": store,
        "ecmwf_grid": ecmwf_grid,
        "hrrr_grid": hrrr_grid,
        "hrrr_alaska_grid": hrrr_alaska_grid,
        "hrdps_grid": hrdps_grid,
        "arome_antilles_grid": arome_antilles_grid,
        "wrf_smn_grid": wrf_smn_grid,
        "icon_eu_grid": icon_eu_grid,
        "dmi_dini_grid": dmi_dini_grid,
        "cloud_grid": cloud_grid,
        "nowcast_store": nowcast_store,
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
    ecmwf_grid = stores["ecmwf_grid"]
    hrrr_grid = stores["hrrr_grid"]
    hrrr_alaska_grid = stores["hrrr_alaska_grid"]
    hrdps_grid = stores["hrdps_grid"]
    arome_antilles_grid = stores["arome_antilles_grid"]
    wrf_smn_grid = stores["wrf_smn_grid"]
    icon_eu_grid = stores["icon_eu_grid"]
    dmi_dini_grid = stores["dmi_dini_grid"]
    cloud_grid = stores["cloud_grid"]
    nowcast_store = stores["nowcast_store"]

    enabled = settings.get_enabled_regions()
    coverage_overrides = (
        MRMS_STATIONS if settings.na_source in ("mrms", "mrms_fallback") else None
    )
    build_coverage_masks(station_overrides=coverage_overrides)
    build_feather_masks()

    chain_sources = []
    for grid in (
        hrrr_grid, hrrr_alaska_grid, hrdps_grid,
        arome_antilles_grid, dmi_dini_grid, icon_eu_grid,
        wrf_smn_grid, ecmwf_grid,
    ):
        if grid is not None:
            chain_sources.append(grid)
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
    routes.ecmwf_grid = ecmwf_grid
    routes.hrrr_grid = hrrr_grid
    routes.hrrr_alaska_grid = hrrr_alaska_grid
    routes.hrdps_grid = hrdps_grid
    routes.arome_antilles_grid = arome_antilles_grid
    routes.wrf_smn_grid = wrf_smn_grid
    routes.icon_eu_grid = icon_eu_grid
    routes.dmi_dini_grid = dmi_dini_grid
    routes.nwp_chain = nwp_chain
    routes.cloud_grid = cloud_grid
    routes.tile_warmer = None
    routes.nowcast_store = nowcast_store
    routes.tile_request_tracker = tile_request_tracker
    routes.start_time = time.time()
    routes.enabled_regions = enabled
    routes.radar_cache = None
    routes.radar_fetcher = None
    routes.alerts_enabled = False  # alerts ride the pipeline; render-only is read-only

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
        if cloud_grid is not None:
            await cloud_grid.close()
        logger.info("Render-only worker shutdown complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.render_only:
        async with _render_only_lifespan(app):
            yield
        return

    store = FrameStore(max_frames=settings.max_frames)
    cache = TileCache(max_mb=settings.tile_cache_mb)
    ecmwf_grid = ECMWFGrid() if settings.ecmwf_enabled else None
    from pathlib import Path
    nwp_cache_dir = Path(settings.cache_dir) if settings.cache_dir else None
    # ``na_nwp_source=hrrr`` enables BOTH HRRR-CONUS and HRRR-Alaska.
    # They are the same NCEP model running on disjoint domains and sit
    # on the same anonymous S3 bucket, so a single toggle covers both.
    # The chain order is CONUS → Alaska → … (order between them is
    # irrelevant since their domains don't overlap).
    if settings.na_nwp_source == "hrrr":
        hrrr_grid = HRRRGrid(cache_dir=nwp_cache_dir)
        hrrr_alaska_grid = HRRRAlaskaGrid(cache_dir=nwp_cache_dir)
    else:
        hrrr_grid = None
        hrrr_alaska_grid = None
    # Both DINI and ICON-EU may be active simultaneously: DINI takes
    # precedence over its (smaller, higher-res) domain and ICON-EU fills
    # the broader European coverage outside DINI's footprint.  The
    # profile name controls which subset is instantiated.
    if settings.eu_nwp_profile in ("icon_eu_only", "dini_with_icon_eu"):
        icon_eu_grid = ICONEUGrid(cache_dir=nwp_cache_dir)
    else:
        icon_eu_grid = None
    if settings.eu_nwp_profile == "dini_with_icon_eu":
        dmi_dini_grid = DMIDiniGrid(cache_dir=nwp_cache_dir)
    else:
        dmi_dini_grid = None
    # HRDPS is independent of na_nwp_source: HRRR's CONUS focus and
    # HRDPS's Canadian focus are disjoint enough that running both
    # together is the common case — HRRR wins where it's denser inside
    # CONUS, HRDPS fills Canada / northern fringe / the Atlantic.
    if settings.hrdps_enabled:
        hrdps_grid = HRDPSGrid(cache_dir=nwp_cache_dir)
    else:
        hrdps_grid = None
    # AROME Antilles fills the eastern Caribbean — disjoint from every
    # other regional source in the chain, so position relative to the
    # other regionals doesn't matter.  Sits ahead of IFS to win inside
    # its domain.
    if settings.arome_antilles_enabled:
        arome_antilles_grid = AROMEAntillesGrid(cache_dir=nwp_cache_dir)
    else:
        arome_antilles_grid = None
    # WRF-SMN covers the South American Southern Cone (Argentina + Chile
    # + Uruguay + Paraguay + Bolivia + S. Brazil) — disjoint from every
    # other regional source so position only matters relative to IFS.
    if settings.wrf_smn_enabled:
        wrf_smn_grid = WRFSMNGrid(cache_dir=nwp_cache_dir)
    else:
        wrf_smn_grid = None
    # Chain order = specificity (narrowest domain first), so HRRR fills
    # CONUS, HRDPS fills Canada, AROME Antilles fills the Caribbean,
    # DMI DINI fills NW + central Europe at 2 km, ICON-EU fills the
    # rest of Europe at 7 km, WRF-SMN fills S. America's Southern Cone,
    # IFS catches everything else globally.
    chain_sources = []
    if hrrr_grid:
        chain_sources.append(hrrr_grid)
    if hrrr_alaska_grid:
        chain_sources.append(hrrr_alaska_grid)
    if hrdps_grid:
        chain_sources.append(hrdps_grid)
    if arome_antilles_grid:
        chain_sources.append(arome_antilles_grid)
    if dmi_dini_grid:
        chain_sources.append(dmi_dini_grid)
    if icon_eu_grid:
        chain_sources.append(icon_eu_grid)
    if wrf_smn_grid:
        chain_sources.append(wrf_smn_grid)
    if ecmwf_grid is not None:
        chain_sources.append(ecmwf_grid)
    nwp_chain = NWPChain(chain_sources)
    logger.info("NWP chain: [%s]", ", ".join(s.name for s in nwp_chain.sources))
    cloud = CloudGrid() if settings.satellite_enabled else None
    enabled = settings.get_enabled_regions()

    # Precompute radar station coverage masks used by the ECMWF fallback
    # to distinguish "outside radar range" from "clear sky within range".
    # When MRMS is the NA source, use combined NEXRAD+Canada stations since
    # MRMS ingests both networks — this gives correct coverage for USCOMP
    # and CACOMP.  When IEM is the source, use the default per-region
    # stations (NEXRAD-only for USCOMP, Canada-only for CACOMP).
    coverage_overrides = None
    if settings.na_source in ("mrms", "mrms_fallback"):
        coverage_overrides = MRMS_STATIONS
    build_coverage_masks(station_overrides=coverage_overrides)
    build_feather_masks()

    # Nowcast store and generator
    nowcast_store = None
    nowcast_generator = None
    if settings.nowcast_enabled:
        nowcast_store = NowcastStore()
        nowcast_generator = NowcastGenerator(store, nowcast_store, cache=cache)
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
    routes.ecmwf_grid = ecmwf_grid
    routes.hrrr_grid = hrrr_grid
    routes.hrrr_alaska_grid = hrrr_alaska_grid
    routes.hrdps_grid = hrdps_grid
    routes.arome_antilles_grid = arome_antilles_grid
    routes.wrf_smn_grid = wrf_smn_grid
    routes.icon_eu_grid = icon_eu_grid
    routes.dmi_dini_grid = dmi_dini_grid
    routes.nwp_chain = nwp_chain
    routes.cloud_grid = cloud
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
        ecmwf_grid=ecmwf_grid,
        hrrr_grid=hrrr_grid,
        hrrr_alaska_grid=hrrr_alaska_grid,
        hrdps_grid=hrdps_grid,
        arome_antilles_grid=arome_antilles_grid,
        wrf_smn_grid=wrf_smn_grid,
        icon_eu_grid=icon_eu_grid,
        dmi_dini_grid=dmi_dini_grid,
        cloud_grid=cloud,
        nowcast_generator=nowcast_generator,
        warmer=warmer,
        radar_cache=radar_cache,
    )
    routes.radar_cache = radar_cache
    routes.radar_fetcher = fetcher
    logger.info(
        "Starting LibreWXR (public_url=%s, max_zoom=%d, regions=%s, "
        "tile_cache=%d MB, memory_limit=%d MB, nowcast=%s, satellite=%s, "
        "alerts=%s, cache_dir=%s)",
        settings.public_url,
        settings.max_zoom,
        ", ".join(enabled),
        settings.tile_cache_mb,
        mem_limit,
        f"{settings.nowcast_frames} frames" if settings.nowcast_enabled else "off",
        f"{settings.satellite_max_frames} frames" if settings.satellite_enabled else "off",
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
    if cloud is not None:
        await cloud.close()
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
    uvicorn.run(
        "librewxr.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
