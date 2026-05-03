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
from librewxr.data.cloud_grid import CloudGrid
from librewxr.data.coverage import build_coverage_masks, build_feather_masks
from librewxr.data.ecmwf_grid import ECMWFGrid
from librewxr.data.fetcher import RadarFetcher
from librewxr.data.hrrr_grid import HRRRGrid
from librewxr.data.icon_eu_grid import ICONEUGrid
from librewxr.data.nowcast import NowcastGenerator, NowcastStore
from librewxr.data.nwp_source import NWPChain
from librewxr.data.radar_stations import MRMS_STATIONS
from librewxr.data.store import FrameStore
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
    "librewxr.data.icon_eu_grid": "icon-eu",
    "librewxr.data.cloud_grid": "cloud",
    "librewxr.data.cloud_cache": "cloud",
    "librewxr.data.nowcast": "nowcast",
    "librewxr.tiles.warmer": "warmer",
    "librewxr.tiles.cache": "tiles",
    "librewxr.tiles.renderer": "tiles",
    "librewxr.tiles.satellite_renderer": "tiles",
    "librewxr.tiles.coordinates": "tiles",
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = FrameStore(max_frames=settings.max_frames)
    cache = TileCache(max_mb=settings.tile_cache_mb)
    ecmwf_grid = ECMWFGrid()
    from pathlib import Path
    nwp_cache_dir = Path(settings.cache_dir) if settings.cache_dir else None
    if settings.na_nwp_source == "hrrr":
        hrrr_grid = HRRRGrid(cache_dir=nwp_cache_dir)
    else:
        hrrr_grid = None
    if settings.eu_nwp_source == "icon_eu":
        icon_eu_grid = ICONEUGrid(cache_dir=nwp_cache_dir)
    else:
        icon_eu_grid = None
    # Chain order = specificity (narrowest domain first), so HRRR fills
    # CONUS, ICON-EU fills Europe, IFS catches everything else.
    chain_sources = []
    if hrrr_grid:
        chain_sources.append(hrrr_grid)
    if icon_eu_grid:
        chain_sources.append(icon_eu_grid)
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

    # Wire up the shared state
    routes.frame_store = store
    routes.tile_cache = cache
    routes.ecmwf_grid = ecmwf_grid
    routes.hrrr_grid = hrrr_grid
    routes.icon_eu_grid = icon_eu_grid
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
        icon_eu_grid=icon_eu_grid,
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
        "cache_dir=%s)",
        settings.public_url,
        settings.max_zoom,
        ", ".join(enabled),
        settings.tile_cache_mb,
        mem_limit,
        f"{settings.nowcast_frames} frames" if settings.nowcast_enabled else "off",
        f"{settings.satellite_max_frames} frames" if settings.satellite_enabled else "off",
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

    # Pre-render overview tiles in the background so zoomed-out views are
    # served instantly from cache.  The fetcher re-triggers this after each
    # fetch cycle so newly-arrived frames are warmed automatically.
    if settings.warm_overview_zoom >= 0:
        asyncio.create_task(
            warmer.warm_overview(
                ecmwf_grid=ecmwf_grid,
                nwp_chain=nwp_chain,
                max_zoom=settings.warm_overview_zoom,
                max_zoom_regional=settings.warm_overview_zoom_regional,
            )
        )

    yield

    await monitor.stop()
    await fetcher.stop()
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
