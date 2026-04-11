# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from librewxr.api import routes
from librewxr.config import settings
from librewxr.data.coverage import build_coverage_masks, build_feather_masks
from librewxr.data.ecmwf_grid import ECMWFGrid
from librewxr.data.fetcher import RadarFetcher
from librewxr.data.nowcast import NowcastGenerator, NowcastStore
from librewxr.data.store import FrameStore
from librewxr.memory import MemoryMonitor, detect_memory_limit_mb
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import (
    region_pixel_indices,
    region_pixel_indices_fractional,
    region_pixel_indices_padded,
    tile_pixel_indices,
    tile_pixel_indices_fractional,
    tile_pixel_indices_padded,
    tile_pixel_latlons,
    tile_pixel_latlons_padded,
)
from librewxr.tiles.warmer import TileWarmer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _clear_coord_caches() -> None:
    """Clear all coordinate LRU caches."""
    region_pixel_indices.cache_clear()
    region_pixel_indices_padded.cache_clear()
    region_pixel_indices_fractional.cache_clear()
    tile_pixel_latlons.cache_clear()
    tile_pixel_latlons_padded.cache_clear()
    tile_pixel_indices.cache_clear()
    tile_pixel_indices_padded.cache_clear()
    tile_pixel_indices_fractional.cache_clear()
    logger.info("Coordinate caches cleared by memory monitor")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = FrameStore(max_frames=settings.max_frames)
    cache = TileCache(max_mb=settings.tile_cache_mb)
    ecmwf_grid = ECMWFGrid()
    enabled = settings.get_enabled_regions()

    # Precompute radar station coverage masks used by the ECMWF fallback
    # to distinguish "outside radar range" from "clear sky within range".
    build_coverage_masks()
    build_feather_masks()

    # Nowcast store and generator
    nowcast_store = None
    nowcast_generator = None
    if settings.nowcast_enabled:
        nowcast_store = NowcastStore()
        nowcast_generator = NowcastGenerator(store, nowcast_store, cache=cache)
        logger.info("Nowcast enabled: %d frames", settings.nowcast_frames)

    warmer = TileWarmer(
        store, cache,
        max_workers=settings.warmer_threads,
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

    # Wire up the shared state
    routes.frame_store = store
    routes.tile_cache = cache
    routes.ecmwf_grid = ecmwf_grid
    routes.tile_warmer = warmer
    routes.nowcast_store = nowcast_store
    routes.start_time = time.time()
    routes.enabled_regions = enabled

    fetcher = RadarFetcher(
        store, cache,
        ecmwf_grid=ecmwf_grid,
        nowcast_generator=nowcast_generator,
    )
    logger.info(
        "Starting LibreWXR (public_url=%s, max_zoom=%d, regions=%s, "
        "tile_cache=%d MB, memory_limit=%d MB, nowcast=%s)",
        settings.public_url,
        settings.max_zoom,
        ", ".join(enabled),
        settings.tile_cache_mb,
        mem_limit,
        f"{settings.nowcast_frames} frames" if settings.nowcast_enabled else "off",
    )
    await fetcher.start()
    await monitor.start()

    yield

    await monitor.stop()
    await fetcher.stop()
    warmer.shutdown()
    if nowcast_store is not None:
        nowcast_store.clear()
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


def main():
    import uvicorn
    uvicorn.run(
        "librewxr.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
