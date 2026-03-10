# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from librewrx.api import routes
from librewrx.config import settings
from librewrx.data.fetcher import RadarFetcher
from librewrx.data.store import FrameStore
from librewrx.data.temperature import TemperatureGrid
from librewrx.tiles.cache import TileCache
from librewrx.tiles.warmer import TileWarmer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = FrameStore(max_frames=settings.max_frames)
    cache = TileCache(max_size=settings.tile_cache_size)
    temp_grid = TemperatureGrid()
    enabled = settings.get_enabled_regions()

    warmer = TileWarmer(
        store, cache,
        max_workers=settings.warmer_threads,
        enabled_regions=enabled,
    )

    # Wire up the shared state
    routes.frame_store = store
    routes.tile_cache = cache
    routes.temperature_grid = temp_grid
    routes.tile_warmer = warmer
    routes.start_time = time.time()
    routes.enabled_regions = enabled

    fetcher = RadarFetcher(store, cache, temperature_grid=temp_grid)
    logger.info(
        "Starting LibreWRX (public_url=%s, max_zoom=%d, regions=%s)",
        settings.public_url,
        settings.max_zoom,
        ", ".join(enabled),
    )
    await fetcher.start()

    yield

    await fetcher.stop()
    warmer.shutdown()
    cache.clear()
    logger.info("LibreWRX shutdown complete")


app = FastAPI(title="LibreWRX", version="0.1.0", lifespan=lifespan)

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
        "librewrx.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
