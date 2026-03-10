# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from librewrx.data.store import FrameStore
from librewrx.tiles.cache import TileCache
from librewrx.tiles.renderer import render_tile

logger = logging.getLogger(__name__)


class TileWarmer:
    """Pre-renders tiles for other timestamps when a cache miss occurs."""

    def __init__(
        self,
        store: FrameStore,
        cache: TileCache,
        max_workers: int = 4,
        enabled_regions: list[str] | None = None,
    ):
        self._store = store
        self._cache = cache
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: set[tuple] = set()
        self._lock = asyncio.Lock()
        self._enabled_regions = enabled_regions

    async def warm(
        self,
        triggered_timestamp: int,
        z: int,
        x: int,
        y: int,
        tile_size: int,
        color: int,
        smooth: bool,
        snow: bool,
        ext: str,
        temperature_grid=None,
    ) -> None:
        """Schedule background renders for all other timestamps."""
        timestamps = await self._store.get_timestamps()
        loop = asyncio.get_running_loop()

        for ts in timestamps:
            if ts == triggered_timestamp:
                continue

            cache_key = (ts, z, x, y, tile_size, color, smooth, snow, ext)

            if self._cache.get(cache_key) is not None:
                continue

            async with self._lock:
                if cache_key in self._pending:
                    continue
                self._pending.add(cache_key)

            frame = await self._store.get_frame(ts)
            if frame is None:
                async with self._lock:
                    self._pending.discard(cache_key)
                continue

            frame_regions = frame.regions
            loop.run_in_executor(
                self._executor,
                self._render_and_cache,
                cache_key,
                frame_regions,
                z, x, y,
                tile_size,
                color,
                smooth,
                snow,
                ext,
                temperature_grid,
            )

    def _render_and_cache(
        self,
        cache_key: tuple,
        frame_regions: dict,
        z: int,
        x: int,
        y: int,
        tile_size: int,
        color: int,
        smooth: bool,
        snow: bool,
        ext: str,
        temperature_grid,
    ) -> None:
        """Render a tile and store it in the cache (runs in thread pool)."""
        try:
            tile_bytes = render_tile(
                frame_regions=frame_regions,
                z=z, x=x, y=y,
                tile_size=tile_size,
                color_scheme=color,
                smooth=smooth,
                snow=snow,
                fmt=ext,
                temperature_grid=temperature_grid,
                enabled_regions=self._enabled_regions,
            )
            self._cache.put(cache_key, tile_bytes)
        except Exception:
            logger.debug("Warm render failed for key %s", cache_key[:5])
        finally:
            self._pending.discard(cache_key)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
