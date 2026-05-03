# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from librewxr.data.store import FrameStore
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import overlapping_regions
from librewxr.tiles.renderer import render_tile

logger = logging.getLogger(__name__)


class TileWarmer:
    """Pre-renders tiles for other timestamps when a cache miss occurs."""

    def __init__(
        self,
        store: FrameStore,
        cache: TileCache,
        executor: ThreadPoolExecutor,
        enabled_regions: list[str] | None = None,
        nowcast_store=None,
    ):
        self._store = store
        self._cache = cache
        self._executor = executor
        self._pending: dict[tuple, float] = {}
        self._lock = asyncio.Lock()
        self._pending_ttl = 300.0  # seconds before a pending entry expires
        self._enabled_regions = enabled_regions
        self._nowcast_store = nowcast_store

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
        ecmwf_grid=None,
        nwp_chain=None,
        frame_timestamp: int | None = None,
    ) -> None:
        """Schedule background renders for all other timestamps."""
        # Collect timestamps from both radar and nowcast stores
        timestamps = await self._store.get_timestamps()
        nowcast_timestamps: set[int] = set()
        if self._nowcast_store is not None:
            nc_ts = await self._nowcast_store.get_timestamps()
            nowcast_timestamps = set(nc_ts)
            timestamps = list(set(timestamps) | nowcast_timestamps)
            timestamps.sort(reverse=True)

        loop = asyncio.get_running_loop()

        for ts in timestamps:
            if ts == triggered_timestamp:
                continue

            cache_key = (ts, z, x, y, tile_size, color, smooth, snow, ext, "")

            if self._cache.get(cache_key) is not None:
                continue

            async with self._lock:
                if cache_key in self._pending:
                    if time.monotonic() - self._pending[cache_key] < self._pending_ttl:
                        continue
                self._pending[cache_key] = time.monotonic()

            # Try radar store first, then nowcast store
            nowcast_blend = None
            frame = await self._store.get_frame(ts)
            if frame is None and self._nowcast_store is not None:
                nc_frame, nowcast_blend = await self._nowcast_store.get_frame(ts)
                if nc_frame is not None:
                    frame = nc_frame
            if frame is None:
                async with self._lock:
                    self._pending.pop(cache_key, None)
                continue

            frame_regions = frame.regions
            self._submit_render(
                loop, cache_key, frame_regions,
                z, x, y, tile_size, color, smooth, snow, ext,
                ecmwf_grid, nwp_chain, ts, nowcast_blend,
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
        ecmwf_grid,
        nwp_chain,
        frame_timestamp: int | None = None,
        nowcast_blend: float | None = None,
    ) -> None:
        """Render a tile and store it in the cache (runs in thread pool)."""
        try:
            # A direct request may have cached this tile while we were queued
            if self._cache.get(cache_key) is not None:
                return
            tile_bytes = render_tile(
                frame_regions=frame_regions,
                z=z, x=x, y=y,
                tile_size=tile_size,
                color_scheme=color,
                smooth=smooth,
                snow=snow,
                fmt=ext,
                ecmwf_grid=ecmwf_grid,
                nwp_chain=nwp_chain,
                enabled_regions=self._enabled_regions,
                frame_timestamp=frame_timestamp,
                nowcast_blend=nowcast_blend,
                arrow_style="",
            )
            self._cache.put(cache_key, tile_bytes)
        except Exception:
            logger.debug("Warm render failed for key %s", cache_key[:5])
        finally:
            self._pending.pop(cache_key, None)

    async def warm_overview(
        self,
        ecmwf_grid=None,
        nwp_chain=None,
        max_zoom: int = 4,
        max_zoom_regional: int = -1,
        tile_size: int = 256,
        color: int = 7,
        smooth: bool = False,
        snow: bool = False,
        ext: str = "png",
    ) -> None:
        """Pre-render overview tiles for every timestamp.

        Two passes:
        - Zooms 0..``max_zoom`` render every tile (global view).
        - Zooms ``max_zoom+1``..``max_zoom_regional`` render only tiles
          whose Web Mercator bbox overlaps an enabled region's bbox,
          dropping ocean/desert tiles that no one would zoom into.

        ``max_zoom_regional`` <= ``max_zoom`` disables the regional pass.
        """
        timestamps = await self._store.get_timestamps()
        if self._nowcast_store is not None:
            nc_ts = await self._nowcast_store.get_timestamps()
            timestamps = list(set(timestamps) | set(nc_ts))
            timestamps.sort(reverse=True)

        max_zoom_total = max(max_zoom, max_zoom_regional)
        if max_zoom_total < 0 or not timestamps:
            return

        # Precompute the tile coordinate list for each zoom once — overlap
        # filtering is timestamp-independent, so doing this per timestamp
        # would waste tens of thousands of bbox checks.
        tiles_by_zoom: dict[int, list[tuple[int, int]]] = {}
        for z in range(max_zoom_total + 1):
            n = 2**z
            if z <= max_zoom:
                tiles_by_zoom[z] = [(x, y) for y in range(n) for x in range(n)]
            else:
                tiles_by_zoom[z] = [
                    (x, y) for y in range(n) for x in range(n)
                    if overlapping_regions(z, x, y, self._enabled_regions)
                ]

        loop = asyncio.get_running_loop()
        for ts in timestamps:
            # Try radar store first, then nowcast store
            nowcast_blend = None
            frame = await self._store.get_frame(ts)
            if frame is None and self._nowcast_store is not None:
                nc_frame, nowcast_blend = await self._nowcast_store.get_frame(ts)
                if nc_frame is not None:
                    frame = nc_frame
            if frame is None:
                continue
            frame_regions = frame.regions

            submitted = 0
            for z in range(max_zoom_total + 1):
                for x, y in tiles_by_zoom[z]:
                    submitted += 1
                    if submitted % 500 == 0:
                        await asyncio.sleep(0)
                    cache_key = (ts, z, x, y, tile_size, color, smooth, snow, ext, "")
                    if self._cache.get(cache_key) is not None:
                        continue
                    async with self._lock:
                        if cache_key in self._pending:
                            if time.monotonic() - self._pending[cache_key] < self._pending_ttl:
                                continue
                        self._pending[cache_key] = time.monotonic()
                    self._submit_render(
                        loop, cache_key, frame_regions,
                        z, x, y, tile_size, color, smooth, snow, ext,
                        ecmwf_grid, nwp_chain, ts, nowcast_blend,
                    )

    def _submit_render(
        self,
        loop: asyncio.AbstractEventLoop,
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
        ecmwf_grid,
        nwp_chain,
        ts: int,
        nowcast_blend: float | None,
    ) -> None:
        """Schedule a render on the executor with exception logging.

        ``_render_and_cache`` catches its own exceptions, but anything that
        escapes (or a scheduling failure on the executor itself) would
        otherwise vanish into a discarded Future.
        """
        future = loop.run_in_executor(
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
            ecmwf_grid,
            nwp_chain,
            ts,
            nowcast_blend,
        )
        future.add_done_callback(self._log_render_exception)

    @staticmethod
    def _log_render_exception(future) -> None:
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            logger.warning("Warm render task raised: %r", exc)

    def shutdown(self) -> None:
        pass  # Executor is shared; lifecycle managed by main.py
