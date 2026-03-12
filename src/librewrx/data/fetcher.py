# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import logging
import time
from datetime import datetime, timezone

import numpy as np

from librewrx.config import settings
from librewrx.data.gfs_reflectivity import GFSReflectivityGrid
from librewrx.data.regions import REGIONS, RegionDef
from librewrx.data.sources import DWDSource, IEMSource, METNordicSource
from librewrx.data.store import FrameStore, RadarFrame
from librewrx.data.temperature import TemperatureGrid
from librewrx.tiles.cache import TileCache

logger = logging.getLogger(__name__)


class RadarFetcher:
    """Background task that periodically fetches radar frames."""

    def __init__(
        self,
        store: FrameStore,
        cache: TileCache,
        temperature_grid: TemperatureGrid | None = None,
        reflectivity_grid: GFSReflectivityGrid | None = None,
    ):
        self._store = store
        self._cache = cache
        self._temp_grid = temperature_grid
        self._refl_grid = reflectivity_grid
        self._task: asyncio.Task | None = None
        self._enabled_regions = [
            REGIONS[name] for name in settings.get_enabled_regions()
        ]

        # Build a source for each enabled region based on its group
        self._sources: dict[str, IEMSource | METNordicSource | DWDSource] = {}
        iem_source: IEMSource | None = None
        nordic_source: METNordicSource | None = None
        dwd_source: DWDSource | None = None
        for region in self._enabled_regions:
            if region.group == "NORDIC":
                if nordic_source is None:
                    nordic_source = METNordicSource(settings.met_nordic_base_url)
                self._sources[region.name] = nordic_source
            elif region.group == "GERMANY":
                if dwd_source is None:
                    dwd_source = DWDSource(settings.dwd_base_url)
                self._sources[region.name] = dwd_source
            else:
                if iem_source is None:
                    iem_source = IEMSource(settings.iem_base_url)
                self._sources[region.name] = iem_source

    async def start(self) -> None:
        """Start the background fetch loop.

        Fetches auxiliary grids and the latest radar frame immediately so
        the server can start serving tiles within seconds.  Historical
        frames are backfilled in a background task.
        """
        region_names = [r.name for r in self._enabled_regions]
        logger.info("Fetching regions: %s", ", ".join(region_names))
        await self._fetch_initial()
        self._task = asyncio.create_task(self._backfill_then_loop())
        logger.info("Radar fetcher started (backfill running in background)")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Close all unique sources
        closed: set[int] = set()
        for source in self._sources.values():
            if id(source) not in closed:
                await source.close()
                closed.add(id(source))
        if self._temp_grid:
            await self._temp_grid.close()
        if self._refl_grid:
            await self._refl_grid.close()
        logger.info("Radar fetcher stopped")

    async def _backfill_then_loop(self) -> None:
        """Backfill historical frames, then enter the regular refresh loop."""
        try:
            await self._fetch_all_frames()
        except Exception:
            logger.exception("Error in initial backfill")

        while True:
            await asyncio.sleep(settings.fetch_interval)
            try:
                await self._fetch_all_frames()
            except Exception:
                logger.exception("Error in fetch loop")

    async def _fetch_initial(self) -> None:
        """Quick startup: fetch auxiliary grids and latest radar frame only."""
        await self._fetch_auxiliary_grids()

        now_rounded = int(time.time() // 300) * 300
        await self._fetch_timestamps([(now_rounded, "live", 0)])

    async def _fetch_auxiliary_grids(self) -> None:
        """Fetch temperature and GFS reflectivity grids."""
        if self._temp_grid is not None:
            try:
                await self._temp_grid.fetch()
            except Exception:
                logger.warning("Temperature fetch failed, snow rendering may be stale")

        if self._refl_grid is not None:
            try:
                await self._refl_grid.fetch()
            except Exception:
                logger.warning("GFS reflectivity fetch failed, global fallback may be stale")

    async def _fetch_all_frames(self) -> None:
        """Fetch frames for all enabled regions to fill the store."""
        await self._fetch_auxiliary_grids()

        now = time.time()
        now_rounded = int(now // 300) * 300

        # Build timestamp list: live frames (10-min spacing) + archive frames
        live_indices = list(range(0, 12, 2))
        ts_and_sources: list[tuple[int, str, int | datetime]] = []

        for idx in live_indices:
            minutes_ago = idx * 5
            ts = now_rounded - minutes_ago * 60
            ts_and_sources.append((ts, "live", minutes_ago))

        remaining = settings.max_frames - len(live_indices)
        for i in range(remaining):
            minutes_ago = 60 + (i + 1) * 10
            ts = now_rounded - minutes_ago * 60
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            ts_and_sources.append((ts, "archive", dt))

        await self._fetch_timestamps(ts_and_sources)

    async def _fetch_timestamps(
        self, ts_and_sources: list[tuple[int, str, int | datetime]]
    ) -> None:
        """Fetch all enabled regions for the given timestamps."""
        # For each timestamp, fetch all enabled regions in parallel
        tasks = []
        task_meta: list[tuple[int, RegionDef]] = []

        for ts, source_type, source_arg in ts_and_sources:
            for region in self._enabled_regions:
                source = self._sources[region.name]
                if source_type == "live":
                    tasks.append(source.fetch_frame(region, source_arg))
                else:
                    tasks.append(source.fetch_archive_frame(region, source_arg))
                task_meta.append((ts, region))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Group results by timestamp and build frames
        frames_by_ts: dict[int, dict[str, np.ndarray]] = {}
        for (ts, region), result in zip(task_meta, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Failed to fetch %s for ts=%d: %s", region.name, ts, result
                )
                continue
            if result is None:
                continue

            if settings.despeckle_min_neighbors > 0:
                result = _despeckle(result, settings.despeckle_min_neighbors)

            if ts not in frames_by_ts:
                frames_by_ts[ts] = {}
            frames_by_ts[ts][region.name] = result

        # Store frames
        added = 0
        for ts, regions_data in frames_by_ts.items():
            frame = RadarFrame(timestamp=ts, regions=regions_data)
            evicted_ts = await self._store.add_frame(frame)
            if evicted_ts is not None:
                self._cache.invalidate_timestamp(evicted_ts)
            added += 1

        count = await self._store.frame_count()
        region_summary = ", ".join(
            f"{r.name}" for r in self._enabled_regions
        )
        logger.info(
            "Fetch complete: %d frames added, %d total in store (%s)",
            added, count, region_summary,
        )


def _despeckle(data: np.ndarray, min_neighbors: int) -> np.ndarray:
    """Remove isolated pixels (ground clutter / AP artifacts).

    Uses padded slicing instead of np.roll for ~2.4x speedup on large
    arrays.  Slicing also avoids the wrap-around artifact that np.roll
    produces at array edges.
    """
    mask = data > 0
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    count = np.zeros((h, w), dtype=np.int8)
    for dr in range(3):
        for dc in range(3):
            if dr == 1 and dc == 1:
                continue
            count += padded[dr:dr + h, dc:dc + w]

    result = data.copy()
    result[mask & (count < min_neighbors)] = 0
    return result
