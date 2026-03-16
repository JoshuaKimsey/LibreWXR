# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Memory pressure monitor — safety net to prevent OOM kills.

Periodically checks process RSS against the container/system memory
limit and proactively evicts caches before the OOM killer intervenes.
"""
import asyncio
import logging
from pathlib import Path

import psutil

from librewxr.tiles.cache import TileCache

logger = logging.getLogger(__name__)

# Eviction thresholds (fraction of memory limit)
_WARN_THRESHOLD = 0.80
_EVICT_TILES_THRESHOLD = 0.85
_EVICT_ALL_THRESHOLD = 0.90


def detect_memory_limit_mb(override_mb: int = 0) -> int:
    """Detect container memory limit in MB.

    Priority: explicit override > cgroup v2 > cgroup v1 > system RAM.
    """
    if override_mb > 0:
        return override_mb

    # cgroup v2
    try:
        cg2 = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if cg2 != "max":
            return int(cg2) // (1024 * 1024)
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    # cgroup v1
    try:
        cg1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip()
        limit = int(cg1)
        # cgroup v1 reports a huge number when unlimited
        if limit < psutil.virtual_memory().total * 2:
            return limit // (1024 * 1024)
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    # Fallback: system RAM
    return psutil.virtual_memory().total // (1024 * 1024)


class MemoryMonitor:
    """Background task that monitors memory and evicts caches under pressure."""

    def __init__(
        self,
        tile_cache: TileCache,
        coord_cache_clear_fn,
        memory_limit_mb: int,
        check_interval: int = 30,
    ):
        self._tile_cache = tile_cache
        self._clear_coord_caches = coord_cache_clear_fn
        self._limit_bytes = memory_limit_mb * 1024 * 1024
        self._limit_mb = memory_limit_mb
        self._check_interval = check_interval
        self._task: asyncio.Task | None = None
        self._process = psutil.Process()

    async def start(self) -> None:
        logger.info(
            "Memory monitor started (limit=%d MB, check every %ds, "
            "warn=%.0f%%, evict_tiles=%.0f%%, evict_all=%.0f%%)",
            self._limit_mb, self._check_interval,
            _WARN_THRESHOLD * 100, _EVICT_TILES_THRESHOLD * 100,
            _EVICT_ALL_THRESHOLD * 100,
        )
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._check_interval)
            try:
                self._check()
            except Exception:
                logger.exception("Memory monitor check failed")

    def _check(self) -> None:
        rss = self._process.memory_info().rss
        usage = rss / self._limit_bytes

        if usage >= _EVICT_ALL_THRESHOLD:
            logger.warning(
                "Memory critical: %d MB / %d MB (%.0f%%) — clearing tile + coord caches",
                rss // (1024 * 1024), self._limit_mb, usage * 100,
            )
            self._tile_cache.clear()
            self._clear_coord_caches()

        elif usage >= _EVICT_TILES_THRESHOLD:
            freed = self._tile_cache.evict_half()
            logger.warning(
                "Memory pressure: %d MB / %d MB (%.0f%%) — evicted %.1f MB of tiles",
                rss // (1024 * 1024), self._limit_mb, usage * 100,
                freed / (1024 * 1024),
            )

        elif usage >= _WARN_THRESHOLD:
            logger.info(
                "Memory usage elevated: %d MB / %d MB (%.0f%%)",
                rss // (1024 * 1024), self._limit_mb, usage * 100,
            )
