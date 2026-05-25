# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Memory pressure monitor — safety net to prevent OOM kills.

Periodically checks process RSS against the container/system memory
limit and proactively evicts caches before the OOM killer intervenes.
"""
import asyncio
import ctypes
import gc
import logging
from pathlib import Path

import psutil

from librewxr.tiles.cache import TileCache

logger = logging.getLogger(__name__)


def release_memory() -> None:
    """Force Python garbage collection and return freed pages to the OS.

    Python's garbage collector doesn't run eagerly for non-cyclic objects,
    and glibc's malloc never returns freed heap pages to the OS on its own.
    Calling gc.collect() + malloc_trim(0) after heavy operations (ECMWF
    regridding, nowcast optical flow) reclaims hundreds of MB that would
    otherwise show up as "other" in the memory breakdown.
    """
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass  # Non-glibc platform (musl, macOS) — gc.collect() is enough

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


def _read_cgroup_memory_usage() -> int | None:
    """Return the cgroup's current memory usage in bytes, or None.

    Captures every process in the container — important in multi-worker
    mode where each render worker's own RSS is only a fraction of the
    container's total.  Falls back to ``None`` outside containers so
    callers can use per-process RSS instead.
    """
    # cgroup v2
    try:
        v2 = Path("/sys/fs/cgroup/memory.current").read_text().strip()
        return int(v2)
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    # cgroup v1
    try:
        v1 = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text().strip()
        return int(v1)
    except (FileNotFoundError, ValueError, PermissionError):
        pass

    return None


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
        scope = "container (cgroup)" if _read_cgroup_memory_usage() is not None else "process"
        logger.info(
            "Memory monitor started (scope=%s, limit=%d MB, check every %ds, "
            "warn=%.0f%%, evict_tiles=%.0f%%, evict_all=%.0f%%)",
            scope, self._limit_mb, self._check_interval,
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
        # In multi-worker deployments the container holds N render
        # workers, each with its own ``psutil.Process``.  Comparing one
        # worker's RSS to the container-wide cgroup limit never trips
        # the thresholds because no single worker holds more than ~1/N
        # of the limit.  Read the cgroup's own usage when available so
        # every worker sees the same shared pressure and they all evict
        # their local caches in concert.  Falls back to per-process RSS
        # outside containers (local dev, single-process deployments).
        cgroup_usage = _read_cgroup_memory_usage()
        if cgroup_usage is not None:
            rss = cgroup_usage
        else:
            rss = self._process.memory_info().rss
        usage = rss / self._limit_bytes

        if usage >= _EVICT_ALL_THRESHOLD:
            logger.warning(
                "Memory critical: %d MB / %d MB (%.0f%%) — clearing tile + coord caches",
                rss // (1024 * 1024), self._limit_mb, usage * 100,
            )
            self._tile_cache.clear()
            self._clear_coord_caches()
            release_memory()

        elif usage >= _EVICT_TILES_THRESHOLD:
            freed = self._tile_cache.evict_half()
            release_memory()
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
