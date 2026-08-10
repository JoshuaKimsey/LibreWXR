# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Memory pressure monitor — safety net to prevent OOM kills.

Periodically checks the container's reclaimable (non-file-backed) cgroup
memory against the container/system memory limit and proactively evicts
caches before the OOM killer intervenes.  The decision metric excludes
clean file-backed page cache: in multi mode every render worker memmaps
the same snapshot files, so those pages are shared, clean, and
kernel-reclaimable — they are not actionable pressure.  Each worker also
jitters its thresholds by a small fixed offset and requires two
consecutive over-threshold checks before evicting, so one cgroup spike
does not trip all workers in the same instant.
"""
import asyncio
import ctypes
import gc
import logging
import random
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CgroupUsage:
    """Reclaimable-aware container usage snapshot.

    ``decision_bytes`` is the figure threshold checks compare against the
    limit: ``anon + shmem`` on cgroup v2 (from ``memory.stat``),
    ``rss + shmem`` on cgroup v1, falling back to the raw cgroup usage
    file when the stat file can't be parsed.  ``total_bytes`` is the raw
    cgroup usage (``memory.current`` / ``memory.usage_in_bytes``) kept
    for logging and diagnostics — it includes the clean file-backed page
    cache the kernel reclaims on its own.

    ``anon_bytes`` / ``file_bytes`` / ``shmem_bytes`` are the individual
    stat counters (v2: ``anon`` / ``file`` / ``shmem``; v1: ``rss`` /
    ``cache`` / ``shmem``) exposed for the cluster /health aggregation.
    They are 0 on the raw-usage fallback paths where no stat file was
    parsed.
    """

    decision_bytes: int
    total_bytes: int
    label: str
    stat_based: bool
    anon_bytes: int = 0
    file_bytes: int = 0
    shmem_bytes: int = 0


def _read_int(path: Path) -> int | None:
    """Read a single unsigned integer from ``path``, or None on any failure."""
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError, PermissionError, OSError):
        return None


def _read_stat_fields(path: Path, fields: tuple[str, ...]) -> dict[str, int] | None:
    """Read individual counters from a cgroup ``memory.stat`` file.

    Returns a dict of the requested counters, or ``None`` when the file
    is missing/unreadable or any requested counter is absent or
    malformed, so callers can fall back to the raw usage file instead of
    crashing the monitor.  Unrequested lines are ignored, so a malformed
    counter we don't care about never poisons the parse.
    """
    try:
        lines = path.read_text().splitlines()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    values: dict[str, int] = {}
    for line in lines:
        key, _, raw = line.partition(" ")
        if key in fields:
            try:
                values[key] = int(raw)
            except ValueError:
                return None
    return values if len(values) == len(fields) else None


def _read_stat_sum(path: Path, fields: tuple[str, ...]) -> int | None:
    """Sum the named counters from a cgroup ``memory.stat`` file.

    Returns ``None`` when the file is missing/unreadable or any requested
    counter is absent or malformed, so callers can fall back to the raw
    usage file instead of crashing the monitor.
    """
    values = _read_stat_fields(path, fields)
    return sum(values.values()) if values is not None else None


def _read_cgroup_memory_usage() -> CgroupUsage | None:
    """Return a reclaimable-aware cgroup usage snapshot, or None.

    Captures every process in the container — important in multi-worker
    mode where each render worker's own RSS is only a fraction of the
    container's total.  Falls back to ``None`` outside containers so
    callers can use per-process RSS instead.

    The decision metric deliberately excludes clean file-backed page
    cache.  In multi mode all workers memmap the same snapshot files;
    those pages are shared, clean, and kernel-reclaimable, so the kernel
    reclaims them on its own under pressure and counting them would make
    every worker act on pressure that is not actionable.  ``anon`` (private
    heap) and ``shmem`` (tmpfs/shared pages — NOT cleanly reclaimable) are
    the parts that matter; tmpfs-backed cache dirs stay correctly counted
    because they live in ``shmem``.  The bulk of the memmap page cache
    lives in ``file`` and is intentionally excluded.
    """
    # cgroup v2 — decision metric is anon + shmem from memory.stat.  The
    # individual ``anon`` / ``file`` / ``shmem`` counters ride along for
    # the cluster /health split.
    v2_stat = _read_stat_fields(
        Path("/sys/fs/cgroup/memory.stat"), ("anon", "shmem", "file")
    )
    total = _read_int(Path("/sys/fs/cgroup/memory.current"))
    if v2_stat is not None:
        decision = v2_stat["anon"] + v2_stat["shmem"]
        return CgroupUsage(
            decision_bytes=decision,
            total_bytes=total if total is not None else decision,
            anon_bytes=v2_stat["anon"],
            file_bytes=v2_stat["file"],
            shmem_bytes=v2_stat["shmem"],
            label="anon+shmem",
            stat_based=True,
        )
    if total is not None:
        return CgroupUsage(
            decision_bytes=total,
            total_bytes=total,
            label="cgroup",
            stat_based=False,
        )

    # cgroup v1 — decision metric is rss + shmem from memory.stat, with
    # the individual counters mapped rss->anon / cache->file.
    v1_stat = _read_stat_fields(
        Path("/sys/fs/cgroup/memory/memory.stat"), ("rss", "shmem", "cache")
    )
    total = _read_int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    if v1_stat is not None:
        decision = v1_stat["rss"] + v1_stat["shmem"]
        return CgroupUsage(
            decision_bytes=decision,
            total_bytes=total if total is not None else decision,
            anon_bytes=v1_stat["rss"],
            file_bytes=v1_stat["cache"],
            shmem_bytes=v1_stat["shmem"],
            label="rss+shmem",
            stat_based=True,
        )
    if total is not None:
        return CgroupUsage(
            decision_bytes=total,
            total_bytes=total,
            label="cgroup",
            stat_based=False,
        )

    return None


def _jittered_thresholds(
    warn: float = _WARN_THRESHOLD,
    evict_tiles: float = _EVICT_TILES_THRESHOLD,
    evict_all: float = _EVICT_ALL_THRESHOLD,
) -> tuple[float, float, float]:
    """Apply the per-process random offsets, preserving warn < evict < clear.

    The offsets are fixed for the process lifetime (applied once at
    monitor init), so 16 workers in one cgroup trip at slightly different
    usage levels instead of all firing in lock-step.  With the base
    constants the ranges can't overlap, but clamp defensively anyway in
    case the bases ever drift closer together.
    """
    warn = warn + random.uniform(-0.01, 0.01)
    evict_tiles = evict_tiles + random.uniform(-0.02, 0.02)
    evict_all = evict_all + random.uniform(-0.02, 0.02)
    if evict_tiles <= warn:
        evict_tiles = warn + 0.001
    if evict_all <= evict_tiles:
        evict_all = evict_tiles + 0.001
    return warn, evict_tiles, evict_all


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
        # Per-process threshold jitter: each worker in a shared cgroup
        # gets a small fixed offset so they don't all trip at the exact
        # same usage level.  Jittered into instance attributes only —
        # the module-level base constants are never mutated.
        (
            self._warn_threshold,
            self._evict_tiles_threshold,
            self._evict_all_threshold,
        ) = _jittered_thresholds()
        # Hysteresis counters: an eviction level only acts after two
        # consecutive checks above its (jittered) threshold; the counter
        # resets whenever a check falls below the level.
        self._evict_streak = 0
        self._clear_streak = 0
        # One-time debug log when memory.stat can't be parsed and the raw
        # cgroup usage file is used for decisions instead.
        self._logged_usage_fallback = False
        # Raw cgroup usage (MB) from the most recent check, for
        # diagnostics; None outside containers.
        self._cgroup_total_mb: int | None = None
        # Last full cgroup reading (decision + anon/file/shmem split) for
        # the cluster /health aggregation; None outside containers.
        self._last_cgroup_usage: CgroupUsage | None = None

    @property
    def cgroup_total_mb(self) -> int | None:
        """Raw cgroup usage in MB from the most recent check (None outside containers)."""
        return self._cgroup_total_mb

    @property
    def cgroup_memory_mb(self) -> dict | None:
        """Most recent cgroup anon/file/shmem split + limit in MB.

        Returns ``None`` outside containers (the per-process psutil
        fallback carries no cgroup split).  ``limit_mb`` is the monitor's
        detected container limit, matching the value the /health top-level
        ``memory.limit_mb`` reports.
        """
        if self._last_cgroup_usage is None:
            return None
        usage = self._last_cgroup_usage
        return {
            "anon_mb": usage.anon_bytes // (1024 * 1024),
            "file_mb": usage.file_bytes // (1024 * 1024),
            "shmem_mb": usage.shmem_bytes // (1024 * 1024),
            "limit_mb": self._limit_mb,
        }

    async def start(self) -> None:
        scope = "container (cgroup)" if _read_cgroup_memory_usage() is not None else "process"
        logger.debug(
            "Memory monitor base thresholds: warn=%.0f%%, evict_tiles=%.0f%%, "
            "evict_all=%.0f%% (jittered per-worker: warn=%.0f%%, evict_tiles=%.0f%%, "
            "evict_all=%.0f%%)",
            _WARN_THRESHOLD * 100, _EVICT_TILES_THRESHOLD * 100,
            _EVICT_ALL_THRESHOLD * 100,
            self._warn_threshold * 100, self._evict_tiles_threshold * 100,
            self._evict_all_threshold * 100,
        )
        logger.debug(
            "Memory monitor started (scope=%s, limit=%d MB, check every %ds, "
            "warn=%.0f%%, evict_tiles=%.0f%%, evict_all=%.0f%%)",
            scope, self._limit_mb, self._check_interval,
            self._warn_threshold * 100, self._evict_tiles_threshold * 100,
            self._evict_all_threshold * 100,
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
        #
        # The decision metric is the cgroup's kernel-irreclaimable share
        # (anon + shmem on v2, rss + shmem on v1) rather than the raw
        # total: in multi mode all workers memmap the same snapshot
        # files, and those file-backed pages are shared, clean, and
        # kernel-reclaimable — the kernel reclaims them on its own under
        # pressure, so counting them would make every worker act on
        # pressure that is not actionable.
        usage_info = _read_cgroup_memory_usage()
        # Keep the full reading for the cluster /health split (anon/file/
        # shmem); None outside containers.
        self._last_cgroup_usage = usage_info
        if usage_info is not None:
            decision_bytes = usage_info.decision_bytes
            total_bytes = usage_info.total_bytes
            if not usage_info.stat_based and not self._logged_usage_fallback:
                self._logged_usage_fallback = True
                logger.debug(
                    "cgroup memory.stat unreadable — falling back to raw cgroup usage file"
                )
        else:
            decision_bytes = self._process.memory_info().rss
            total_bytes = decision_bytes
        usage = decision_bytes / self._limit_bytes

        decision_mb = decision_bytes // (1024 * 1024)
        total_mb = total_bytes // (1024 * 1024)
        self._cgroup_total_mb = total_mb if usage_info is not None else None
        label = usage_info.label if usage_info is not None else "rss"

        if usage >= self._evict_all_threshold:
            self._clear_streak += 1
            self._evict_streak = 0
            if self._clear_streak >= 2:
                logger.warning(
                    "Memory critical: %d MB (%s) / %d MB (%.0f%%; cgroup total %d MB) — "
                    "clearing tile + coord caches",
                    decision_mb, label, self._limit_mb, usage * 100, total_mb,
                )
                self._tile_cache.clear()
                self._clear_coord_caches()
                release_memory()

        elif usage >= self._evict_tiles_threshold:
            self._evict_streak += 1
            self._clear_streak = 0
            if self._evict_streak >= 2:
                freed = self._tile_cache.evict_half()
                release_memory()
                logger.warning(
                    "Memory pressure: %d MB (%s) / %d MB (%.0f%%; cgroup total %d MB) — "
                    "evicted %.1f MB of tiles",
                    decision_mb, label, self._limit_mb, usage * 100, total_mb,
                    freed / (1024 * 1024),
                )

        elif usage >= self._warn_threshold:
            self._evict_streak = 0
            self._clear_streak = 0
            logger.info(
                "Memory usage elevated: %d MB (%s) / %d MB (%.0f%%; cgroup total %d MB)",
                decision_mb, label, self._limit_mb, usage * 100, total_mb,
            )

        else:
            self._evict_streak = 0
            self._clear_streak = 0
