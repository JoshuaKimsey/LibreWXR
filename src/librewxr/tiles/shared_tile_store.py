# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Shared on-disk encoded-tile store for multi-mode render workers.

In a multi-worker deployment every render worker independently encodes the
same tile for the same frame — the encode tail (LUT colorize, blur, WebP)
is the expensive per-request part of the render path, so N workers mean N
redundant encodes across the fleet.  This store moves those encoded bytes
onto the shared cache volume: the first worker to encode a tile publishes
it and every other worker serves the same bytes.  One encode serves all
workers.

Keys are content-versioned by the CALLER: the caller folds the frame's
content version (see ``FrameStore.frame_version``) into the key after the
leading timestamp, so the moment a frame's content changes (version bump)
or the frame is evicted, the old key becomes unreachable and the new
content is encoded under a fresh key — stale bytes are never served and
are only garbage the pruner reclaims.  Keys start with the frame timestamp
followed by "-" (e.g. ``"1712345600-v1-7-70-63-512"``); that timestamp
picks both the 64-way shard directory (``ts % 64``) and the invalidation
prefix, so invalidating one timestamp touches exactly one directory.

Publishes are atomic (``.<name>.tmp`` + ``os.replace`` inside the shard),
so any number of workers may publish concurrently — readers only ever see
a complete file, and concurrent identical publishes converge on the last
``os.replace`` (same content, last write wins).  All file mutations are
best-effort and never raise; transient OSErrors surface as a debug log on
read and a warning on write, matching the project's no-fsync, tolerate-
and-continue convention.  ``prune`` is a full on-disk scan and must only
be called off the event loop.

This module is a pure storage class: no ``librewxr.config`` settings are
imported.  Callers pass ``cache_dir`` and ``max_mb``.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Subdirectory of the cache dir holding the encoded tiles.
_ROOT_DIRNAME = "tiles_shared"
# One shard dir per 64 timestamps (``ts % 64``).  All files for one
# timestamp land in the same shard so ``invalidate_timestamp`` only ever
# has to look inside a single directory.
_N_SHARDS = 64
# Keys whose leading timestamp cannot be parsed store here.  They can be
# read and pruned like any other entry but can never be invalidated by
# timestamp (see ``_ts_of``) — only prune()/clear() remove them.
_FALLBACK_SHARD = "00"
# Suffix appended to every published file; a store for one key is always
# exactly one ``<key>.tile`` file.
_TILE_SUFFIX = ".tile"


class SharedTileStore:
    """On-disk, size-capped store of encoded tile bytes under ``cache_dir``.

    Layout: ``<cache_dir>/tiles_shared/<ts % 64:02d>/<key>.tile``.  Each
    entry is one complete encoded tile; files are self-describing by key,
    so no manifest or index needs to be kept in sync across processes.
    """

    def __init__(self, cache_dir: Path, max_mb: int) -> None:
        self._root = Path(cache_dir) / _ROOT_DIRNAME
        self._max_bytes = max_mb * 1024 * 1024
        # Approximate in-memory counters.  They drift from the real on-disk
        # state (overwrites re-count, other workers publish concurrently)
        # and are resynced from a full scan by ``prune``.
        self._size = 0
        self._total_bytes = 0
        self._root.mkdir(parents=True, exist_ok=True)
        # Sweep stale ``*.tmp`` files left by a crash mid-publish so
        # subsequent atomic writes don't accumulate garbage (FrameStore
        # convention).  A live tmp from a concurrent writer is the writer's
        # problem — os.replace on a vanished tmp is handled by publish().
        for path in self._root.rglob("*.tmp"):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.debug("shared_tile_store: could not sweep %s", path)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def size(self) -> int:
        """Approximate entry count (stale after cross-process writes)."""
        return self._size

    @property
    def total_bytes(self) -> int:
        """Approximate on-disk bytes (stale after cross-process writes)."""
        return self._total_bytes

    @staticmethod
    def _ts_of(key: str) -> int:
        """Leading integer of the key before the first "-".

        Callers always provide ts-prefixed keys, so this identifies the
        shard AND the invalidation prefix in one parse.  A key with no
        leading integer (or a non-integer lead) raises ValueError: it is
        treated as unshardable — stored under ``_FALLBACK_SHARD``, readable
        and prunable, but invisible to ``invalidate_timestamp``.
        """
        dash = key.find("-")
        if dash <= 0:
            raise ValueError(f"shared_tile_store: key has no leading timestamp: {key!r}")
        return int(key[:dash])

    def _shard_for_key(self, key: str) -> str:
        """Shard dir name for a key (fallback shard when unshardable)."""
        try:
            return f"{self._ts_of(key) % _N_SHARDS:02d}"
        except ValueError:
            return _FALLBACK_SHARD

    def _path_for(self, key: str) -> Path:
        """Filesystem path for a key (does not touch the disk)."""
        return self._root / self._shard_for_key(key) / f"{key}{_TILE_SUFFIX}"

    def get(self, key: str) -> bytes | None:
        """Return the encoded tile bytes for ``key``, or None on any miss.

        A missing file and any read error are equivalent: the caller falls
        back to encoding and publishing.  Logged at DEBUG once per call —
        a miss is an expected part of normal operation (first worker to
        encode a tile), not an error worth WARNING noise.
        """
        path = self._path_for(key)
        try:
            return path.read_bytes()
        except OSError:
            # FileNotFoundError subsumed: raced with a concurrent
            # prune/invalidate/clear, or EIO/EAGAIN — serve nothing and
            # let the caller re-encode.
            logger.debug("shared_tile_store: miss on %s", key)
            return None

    def publish(self, key: str, data: bytes) -> None:
        """Atomically persist ``data`` under ``key``.

        Writes to a ``.<name>.tmp`` inside the shard then ``os.replace``s
        onto the final path, so a concurrent reader either sees the old
        file or the complete new one, never a partial write.  Concurrent
        identical publishes from different workers are safe: both write
        identical tmp files and the last ``os.replace`` wins with the same
        content.  The in-memory counters increment unconditionally and are
        NOT recounted on overwrite — they are approximate by design and
        resynced by ``prune``.  Best-effort: failures log a warning and
        leave the previous entry (if any) untouched.
        """
        path = self._path_for(key)
        tmp: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except OSError:
            logger.warning(
                "shared_tile_store: publish failed for %s", key, exc_info=True,
            )
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            return
        self._total_bytes += len(data)
        self._size += 1

    def invalidate_timestamp(self, timestamp: int) -> None:
        """Remove every entry whose key starts with ``f"{timestamp}-"``.

        All files for a timestamp share one shard (``ts % 64``), so this
        is a single-directory scan.  Missing shard dir → no-op.  Counters
        are adjusted by ``path.stat().st_size`` where the stat succeeds —
        best-effort, so failures leave the approximate counters slightly
        high until the next ``prune`` resync.  In-flight ``.tmp`` writes
        from a concurrent publish of the same timestamp start with "."
        and are deliberately left alone.
        """
        shard = self._root / f"{timestamp % _N_SHARDS:02d}"
        if not shard.is_dir():
            return
        prefix = f"{timestamp}-"
        try:
            names = list(shard.iterdir())
        except OSError:
            # Raced with a concurrent clear() — nothing to do.
            return
        for path in names:
            if not path.name.startswith(prefix):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            try:
                path.unlink()
            except OSError:
                continue
            if size:
                self._total_bytes = max(0, self._total_bytes - size)
                self._size = max(0, self._size - 1)

    def clear(self) -> None:
        """Remove the whole store tree and recreate the root.

        Used on startup/fetch reset to drop every cached encode in one
        shot; cheaper than timestamp-by-timestamp invalidation.  Counters
        reset to zero (a concurrent writer could repopulate the tree
        after the rmtree — the next prune resyncs).
        """
        shutil.rmtree(self._root, ignore_errors=True)
        self._root.mkdir(parents=True, exist_ok=True)
        self._size = 0
        self._total_bytes = 0

    def prune(self) -> None:
        """Evict oldest-mtime entries until the store fits the MB budget.

        Full scan of every shard sums the real on-disk sizes; when the
        total exceeds ``max_mb * 1024 * 1024``, entries are unlinked
        oldest-mtime-first until the remainder fits under budget, then the
        in-memory counters are resynced exactly from a final scan — any
        drift from overwrites, failed publishes, or concurrent writers
        self-corrects here.  Live ``.tmp`` files are skipped: they belong
        to a concurrent publish that os.replace will finish momentarily.
        O(files log files) with multiple stat calls — only ever call this
        off the event loop.
        """
        entries: list[tuple[Path, int, float]] = []  # (path, size, mtime)
        total = 0
        if self._root.is_dir():
            for path in self._root.rglob("*"):
                if not path.is_file():
                    continue
                if path.name.endswith(".tmp"):
                    continue
                try:
                    st = path.stat()
                except OSError:
                    continue
                entries.append((path, st.st_size, st.st_mtime))
                total += st.st_size

        if total > self._max_bytes:
            entries.sort(key=lambda e: e[2])  # oldest mtime first
            for path, size, _mtime in entries:
                if total <= self._max_bytes:
                    break
                try:
                    path.unlink()
                except OSError:
                    continue
                total -= size

        # Resync counters exactly from what survives (approximate counters
        # are not trustworthy after evictions + any concurrent writes).
        self._size = 0
        self._total_bytes = 0
        if self._root.is_dir():
            for path in self._root.rglob("*"):
                if not path.is_file():
                    continue
                if path.name.endswith(".tmp"):
                    continue
                try:
                    st = path.stat()
                except OSError:
                    continue
                self._size += 1
                self._total_bytes += st.st_size
