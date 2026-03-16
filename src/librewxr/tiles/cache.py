# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
from collections import OrderedDict
from threading import Lock


class TileCache:
    """Thread-safe LRU cache for rendered tiles, capped by total byte size."""

    def __init__(self, max_mb: int = 200):
        self._max_bytes = max_mb * 1024 * 1024
        self._cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._total_bytes = 0
        self._lock = Lock()

    def get(self, key: tuple) -> bytes | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, key: tuple, value: bytes) -> None:
        with self._lock:
            if key in self._cache:
                self._total_bytes -= len(self._cache[key])
                self._cache.move_to_end(key)
                self._cache[key] = value
                self._total_bytes += len(value)
            else:
                self._cache[key] = value
                self._total_bytes += len(value)
            self._evict_to_budget()

    def evict_half(self) -> int:
        """Evict the oldest half of entries. Returns bytes freed."""
        with self._lock:
            target = len(self._cache) // 2
            freed = 0
            for _ in range(target):
                if not self._cache:
                    break
                _, v = self._cache.popitem(last=False)
                freed += len(v)
            self._total_bytes -= freed
            return freed

    def invalidate_timestamp(self, timestamp: int) -> None:
        """Remove all entries for a given timestamp."""
        with self._lock:
            keys_to_remove = [k for k in self._cache if k[0] == timestamp]
            for k in keys_to_remove:
                self._total_bytes -= len(self._cache[k])
                del self._cache[k]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._total_bytes = 0

    def _evict_to_budget(self) -> None:
        """Evict oldest entries until total bytes is within budget."""
        while self._total_bytes > self._max_bytes and self._cache:
            _, v = self._cache.popitem(last=False)
            self._total_bytes -= len(v)

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes
