# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
from collections import OrderedDict
from threading import Lock


class TileCache:
    """Thread-safe LRU cache for rendered tiles."""

    def __init__(self, max_size: int = 50_000):
        self._max_size = max_size
        self._cache: OrderedDict[tuple, bytes] = OrderedDict()
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
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                self._cache[key] = value
                while len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)

    def invalidate_timestamp(self, timestamp: int) -> None:
        """Remove all entries for a given timestamp."""
        with self._lock:
            keys_to_remove = [k for k in self._cache if k[0] == timestamp]
            for k in keys_to_remove:
                del self._cache[k]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
