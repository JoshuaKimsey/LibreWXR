# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol


class _SizedValue(Protocol):
    """Anything the cache stores must expose its byte size."""
    @property
    def nbytes(self) -> int: ...


def _size_of(value: Any) -> int:
    """Byte size of a cache entry.  Supports bytes and anything with ``nbytes``."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    return int(value.nbytes)


@dataclass
class CachedRender:
    """An encoded tile plus its ETag, stored under a present-stage cache key.

    ``TileCache`` already accepts any object exposing ``nbytes``; this wraps the
    encoded bytes so the ETag is reused on cache hits without re-hashing.
    """
    data: bytes
    etag: str

    @property
    def nbytes(self) -> int:
        return len(self.data) + len(self.etag.encode("ascii"))


class TileCache:
    """Thread-safe LRU cache for tile data, capped by total byte size.

    Stores any value that exposes a byte size — either ``bytes`` (encoded
    tile output) or an object with a ``.nbytes`` property (e.g. the
    ``TileGeometry`` records produced by the renderer's compute step).
    The cap is enforced on the sum of those sizes.
    """

    def __init__(self, max_mb: int = 200):
        self._max_bytes = max_mb * 1024 * 1024
        self._cache: OrderedDict[tuple, Any] = OrderedDict()
        self._total_bytes = 0
        self._lock = Lock()

    def get(self, key: tuple) -> Any | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, key: tuple, value: Any) -> None:
        with self._lock:
            new_size = _size_of(value)
            if key in self._cache:
                self._total_bytes -= _size_of(self._cache[key])
                self._cache.move_to_end(key)
            self._cache[key] = value
            self._total_bytes += new_size
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
                freed += _size_of(v)
            self._total_bytes -= freed
            return freed

    def invalidate_timestamp(self, timestamp: int) -> None:
        """Remove all entries for a given timestamp."""
        with self._lock:
            keys_to_remove = [k for k in self._cache if k[0] == timestamp]
            for k in keys_to_remove:
                self._total_bytes -= _size_of(self._cache[k])
                del self._cache[k]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._total_bytes = 0

    def entries(self) -> list[tuple[tuple, int]]:
        """Return ``(key, size_bytes)`` for every cached entry (read-only snapshot)."""
        with self._lock:
            return [(key, _size_of(value)) for key, value in self._cache.items()]

    def _evict_to_budget(self) -> None:
        """Evict oldest entries until total bytes is within budget."""
        while self._total_bytes > self._max_bytes and self._cache:
            _, v = self._cache.popitem(last=False)
            self._total_bytes -= _size_of(v)

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def max_bytes(self) -> int:
        return self._max_bytes
