# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Counts tile requests above a zoom threshold to surface usage hotspots.

Purely observational: a future adaptive-warming pass can read the same
counters to decide which tiles to keep warm, but this module never
schedules work itself. Designed to be cheap on the hot path — a dict
update under a Lock — so it can sit inline in the tile endpoint.

State is in-memory only; restarts wipe the counters. That's fine for
the diagnostic phase: we just want a few days of distribution to see
whether traffic is power-law (a few hot tiles) or diffuse.
"""
from collections import Counter
from threading import Lock


def _avg_ms(total_ns: int, count: int) -> float:
    """Mean latency in milliseconds from ns totals; 0.0 when empty."""
    if count == 0:
        return 0.0
    return round(total_ns / count / 1e6, 2)


class TileRequestTracker:
    """Bounded per-tile request counter, tracking only z >= ``min_zoom``."""

    def __init__(self, min_zoom: int = 7, max_entries: int = 10_000):
        self._min_zoom = min_zoom
        self._max_entries = max_entries
        self._counts: Counter[tuple[int, int, int]] = Counter()
        self._fast_path_counts: dict[str, int] = {}  # reason -> count
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        # Latency accumulators (ns totals + counts per stage).  Computes
        # and presents are only counted when the stage actually ran
        # (cache hits contribute to the request totals but not the stage
        # totals).
        self._lat_request_ns_total = 0
        self._lat_request_count = 0
        self._lat_compute_ns_total = 0
        self._lat_compute_count = 0
        self._lat_present_ns_total = 0
        self._lat_present_count = 0
        self._lock = Lock()

    def record(self, z: int, x: int, y: int) -> None:
        """Increment the counter for one tile request.

        Calls below ``min_zoom`` are no-ops — overview zooms are already
        warmed eagerly, so tracking them adds noise without insight.
        """
        if z < self._min_zoom:
            return
        with self._lock:
            self._counts[(z, x, y)] += 1
            if len(self._counts) > self._max_entries:
                self._evict_cold()

    def record_request(self, z: int, x: int, y: int, cache_hit: bool) -> None:
        """Batched request bookkeeping in a single lock acquisition.

        Combines ``record`` (per-tile counter, gated on ``min_zoom``)
        with the geometry-cache hit/miss tally.  The hit/miss tally is
        NOT gated on ``min_zoom`` — below ``min_zoom`` the (z, x, y)
        counter part is skipped but the cache outcome still counts,
        matching the existing semantics of ``record`` vs
        ``record_cache_hit``/``record_cache_miss``.
        """
        with self._lock:
            if z >= self._min_zoom:
                self._counts[(z, x, y)] += 1
                if len(self._counts) > self._max_entries:
                    self._evict_cold()
            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def _evict_cold(self) -> None:
        """Drop the bottom half of entries by count.

        Must be called with ``self._lock`` held. Halving on overflow keeps
        the amortized cost of eviction O(1) per request — we pay an
        O(n log n) cull once per ``max_entries / 2`` records.
        """
        keep = self._counts.most_common(self._max_entries // 2)
        self._counts = Counter(dict(keep))

    def record_fast_path(self, reason: str) -> None:
        """Record that a fast-path transparent geometry was produced."""
        with self._lock:
            self._fast_path_counts[reason] = self._fast_path_counts.get(reason, 0) + 1

    def record_cache_hit(self) -> None:
        """Record a geometry-stage cache hit."""
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a geometry-stage cache miss."""
        with self._lock:
            self._cache_misses += 1

    def record_latency(
        self,
        request_ns: int,
        compute_ns: int | None,
        present_ns: int | None,
    ) -> None:
        """Accumulate request/compute/present latency in nanoseconds.

        The request duration is always counted; compute and present are
        counted only when not ``None`` (they are ``None`` for cache hits
        where the stage didn't run).
        """
        with self._lock:
            self._lat_request_ns_total += request_ns
            self._lat_request_count += 1
            if compute_ns is not None:
                self._lat_compute_ns_total += compute_ns
                self._lat_compute_count += 1
            if present_ns is not None:
                self._lat_present_ns_total += present_ns
                self._lat_present_count += 1

    def latency_snapshot(self) -> dict:
        """Read-only snapshot of the latency accumulators (under lock)."""
        with self._lock:
            return {
                "request_ns_total": self._lat_request_ns_total,
                "request_count": self._lat_request_count,
                "compute_ns_total": self._lat_compute_ns_total,
                "compute_count": self._lat_compute_count,
                "present_ns_total": self._lat_present_ns_total,
                "present_count": self._lat_present_count,
            }

    def stats(self, top_n: int = 10, hot_threshold: int = 5) -> dict:
        """Snapshot for the /health endpoint.

        Args:
            top_n: How many of the most-requested tiles to include verbatim.
            hot_threshold: Count threshold for the ``hot_tiles`` summary —
                tiles at or above this count are the candidates that an
                adaptive warmer would target.
        """
        with self._lock:
            tracked = len(self._counts)
            top_items = self._counts.most_common(top_n)
            total_requests = sum(self._counts.values())
            hot_tiles = sum(1 for c in self._counts.values() if c >= hot_threshold)
            by_zoom: dict[int, dict[str, int]] = {}
            for (z, _x, _y), count in self._counts.items():
                bucket = by_zoom.setdefault(z, {"tiles": 0, "requests": 0})
                bucket["tiles"] += 1
                bucket["requests"] += count
            fast_path_total = sum(self._fast_path_counts.values())
            fast_path_by_reason = dict(self._fast_path_counts)
            cache_hits = self._cache_hits
            cache_misses = self._cache_misses
            latency = {
                "requests": self._lat_request_count,
                "avg_request_ms": _avg_ms(
                    self._lat_request_ns_total, self._lat_request_count,
                ),
                "computes": self._lat_compute_count,
                "avg_compute_ms": _avg_ms(
                    self._lat_compute_ns_total, self._lat_compute_count,
                ),
                "presents": self._lat_present_count,
                "avg_present_ms": _avg_ms(
                    self._lat_present_ns_total, self._lat_present_count,
                ),
            }
        return {
            "min_zoom": self._min_zoom,
            "max_entries": self._max_entries,
            "tracked_tiles": tracked,
            "total_requests": total_requests,
            "hot_threshold": hot_threshold,
            "hot_tiles": hot_tiles,
            "by_zoom": {z: by_zoom[z] for z in sorted(by_zoom)},
            "top": [
                {"z": z, "x": x, "y": y, "count": count}
                for (z, x, y), count in top_items
            ],
            "fast_path": {
                "total": fast_path_total,
                "by_reason": fast_path_by_reason,
            },
            "cache": {
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate": (
                    cache_hits / (cache_hits + cache_misses)
                    if (cache_hits + cache_misses) > 0
                    else 0.0
                ),
            },
            "latency": latency,
        }
