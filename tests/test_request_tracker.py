# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import pytest

pytestmark = pytest.mark.tiles

from librewxr.tiles.request_tracker import TileRequestTracker


class TestTileRequestTracker:
    def test_records_above_min_zoom(self):
        tracker = TileRequestTracker(min_zoom=7)
        tracker.record(7, 1, 2)
        tracker.record(7, 1, 2)
        tracker.record(8, 5, 3)

        stats = tracker.stats()
        assert stats["tracked_tiles"] == 2
        assert stats["total_requests"] == 3

    def test_skips_below_min_zoom(self):
        tracker = TileRequestTracker(min_zoom=7)
        for _ in range(100):
            tracker.record(3, 0, 0)
            tracker.record(6, 1, 1)

        stats = tracker.stats()
        assert stats["tracked_tiles"] == 0
        assert stats["total_requests"] == 0

    def test_top_returns_hottest_tiles(self):
        tracker = TileRequestTracker(min_zoom=7)
        for _ in range(10):
            tracker.record(8, 100, 200)
        for _ in range(3):
            tracker.record(8, 50, 50)
        tracker.record(9, 1, 1)

        stats = tracker.stats(top_n=2)
        assert stats["top"][0] == {"z": 8, "x": 100, "y": 200, "count": 10}
        assert stats["top"][1] == {"z": 8, "x": 50, "y": 50, "count": 3}
        assert len(stats["top"]) == 2

    def test_hot_threshold_count(self):
        tracker = TileRequestTracker(min_zoom=7)
        for _ in range(7):
            tracker.record(8, 1, 1)  # >= 5
        for _ in range(5):
            tracker.record(8, 2, 2)  # >= 5
        for _ in range(2):
            tracker.record(8, 3, 3)  # < 5

        stats = tracker.stats(hot_threshold=5)
        assert stats["hot_tiles"] == 2

    def test_by_zoom_breakdown(self):
        tracker = TileRequestTracker(min_zoom=7)
        tracker.record(7, 1, 1)
        tracker.record(7, 2, 2)
        tracker.record(7, 2, 2)
        tracker.record(9, 5, 5)

        by_zoom = tracker.stats()["by_zoom"]
        assert by_zoom[7] == {"tiles": 2, "requests": 3}
        assert by_zoom[9] == {"tiles": 1, "requests": 1}

    def test_evicts_when_over_cap(self):
        # Cap=4 → eviction triggers on the 5th distinct tile, halving to 2.
        tracker = TileRequestTracker(min_zoom=7, max_entries=4)
        # Hot tiles get many hits — they should survive eviction.
        for _ in range(10):
            tracker.record(8, 0, 0)
        for _ in range(8):
            tracker.record(8, 1, 1)
        # Cold tiles get one hit each.
        tracker.record(8, 2, 2)
        tracker.record(8, 3, 3)
        # This 5th distinct tile triggers the cull down to 2.
        tracker.record(8, 4, 4)

        stats = tracker.stats()
        assert stats["tracked_tiles"] == 2
        kept = {(t["x"], t["y"]) for t in stats["top"]}
        assert kept == {(0, 0), (1, 1)}

    def test_record_fast_path_increments_reason_and_total(self):
        tracker = TileRequestTracker()
        tracker.record_fast_path("tier2_past_radar")
        tracker.record_fast_path("tier2_past_radar")
        tracker.record_fast_path("case_a_no_nwp_empty_radar")

        stats = tracker.stats()
        assert stats["fast_path"]["total"] == 3
        assert stats["fast_path"]["by_reason"]["tier2_past_radar"] == 2
        assert stats["fast_path"]["by_reason"]["case_a_no_nwp_empty_radar"] == 1

    def test_record_cache_hit_miss_and_hit_rate(self):
        tracker = TileRequestTracker()
        tracker.record_cache_hit()
        tracker.record_cache_hit()
        tracker.record_cache_miss()
        tracker.record_cache_miss()
        tracker.record_cache_miss()

        stats = tracker.stats()
        assert stats["cache"]["hits"] == 2
        assert stats["cache"]["misses"] == 3
        assert stats["cache"]["hit_rate"] == pytest.approx(0.4)

        # Fresh tracker with no activity: hit_rate defaults to 0.0.
        empty = TileRequestTracker().stats()
        assert empty["cache"]["hits"] == 0
        assert empty["cache"]["misses"] == 0
        assert empty["cache"]["hit_rate"] == 0.0

    def test_stats_keeps_all_existing_keys(self):
        tracker = TileRequestTracker()
        tracker.record(7, 1, 1)
        tracker.record_fast_path("tier2_past_radar")
        tracker.record_cache_hit()

        stats = tracker.stats()
        for key in (
            "min_zoom", "max_entries", "tracked_tiles", "total_requests",
            "hot_threshold", "hot_tiles", "by_zoom", "top",
        ):
            assert key in stats
        # Additive keys also present.
        assert "fast_path" in stats
        assert "cache" in stats

    def test_capture_snapshot_records_full_segment(self):
        tracker = TileRequestTracker(min_zoom=7)
        for _ in range(6):
            tracker.record(7, 1, 2)
        tracker.record(8, 5, 3)
        tracker.record(8, 5, 3)
        tracker.record_request(8, 5, 3, cache_hit=True)
        tracker.record_cache_miss()
        tracker.record_fast_path("tier2_past_radar")
        tracker.record_fast_path("case_a_no_nwp_empty_radar")
        tracker.record_latency(2_000_000, 1_000_000, 500_000)

        snap = tracker.capture_snapshot()

        for key in (
            "t", "total_requests", "cache_hits", "cache_misses", "hit_rate",
            "fast_path_total", "tracked_tiles", "hot_tiles",
            "avg_request_ms", "avg_compute_ms", "avg_present_ms",
            "by_zoom", "top",
        ):
            assert key in snap
        assert snap["total_requests"] == 9
        assert snap["tracked_tiles"] == 2
        assert snap["hot_tiles"] == 1  # only the 6-hit tile clears 5
        assert snap["cache_hits"] == 1
        assert snap["cache_misses"] == 1
        assert snap["hit_rate"] == pytest.approx(0.5)
        assert snap["fast_path_total"] == 2
        assert snap["avg_request_ms"] == pytest.approx(2.0)
        assert snap["avg_compute_ms"] == pytest.approx(1.0)
        assert snap["avg_present_ms"] == pytest.approx(0.5)
        # by_zoom keys are strings, matching the /health JSON shape.
        assert snap["by_zoom"] == {
            "7": {"tiles": 1, "requests": 6},
            "8": {"tiles": 1, "requests": 3},
        }
        assert all(isinstance(z, str) for z in snap["by_zoom"])
        assert snap["top"][0] == {"z": 7, "x": 1, "y": 2, "count": 6}
        assert snap["top"][1] == {"z": 8, "x": 5, "y": 3, "count": 3}
        for entry in snap["top"]:
            assert set(entry) == {"z", "x", "y", "count"}

    def test_history_ring_is_bounded(self, monkeypatch):
        import librewxr.tiles.request_tracker as tracker_module

        class _Clock:
            def __init__(self):
                self.now = 1000

            def time(self):
                self.now += 1
                return self.now

        monkeypatch.setattr(tracker_module, "time", _Clock())

        tracker = TileRequestTracker(history_max_entries=5)
        for _ in range(10):
            tracker.capture_snapshot()

        hist = tracker.history()
        assert len(hist) == 5
        # Oldest five were evicted; 1006 is the first survivor.
        assert [snap["t"] for snap in hist] == [1006, 1007, 1008, 1009, 1010]

    def test_capture_snapshots_are_cumulative(self):
        tracker = TileRequestTracker(min_zoom=7)
        tracker.record(7, 1, 1)
        first = tracker.capture_snapshot()

        tracker.record(7, 1, 1)
        tracker.record_request(7, 2, 2, cache_hit=False)
        second = tracker.capture_snapshot()

        assert second["total_requests"] > first["total_requests"]
        assert second["cache_misses"] >= first["cache_misses"]
        assert second["tracked_tiles"] >= first["tracked_tiles"]
        # The ring holds both captures oldest-first.
        assert [snap["t"] for snap in tracker.history()] == [first["t"], second["t"]]
