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