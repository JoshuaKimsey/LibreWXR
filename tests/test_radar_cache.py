# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import json
from dataclasses import dataclass

import numpy as np
import pytest

pytestmark = pytest.mark.store

from librewxr.data.radar_cache import SCHEMA_VERSION, RadarFrameCache
from librewxr.data.store import RadarFrame


@dataclass
class FakeRegion:
    """Minimal RegionDef stand-in for cache shape validation."""
    height: int
    width: int


def _make_frame(ts: int, regions: dict[str, tuple[int, int]]) -> RadarFrame:
    """Build a frame with deterministic uint8 data per region."""
    return RadarFrame(
        timestamp=ts,
        regions={
            name: np.full(shape, ts % 256, dtype=np.uint8)
            for name, shape in regions.items()
        },
    )


class TestRadarFrameCache:
    def test_write_then_load_round_trip(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        regions = {"USCOMP": FakeRegion(10, 20), "AKCOMP": FakeRegion(5, 8)}

        frame = _make_frame(1000, {"USCOMP": (10, 20), "AKCOMP": (5, 8)})
        cache.write_frame(frame)
        cache.save_metadata(regions, [1000])

        loaded = cache.load_frames(regions)
        assert len(loaded) == 1
        assert loaded[0].timestamp == 1000
        assert set(loaded[0].regions) == {"USCOMP", "AKCOMP"}
        np.testing.assert_array_equal(loaded[0].regions["USCOMP"],
                                      frame.regions["USCOMP"])
        np.testing.assert_array_equal(loaded[0].regions["AKCOMP"],
                                      frame.regions["AKCOMP"])

    def test_schema_version_mismatch_invalidates_cache(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        regions = {"USCOMP": FakeRegion(10, 20)}

        cache.write_frame(_make_frame(1000, {"USCOMP": (10, 20)}))
        cache.save_metadata(regions, [1000])

        # Stomp the metadata to a different schema version.
        meta = json.loads((tmp_path / "radar" / "metadata.json").read_text())
        meta["schema_version"] = SCHEMA_VERSION + 99
        (tmp_path / "radar" / "metadata.json").write_text(json.dumps(meta))

        assert cache.load_frames(regions) == []

    def test_shape_mismatch_drops_only_that_region(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        old_regions = {"USCOMP": FakeRegion(10, 20), "AKCOMP": FakeRegion(5, 8)}

        cache.write_frame(_make_frame(1000, {"USCOMP": (10, 20), "AKCOMP": (5, 8)}))
        cache.save_metadata(old_regions, [1000])

        # Pretend USCOMP got reshaped in code; AKCOMP is unchanged.
        new_regions = {"USCOMP": FakeRegion(15, 25), "AKCOMP": FakeRegion(5, 8)}
        loaded = cache.load_frames(new_regions)
        assert len(loaded) == 1
        assert set(loaded[0].regions) == {"AKCOMP"}
        # The bad file should have been dropped.
        assert not (tmp_path / "radar" / "radar_1000_USCOMP.dat").exists()

    def test_cleanup_removes_inactive_timestamps(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        regions = {"USCOMP": FakeRegion(10, 20)}

        for ts in (1000, 2000, 3000):
            cache.write_frame(_make_frame(ts, {"USCOMP": (10, 20)}))

        cache.cleanup(active_timestamps=[2000, 3000])

        assert not (tmp_path / "radar" / "radar_1000_USCOMP.dat").exists()
        assert (tmp_path / "radar" / "radar_2000_USCOMP.dat").exists()
        assert (tmp_path / "radar" / "radar_3000_USCOMP.dat").exists()

    def test_cleanup_grace_retains_then_deletes(self, tmp_path, monkeypatch):
        """With grace_s an inactive ts is clocked, then reaped once elapsed.

        The published state.json still names a just-evicted timestamp, so
        deleting on first sight would break a consumer applying it.  The
        inactive frame carries TWO regions so the pass that reaps it must
        drop every sibling file at once, not one per cycle (the clock is
        dropped when the grace elapses; a sibling seen afterwards must not
        be mistaken for a first sighting and re-clock the timestamp).
        """
        from librewxr.data import radar_cache as radar_cache_module

        class _Clock:
            """Controllable stand-in for the ``time`` module (monotonic only)."""

            def __init__(self) -> None:
                self.now = 1000.0

            def monotonic(self) -> float:
                return self.now

        clock = _Clock()
        monkeypatch.setattr(radar_cache_module, "time", clock)

        cache = RadarFrameCache(tmp_path, grace_s=100.0)
        stale_regions = {"USCOMP": (10, 20), "SVCOMP": (30, 40)}
        cache.write_frame(_make_frame(1000, stale_regions))
        cache.write_frame(_make_frame(2000, {"USCOMP": (10, 20)}))
        stale_paths = [
            tmp_path / "radar" / f"radar_1000_{name}.dat"
            for name in stale_regions
        ]

        # First sight: clocked, not deleted.
        cache.cleanup(active_timestamps=[2000])
        assert all(path.exists() for path in stale_paths)
        assert cache._inactive_since == {1000: 1000.0}

        # Still inside the grace window: kept.
        clock.now += 50.0
        cache.cleanup(active_timestamps=[2000])
        assert all(path.exists() for path in stale_paths)

        # Past the grace: every region file of the timestamp goes in ONE
        # pass, and the clock is dropped.
        clock.now += 51.0
        cache.cleanup(active_timestamps=[2000])
        assert not any(path.exists() for path in stale_paths)
        assert cache._inactive_since == {}
        assert (tmp_path / "radar" / "radar_2000_USCOMP.dat").exists()

    def test_cleanup_drops_inactive_clock_when_ts_active_again(self, tmp_path):
        """A restored timestamp must not carry an eviction clock."""
        cache = RadarFrameCache(tmp_path, grace_s=100.0)
        for ts in (1000, 2000):
            cache.write_frame(_make_frame(ts, {"USCOMP": (10, 20)}))

        cache.cleanup(active_timestamps=[2000])
        assert 1000 in cache._inactive_since

        # 1000 comes back in the next cycle active set.
        cache.cleanup(active_timestamps=[1000, 2000])
        assert cache._inactive_since == {}
        assert (tmp_path / "radar" / "radar_1000_USCOMP.dat").exists()

    def test_load_with_empty_cache_returns_empty(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        assert cache.load_frames({"USCOMP": FakeRegion(10, 20)}) == []

    def test_load_recovers_orphan_files_when_metadata_missing(self, tmp_path):
        """SIGBUS or crash mid-backfill leaves .dat files but stale metadata —
        load_frames should disk-scan and recover them."""
        cache = RadarFrameCache(tmp_path)
        regions = {"USCOMP": FakeRegion(10, 20)}

        for ts in (1000, 2000, 3000):
            cache.write_frame(_make_frame(ts, {"USCOMP": (10, 20)}))
        # Note: no save_metadata call — simulates a crash before metadata write.

        loaded = cache.load_frames(regions)
        assert [f.timestamp for f in loaded] == [1000, 2000, 3000]
        for frame in loaded:
            assert set(frame.regions) == {"USCOMP"}

    def test_load_unions_metadata_and_disk_timestamps(self, tmp_path):
        """If metadata is partially stale (older save), disk-scanned orphans
        get added without losing the recorded ones."""
        cache = RadarFrameCache(tmp_path)
        regions = {"USCOMP": FakeRegion(10, 20)}

        cache.write_frame(_make_frame(1000, {"USCOMP": (10, 20)}))
        cache.save_metadata(regions, [1000])
        # New frame written but metadata not refreshed yet.
        cache.write_frame(_make_frame(2000, {"USCOMP": (10, 20)}))

        loaded = cache.load_frames(regions)
        assert [f.timestamp for f in loaded] == [1000, 2000]

    def test_partial_frame_when_only_some_regions_cached(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        # Write only USCOMP, but request both on load.
        regions = {"USCOMP": FakeRegion(10, 20)}
        cache.write_frame(_make_frame(1000, {"USCOMP": (10, 20)}))
        cache.save_metadata(regions, [1000])

        load_regions = {"USCOMP": FakeRegion(10, 20), "AKCOMP": FakeRegion(5, 8)}
        loaded = cache.load_frames(load_regions)
        assert len(loaded) == 1
        assert set(loaded[0].regions) == {"USCOMP"}

    def test_atomic_write_no_tmp_left_behind(self, tmp_path):
        cache = RadarFrameCache(tmp_path)
        cache.write_frame(_make_frame(1000, {"USCOMP": (10, 20)}))
        cache.save_metadata({"USCOMP": FakeRegion(10, 20)}, [1000])

        tmp_files = list((tmp_path / "radar").glob("*.tmp"))
        assert tmp_files == []
