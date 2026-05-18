# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio

import numpy as np
import pytest

pytestmark = pytest.mark.store

from librewxr.data.fetcher import RadarFetcher
from librewxr.data.radar_cache import RadarFrameCache
from librewxr.data.regions import RegionDef
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH


class TestFrameStore:
    @pytest.mark.asyncio
    async def test_add_and_get(self):
        store = FrameStore(max_frames=3)
        data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
        frame = RadarFrame(timestamp=100, regions={"USCOMP": data})
        await store.add_frame(frame)

        result = await store.get_frame(100)
        assert result is not None
        assert result.timestamp == 100

    @pytest.mark.asyncio
    async def test_eviction(self):
        store = FrameStore(max_frames=2)
        data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)

        await store.add_frame(RadarFrame(timestamp=100, regions={"USCOMP": data}))
        await store.add_frame(RadarFrame(timestamp=200, regions={"USCOMP": data}))
        evicted_ts, merged = await store.add_frame(RadarFrame(timestamp=300, regions={"USCOMP": data}))

        assert evicted_ts == 100
        assert merged is False
        assert await store.get_frame(100) is None
        assert await store.get_frame(200) is not None
        assert await store.get_frame(300) is not None

    @pytest.mark.asyncio
    async def test_duplicate_timestamp_merges_regions(self):
        store = FrameStore(max_frames=3)
        data1 = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
        data2 = np.ones((100, 100), dtype=np.uint8)

        _, merged1 = await store.add_frame(RadarFrame(timestamp=100, regions={"USCOMP": data1}))
        _, merged2 = await store.add_frame(RadarFrame(timestamp=100, regions={"AKCOMP": data2}))

        assert merged1 is False
        assert merged2 is True
        assert await store.frame_count() == 1
        frame = await store.get_frame(100)
        assert "USCOMP" in frame.regions
        assert "AKCOMP" in frame.regions

    @pytest.mark.asyncio
    async def test_sorted_order(self):
        store = FrameStore(max_frames=5)
        data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)

        await store.add_frame(RadarFrame(timestamp=300, regions={"USCOMP": data}))
        await store.add_frame(RadarFrame(timestamp=100, regions={"USCOMP": data}))
        await store.add_frame(RadarFrame(timestamp=200, regions={"USCOMP": data}))

        timestamps = await store.get_timestamps()
        assert timestamps == [100, 200, 300]

    @pytest.mark.asyncio
    async def test_get_latest(self):
        store = FrameStore(max_frames=5)
        data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)

        await store.add_frame(RadarFrame(timestamp=100, regions={"USCOMP": data}))
        await store.add_frame(RadarFrame(timestamp=300, regions={"USCOMP": data}))
        await store.add_frame(RadarFrame(timestamp=200, regions={"USCOMP": data}))

        latest = await store.get_latest_frame()
        assert latest.timestamp == 300


class TestTileCache:
    def test_put_and_get(self):
        cache = TileCache(max_mb=10)
        key = (100, 4, 3, 5, 256, 2, False, False, "png")
        cache.put(key, b"tile_data")
        assert cache.get(key) == b"tile_data"

    def test_byte_eviction(self):
        # Create a cache with a 10-byte limit
        cache = TileCache.__new__(TileCache)
        cache._max_bytes = 10
        cache._cache = __import__("collections").OrderedDict()
        cache._total_bytes = 0
        cache._lock = __import__("threading").Lock()

        k1 = (1,)
        k2 = (2,)
        k3 = (3,)
        cache.put(k1, b"12345")  # 5 bytes, total=5
        cache.put(k2, b"12345")  # 5 bytes, total=10
        cache.put(k3, b"12345")  # 5 bytes, would be 15 -> evicts k1, total=10

        assert cache.get(k1) is None  # evicted
        assert cache.get(k2) == b"12345"
        assert cache.get(k3) == b"12345"
        assert cache.total_bytes == 10

    def test_tracks_bytes(self):
        cache = TileCache(max_mb=10)
        cache.put((1,), b"hello")
        cache.put((2,), b"world!")
        assert cache.total_bytes == 11
        assert cache.size == 2

    def test_invalidate_timestamp(self):
        cache = TileCache(max_mb=10)
        cache.put((100, 4, 3, 5), b"a")
        cache.put((100, 4, 3, 6), b"b")
        cache.put((200, 4, 3, 5), b"c")

        cache.invalidate_timestamp(100)
        assert cache.get((100, 4, 3, 5)) is None
        assert cache.get((100, 4, 3, 6)) is None
        assert cache.get((200, 4, 3, 5)) == b"c"
        assert cache.total_bytes == 1

    def test_evict_half(self):
        cache = TileCache(max_mb=10)
        cache.put((1,), b"aaa")
        cache.put((2,), b"bbb")
        cache.put((3,), b"ccc")
        cache.put((4,), b"ddd")

        freed = cache.evict_half()
        assert freed == 6  # evicted 2 oldest entries (3 bytes each)
        assert cache.size == 2
        assert cache.total_bytes == 6
        assert cache.get((1,)) is None
        assert cache.get((2,)) is None
        assert cache.get((3,)) == b"ccc"
        assert cache.get((4,)) == b"ddd"


class _FakeSource:
    """Returns a deterministic uint8 array sized to the region grid."""

    def __init__(self):
        self.live_calls: list[tuple[str, int]] = []
        self.archive_calls: list[tuple[str, int]] = []

    async def fetch_frame(self, region, minutes_ago):
        self.live_calls.append((region.name, minutes_ago))
        return np.full((region.height, region.width), 50, dtype=np.uint8)

    async def fetch_archive_frame(self, region, dt):
        self.archive_calls.append((region.name, int(dt.timestamp())))
        return np.full((region.height, region.width), 50, dtype=np.uint8)


def _build_fetcher(store, tile_cache, radar_cache, region):
    """Bypass __init__ so we don't drag in real source dispatch / settings."""
    fetcher = RadarFetcher.__new__(RadarFetcher)
    fetcher._store = store
    fetcher._cache = tile_cache
    fetcher._ecmwf_grid = None
    fetcher._cloud_grid = None
    fetcher._nowcast_generator = None
    fetcher._warmer = None
    fetcher._radar_cache = radar_cache
    fetcher._task = None
    fetcher._cloud_task = None
    fetcher._warm_task = None
    fetcher._enabled_regions = [region]
    fetcher._na_source = "iem"
    fetcher._ca_source = "msc"
    source = _FakeSource()
    fetcher._sources = {region.name: source}
    fetcher._cacomp_msc_source = None
    fetcher._iem_fallback = None
    fetcher._cacomp_msc_available = None
    fetcher._on_cycle_complete = None
    return fetcher, source


class TestFetcherRadarCacheWiring:
    @pytest.fixture
    def small_region(self):
        # Explicit grid_width/height keeps arrays tiny so despeckle's
        # neighbor scan stays cheap even with the default min_neighbors=3.
        return RegionDef(
            name="TESTREG",
            west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="US",
            grid_width=32, grid_height=32,
        )

    @pytest.mark.asyncio
    async def test_fetcher_persists_frames_to_radar_cache(
        self, tmp_path, small_region
    ):
        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        radar_cache = RadarFrameCache(tmp_path)
        fetcher, source = _build_fetcher(store, tile_cache, radar_cache, small_region)

        await fetcher._fetch_timestamps([
            (1000, "live", 0),
            (2000, "live", 10),
        ])

        # .dat files should exist for both timestamps.
        assert (tmp_path / "radar" / "radar_1000_TESTREG.dat").exists()
        assert (tmp_path / "radar" / "radar_2000_TESTREG.dat").exists()
        # metadata.json should record both timestamps and the region shape.
        meta_path = tmp_path / "radar" / "metadata.json"
        assert meta_path.exists()
        import json
        meta = json.loads(meta_path.read_text())
        assert sorted(meta["timestamps"]) == [1000, 2000]
        assert meta["regions"]["TESTREG"]["shape"] == [32, 32]

    @pytest.mark.asyncio
    async def test_fetcher_cleanup_removes_evicted_timestamps(
        self, tmp_path, small_region
    ):
        # max_frames=2 forces the oldest timestamp to be evicted on the
        # third write; cache.cleanup should follow the store's lead and
        # delete the corresponding .dat file.
        store = FrameStore(max_frames=2)
        tile_cache = TileCache(max_mb=1)
        radar_cache = RadarFrameCache(tmp_path)
        fetcher, _source = _build_fetcher(store, tile_cache, radar_cache, small_region)

        await fetcher._fetch_timestamps([(1000, "live", 0)])
        await fetcher._fetch_timestamps([(2000, "live", 10)])
        await fetcher._fetch_timestamps([(3000, "live", 20)])

        # Store should hold only the newest two; cache should match.
        assert sorted(await store.get_timestamps()) == [2000, 3000]
        assert not (tmp_path / "radar" / "radar_1000_TESTREG.dat").exists()
        assert (tmp_path / "radar" / "radar_2000_TESTREG.dat").exists()
        assert (tmp_path / "radar" / "radar_3000_TESTREG.dat").exists()

    @pytest.mark.asyncio
    async def test_fetcher_without_radar_cache_does_not_crash(
        self, tmp_path, small_region
    ):
        # When cache_dir is unset in production, _radar_cache is None;
        # _fetch_timestamps should still drive the store cleanly.
        store = FrameStore(max_frames=2)
        tile_cache = TileCache(max_mb=1)
        fetcher, _source = _build_fetcher(store, tile_cache, None, small_region)

        await fetcher._fetch_timestamps([(1000, "live", 0)])
        assert await store.get_timestamps() == [1000]


class TestOnCycleCompleteHook:
    @pytest.fixture
    def small_region(self):
        return RegionDef(
            name="TESTREG",
            west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="US",
            grid_width=32, grid_height=32,
        )

    @pytest.mark.asyncio
    async def test_async_hook_runs_after_each_cycle(self, small_region):
        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        fetcher, _src = _build_fetcher(store, tile_cache, None, small_region)

        calls = 0

        async def hook():
            nonlocal calls
            calls += 1

        fetcher._on_cycle_complete = hook
        await fetcher._fire_cycle_complete()
        await fetcher._fire_cycle_complete()
        assert calls == 2

    @pytest.mark.asyncio
    async def test_sync_hook_supported(self, small_region):
        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        fetcher, _src = _build_fetcher(store, tile_cache, None, small_region)

        calls = 0

        def hook():
            nonlocal calls
            calls += 1

        fetcher._on_cycle_complete = hook
        await fetcher._fire_cycle_complete()
        assert calls == 1

    @pytest.mark.asyncio
    async def test_hook_failure_does_not_propagate(self, small_region):
        # A failed snapshot dump must never kill the fetcher loop.
        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        fetcher, _src = _build_fetcher(store, tile_cache, None, small_region)

        async def hook():
            raise RuntimeError("disk full")

        fetcher._on_cycle_complete = hook
        await fetcher._fire_cycle_complete()  # should not raise

    @pytest.mark.asyncio
    async def test_no_hook_is_silent(self, small_region):
        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        fetcher, _src = _build_fetcher(store, tile_cache, None, small_region)
        assert fetcher._on_cycle_complete is None
        await fetcher._fire_cycle_complete()  # should not raise

    @pytest.mark.asyncio
    async def test_constructor_accepts_hook_kwarg(self):
        # Smoke check that the public constructor accepts on_cycle_complete.
        # We bypass __init__ for the body of the test, but verify the
        # signature includes the kwarg so future refactors don't drop it.
        import inspect as _inspect
        sig = _inspect.signature(RadarFetcher.__init__)
        assert "on_cycle_complete" in sig.parameters
