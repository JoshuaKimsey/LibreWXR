# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio

import numpy as np
import pytest

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
        evicted_ts = await store.add_frame(RadarFrame(timestamp=300, regions={"USCOMP": data}))

        assert evicted_ts == 100
        assert await store.get_frame(100) is None
        assert await store.get_frame(200) is not None
        assert await store.get_frame(300) is not None

    @pytest.mark.asyncio
    async def test_duplicate_timestamp_merges_regions(self):
        store = FrameStore(max_frames=3)
        data1 = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
        data2 = np.ones((100, 100), dtype=np.uint8)

        await store.add_frame(RadarFrame(timestamp=100, regions={"USCOMP": data1}))
        await store.add_frame(RadarFrame(timestamp=100, regions={"AKCOMP": data2}))

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
