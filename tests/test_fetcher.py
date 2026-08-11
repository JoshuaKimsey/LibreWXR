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
        # The timestamp index is initialized in __init__; a __new__
        # instance must wire it by hand (see tiles/cache.py).
        cache._by_ts = {}

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
    """Returns a deterministic uint8 array sized to the region grid.

    ``fill_value`` is configurable so tests can produce different data
    on different calls (proving carry-forward really copies prior data
    rather than fetching fresh).  Set ``next_return = None`` to
    simulate a silent drop on the next call.
    """

    def __init__(self, fill_value: int = 50):
        self.live_calls: list[tuple[str, int]] = []
        self.archive_calls: list[tuple[str, int]] = []
        self.fill_value = fill_value
        self.next_return: object = ...  # ... = use default array

    def _build_array(self, region):
        return np.full(
            (region.height, region.width), self.fill_value, dtype=np.uint8,
        )

    async def fetch_frame(self, region, minutes_ago):
        self.live_calls.append((region.name, minutes_ago))
        if self.next_return is not ...:
            val = self.next_return
            self.next_return = ...
            return val
        return self._build_array(region)

    async def fetch_archive_frame(self, region, dt):
        self.archive_calls.append((region.name, int(dt.timestamp())))
        if self.next_return is not ...:
            val = self.next_return
            self.next_return = ...
            return val
        return self._build_array(region)


def _build_fetcher(store, tile_cache, radar_cache, region):
    """Bypass __init__ so we don't drag in real source dispatch / settings."""
    fetcher = RadarFetcher.__new__(RadarFetcher)
    fetcher._store = store
    fetcher._cache = tile_cache
    fetcher._nwp_contributions = []
    fetcher._satellite_contributions = []
    fetcher._satellite_tasks = {}
    fetcher._carried_regions = {}
    fetcher._nowcast_generator = None
    fetcher._warmer = None
    fetcher._radar_cache = radar_cache
    fetcher._task = None
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


class TestCarryForward:
    """When a fetch returns no data for an enabled region, the fetcher
    fills the new frame from the most recent prior frame in the store
    (up to ``_CARRY_FORWARD_MAX_INTERVALS`` lookback).  Users see
    continuous radar instead of intermittent NWP-fallback flicker.
    """

    @pytest.fixture
    def small_region(self):
        return RegionDef(
            name="TESTREG",
            west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="US",
            grid_width=32, grid_height=32,
        )

    @pytest.mark.asyncio
    async def test_silent_drop_carries_forward_from_prev_frame(
        self, small_region,
    ):
        # settings.fetch_interval defaults to 600 — match that so the
        # lookback math (ts - N*interval) lines up with our test ts.
        from librewxr.config import settings
        interval = settings.fetch_interval

        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        fetcher, source = _build_fetcher(store, tile_cache, None, small_region)

        # First fetch lands a real frame at ts=1000.
        source.fill_value = 77
        await fetcher._fetch_timestamps([(1000, "live", 0)])
        assert (await store.get_frame(1000)).regions["TESTREG"][0, 0] == 77

        # Second fetch: source returns None (simulating a silent drop).
        source.fill_value = 99  # any later real value should NOT appear
        source.next_return = None
        await fetcher._fetch_timestamps([(1000 + interval, "live", 10)])

        # The new frame must exist AND contain the carried-forward data
        # from ts=1000 — value 77, not the 99 default.
        new_frame = await store.get_frame(1000 + interval)
        assert new_frame is not None
        assert "TESTREG" in new_frame.regions
        assert new_frame.regions["TESTREG"][0, 0] == 77

    @pytest.mark.asyncio
    async def test_carry_forward_respects_staleness_limit(
        self, small_region,
    ):
        """If the only prior frame is more than _CARRY_FORWARD_MAX_INTERVALS
        old, no carry-forward happens — the region drops cleanly."""
        from librewxr.config import settings
        interval = settings.fetch_interval

        store = FrameStore(max_frames=8)
        tile_cache = TileCache(max_mb=1)
        fetcher, source = _build_fetcher(store, tile_cache, None, small_region)

        # Anchor frame at ts=1000.
        source.fill_value = 77
        await fetcher._fetch_timestamps([(1000, "live", 0)])

        # Silent drop 3 intervals later — past the 2-interval limit.
        source.next_return = None
        far_ts = 1000 + 3 * interval
        await fetcher._fetch_timestamps([(far_ts, "live", 30)])

        # The far frame should NOT contain TESTREG — the stale data is
        # too old to carry forward.  Either the frame doesn't exist or
        # the region key is absent.
        far_frame = await store.get_frame(far_ts)
        if far_frame is not None:
            assert "TESTREG" not in far_frame.regions

    @pytest.mark.asyncio
    async def test_successful_refetch_overrides_carried_data(
        self, small_region,
    ):
        """A later real fetch for the same ts replaces the carry-forward
        copy via the FrameStore's merge-on-duplicate-ts behaviour."""
        from librewxr.config import settings
        interval = settings.fetch_interval

        store = FrameStore(max_frames=4)
        tile_cache = TileCache(max_mb=1)
        fetcher, source = _build_fetcher(store, tile_cache, None, small_region)

        # Establish prev frame, then carry-forward into a dropped ts.
        source.fill_value = 77
        await fetcher._fetch_timestamps([(1000, "live", 0)])

        source.next_return = None
        await fetcher._fetch_timestamps([(1000 + interval, "live", 10)])
        assert (await store.get_frame(1000 + interval)).regions["TESTREG"][0, 0] == 77

        # Re-fetch the same ts with real data — should override.
        source.fill_value = 111
        await fetcher._fetch_timestamps([(1000 + interval, "live", 10)])
        assert (await store.get_frame(1000 + interval)).regions["TESTREG"][0, 0] == 111

    @pytest.mark.asyncio
    async def test_carry_forward_is_independent_copy(
        self, small_region,
    ):
        """The carried-forward array must be detached from the source
        frame's memmap, so eviction of the source can't invalidate it."""
        from librewxr.config import settings
        interval = settings.fetch_interval

        # max_frames=2 forces the original ts=1000 frame to be evicted
        # once a third timestamp is written.
        store = FrameStore(max_frames=2)
        tile_cache = TileCache(max_mb=1)
        fetcher, source = _build_fetcher(store, tile_cache, None, small_region)

        source.fill_value = 77
        await fetcher._fetch_timestamps([(1000, "live", 0)])

        # Carry-forward into ts=1000+interval.
        source.next_return = None
        await fetcher._fetch_timestamps([(1000 + interval, "live", 10)])

        # Push a third timestamp — evicts ts=1000.  The carried-forward
        # data at ts=1000+interval should survive because it was copied,
        # not memmap-shared.
        source.fill_value = 200
        await fetcher._fetch_timestamps([(1000 + 2 * interval, "live", 20)])

        carried = (await store.get_frame(1000 + interval)).regions["TESTREG"]
        assert carried[0, 0] == 77  # still readable, original value


class TestCarryForwardRefetch:
    """A carried-forward region must not permanently mask the slot.

    Sources like Météo-France serve missed slots from a short archive
    window (the DPPaquetRadar packet, ~15 min).  When a live fetch
    fails and the slot is bridged by carry-forward, later cycles must
    re-fetch that slot while it is still recoverable — otherwise the
    stale duplicate frame is frozen into the timeline forever (visible
    as a stutter/gap in the animation).
    """

    @pytest.fixture
    def small_region(self):
        return RegionDef(
            name="TESTREG",
            west=0.0, east=3.2, south=0.0, north=3.2,
            pixel_size=0.1, group="US",
            grid_width=32, grid_height=32,
        )

    def _clocked_fetcher(self, monkeypatch, small_region, max_frames=3):
        from librewxr.config import settings
        interval = settings.fetch_interval

        monkeypatch.setattr(settings, "max_frames", max_frames)
        clock = {"now": float(1000 * interval)}
        monkeypatch.setattr(
            "librewxr.data.fetcher.time.time", lambda: clock["now"],
        )

        store = FrameStore(max_frames=max_frames + 4)
        tile_cache = TileCache(max_mb=1)
        fetcher, source = _build_fetcher(
            store, tile_cache, None, small_region,
        )
        return fetcher, source, store, clock, interval

    @pytest.mark.asyncio
    async def test_carried_region_is_refetched_on_next_cycle(
        self, monkeypatch, small_region,
    ):
        fetcher, source, store, clock, interval = self._clocked_fetcher(
            monkeypatch, small_region,
        )
        ts0 = int(clock["now"])

        # Cycle 1: healthy — three timestamps land with value 77.
        source.fill_value = 77
        await fetcher._fetch_all_frames()

        # Cycle 2: live slot drops silently -> carry-forward from ts0.
        clock["now"] += interval
        dropped_ts = int(clock["now"])
        source.next_return = None
        source.fill_value = 99
        await fetcher._fetch_all_frames()
        assert (await store.get_frame(dropped_ts)).regions["TESTREG"][0, 0] == 77

        # Cycle 3: the source can now serve the slot.  The fetcher must
        # re-fetch the carried slot instead of treating it as complete.
        clock["now"] += interval
        source.fill_value = 111
        await fetcher._fetch_all_frames()
        assert (await store.get_frame(dropped_ts)).regions["TESTREG"][0, 0] == 111

    @pytest.mark.asyncio
    async def test_successful_refetch_clears_the_retry(
        self, monkeypatch, small_region,
    ):
        """Once real data replaces the carried copy, later cycles skip
        the slot again — no endless re-downloads."""
        fetcher, source, store, clock, interval = self._clocked_fetcher(
            monkeypatch, small_region,
        )

        source.fill_value = 77
        await fetcher._fetch_all_frames()

        clock["now"] += interval
        source.next_return = None
        await fetcher._fetch_all_frames()

        clock["now"] += interval
        source.fill_value = 111
        await fetcher._fetch_all_frames()

        # Cycle 4: only the brand-new live slot should be fetched.
        clock["now"] += interval
        n_before = len(source.live_calls)
        await fetcher._fetch_all_frames()
        assert source.live_calls[n_before:] == [("TESTREG", 0)]

    @pytest.mark.asyncio
    async def test_refetch_stops_outside_recovery_window(
        self, monkeypatch, small_region,
    ):
        """A carried slot older than the recovery window is left alone —
        the source archive can no longer serve it, so retrying is waste."""
        fetcher, source, store, clock, interval = self._clocked_fetcher(
            monkeypatch, small_region, max_frames=8,
        )

        source.fill_value = 77
        await fetcher._fetch_all_frames()

        clock["now"] += interval
        stale_ts = int(clock["now"])
        source.next_return = None
        await fetcher._fetch_all_frames()

        # Jump past the recovery window in one leap.  The intervening
        # slots are brand-new fetches; the carried slot must NOT be
        # among the fetched timestamps.
        leap = RadarFetcher._CARRY_FORWARD_REFETCH_MAX_INTERVALS + 1
        clock["now"] += leap * interval
        source.fill_value = 111
        n_before = len(source.live_calls) + len(source.archive_calls)
        await fetcher._fetch_all_frames()

        new_live = source.live_calls[n_before:]
        fetched_ts = {
            int(clock["now"]) - m * 60 for _, m in new_live
        }
        assert stale_ts not in fetched_ts
        assert (await store.get_frame(stale_ts)).regions["TESTREG"][0, 0] == 77

    @pytest.mark.asyncio
    async def test_failed_refetch_leaves_frame_untouched_and_retries(
        self, monkeypatch, small_region,
    ):
        """A refetch that fails again must not rewrite the carried frame
        (no version bump / tile-cache churn) and must stay eligible for
        retry on the following cycle."""
        fetcher, source, store, clock, interval = self._clocked_fetcher(
            monkeypatch, small_region,
        )

        source.fill_value = 77
        await fetcher._fetch_all_frames()

        clock["now"] += interval
        dropped_ts = int(clock["now"])
        source.next_return = None
        await fetcher._fetch_all_frames()
        version_after_carry = store._frame_versions[dropped_ts]

        # Cycle 3: everything fails (new slot AND the refetch).
        async def always_none(region, minutes_ago):
            source.live_calls.append((region.name, minutes_ago))
            return None

        real_fetch = source.fetch_frame
        source.fetch_frame = always_none
        clock["now"] += interval
        await fetcher._fetch_all_frames()

        assert (await store.get_frame(dropped_ts)).regions["TESTREG"][0, 0] == 77
        assert store._frame_versions[dropped_ts] == version_after_carry

        # Cycle 4: source recovers — the slot is still retried and healed.
        source.fetch_frame = real_fetch
        source.fill_value = 111
        clock["now"] += interval
        await fetcher._fetch_all_frames()
        assert (await store.get_frame(dropped_ts)).regions["TESTREG"][0, 0] == 111


class _StubSatelliteInstance:
    """Satellite source stand-in exposing only what the background task uses."""

    def __init__(self, behavior):
        self._behavior = behavior

    async def fetch(self) -> bool:
        return await self._behavior()


class _StubSatelliteContribution:
    def __init__(self, behavior, name="GMGSI LW"):
        self.name = name
        self.instance = _StubSatelliteInstance(behavior)


def _build_satellite_fetcher():
    """Bare fetcher with just the state _fetch_satellite_background touches."""
    fetcher = RadarFetcher.__new__(RadarFetcher)
    fetcher._on_cycle_complete = None
    fetcher._satellite_tasks = {}
    return fetcher


class TestSatelliteBackgroundFetch:
    async def test_hung_fetch_times_out_and_frees_the_skip_gate(
        self, monkeypatch, caplog
    ):
        """A fetch that hangs must finish (via deadline) so later cycles retry.

        The scheduler skips a channel while its previous task is pending;
        before the deadline existed, one hung S3 call froze the channel
        until restart with nothing logged above DEBUG.
        """
        from librewxr.config import settings

        monkeypatch.setattr(settings, "satellite_fetch_timeout", 0.05)
        fetcher = _build_satellite_fetcher()

        async def hang() -> bool:
            await asyncio.sleep(30)
            return True

        contrib = _StubSatelliteContribution(hang)
        task = asyncio.create_task(fetcher._fetch_satellite_background(contrib))
        with caplog.at_level("WARNING"):
            await asyncio.wait_for(task, timeout=5)  # must not take ~30s

        assert task.done()
        assert any("timed out" in r.message for r in caplog.records)

    async def test_successful_fetch_fires_cycle_complete(self):
        fetcher = _build_satellite_fetcher()
        fired = asyncio.Event()

        async def on_complete() -> None:
            fired.set()

        fetcher._on_cycle_complete = on_complete

        async def ok() -> bool:
            return True

        await fetcher._fetch_satellite_background(_StubSatelliteContribution(ok))
        assert fired.is_set()

    async def test_failed_fetch_is_dropped_with_a_warning(self, caplog):
        fetcher = _build_satellite_fetcher()

        async def boom() -> bool:
            raise RuntimeError("s3 exploded")

        with caplog.at_level("WARNING"):
            await fetcher._fetch_satellite_background(_StubSatelliteContribution(boom))
        assert any("fetch failed" in r.message for r in caplog.records)
