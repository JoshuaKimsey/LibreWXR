# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the stitched global precip mask (PrecipMaskStore).

Covers ``_build_timestamp_mask_sync`` (projection, OR, dilation), the
public async ``build`` (timestamp union, NWP signature-gated cache,
nowcast folding), ``has_precip_in_bbox`` conservatism, the memmap state
round-trip, and stale-file cleanup.
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from librewxr.config import settings
from librewxr.data.regions import REGIONS
from librewxr.data.precip_mask import PrecipMaskStore

PIXEL = PrecipMaskStore.PIXEL_SIZE
WEST = PrecipMaskStore.WEST
NORTH = PrecipMaskStore.NORTH
GW = PrecipMaskStore.GRID_WIDTH
GH = PrecipMaskStore.GRID_HEIGHT

# A coarse cell inside USCOMP (CONUS): lat ~34.75, lon ~-94.75.
_CELL = (110, 170)
_FAR_CELL = (110, 500)  # lon 70 — Indian Ocean, outside every radar region
_TS = 1700000000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cell_bbox(row: int, col: int) -> tuple:
    """Bounding box of coarse cell (row, col): (west, south, east, north)."""
    west = WEST + col * PIXEL
    north = NORTH - row * PIXEL
    return (west, north - PIXEL, west + PIXEL, north)


def _meshgrid_latlon(row: int, col: int) -> tuple[float, float]:
    """The meshgrid lat/lon that ``build`` samples at coarse cell (row, col)."""
    lat = (NORTH - 0.125) - row * (2 * (NORTH - 0.125)) / (GH - 1)
    lon = (WEST + 0.125) + col * (2 * (-WEST - 0.125)) / (GW - 1)
    return lat, lon


def _placed_uscomp(row: int, col: int, value: int = 200) -> np.ndarray:
    """USCOMP array with ``value`` at the meshgrid sample point of (row, col).

    The pixel is placed exactly where ``_project_region`` samples it, so
    the only True coarse cell is (row, col) — everything else derives from
    the 1-cell dilation.
    """
    region = REGIONS["USCOMP"]
    lat, lon = _meshgrid_latlon(row, col)
    r = int(np.rint((region.north - lat) / region._ps_y))
    c = int(np.rint((lon - region.west) / region.pixel_size))
    arr = np.zeros((region.height, region.width), dtype=np.uint8)
    arr[r, c] = value
    return arr


def _empty_uscomp() -> np.ndarray:
    region = REGIONS["USCOMP"]
    return np.zeros((region.height, region.width), dtype=np.uint8)


class _FakeRadarFrame:
    def __init__(self, ts: int, regions: dict[str, np.ndarray]):
        self.timestamp = ts
        self.regions = regions


class _FakeFrameStore:
    """Async frame_store double: ts -> regions dict (or absent = no frame)."""

    def __init__(self, frames: dict[int, dict[str, np.ndarray]] | None = None):
        self._frames = dict(frames or {})

    async def get_timestamps(self) -> list[int]:
        return sorted(self._frames.keys())

    async def get_frame(self, ts: int):
        regions = self._frames.get(ts)
        if regions is None:
            return None
        return _FakeRadarFrame(ts, regions)


class _FakeNowcastStore:
    """Async nowcast_store double: ts -> (frame, blend)."""

    def __init__(self, frames: dict[int, dict[str, np.ndarray]] | None = None):
        self._frames = {
            ts: _FakeRadarFrame(ts, regions)
            for ts, regions in (frames or {}).items()
        }

    async def get_timestamps(self) -> list[int]:
        return sorted(self._frames.keys())

    async def get_frame(self, ts: int):
        frame = self._frames.get(ts)
        if frame is None:
            return None, 0.0
        return frame, 0.5


class _FakeNWPChain:
    """Sync nwp_chain double with a sample call counter."""

    def __init__(self, sample_fn=None, sources=None):
        self._sample_fn = sample_fn or (
            lambda lat, lon, ts, bilinear=False: np.zeros(lat.shape, dtype=np.uint8)
        )
        self.sources = sources or []
        self.calls = 0

    def has_data(self) -> bool:
        return True

    def sample(self, lat, lon, timestamp=None, bilinear=False) -> np.ndarray:
        self.calls += 1
        return self._sample_fn(lat, lon, timestamp, bilinear)


class _FakeSrc:
    """Minimal source for the NWP signature tests."""

    def __init__(self, name="fake", count=0, latest=None, reference_time=None):
        self.name = name
        self._sorted_timestamps = [0] * count
        self._latest_run_ts = latest
        if reference_time is not None:
            self.reference_time = reference_time
            self._timesteps = {i: None for i in range(count)}


def _nwp_with_cell(row: int, col: int, value: int = 200):
    """sample() returning a (360, 720) grid with ``value`` at (row, col)."""

    def _fn(lat, lon, ts, bilinear=False):
        arr = np.zeros((GH, GW), dtype=np.uint8)
        arr[row, col] = value
        return arr

    return _fn


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


class TestBasicBehavior:
    def test_empty_stores_conservative_true(self):
        store = PrecipMaskStore(cache_dir=None)
        asyncio.run(store.build({}, None, settings))
        assert store.has_precip_in_bbox(_TS, _cell_bbox(*_CELL)) is True

    async def test_unknown_timestamp_conservative_true(self):
        store = PrecipMaskStore(cache_dir=None)
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _placed_uscomp(*_CELL)}})
        await store.build({"frame_store": frame_store}, _FakeNWPChain(), settings)
        assert store.has_precip_in_bbox(_TS, _cell_bbox(*_CELL)) is True
        assert store.has_precip_in_bbox(_TS + 100, _cell_bbox(*_CELL)) is True

    async def test_populated_radar_mention_true_at_cell(self):
        store = PrecipMaskStore(cache_dir=None)
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _placed_uscomp(*_CELL)}})
        await store.build({"frame_store": frame_store}, _FakeNWPChain(), settings)
        assert store.has_precip_in_bbox(_TS, _cell_bbox(*_CELL)) is True
        assert store.has_precip_in_bbox(_TS, _cell_bbox(*_FAR_CELL)) is False

    async def test_no_precip_consistently_false(self):
        store = PrecipMaskStore(cache_dir=None)
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _empty_uscomp()}})
        await store.build({"frame_store": frame_store}, _FakeNWPChain(), settings)
        for cell in (_CELL, _FAR_CELL):
            assert store.has_precip_in_bbox(_TS, _cell_bbox(*cell)) is False

    async def test_dilation_by_one_cell(self):
        store = PrecipMaskStore(cache_dir=None)
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _placed_uscomp(*_CELL)}})
        await store.build({"frame_store": frame_store}, _FakeNWPChain(), settings)
        row, col = _CELL
        # Adjacent cell (east) is not directly hit but the 1-cell dilation
        # spills the True cell into it.
        assert store.has_precip_in_bbox(_TS, _cell_bbox(row, col + 1)) is True
        # Two cells away is beyond the dilation reach -> False.
        assert store.has_precip_in_bbox(_TS, _cell_bbox(row, col + 2)) is False

    async def test_antimeridian_wrap(self):
        store = PrecipMaskStore(cache_dir=None)
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _empty_uscomp()}})
        chain = _FakeNWPChain(sample_fn=_nwp_with_cell(150, GW - 1))
        await store.build({"frame_store": frame_store, "nowcast_store": None}, chain, settings)
        # Cell at column 719 is True directly.
        assert store.has_precip_in_bbox(_TS, _cell_bbox(150, GW - 1)) is True
        # Dilation wraps the antimeridian: column 0 picks up column 719.
        assert store.has_precip_in_bbox(_TS, _cell_bbox(150, 0)) is True

    async def test_antimeridian_straddling_bbox_conservative(self):
        store = PrecipMaskStore(cache_dir=None)
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _empty_uscomp()}})
        await store.build({"frame_store": frame_store}, _FakeNWPChain(), settings)
        # west > east (non-wrapped form) -> conservative True regardless
        # of the (all-False) mask contents.
        assert store.has_precip_in_bbox(_TS, (179.0, -10.0, -179.0, 10.0)) is True

    async def test_nwp_sample_contributes_to_mask(self):
        store = PrecipMaskStore(cache_dir=None)
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _empty_uscomp()}})
        chain = _FakeNWPChain(sample_fn=_nwp_with_cell(*_CELL))
        await store.build({"frame_store": frame_store}, chain, settings)
        assert store.has_precip_in_bbox(_TS, _cell_bbox(*_CELL)) is True
        assert store.has_precip_in_bbox(_TS, _cell_bbox(*_FAR_CELL)) is False

    async def test_nowcast_contributes_to_mask(self):
        store = PrecipMaskStore(cache_dir=None)
        # Radar and nowcast timestamps are disjoint in the pipeline (past
        # vs future); give the nowcast frame its own timestamp.
        frame_store = _FakeFrameStore({100: {"USCOMP": _empty_uscomp()}})
        nowcast_store = _FakeNowcastStore({200: {"USCOMP": _placed_uscomp(*_CELL)}})
        await store.build(
            {"frame_store": frame_store, "nowcast_store": nowcast_store},
            _FakeNWPChain(), settings,
        )
        assert store.has_precip_in_bbox(200, _cell_bbox(*_CELL)) is True
        assert store.has_precip_in_bbox(200, _cell_bbox(*_FAR_CELL)) is False

    async def test_multi_source_or_in_same_cell(self):
        store = PrecipMaskStore(cache_dir=None)
        # Radar at ts=100, nowcast at ts=200 (a timestamp has one owner),
        # NWP contributing to both — the OR is idempotent (bool mask).
        frame_store = _FakeFrameStore({100: {"USCOMP": _placed_uscomp(*_CELL)}})
        nowcast_store = _FakeNowcastStore({200: {"USCOMP": _placed_uscomp(*_CELL)}})
        chain = _FakeNWPChain(sample_fn=_nwp_with_cell(*_CELL))
        await store.build(
            {"frame_store": frame_store, "nowcast_store": nowcast_store},
            chain, settings,
        )
        assert store.has_precip_in_bbox(100, _cell_bbox(*_CELL)) is True
        assert store.has_precip_in_bbox(200, _cell_bbox(*_CELL)) is True
        assert store.has_precip_in_bbox(100, _cell_bbox(*_FAR_CELL)) is False

    async def test_cross_timestamp_no_bleed(self):
        store = PrecipMaskStore(cache_dir=None)
        frame_store = _FakeFrameStore({
            100: {"USCOMP": _placed_uscomp(*_CELL)},
            200: {"USCOMP": _empty_uscomp()},
        })
        await store.build({"frame_store": frame_store}, _FakeNWPChain(), settings)
        assert store.has_precip_in_bbox(100, _cell_bbox(*_CELL)) is True
        assert store.has_precip_in_bbox(100, _cell_bbox(*_FAR_CELL)) is False
        assert store.has_precip_in_bbox(200, _cell_bbox(*_CELL)) is False


# ---------------------------------------------------------------------------
# NWP cache signature gate
# ---------------------------------------------------------------------------


class TestNWPSignatureGate:
    async def test_nwp_signature_change_rebuilds(self):
        store = PrecipMaskStore(cache_dir=None)
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _empty_uscomp()}})
        ifs = _FakeSrc(
            name="ecmwf_ifs", count=3, latest=1000, reference_time="run-A",
        )
        chain = _FakeNWPChain(sources=[ifs])
        await store.build({"frame_store": frame_store}, chain, settings)
        calls_after_first = chain.calls
        assert calls_after_first == 1

        # Same signature -> cached masks reused.
        await store.build({"frame_store": frame_store}, chain, settings)
        assert chain.calls == calls_after_first

        # IFS reference_time changed -> signature differs -> re-sample.
        ifs.reference_time = "run-B"
        await store.build({"frame_store": frame_store}, chain, settings)
        assert chain.calls == calls_after_first + 1

    async def test_nwp_signature_match_reuses(self):
        store = PrecipMaskStore(cache_dir=None)
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _empty_uscomp()}})
        chain = _FakeNWPChain(sources=[_FakeSrc(count=2, latest=1000)])
        await store.build({"frame_store": frame_store}, chain, settings)
        calls_after_first = chain.calls
        assert calls_after_first == 1
        await store.build({"frame_store": frame_store}, chain, settings)
        assert chain.calls == calls_after_first


# ---------------------------------------------------------------------------
# State round-trip
# ---------------------------------------------------------------------------


class TestStateRoundTrip:
    def test_getstate_emits_string_keys_and_basenames(self, tmp_path):
        store = PrecipMaskStore(cache_dir=tmp_path / "cache")
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _placed_uscomp(*_CELL)}})
        asyncio.run(store.build({"frame_store": frame_store}, _FakeNWPChain(), settings))

        state = store.__getstate__()
        assert set(state["masks"].keys()) == {str(_TS)}
        assert state["masks"][str(_TS)] == [f"{_TS}.dat", "bool", [GH, GW]]
        assert state["version"] == 1

    def test_setstate_remmaps_and_query_works(self, tmp_path):
        producer = PrecipMaskStore(cache_dir=tmp_path / "cache")
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _placed_uscomp(*_CELL)}})
        asyncio.run(producer.build({"frame_store": frame_store}, _FakeNWPChain(), settings))

        # JSON round trip: keys stay strings, lists stay lists.
        state = json.loads(json.dumps(producer.__getstate__()))
        consumer = PrecipMaskStore.__new__(PrecipMaskStore)
        consumer.__setstate__(state)
        assert consumer.has_precip_in_bbox(_TS, _cell_bbox(*_CELL)) is True
        assert consumer.has_precip_in_bbox(_TS, _cell_bbox(*_FAR_CELL)) is False

    def test_setstate_handles_missing_file(self, tmp_path):
        producer = PrecipMaskStore(cache_dir=tmp_path / "cache")
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _placed_uscomp(*_CELL)}})
        asyncio.run(producer.build({"frame_store": frame_store}, _FakeNWPChain(), settings))

        state = producer.__getstate__()
        mask_file = tmp_path / "cache" / "mask" / f"{_TS}.dat"
        assert mask_file.exists()
        mask_file.unlink()

        consumer = PrecipMaskStore.__new__(PrecipMaskStore)
        consumer.__setstate__(state)
        assert consumer._masks == {}
        assert consumer.has_precip_in_bbox(_TS, _cell_bbox(*_CELL)) is True

    def test_setstate_backward_compat_no_masks_key(self):
        store = PrecipMaskStore.__new__(PrecipMaskStore)
        store.__setstate__({"version": 5})
        assert store._masks == {}
        assert store._version == 0
        assert store.has_precip_in_bbox(_TS, _cell_bbox(*_CELL)) is True

    def test_in_memory_mode_cache_dir_none(self):
        store = PrecipMaskStore(cache_dir=None)
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _placed_uscomp(*_CELL)}})
        asyncio.run(store.build({"frame_store": frame_store}, _FakeNWPChain(), settings))

        # Heap masks lack .filename -> not serializable; state is empty.
        state = store.__getstate__()
        assert state == {"version": 1, "masks": {}}
        # In-process heap lookup still answers.
        assert store.has_precip_in_bbox(_TS, _cell_bbox(*_CELL)) is True


# ---------------------------------------------------------------------------
# Stale memmap cleanup
# ---------------------------------------------------------------------------


class TestStaleCleanup:
    def test_cleanup_old_mask_files(self, tmp_path):
        store = PrecipMaskStore(cache_dir=tmp_path / "cache")
        mask_dir = tmp_path / "cache" / "mask"

        frame_store = _FakeFrameStore({
            100: {"USCOMP": _placed_uscomp(*_CELL)},
            200: {"USCOMP": _empty_uscomp()},
            300: {"USCOMP": _empty_uscomp()},
        })
        asyncio.run(store.build({"frame_store": frame_store}, _FakeNWPChain(), settings))
        assert (mask_dir / "100.dat").exists()
        assert (mask_dir / "200.dat").exists()
        assert (mask_dir / "300.dat").exists()

        # Second cycle drops ts=200, adds ts=400.
        frame_store = _FakeFrameStore({
            100: {"USCOMP": _placed_uscomp(*_CELL)},
            300: {"USCOMP": _empty_uscomp()},
            400: {"USCOMP": _empty_uscomp()},
        })
        asyncio.run(store.build({"frame_store": frame_store}, _FakeNWPChain(), settings))
        assert not (mask_dir / "200.dat").exists()
        assert (mask_dir / "100.dat").exists()
        assert (mask_dir / "300.dat").exists()
        assert (mask_dir / "400.dat").exists()


# ---------------------------------------------------------------------------
# Sync helper (no async scaffolding)
# ---------------------------------------------------------------------------


class TestSyncHelper:
    def test_build_timestamp_mask_sync_direct(self):
        store = PrecipMaskStore(cache_dir=None)
        threshold = int((settings.noise_floor_dbz + 32) * 2)
        region_arrays = {"USCOMP": _placed_uscomp(*_CELL)}
        nwp_mask = np.zeros((GH, GW), dtype=bool)
        mask = store._build_timestamp_mask_sync(_TS, region_arrays, nwp_mask, threshold)
        row, col = _CELL
        assert mask[row, col]
        assert mask[row, col + 1]  # dilated
        assert not mask[row, col + 2]  # beyond dilation

    def test_build_timestamp_mask_sync_nwp_or(self):
        store = PrecipMaskStore(cache_dir=None)
        threshold = int((settings.noise_floor_dbz + 32) * 2)
        nwp_mask = np.zeros((GH, GW), dtype=bool)
        nwp_mask[300, 100] = True
        mask = store._build_timestamp_mask_sync(_TS, {}, nwp_mask, threshold)
        assert mask[300, 100]
        assert mask[300, 101]  # dilated east
        assert not mask[300, 102]
