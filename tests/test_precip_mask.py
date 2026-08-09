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
import os
import re
from pathlib import Path

import numpy as np
import pytest

from librewxr.config import settings
from librewxr.data.regions import REGIONS, RegionDef
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
    lat = NORTH - (row + 0.5) * PIXEL
    lon = WEST + (col + 0.5) * PIXEL
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
    """sample() returning a fine (2*GH, 2*GW) grid with ``value`` at the
    fine cell (2*row, 2*col) inside coarse cell (row, col)."""

    def _fn(lat, lon, ts, bilinear=False):
        arr = np.zeros((2 * GH, 2 * GW), dtype=np.uint8)
        arr[2 * row, 2 * col] = value
        return arr

    return _fn


def _nwp_fine_cell(fine_row: int, fine_col: int, value: int = 200):
    """sample() returning a fine (2*GH, 2*GW) grid with ``value`` at exactly
    one fine cell (``fine_row``, ``fine_col``)."""

    def _fn(lat, lon, ts, bilinear=False):
        arr = np.zeros((2 * GH, 2 * GW), dtype=np.uint8)
        arr[fine_row, fine_col] = value
        return arr

    return _fn


# ---------------------------------------------------------------------------
# Meshgrid alignment
# ---------------------------------------------------------------------------


class TestMeshgrid:
    def test_centers_align_exactly_to_half_cell_offsets(self):
        store = PrecipMaskStore(cache_dir=None)
        store._ensure_meshgrid()
        lat_grid, lon_grid = store._latlon_meshgrid
        rows = np.arange(GH)
        cols = np.arange(GW)
        # Center of coarse cell (r, c): lat 90-(r+0.5)*0.5, lon -180+(c+0.5)*0.5.
        np.testing.assert_allclose(
            lat_grid[:, 0], NORTH - (rows + 0.5) * PIXEL, rtol=0, atol=1e-6,
        )
        np.testing.assert_allclose(
            lon_grid[0, :], WEST + (cols + 0.5) * PIXEL, rtol=0, atol=1e-6,
        )
        # Exactly 0.5-deg spacing — the old 0.125 offset drifted up to
        # 0.125 deg from the bucket math in ``has_precip_in_bbox``.
        np.testing.assert_allclose(
            np.diff(lat_grid[:, 0]), np.full(GH - 1, -PIXEL), rtol=0, atol=1e-6,
        )
        np.testing.assert_allclose(
            np.diff(lon_grid[0, :]), np.full(GW - 1, PIXEL), rtol=0, atol=1e-6,
        )

    def test_nwp_meshgrid_is_exactly_2x_fine_per_coarse_axis(self):
        store = PrecipMaskStore(cache_dir=None)
        store._ensure_meshgrid()
        store._ensure_nwp_meshgrid()
        lat_grid, lon_grid = store._latlon_meshgrid
        fine_lat, fine_lon = store._nwp_meshgrid
        assert fine_lat.shape == (2 * GH, 2 * GW)
        assert fine_lon.shape == (2 * GH, 2 * GW)
        # Fine centers flank each coarse center at +/- 0.125 deg.
        r, c = _CELL
        np.testing.assert_allclose(
            fine_lat[2 * r], lat_grid[r, 0] + 0.125, rtol=0, atol=1e-6,
        )
        np.testing.assert_allclose(
            fine_lat[2 * r + 1], lat_grid[r, 0] - 0.125, rtol=0, atol=1e-6,
        )
        np.testing.assert_allclose(
            fine_lon[0, 2 * c], lon_grid[0, c] - 0.125, rtol=0, atol=1e-6,
        )
        np.testing.assert_allclose(
            fine_lon[0, 2 * c + 1], lon_grid[0, c] + 0.125, rtol=0, atol=1e-6,
        )


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
# Area-conservative projection (regression)
# ---------------------------------------------------------------------------


class TestAreaConservativeProjection:
    @pytest.fixture
    def corner_blob_region(self, monkeypatch):
        """Register a small synthetic latlon region; return (region, blob px).

        The blob is planted on the coarse-cell corner at lat 32.0 / lon
        -88.0 (between coarse rows 115/116 and cols 183/184).
        """
        region = RegionDef(
            name="TESTCORNER",
            west=-100.0, east=-70.0, south=20.0, north=40.0,
            pixel_size=0.01, group="TEST",
        )
        monkeypatch.setitem(REGIONS, "TESTCORNER", region)
        rp = int(np.rint((region.north - 32.0) / region._ps_y))
        cp = int(np.rint((-88.0 - region.west) / region.pixel_size))
        return region, (rp, cp)

    async def test_blob_between_coarse_cell_centers_is_captured(
        self, corner_blob_region,
    ):
        """Regression: a tiny blob straddling a coarse cell corner must trip the gate.

        The old point-sampling read the region array at coarse cell
        centers; the blob sits >= 0.25 deg (25 region pixels) from every
        surrounding center, so all four covering cells came out False and
        high-zoom tiles over the blob rendered transparent.  The
        dilate-then-sample projection marks every coarse cell that
        contains a blob pixel.
        """
        store = PrecipMaskStore(cache_dir=None)
        region, (rp, cp) = corner_blob_region
        arr = np.zeros((region.height, region.width), dtype=np.uint8)
        arr[rp - 1:rp + 2, cp - 1:cp + 2] = 200
        frame_store = _FakeFrameStore({_TS: {"TESTCORNER": arr}})
        await store.build({"frame_store": frame_store}, _FakeNWPChain(), settings)

        # All four coarse cells touching the corner contain blob pixels.
        for row in (115, 116):
            for col in (183, 184):
                assert store.has_precip_in_bbox(_TS, _cell_bbox(row, col)) is True
        # A small high-zoom-style bbox right over the blob trips the gate.
        assert store.has_precip_in_bbox(_TS, (-88.05, 31.95, -87.95, 32.05)) is True
        # Control: a cell far outside the region stays False.
        assert store.has_precip_in_bbox(_TS, _cell_bbox(*_FAR_CELL)) is False

    async def test_region_smaller_than_a_coarse_cell_falls_back_conservative(
        self, monkeypatch,
    ):
        """A region narrower than one coarse cell marks every in-bounds cell."""
        region = RegionDef(
            name="TINY",
            west=-94.5, east=-94.0, south=34.0, north=34.5,
            pixel_size=0.01, group="TEST",
        )
        monkeypatch.setitem(REGIONS, "TINY", region)
        # Tiny region: lat 34..34.5, lon -94.5..-94.0 straddles the corner
        # of coarse cells (111, 171) / (111, 172) / (112, 171) / (112, 172)
        # (corner lat 34.0 = row boundary 111/112, lon -94.0 = col 171/172).
        arr = np.zeros((region.height, region.width), dtype=np.uint8)
        arr[region.height // 2, region.width // 2] = 200
        store = PrecipMaskStore(cache_dir=None)
        frame_store = _FakeFrameStore({_TS: {"TINY": arr}})
        await store.build({"frame_store": frame_store}, _FakeNWPChain(), settings)
        # Conservative fallback marks the in-bounds cells (dilation spreads
        # the hit to the full 2x2 corner neighbourhood).
        for row in (111, 112):
            for col in (171, 172):
                assert store.has_precip_in_bbox(_TS, _cell_bbox(row, col)) is True


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
# NWP 0.25-deg supersample (max-pool)
# ---------------------------------------------------------------------------


class TestNWPSupersample:
    async def test_single_fine_cell_pools_to_its_coarse_cell_only(self):
        store = PrecipMaskStore(cache_dir=None)
        frame_store = _FakeFrameStore({_TS: {"USCOMP": _empty_uscomp()}})
        row, col = _CELL
        # Hit in exactly one fine cell — the bottom-right fine cell of
        # coarse cell (row, col), 0.125 deg from the coarse center.  The
        # 2x2 max-pool must still land it in (row, col).
        chain = _FakeNWPChain(sample_fn=_nwp_fine_cell(2 * row + 1, 2 * col + 1))
        await store.build({"frame_store": frame_store}, chain, settings)
        # Pre-dilation pooled NWP mask (what ``_nwp_cache`` holds) has
        # exactly the one covering coarse cell True.
        coarse = store._nwp_cache[_TS]
        assert coarse.shape == (GH, GW)
        assert coarse[row, col]
        assert int(coarse.sum()) == 1
        # The built (1-cell-dilated) mask still answers the gate.
        assert store.has_precip_in_bbox(_TS, _cell_bbox(row, col)) is True
        assert store.has_precip_in_bbox(_TS, _cell_bbox(*_FAR_CELL)) is False


# ---------------------------------------------------------------------------
# Incremental NWP cache
# ---------------------------------------------------------------------------


class TestNWPCacheIncremental:
    async def test_unchanged_signature_samples_only_new_timestamp(self):
        store = PrecipMaskStore(cache_dir=None)
        chain = _FakeNWPChain(sources=[_FakeSrc(count=1, latest=1000)])
        frame_store = _FakeFrameStore({100: {"USCOMP": _empty_uscomp()}})
        await store.build({"frame_store": frame_store}, chain, settings)
        assert chain.calls == 1

        # Same signature, one new timestamp -> only it is sampled.
        frame_store = _FakeFrameStore({
            100: {"USCOMP": _empty_uscomp()},
            200: {"USCOMP": _empty_uscomp()},
        })
        await store.build({"frame_store": frame_store}, chain, settings)
        assert chain.calls == 2
        assert set(store._nwp_cache) == {100, 200}

        # A dropped timestamp is evicted from the cache without sampling.
        frame_store = _FakeFrameStore({100: {"USCOMP": _empty_uscomp()}})
        await store.build({"frame_store": frame_store}, chain, settings)
        assert chain.calls == 2
        assert set(store._nwp_cache) == {100}

    async def test_changed_signature_resamples_all_timestamps(self):
        store = PrecipMaskStore(cache_dir=None)
        src = _FakeSrc(count=1, latest=1000)
        chain = _FakeNWPChain(sources=[src])
        frame_store = _FakeFrameStore({
            100: {"USCOMP": _empty_uscomp()},
            200: {"USCOMP": _empty_uscomp()},
        })
        await store.build({"frame_store": frame_store}, chain, settings)
        assert chain.calls == 2

        # Signature change -> full rebuild of every timestamp.
        src._latest_run_ts = 2000
        await store.build({"frame_store": frame_store}, chain, settings)
        assert chain.calls == 4


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
# Cross-process tmp-file isolation on the shared mask dir
# ---------------------------------------------------------------------------


class TestSaveMaskTmpIsolation:
    """Unique (pid+uuid) tmp names in ``_save_mask``.

    The pipeline is the only writer of ``mask/*.dat`` files, but two
    overlapping pipeline processes during a deploy both build the same
    timestamps - the deterministic-tmp race NowcastStore hit in
    production.  pid+uuid keeps writers independent; the last rename
    wins the final name atomically.
    """

    def test_save_mask_tmp_names_are_unique(self, tmp_path, monkeypatch):
        """Two writes to the same final name must never share a tmp file.

        The tmp name embeds pid + uuid (mirroring ``coord_store.publish``),
        so concurrent writers can't collide on the same ``.tmp`` path and
        steal each other's in-flight file.  The pre-fix deterministic
        ``<ts>.dat.tmp`` produced identical paths; this test must fail
        against that code.
        """
        store = PrecipMaskStore(cache_dir=tmp_path / "cache")
        replaced_srcs: list[str] = []

        real_replace = os.replace

        def _capture_replace(src, dst):
            replaced_srcs.append(str(src))
            return real_replace(src, dst)

        monkeypatch.setattr(
            "librewxr.data.precip_mask.os.replace", _capture_replace,
        )
        mask = np.zeros((GH, GW), dtype=bool)
        mask[_CELL] = True
        store._save_mask({}, _TS, mask)
        store._save_mask({}, _TS, mask)

        assert len(replaced_srcs) == 2
        # pid + uuid naming (mirrors coord_store.publish), distinct per
        # write.  The old ``1700000000.dat.tmp`` fails the regex AND
        # produces two identical paths.
        pattern = re.compile(rf"^{_TS}\.dat\.\d+\.[0-9a-f]{{32}}\.tmp$")
        assert all(pattern.match(Path(name).name) for name in replaced_srcs)
        assert replaced_srcs[0] != replaced_srcs[1]


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
