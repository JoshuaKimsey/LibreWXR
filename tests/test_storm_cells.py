# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for storm-cell detection, StormCellStore state round-trip, and
StormCellGenerator orchestration."""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from librewxr.data.regions import RegionDef
from librewxr.data.storm_cells import (
    MAX_CELLS_PER_REGION,
    _CELL_DTYPE,
    StormCellGenerator,
    StormCellStore,
    detect_storm_cells,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Default pixel_size = 0.01 deg => 1 pixel ~ 1.2321 km^2
_DEFAULT_PS = 0.01
_DEFAULT_PS_LON_KM = _DEFAULT_PS * 111.0
_DEFAULT_PS_LAT_KM = _DEFAULT_PS * 111.0
_DEFAULT_PX_TO_KM2 = _DEFAULT_PS_LON_KM * _DEFAULT_PS_LAT_KM  # ~1.2321


def _make_region(
    name: str = "SYNTH_REGION",
    pixel_size: float = _DEFAULT_PS,
) -> RegionDef:
    return RegionDef(
        name=name,
        west=0.0, east=10.0, south=0.0, north=10.0,
        pixel_size=pixel_size, group="TEST",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def syn_regions(monkeypatch):
    """Add two synthetic test regions to REGIONS and remove after the test."""
    from librewxr.data import regions as _regions_mod
    r1 = _make_region("SYNTH_REGION")
    r2 = _make_region("SYNTH_REGION_2")
    monkeypatch.setitem(_regions_mod.REGIONS, "SYNTH_REGION", r1)
    monkeypatch.setitem(_regions_mod.REGIONS, "SYNTH_REGION_2", r2)
    return {"SYNTH_REGION": r1, "SYNTH_REGION_2": r2}


# ---------------------------------------------------------------------------
# Algorithm tests — detect_storm_cells
# ---------------------------------------------------------------------------


class TestDetectStormCells:
    """Direct tests of the detection algorithm (synchronous, CPU-bound)."""

    @pytest.mark.storm_cells
    def test_empty_frame(self, syn_regions):
        """All-zero frame => no cells."""
        frame = np.zeros((100, 100), dtype=np.uint8)
        result = detect_storm_cells(
            latest_frame_regions={"SYNTH_REGION": frame},
            enabled_regions=["SYNTH_REGION"],
            flows_by_region=None,
            min_dbz=40,
            min_area_km2=2.0,
            fetch_interval_s=600,
        )
        assert "SYNTH_REGION" in result
        assert len(result["SYNTH_REGION"]) == 0

    @pytest.mark.storm_cells
    def test_single_cell(self, syn_regions):
        """A 10x10 block at pixel=160 (48 dBZ) => one detected cell."""
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[20:30, 30:40] = 160  # 48 dBZ, above 40 dBZ threshold
        result = detect_storm_cells(
            latest_frame_regions={"SYNTH_REGION": frame},
            enabled_regions=["SYNTH_REGION"],
            flows_by_region=None,
            min_dbz=40,
            min_area_km2=2.0,
            fetch_interval_s=600,
        )
        assert "SYNTH_REGION" in result
        arr = result["SYNTH_REGION"]
        assert len(arr) == 1
        # Centroid at centre of the 10x10 block (rows 20-29, cols 30-39)
        assert arr[0]["centroid_row"] == pytest.approx(24.5, abs=0.1)
        assert arr[0]["centroid_col"] == pytest.approx(34.5, abs=0.1)
        assert arr[0]["area_px"] == pytest.approx(100.0)
        # Area in km^2
        expected_km2 = 100.0 * _DEFAULT_PX_TO_KM2
        assert arr[0]["area_km2"] == pytest.approx(expected_km2, abs=0.01)
        # Max dBZ decoded from pixel 160
        assert arr[0]["max_dbz"] == pytest.approx(48.0, abs=0.1)

    @pytest.mark.storm_cells
    def test_two_cells(self, syn_regions):
        """Two disjoint 10x10 blocks => two cells."""
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[10:20, 10:20] = 160
        frame[50:60, 50:60] = 160
        result = detect_storm_cells(
            latest_frame_regions={"SYNTH_REGION": frame},
            enabled_regions=["SYNTH_REGION"],
            flows_by_region=None,
            min_dbz=40,
            min_area_km2=2.0,
            fetch_interval_s=600,
        )
        assert len(result["SYNTH_REGION"]) == 2

    @pytest.mark.storm_cells
    def test_below_threshold_ignored(self, syn_regions):
        """Pixels at 100 (18 dBZ) below the 40 dBZ threshold => no cells."""
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[20:30, 30:40] = 100  # 18 dBZ, below threshold
        result = detect_storm_cells(
            latest_frame_regions={"SYNTH_REGION": frame},
            enabled_regions=["SYNTH_REGION"],
            flows_by_region=None,
            min_dbz=40,
            min_area_km2=2.0,
            fetch_interval_s=600,
        )
        assert len(result["SYNTH_REGION"]) == 0

    @pytest.mark.storm_cells
    def test_min_area_filter(self, syn_regions):
        """1-pixel cell (below min_area) filtered; 4x4 cell (above) detected."""
        frame = np.zeros((100, 100), dtype=np.uint8)
        # 1 pixel -- below min_area (2.0 km^2)
        frame[10, 10] = 160
        # 4x4 block = 16 pixels ~ 19.7 km^2 -- above min_area
        frame[50:54, 50:54] = 160
        result = detect_storm_cells(
            latest_frame_regions={"SYNTH_REGION": frame},
            enabled_regions=["SYNTH_REGION"],
            flows_by_region=None,
            min_dbz=40,
            min_area_km2=2.0,
            fetch_interval_s=600,
        )
        # Only the 4x4 cell survives
        assert len(result["SYNTH_REGION"]) == 1
        assert result["SYNTH_REGION"][0]["area_px"] == pytest.approx(16.0)

    @pytest.mark.storm_cells
    def test_motion_from_flow(self, syn_regions):
        """Flow sampled at centroid yields correct motion vector."""
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[20:30, 30:40] = 160
        # Flow field: (H, W, 2) float32
        flow = np.zeros((100, 100, 2), dtype=np.float32)
        # Centroid at row~24.5, col~34.5 rounds to (24, 34) in Python 3
        flow[24, 34] = [5.0, -3.0]  # 5 px east, 3 px north
        flows = {"SYNTH_REGION": flow}
        result = detect_storm_cells(
            latest_frame_regions={"SYNTH_REGION": frame},
            enabled_regions=["SYNTH_REGION"],
            flows_by_region=flows,
            min_dbz=40,
            min_area_km2=2.0,
            fetch_interval_s=600,
        )
        cell = result["SYNTH_REGION"][0]
        # Motion components
        assert cell["motion_dx_px"] == pytest.approx(5.0, abs=0.01)
        assert cell["motion_dy_px"] == pytest.approx(-3.0, abs=0.01)
        # Speed in km/h: sqrt((5*1.11)^2 + (3*1.11)^2) * 3600/600
        # ≈ sqrt(30.86 + 11.11) * 6 ≈ 41.96 * 6... let's compute:
        # sqrt(5.55^2 + 3.33^2) = sqrt(30.80 + 11.09) = sqrt(41.89) ≈ 6.473
        # 6.473 * 6 = 38.84 km/h
        assert cell["motion_speed_kmh"] == pytest.approx(38.84, abs=0.5)
        # Heading: atan2(5.55, 3.33) ≈ 59 degrees NE
        assert cell["motion_heading_deg"] == pytest.approx(59.0, abs=2.0)
        assert not np.isnan(cell["motion_speed_kmh"])
        assert not np.isnan(cell["motion_heading_deg"])

    @pytest.mark.storm_cells
    def test_no_flow_nan_motion(self, syn_regions):
        """No flows_by_region => motion fields are 0 / NaN."""
        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[20:30, 30:40] = 160
        result = detect_storm_cells(
            latest_frame_regions={"SYNTH_REGION": frame},
            enabled_regions=["SYNTH_REGION"],
            flows_by_region=None,
            min_dbz=40,
            min_area_km2=2.0,
            fetch_interval_s=600,
        )
        cell = result["SYNTH_REGION"][0]
        assert cell["motion_dx_px"] == 0.0
        assert cell["motion_dy_px"] == 0.0
        assert np.isnan(cell["motion_speed_kmh"])
        assert np.isnan(cell["motion_heading_deg"])

    @pytest.mark.storm_cells
    def test_multiple_regions(self, syn_regions):
        """Both input regions produce a cell in the output."""
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame1[20:30, 30:40] = 160
        frame2 = np.zeros((100, 100), dtype=np.uint8)
        frame2[50:60, 60:70] = 160
        result = detect_storm_cells(
            latest_frame_regions={
                "SYNTH_REGION": frame1,
                "SYNTH_REGION_2": frame2,
            },
            enabled_regions=["SYNTH_REGION", "SYNTH_REGION_2"],
            flows_by_region=None,
            min_dbz=40,
            min_area_km2=2.0,
            fetch_interval_s=600,
        )
        assert "SYNTH_REGION" in result
        assert "SYNTH_REGION_2" in result
        assert len(result["SYNTH_REGION"]) == 1
        assert len(result["SYNTH_REGION_2"]) == 1

    @pytest.mark.storm_cells
    def test_skips_disabled_regions(self, syn_regions):
        """Regions not in enabled_regions are absent from the output."""
        frame1 = np.zeros((100, 100), dtype=np.uint8)
        frame1[20:30, 30:40] = 160
        frame2 = np.zeros((100, 100), dtype=np.uint8)
        frame2[50:60, 60:70] = 160
        result = detect_storm_cells(
            latest_frame_regions={
                "SYNTH_REGION": frame1,
                "SYNTH_REGION_2": frame2,
            },
            enabled_regions=["SYNTH_REGION"],  # SYNTH_REGION_2 not enabled
            flows_by_region=None,
            min_dbz=40,
            min_area_km2=2.0,
            fetch_interval_s=600,
        )
        assert "SYNTH_REGION" in result
        assert "SYNTH_REGION_2" not in result


# ---------------------------------------------------------------------------
# StormCellStore tests
# ---------------------------------------------------------------------------


class TestStormCellStore:
    """Memmap-backed store: replace, retrieve, serialise, truncate."""

    @pytest.mark.storm_cells
    async def test_replace_and_get(self):
        """replace_cells round-trips through get_cells and get_counts."""
        store = StormCellStore()
        try:
            arr = np.array([
                (10.0, 20.0, 50.0, 61.6, 45.0,
                 0.0, 0.0, float("nan"), float("nan")),
            ], dtype=_CELL_DTYPE)
            await store.replace_cells({"TEST": arr})

            cells = await store.get_cells()
            assert "TEST" in cells
            assert cells["TEST"].shape == (MAX_CELLS_PER_REGION,)
            assert cells["TEST"][0]["centroid_row"] == pytest.approx(10.0)
            assert cells["TEST"][0]["area_px"] == pytest.approx(50.0)

            counts = await store.get_counts()
            assert counts == {"TEST": 1}
            assert store.total_count == 1
            assert store.last_updated > 0
        finally:
            store.cleanup()

    @pytest.mark.storm_cells
    async def test_state_round_trip(self, tmp_path):
        """__getstate__/__setstate__ round-trips memmap data correctly."""
        store = StormCellStore(cache_dir=tmp_path)
        try:
            arr = np.array([
                (15.0, 25.0, 80.0, 98.57, 50.0,
                 2.0, -1.0, 30.0, 45.0),
            ], dtype=_CELL_DTYPE)
            await store.replace_cells({"RT": arr})

            state = store.__getstate__()
            # Recreate in a new object
            new_store = StormCellStore.__new__(StormCellStore)
            new_store.__setstate__(state)

            cells = await new_store.get_cells()
            assert "RT" in cells
            assert cells["RT"][0]["centroid_row"] == pytest.approx(15.0)
            assert cells["RT"][0]["centroid_col"] == pytest.approx(25.0)
            assert cells["RT"][0]["area_px"] == pytest.approx(80.0)
            assert cells["RT"][0]["max_dbz"] == pytest.approx(50.0)
            assert cells["RT"][0]["motion_dx_px"] == pytest.approx(2.0)
            assert cells["RT"][0]["motion_dy_px"] == pytest.approx(-1.0)
            assert cells["RT"][0]["motion_speed_kmh"] == pytest.approx(30.0)
            assert cells["RT"][0]["motion_heading_deg"] == pytest.approx(45.0)

            counts = await new_store.get_counts()
            assert counts == {"RT": 1}
            assert new_store.total_count == 1
            assert new_store.last_updated == pytest.approx(store.last_updated, abs=0.1)

            new_store.cleanup()
        finally:
            store.cleanup()

    @pytest.mark.storm_cells
    async def test_max_cap_truncation(self):
        """More than MAX_CELLS_PER_REGION cells are truncated to MAX."""
        store = StormCellStore()
        try:
            count = MAX_CELLS_PER_REGION + 10
            rows = []
            for i in range(count):
                rows.append((
                    float(i), float(i), 1.0, 1.23, 48.0,
                    0.0, 0.0, float("nan"), float("nan"),
                ))
            arr = np.array(rows, dtype=_CELL_DTYPE)

            await store.replace_cells({"TRUNC": arr})
            cells = await store.get_cells()
            assert cells["TRUNC"].shape == (MAX_CELLS_PER_REGION,)

            counts = await store.get_counts()
            assert counts["TRUNC"] == MAX_CELLS_PER_REGION
        finally:
            store.cleanup()


# ---------------------------------------------------------------------------
# StormCellGenerator tests
# ---------------------------------------------------------------------------


class TestStormCellGenerator:
    """Orchestration: frame-store integration, flow sampling, no-op on cold start."""

    @pytest.mark.storm_cells
    async def test_no_latest_frame(self):
        """No latest frame => generate() no-ops without calling replace_cells."""
        class MockFrameStore:
            async def get_latest_frame(self):
                return None

        class MockNowcastStore:
            async def get_flows(self):
                return {}

        store = StormCellStore()
        try:
            gen = StormCellGenerator(
                frame_store=MockFrameStore(),
                storm_cell_store=store,
                nowcast_store=MockNowcastStore(),
            )
            await gen.generate()
            counts = await store.get_counts()
            assert counts == {}
        finally:
            store.cleanup()

    @pytest.mark.storm_cells
    async def test_with_frame(self, syn_regions, monkeypatch):
        """With a valid frame, the generator detects cells and stores them."""
        from librewxr.data.storm_cells import settings as _sc_settings
        monkeypatch.setattr(_sc_settings, "enabled_regions", "SYNTH_REGION")

        frame = np.zeros((100, 100), dtype=np.uint8)
        frame[20:30, 30:40] = 160

        class MockFrameStore:
            async def get_latest_frame(self):
                return SimpleNamespace(
                    timestamp=1000,
                    regions={"SYNTH_REGION": frame},
                )

        class MockNowcastStore:
            async def get_flows(self):
                return {}

        store = StormCellStore()
        try:
            gen = StormCellGenerator(
                frame_store=MockFrameStore(),
                storm_cell_store=store,
                nowcast_store=MockNowcastStore(),
            )
            await gen.generate()

            counts = await store.get_counts()
            assert counts.get("SYNTH_REGION") == 1

            cells = await store.get_cells()
            assert cells["SYNTH_REGION"][0]["centroid_row"] == pytest.approx(24.5, abs=0.1)
            assert cells["SYNTH_REGION"][0]["max_dbz"] == pytest.approx(48.0, abs=0.1)
        finally:
            store.cleanup()
