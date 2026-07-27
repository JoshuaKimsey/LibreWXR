# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for storm-cell MCP tools: inverse-projection helpers and the
get_storm_cells tool function with mocked stores."""

import math
import pytest
import numpy as np

from librewxr.data.regions import REGIONS, RegionDef
from librewxr.data.storm_cells import _CELL_DTYPE
from librewxr.mcp.storm_cells import cell_pixel_to_latlon, _build_inverse_grid
from librewxr.mcp.tools import get_storm_cells
from librewxr.tiles.coordinates import _laea_forward, _tmerc_forward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockStormCellStore:
    """Minimal async storm-cell store for testing."""

    def __init__(self, cells_by_region, counts):
        self._cells = {k: v for k, v in cells_by_region.items()}
        self._counts = {k: v for k, v in counts.items()}

    async def get_cells(self):
        return dict(self._cells)

    async def get_counts(self):
        return dict(self._counts)


def _build_single_cell_array(
    centroid_row: float, centroid_col: float,
    area_km2: float = 50.0, max_dbz: float = 45.0,
    motion_speed_kmh: float = 30.0, motion_heading_deg: float = 90.0,
) -> np.ndarray:
    """Build a 1-row structured array matching _CELL_DTYPE."""
    arr = np.zeros(1, dtype=_CELL_DTYPE)
    arr["centroid_row"] = centroid_row
    arr["centroid_col"] = centroid_col
    arr["area_px"] = 100
    arr["area_km2"] = area_km2
    arr["max_dbz"] = max_dbz
    arr["motion_dx_px"] = 1.0
    arr["motion_dy_px"] = 0.0
    arr["motion_speed_kmh"] = motion_speed_kmh
    arr["motion_heading_deg"] = motion_heading_deg
    return arr


# ---------------------------------------------------------------------------
# Inverse-projection helpers
# ---------------------------------------------------------------------------


@pytest.mark.mcp
def test_cell_pixel_to_latlon_latlon():
    """latlon region: direct math produces exact results."""
    region = RegionDef(
        name="TEST_LATLON",
        west=0.0, east=10.0, south=0.0, north=10.0,
        pixel_size=0.01, group="TEST",
    )
    # (row=500, col=300) -> lat = 10 - 500*0.01 = 5.0, lon = 0 + 300*0.01 = 3.0
    lat, lon = cell_pixel_to_latlon(region, row=500.0, col=300.0)
    assert lat == pytest.approx(5.0, abs=1e-9)
    assert lon == pytest.approx(3.0, abs=1e-9)


@pytest.mark.mcp
def test_cell_pixel_to_latlon_laea_roundtrip():
    """laea region: round-trip (forward then inverse) is within ~0.5 deg."""
    region = RegionDef(
        name="TEST_LAEA",
        west=-10.0, east=10.0, south=40.0, north=50.0,
        pixel_size=0.05, group="TEST",
        proj="laea",
        laea_lat0=45.0, laea_lon0=0.0,
        laea_x0=0.0, laea_y0=0.0,
        grid_x_min=-500000.0, grid_y_max=500000.0, grid_scale=5000.0,
        grid_width=200, grid_height=200,
    )
    test_lat, test_lon = 45.5, 2.0

    # Forward project to get pixel coords.
    x, y = _laea_forward(
        np.asarray([test_lon], dtype=np.float64),
        np.asarray([test_lat], dtype=np.float64),
        region,
    )
    col_f = (float(x[0]) - region.grid_x_min) / region.grid_scale
    row_f = (region.grid_y_max - float(y[0])) / region.grid_scale

    # Inverse.
    result_lat, result_lon = cell_pixel_to_latlon(region, row_f, col_f)
    assert result_lat == pytest.approx(test_lat, abs=0.5)
    assert result_lon == pytest.approx(test_lon, abs=0.5)


@pytest.mark.mcp
def test_cell_pixel_to_latlon_tmerc_roundtrip():
    """tmerc region: round-trip (forward then inverse) is within ~0.5 deg."""
    region = RegionDef(
        name="TEST_TMERC",
        west=5.0, east=15.0, south=38.0, north=46.0,
        pixel_size=0.01, group="TEST",
        proj="tmerc",
        tmerc_lat0=42.0, tmerc_lon0=10.0,
        tmerc_radius=6371229.0, tmerc_k0=1.0,
        grid_x_min=-500000.0, grid_y_max=500000.0, grid_scale=1000.0,
        grid_width=1000, grid_height=1000,
    )
    test_lat, test_lon = 43.0, 11.0

    # Forward project to get pixel coords.
    x, y = _tmerc_forward(
        np.asarray([test_lon], dtype=np.float64),
        np.asarray([test_lat], dtype=np.float64),
        region,
    )
    col_f = (float(x[0]) - region.grid_x_min) / region.grid_scale
    row_f = (region.grid_y_max - float(y[0])) / region.grid_scale

    # Inverse.
    result_lat, result_lon = cell_pixel_to_latlon(region, row_f, col_f)
    assert result_lat == pytest.approx(test_lat, abs=0.5)
    assert result_lon == pytest.approx(test_lon, abs=0.5)


# ---------------------------------------------------------------------------
# get_storm_cells
# ---------------------------------------------------------------------------


@pytest.mark.mcp
async def test_get_storm_cells_none_store():
    """None store -> empty list (degraded empty)."""
    result = await get_storm_cells(None, 35.0, -100.0, 100.0)
    assert result == []


@pytest.mark.mcp
async def test_get_storm_cells_empty_store():
    """Empty store -> empty list."""
    store = _MockStormCellStore({}, {})
    result = await get_storm_cells(store, 35.0, -100.0, 100.0)
    assert result == []


@pytest.mark.mcp
async def test_get_storm_cells_with_cells(monkeypatch):
    """Single cell at known pixel -> returned with correct lat/lon."""
    region = RegionDef(
        name="TEST_US",
        west=-100.0, east=-90.0, south=35.0, north=45.0,
        pixel_size=1.0, group="TEST",
    )
    monkeypatch.setitem(REGIONS, "TEST_US", region)

    # (row=5, col=5) -> lat = 45 - 5*1 = 40.0, lon = -100 + 5*1 = -95.0
    cell_arr = _build_single_cell_array(
        centroid_row=5.0, centroid_col=5.0,
        area_km2=120.0, max_dbz=55.0,
        motion_speed_kmh=40.0, motion_heading_deg=180.0,
    )
    store = _MockStormCellStore({"TEST_US": cell_arr}, {"TEST_US": 1})
    result = await get_storm_cells(store, 40.0, -95.0, radius_km=50.0)

    assert len(result) == 1
    r = result[0]
    assert set(r.keys()) == {"lat", "lon", "area_km2", "max_dbz",
                              "motion_speed_kmh", "motion_heading_deg", "region"}
    assert r["lat"] == pytest.approx(40.0, abs=1e-9)
    assert r["lon"] == pytest.approx(-95.0, abs=1e-9)
    assert r["area_km2"] == pytest.approx(120.0, abs=1e-6)
    assert r["max_dbz"] == pytest.approx(55.0, abs=1e-6)
    assert r["motion_speed_kmh"] == 40.0
    assert r["motion_heading_deg"] == 180.0
    assert r["region"] == "TEST_US"

    # Remove monkeypatched region so other tests aren't affected.
    monkeypatch.delitem(REGIONS, "TEST_US", raising=False)


@pytest.mark.mcp
async def test_get_storm_cells_radius_filter(monkeypatch):
    """Two cells: near one is returned, far one is excluded."""
    region = RegionDef(
        name="TEST_RADIUS",
        west=-100.0, east=-80.0, south=30.0, north=50.0,
        pixel_size=0.5, group="TEST",
    )
    monkeypatch.setitem(REGIONS, "TEST_RADIUS", region)

    # Query at (40, -95).
    # Cell at (row=20, col=10) -> (50 - 20*0.5=40, -100 + 10*0.5=-95) = (40, -95) -> within 0 km
    # Cell at (row=39, col=10) -> (50 - 39*0.5=30.5, -100 + 10*0.5=-95) = (30.5, -95)
    #   dlat = 30.5 - 40 = -9.5 deg * 111 km/deg = 1054.5 km -> far
    near_arr = _build_single_cell_array(
        centroid_row=20.0, centroid_col=10.0,
        area_km2=50.0, max_dbz=40.0,
    )
    far_arr = _build_single_cell_array(
        centroid_row=39.0, centroid_col=10.0,
        area_km2=80.0, max_dbz=50.0,
    )
    combined = np.concatenate([near_arr, far_arr])
    store = _MockStormCellStore({"TEST_RADIUS": combined}, {"TEST_RADIUS": 2})
    result = await get_storm_cells(store, 40.0, -95.0, radius_km=50.0)

    assert len(result) == 1
    assert result[0]["area_km2"] == pytest.approx(50.0, abs=1e-6)

    monkeypatch.delitem(REGIONS, "TEST_RADIUS", raising=False)


@pytest.mark.mcp
async def test_get_storm_cells_nan_motion(monkeypatch):
    """NaN motion fields -> None in output (JSON-safe)."""
    region = RegionDef(
        name="TEST_NAN",
        west=-100.0, east=-90.0, south=35.0, north=45.0,
        pixel_size=1.0, group="TEST",
    )
    monkeypatch.setitem(REGIONS, "TEST_NAN", region)

    cell_arr = _build_single_cell_array(
        centroid_row=5.0, centroid_col=5.0,
        motion_speed_kmh=float("nan"), motion_heading_deg=float("nan"),
    )
    store = _MockStormCellStore({"TEST_NAN": cell_arr}, {"TEST_NAN": 1})
    result = await get_storm_cells(store, 40.0, -95.0, radius_km=50.0)

    assert len(result) == 1
    r = result[0]
    assert r["motion_speed_kmh"] is None
    assert r["motion_heading_deg"] is None

    monkeypatch.delitem(REGIONS, "TEST_NAN", raising=False)


@pytest.mark.mcp
async def test_get_storm_cells_motion_present(monkeypatch):
    """Non-NaN motion fields passed through as floats."""
    region = RegionDef(
        name="TEST_MOTION",
        west=-100.0, east=-90.0, south=35.0, north=45.0,
        pixel_size=1.0, group="TEST",
    )
    monkeypatch.setitem(REGIONS, "TEST_MOTION", region)

    cell_arr = _build_single_cell_array(
        centroid_row=5.0, centroid_col=5.0,
        motion_speed_kmh=20.0, motion_heading_deg=59.0,
    )
    store = _MockStormCellStore({"TEST_MOTION": cell_arr}, {"TEST_MOTION": 1})
    result = await get_storm_cells(store, 40.0, -95.0, radius_km=50.0)

    assert len(result) == 1
    r = result[0]
    assert r["motion_speed_kmh"] == 20.0
    assert r["motion_heading_deg"] == 59.0

    monkeypatch.delitem(REGIONS, "TEST_MOTION", raising=False)
