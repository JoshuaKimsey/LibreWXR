# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for the GET /v2/storm-cells REST endpoint (dual geojson/json format)."""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from librewxr.api import routes
from librewxr.data.regions import REGIONS, RegionDef
from librewxr.data.storm_cells import _CELL_DTYPE

pytestmark = pytest.mark.storm_cells


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockStormCellStore:
    """Minimal async storm-cell store for testing.

    Mirrors the mock in ``test_mcp_storm_cells.py`` plus the
    ``last_updated`` accessor the json format uses (a plain attribute;
    the real store exposes it as a read-only property).
    """

    def __init__(self, cells_by_region, counts, last_updated: float = 0.0):
        self._cells = {k: v for k, v in cells_by_region.items()}
        self._counts = {k: v for k, v in counts.items()}
        self.last_updated = last_updated

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


@pytest.fixture
def app(monkeypatch):
    """Minimal FastAPI app with just the router; seeds the routes global.

    Two synthetic regions, each with one detected cell:
    - TEST_US: (row=5, col=5) -> (lat=40.0, lon=-95.0), NaN motion
    - TEST_FAR: (row=5, col=5) -> (lat=5.0, lon=-5.0), motion present

    ``monkeypatch`` restores ``routes.storm_cell_store`` and the REGIONS
    entries after each test.
    """
    monkeypatch.setitem(REGIONS, "TEST_US", RegionDef(
        name="TEST_US",
        west=-100.0, east=-90.0, south=35.0, north=45.0,
        pixel_size=1.0, group="TEST",
    ))
    monkeypatch.setitem(REGIONS, "TEST_FAR", RegionDef(
        name="TEST_FAR",
        west=-10.0, east=10.0, south=-10.0, north=10.0,
        pixel_size=1.0, group="TEST",
    ))

    near_arr = _build_single_cell_array(
        centroid_row=5.0, centroid_col=5.0,
        area_km2=120.0, max_dbz=55.0,
        motion_speed_kmh=float("nan"), motion_heading_deg=float("nan"),
    )
    far_arr = _build_single_cell_array(
        centroid_row=5.0, centroid_col=5.0,
        area_km2=200.0, max_dbz=48.0,
        motion_speed_kmh=25.0, motion_heading_deg=180.0,
    )
    store = _MockStormCellStore(
        {"TEST_US": near_arr, "TEST_FAR": far_arr},
        {"TEST_US": 1, "TEST_FAR": 1},
        last_updated=1_700_000_000.0,
    )
    monkeypatch.setattr(routes, "storm_cell_store", store)

    test_app = FastAPI()
    test_app.include_router(routes.router)
    return test_app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_store_none_returns_503(monkeypatch):
    """storm_cell_store None (detection disabled) -> 503, mirroring /v2/alerts."""
    monkeypatch.setattr(routes, "storm_cell_store", None)
    test_app = FastAPI()
    test_app.include_router(routes.router)
    with TestClient(test_app, raise_server_exceptions=False) as c:
        resp = c.get("/v2/storm-cells")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Storm cells not available"


def test_geojson_default_shape(app):
    """Default format is a FeatureCollection of Point features [lon, lat]."""
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/v2/storm-cells")
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 2

    by_region = {f["properties"]["region"]: f for f in data["features"]}
    assert set(by_region) == {"TEST_US", "TEST_FAR"}

    for feature in data["features"]:
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Point"
        coords = feature["geometry"]["coordinates"]
        assert len(coords) == 2
        lon, lat = coords
        assert -180 <= lon <= 180
        assert -90 <= lat <= 90
        props = feature["properties"]
        # lat/lon live in the geometry only, not the properties.
        assert set(props) == {
            "area_km2", "max_dbz", "motion_speed_kmh",
            "motion_heading_deg", "region",
        }

    # Centroid coords are [lon, lat].
    assert by_region["TEST_US"]["geometry"]["coordinates"] == [-95.0, 40.0]
    assert by_region["TEST_FAR"]["geometry"]["coordinates"] == [-5.0, 5.0]

    # NaN motion serializes as null.
    us_props = by_region["TEST_US"]["properties"]
    assert us_props["motion_speed_kmh"] is None
    assert us_props["motion_heading_deg"] is None
    assert us_props["area_km2"] == 120.0
    assert us_props["max_dbz"] == 55.0


def test_json_format(app):
    """format=json returns generated_at plus raw cells with the MCP keys."""
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/v2/storm-cells?format=json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["generated_at"] == 1_700_000_000
    assert len(data["cells"]) == 2
    keys = {
        "lat", "lon", "area_km2", "max_dbz",
        "motion_speed_kmh", "motion_heading_deg", "region",
    }
    for cell in data["cells"]:
        assert set(cell) == keys


def test_radius_filter_returns_only_in_range(app):
    """lat/lon query keeps only the cell within radius_km."""
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/v2/storm-cells?lat=40.0&lon=-95.0&radius_km=50")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["features"]) == 1
    assert data["features"][0]["properties"]["region"] == "TEST_US"


def test_no_lat_lon_returns_all_cells(app):
    """No lat/lon -> radius filter skipped, all cells returned."""
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/v2/storm-cells")
        # radius_km is silently ignored without lat/lon.
        resp_radius = c.get("/v2/storm-cells?radius_km=1")
    assert resp.status_code == 200
    assert len(resp.json()["features"]) == 2
    assert resp_radius.status_code == 200
    assert len(resp_radius.json()["features"]) == 2


def test_lat_without_lon_returns_400(app):
    """Exactly one of lat/lon -> 400."""
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/v2/storm-cells?lat=40.0")
    assert resp.status_code == 400
    assert resp.json()["detail"] == "lat and lon must be provided together"


def test_bogus_format_returns_422(app):
    """format not in {geojson, json} -> 422 from the Query pattern."""
    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get("/v2/storm-cells?format=bogus")
    assert resp.status_code == 422