# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api

from librewxr.api import routes
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH


def _make_test_app() -> tuple[FastAPI, FrameStore, TileCache, int, int]:
    """Create a minimal FastAPI app with just the router — no lifespan."""
    store = FrameStore(max_frames=12)
    cache = TileCache(max_mb=10)
    ts = int(time.time() // 300) * 300
    ts_prev = ts - 600

    data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    data[2500:2700, 6000:6200] = 128

    import asyncio
    frame = RadarFrame(timestamp=ts, regions={"USCOMP": data})
    asyncio.run(store.add_frame(frame))
    prev_frame = RadarFrame(timestamp=ts_prev, regions={"USCOMP": data})
    asyncio.run(store.add_frame(prev_frame))

    # Wire shared state directly — same as main.py does after lifespan init
    routes.frame_store = store
    routes.tile_cache = cache
    routes.ecmwf_grid = None
    routes.tile_warmer = None
    routes.nowcast_store = None
    routes.start_time = time.time()
    routes.enabled_regions = ["USCOMP"]

    test_app = FastAPI()
    test_app.include_router(routes.router)
    return test_app, store, cache, ts, ts_prev


# Module-scoped: built once, shared across all tests in this file
_app, _store, _cache, _ts, _ts_prev = _make_test_app()


@pytest.fixture(scope="module")
def client():
    with TestClient(_app, raise_server_exceptions=False) as c:
        yield c, _ts, _ts_prev


class TestWeatherMapsEndpoint:
    def test_returns_valid_json(self, client):
        c, ts, ts_prev = client
        resp = c.get("/public/weather-maps.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "2.0"
        assert "generated" in data
        assert "host" in data
        assert "radar" in data
        assert "past" in data["radar"]
        assert "nowcast" in data["radar"]
        assert "satellite" in data

    def test_past_contains_timestamps(self, client):
        c, ts, ts_prev = client
        resp = c.get("/public/weather-maps.json")
        data = resp.json()
        past = data["radar"]["past"]
        assert len(past) >= 1
        # past is sorted oldest-first; ts_prev was added first (earlier)
        assert past[0]["time"] == ts_prev
        assert past[0]["path"] == f"/v2/radar/{ts_prev}"


class TestRadarTileEndpoint:
    def test_valid_tile_request(self, client):
        c, ts, ts_prev = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_webp_format(self, client):
        c, ts, ts_prev = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.webp")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/webp"

    def test_missing_timestamp(self, client):
        c, _, _ = client
        resp = c.get("/v2/radar/9999999999/256/4/3/5/2/0_0.png")
        assert resp.status_code == 404

    def test_latest_frame_cache_header(self, client):
        """Latest frame gets short cache lifetime."""
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png")
        assert "cache-control" in resp.headers
        assert "max-age=300" in resp.headers["cache-control"]

    def test_historical_frame_cache_header(self, client):
        """Historical frames get long cache lifetime since they are immutable."""
        c, _, ts_prev = client
        resp = c.get(f"/v2/radar/{ts_prev}/256/4/3/5/2/0_0.png")
        assert resp.status_code == 200
        assert "cache-control" in resp.headers
        assert "max-age=7200" in resp.headers["cache-control"]


class TestCoverageTileEndpoint:
    def test_valid_coverage_request(self, client):
        c, _, _ = client
        resp = c.get("/v2/coverage/0/256/4/3/5/0/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
