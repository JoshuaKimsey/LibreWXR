# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from librewrx.api import routes
from librewrx.data.store import FrameStore, RadarFrame
from librewrx.main import app
from librewrx.tiles.cache import TileCache
from librewrx.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH


@pytest.fixture
def populated_store():
    """Create a frame store with test data."""
    store = FrameStore(max_frames=12)
    data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    # Add some radar returns
    data[2500:2700, 6000:6200] = 128

    import asyncio
    ts = int(time.time() // 300) * 300
    frame = RadarFrame(timestamp=ts, regions={"USCOMP": data})
    asyncio.run(store.add_frame(frame))
    return store, ts


@pytest.fixture
def client(populated_store):
    """Create a test client with pre-populated data (no background fetcher)."""
    store, ts = populated_store
    cache = TileCache(max_size=100)

    with TestClient(app, raise_server_exceptions=False) as c:
        # Override after lifespan sets its own store/cache
        routes.frame_store = store
        routes.tile_cache = cache
        routes.enabled_regions = ["USCOMP"]
        yield c, ts


class TestWeatherMapsEndpoint:
    def test_returns_valid_json(self, client):
        c, ts = client
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
        c, ts = client
        resp = c.get("/public/weather-maps.json")
        data = resp.json()
        past = data["radar"]["past"]
        assert len(past) >= 1
        assert past[0]["time"] == ts
        assert past[0]["path"] == f"/v2/radar/{ts}"


class TestRadarTileEndpoint:
    def test_valid_tile_request(self, client):
        c, ts = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_webp_format(self, client):
        c, ts = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.webp")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/webp"

    def test_missing_timestamp(self, client):
        c, _ = client
        resp = c.get("/v2/radar/9999999999/256/4/3/5/2/0_0.png")
        assert resp.status_code == 404

    def test_cache_header(self, client):
        c, ts = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png")
        assert "cache-control" in resp.headers


class TestCoverageTileEndpoint:
    def test_valid_coverage_request(self, client):
        c, _ = client
        resp = c.get("/v2/coverage/0/256/4/3/5/0/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
