# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api

from librewxr.api import routes
from librewxr.config import settings
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH


class _StubSatelliteSource:
    """Duck-typed GMGSI grid for the satellite tile route tests.

    Mirrors the ``GMGSISource`` surface the route + renderers touch: a
    ``timestamps`` property and a ``sample(lat, lon, timestamp)`` method
    returning a constant uint8 grid (same trick as the ``_ConstantSource``
    in ``test_gmgsi_composite_renderer.py``).
    """

    def __init__(self, value: int, timestamps: list[int]) -> None:
        self._timestamps = sorted(timestamps)
        self.value = value

    @property
    def timestamps(self) -> list[int]:
        return list(self._timestamps)

    @property
    def data_bytes(self) -> int:
        # GMGSISource exposes this for the /health memory breakdown; the
        # stub keeps no backing array, so report zero footprint.
        return 0

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        return np.full(lat.shape, self.value, dtype=np.uint8)


class _StubStormCellStore:
    """Duck-typed StormCellStore holding one cell centred on a given tile.

    The centroid is derived from ``region_pixel_indices_fractional`` so it
    is guaranteed to land inside the tile's coverage of the region, for
    any projection — the same trick ``test_storm_cell_render.py`` uses.
    """

    def __init__(self, detected_at: int, region: str, z: int, x: int, y: int) -> None:
        from librewxr.data.regions import REGIONS
        from librewxr.data.storm_cells import _CELL_DTYPE
        from librewxr.tiles.coordinates import (
            region_pixel_indices,
            region_pixel_indices_fractional,
        )

        region_def = REGIONS[region]
        row_f, col_f = region_pixel_indices_fractional(region_def, z, x, y, 256)
        row_i, _ = region_pixel_indices(region_def, z, x, y, 256)
        valid = row_i >= 0
        assert valid.any(), f"tile {z}/{x}/{y} does not cover {region}"

        cell = np.zeros((1,), dtype=_CELL_DTYPE)
        cell["centroid_row"][0] = float(row_f[valid].mean())
        cell["centroid_col"][0] = float(col_f[valid].mean())
        cell["area_km2"][0] = 500.0
        cell["max_dbz"][0] = 55.0
        cell["motion_speed_kmh"][0] = np.nan

        self._cells = {region: cell}
        self._counts = {region: 1}
        self.detected_at_timestamp = detected_at

    async def get_cells(self) -> dict[str, np.ndarray]:
        return dict(self._cells)

    async def get_counts(self) -> dict[str, int]:
        return dict(self._counts)


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

    # Duck-typed GMGSI stubs so the satellite route renders (composite:
    # cold LW cloud over VIS=0 night side) instead of returning 503.
    routes.satellite_grids = {
        "gmgsi_lw_grid": _StubSatelliteSource(180, [ts_prev, ts]),
        "gmgsi_vis_grid": _StubSatelliteSource(0, [ts_prev, ts]),
    }

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

    def test_radar_tile_has_etag(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png")
        assert resp.status_code == 200
        etag = resp.headers.get("etag")
        assert etag is not None
        assert etag.startswith('"')
        assert etag.endswith('"')
        assert len(etag) == 18

    def test_radar_tile_304_on_match(self, client):
        c, ts, _ = client
        url = f"/v2/radar/{ts}/256/4/3/5/2/0_0.png"
        first = c.get(url)
        assert first.status_code == 200
        etag = first.headers["etag"]
        resp = c.get(url, headers={"If-None-Match": etag})
        assert resp.status_code == 304
        assert resp.content == b""
        assert resp.headers.get("etag") == etag
        assert resp.headers.get("cache-control", "").startswith("public, max-age=")
        # httpx/TestClient omits Content-Length on 304 (see test_conditional.py)
        assert resp.headers.get("content-length") is None

    def test_radar_tile_304_on_star(self, client):
        c, ts, _ = client
        resp = c.get(
            f"/v2/radar/{ts}/256/4/3/5/2/0_0.png",
            headers={"If-None-Match": "*"},
        )
        assert resp.status_code == 304
        assert resp.content == b""

    def test_radar_tile_304_on_mismatch(self, client):
        c, ts, _ = client
        resp = c.get(
            f"/v2/radar/{ts}/256/4/3/5/2/0_0.png",
            headers={"If-None-Match": '"deadbeefdeadbeef"'},
        )
        assert resp.status_code == 200
        assert resp.content != b""

    def test_radar_tile_etag_stable_across_requests(self, client):
        c, ts, _ = client
        url = f"/v2/radar/{ts}/256/4/3/5/2/0_0.png"
        first = c.get(url)
        second = c.get(url)
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.headers["etag"] == second.headers["etag"]

    def test_radar_tile_overlay_has_etag(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png?arrows=1")
        assert resp.status_code == 200
        etag = resp.headers.get("etag")
        assert etag is not None
        assert etag.startswith('"')
        assert etag.endswith('"')
        assert len(etag) == 18

    def test_cells_overlay_renders_on_warm_geometry_cache(self, client, monkeypatch):
        """?cells= must draw even when the geometry cache already holds the tile.

        Regression: ``need_frame`` only accounted for ``arrow_style``, so a
        cells-only request that hit the geometry cache left ``frame`` unset.
        ``present_tile`` then received ``frame_regions=None``, and
        ``_draw_storm_cells`` fell back to an empty region list — the tile
        came back byte-identical to the plain one, with no error.  In
        production the cache is warm essentially always, so ?cells= looked
        inert while ?arrows= (which forced the fetch) worked.
        """
        c, ts, _ = client
        # 4/3/6 is the z=4 tile that actually covers the fixture's radar
        # block — a transparent tile short-circuits before any overlay.
        url = f"/v2/radar/{ts}/256/4/3/6/2/0_0.png"

        # Warm the geometry cache first — that's the condition that used to
        # silently disable the overlay.
        plain = c.get(url)
        assert plain.status_code == 200

        monkeypatch.setattr(
            routes, "storm_cell_store", _StubStormCellStore(ts, "USCOMP", 4, 3, 6),
        )
        with_cells = c.get(f"{url}?cells=1")
        assert with_cells.status_code == 200
        assert with_cells.content != plain.content

    def test_cells_overlay_skipped_on_other_frames(self, client, monkeypatch):
        """Cells only render on the frame detection actually ran on."""
        c, ts, ts_prev = client
        monkeypatch.setattr(
            routes, "storm_cell_store", _StubStormCellStore(ts, "USCOMP", 4, 3, 6),
        )
        url = f"/v2/radar/{ts_prev}/256/4/3/6/2/0_0.png"
        assert c.get(f"{url}?cells=1").content == c.get(url).content


class TestCoverageTileEndpoint:
    def test_valid_coverage_request(self, client):
        c, _, _ = client
        resp = c.get("/v2/coverage/0/256/4/3/5/0/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_coverage_tile_has_etag(self, client):
        c, _, _ = client
        resp = c.get("/v2/coverage/0/256/4/3/5/0/0_0.png")
        assert resp.status_code == 200
        etag = resp.headers.get("etag")
        assert etag is not None
        assert etag.startswith('"')
        assert etag.endswith('"')
        assert len(etag) == 18

    def test_coverage_tile_304_on_match(self, client):
        c, _, _ = client
        url = "/v2/coverage/0/256/4/3/5/0/0_0.png"
        first = c.get(url)
        assert first.status_code == 200
        etag = first.headers["etag"]
        resp = c.get(url, headers={"If-None-Match": etag})
        assert resp.status_code == 304
        assert resp.content == b""
        assert resp.headers.get("cache-control") == "public, max-age=300"


class TestSatelliteTileEndpoint:
    def test_satellite_tile_has_etag(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/satellite/{ts}/256/4/3/5/0/0_0.png")
        assert resp.status_code == 200
        etag = resp.headers.get("etag")
        assert etag is not None
        assert etag.startswith('"')
        assert etag.endswith('"')
        assert len(etag) == 18

    def test_satellite_tile_304_on_match(self, client):
        c, ts, _ = client
        url = f"/v2/satellite/{ts}/256/4/3/5/0/0_0.png"
        first = c.get(url)
        assert first.status_code == 200
        etag = first.headers["etag"]
        resp = c.get(url, headers={"If-None-Match": etag})
        assert resp.status_code == 304
        assert resp.content == b""
        assert resp.headers.get("etag") == etag
        assert resp.headers.get("cache-control", "").startswith("public, max-age=")


class TestHealthEndpoint:
    def test_health_memory_breakdown_without_store(self, client, monkeypatch):
        """Shared store inactive -> breakdown reports the per-worker lru
        byte estimate under ``coord_caches_mb`` and zero store fields.

        The store singleton is process-global and lazily constructed from
        ``settings.cache_dir``, so the default case must pin cache_dir off
        and reset the singleton (mirrors the ``coord_store_env`` fixture in
        test_coordinates.py) instead of relying on ambient settings.
        """
        from librewxr.tiles.coordinates import _reset_coord_store

        monkeypatch.setattr(settings, "cache_dir", "")
        _reset_coord_store()
        c, _, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        breakdown = data["memory"]["breakdown"]
        assert breakdown["coord_store_mb"] == 0.0
        assert breakdown["coord_store_entries"] == 0
        assert breakdown["coord_caches_mb"] >= 0.0
        # The "coord_caches" block serializes the new ``store`` sub-dict
        # automatically; it is None while the store is inactive.
        assert data["coord_caches"]["store"] is None

    def test_health_memory_breakdown_store_active(self, client, monkeypatch):
        """Store-backed: entries are shared read-only memmap pages, not
        private heap - ``coord_caches_mb`` drops to 0 and the on-disk
        footprint is reported separately."""
        c, _, _ = client
        store_stats = {
            "hits": 1,
            "misses": 0,
            "publishes": 1,
            "entries": 3,
            "bytes": 3 * 1024 * 1024,
        }
        monkeypatch.setattr(
            routes,
            "coord_cache_stats",
            lambda: {"max_size": 2048, "caches": {}, "store": store_stats},
        )
        resp = c.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        breakdown = data["memory"]["breakdown"]
        assert breakdown["coord_store_mb"] == 3.0
        assert breakdown["coord_store_entries"] == 3
        assert breakdown["coord_caches_mb"] == 0.0
        assert data["coord_caches"]["store"] == store_stats
