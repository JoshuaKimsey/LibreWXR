# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Route tests for the Phase 3 lat/lon-centered window mode.

The radar and coverage routes discriminate coordinates exactly the way
RainViewer does: path segments containing a dot are (lat, lon) window
coordinates, plain integers are x/y tile indices.  These tests pin the
window path end to end (frame resolution, window stitching, present
caching, ETag/304, max-age bucketing, coverage stitching, and the
/health window bucket) plus the tile-mode regressions the widened
``x``/``y`` path params must preserve.
"""
import io
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

pytestmark = pytest.mark.api

from librewxr.api import routes
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH


class _StubNowcastStore:
    """Duck-typed NowcastStore returning one nowcast frame.

    Mirrors the real surface the window path touches: ``get_frame``
    returning ``(frame, blend_weight)``.  The radar route only reaches
    the nowcast store when the radar store lacks the timestamp.
    """

    def __init__(self, frame: RadarFrame) -> None:
        self._frame = frame
        self.flow_version = 0

    async def get_frame(self, timestamp: int):
        if timestamp == self._frame.timestamp:
            return self._frame, 0.5
        return None, 0.0


def _fixture_synthetic_frame(timestamp: int) -> RadarFrame:
    """USCOMP frame with a constant reflectivity block.

    USCOMP (sources/regional/north_america/usa/radar/regions.py):
    north=50, west=-126, pixel_size=0.005.  Rows 2500:2700 map to lat
    36.5..37.5 and cols 6000:6200 to lon -96..-95, so the block's centre
    is lat=37.0, lon=-95.5 - the window anchor used throughout.
    """
    data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    data[2500:2700, 6000:6200] = 128
    return RadarFrame(timestamp=timestamp, regions={"USCOMP": data})


_ROUTE_STATE_NAMES = (
    "frame_store", "tile_cache", "ecmwf_grid", "nwp_chain",
    "precip_mask", "tile_warmer", "nowcast_store", "shared_tile_store",
    "tile_request_tracker", "storm_cell_store", "satellite_grids",
    "start_time", "enabled_regions", "_latest_ts_cache",
)


@pytest.fixture(scope="module")
def client():
    """Module-scoped app + state wiring for this file's tests.

    The routes-module singletons are set lazily on first use (NOT at
    import time) and restored on teardown: other test modules build
    their own apps from the same module-level singletons at import
    time, so clobbering them during collection would break their
    later tests (see test_api.py's satellite/health assertions).
    """
    import asyncio

    store = FrameStore(max_frames=12)
    cache = TileCache(max_mb=10)
    ts = int(time.time() // 300) * 300
    ts_prev = ts - 600

    asyncio.run(store.add_frame(_fixture_synthetic_frame(ts)))
    asyncio.run(store.add_frame(_fixture_synthetic_frame(ts_prev)))

    prev = {name: getattr(routes, name) for name in _ROUTE_STATE_NAMES}
    try:
        # Wire shared state directly - same as main.py does after lifespan init
        routes.frame_store = store
        routes.tile_cache = cache
        routes.ecmwf_grid = None
        routes.nwp_chain = None
        routes.precip_mask = None
        routes.tile_warmer = None
        routes.nowcast_store = None
        routes.shared_tile_store = None
        routes.tile_request_tracker = None
        routes.storm_cell_store = None
        routes.satellite_grids = {}
        routes.start_time = time.time()
        routes.enabled_regions = ["USCOMP"]
        routes._latest_ts_cache = None

        test_app = FastAPI()
        test_app.include_router(routes.router)
        with TestClient(test_app, raise_server_exceptions=False) as c:
            yield c, ts, ts_prev
    finally:
        for name, value in prev.items():
            setattr(routes, name, value)


def _png_size(data: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(data)).size


def _png_alpha(data: bytes) -> np.ndarray:
    """Alpha channel of a decoded PNG as a uint8 array."""
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    return np.asarray(img)[..., 3]


class TestTileModeRegression:
    """The widened ``x``/``y`` path params must not change tile-mode
    behavior: integer coordinates keep working, and the old validation
    outcomes hold (malformed non-numeric coordinates now yield 400
    instead of 422, which is the intended, documented change)."""

    def test_int_coords_200(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/5/2/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_negative_x_400(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/-5/5/2/0_0.png")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Tile coordinates out of range"

    def test_negative_y_400(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/-5/2/0_0.png")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Tile coordinates out of range"

    def test_non_numeric_400(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/abc/5/2/0_0.png")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid tile coordinates"

    def test_mixed_dot_and_int_400(self, client):
        """One dot, one int - neither pure window nor pure tile."""
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/4.5/2/0_0.png")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Invalid tile coordinates"

    def test_x_beyond_2z_400(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/16/5/2/0_0.png")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Tile coordinates out of range"

    def test_y_beyond_2z_400(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/4/3/16/2/0_0.png")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Tile coordinates out of range"


class TestRadarWindow:
    """Lat/lon-centered window mode on the fixture's data block (centre
    lat=37.0, lon=-95.5 at z=5 spans tiles (7,11)-(8,12), which carry
    the synthetic echo)."""

    def test_valid_window_256(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/5/37.0/-95.5/2/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert _png_size(resp.content) == (256, 256)

    def test_valid_window_512(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/512/5/37.0/-95.5/2/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert _png_size(resp.content) == (512, 512)

    def test_unknown_timestamp_404(self, client):
        c, _, _ = client
        resp = c.get("/v2/radar/9999999999/256/5/37.0/-95.5/2/0_0.png")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Frame not found"

    def test_lat_out_of_range_400(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/5/95.0/-95.5/2/0_0.png")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Latitude out of range"

    def test_lat_negative_out_of_range_400(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/5/-90.5/-95.5/2/0_0.png")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Latitude out of range"

    def test_high_latitude_200(self, client):
        """lat=89.0 lies inside [-90, 90]; the Mercator forward mapping
        clamps it to MERCATOR_MAX_LAT downstream."""
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/5/89.0/-95.5/2/0_0.png")
        assert resp.status_code == 200

    def test_lon_wrap_at_seam_200(self, client):
        """lon=179.99 at z=1 puts px0=384: the window wraps across the
        +/-180 seam and re-enters from tile x=0 (see test_window.py's
        seam-wrap case)."""
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/1/0.0/179.99/2/0_0.png")
        assert resp.status_code == 200

    def test_z0_size512_200(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/512/0/37.0/-95.5/2/0_0.png")
        assert resp.status_code == 200
        assert _png_size(resp.content) == (512, 512)

    def test_no_data_window_transparent(self, client):
        """A window far outside every enabled region renders the fully
        transparent constant: PNG decodes with an all-zero alpha."""
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/5/-45.0/120.0/2/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert not _png_alpha(resp.content).any()

    def test_nowcast_timestamp_window_200(self, client, monkeypatch):
        """Nowcast-store fallback: a timestamp the radar store lacks but
        the nowcast store has renders through the same window path (the
        nowcast frame's content is radar-quality, so the blended tile is
        indistinguishable from the plain path here - only the 200 and
        the nowcast max-age bucket are asserted)."""
        c, ts, _ = client
        nc_ts = ts + 300
        monkeypatch.setattr(
            routes, "nowcast_store",
            _StubNowcastStore(_fixture_synthetic_frame(nc_ts)),
        )
        resp = c.get(f"/v2/radar/{nc_ts}/256/5/37.0/-95.5/2/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert "max-age=300" in resp.headers["cache-control"]

    def test_smooth_snow_1_1_200(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/5/37.0/-95.5/2/1_1.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert _png_size(resp.content) == (256, 256)

    def test_smooth_snow_0_0_200(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/5/37.0/-95.5/2/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert _png_size(resp.content) == (256, 256)

    def test_webp_window_200(self, client):
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/5/37.0/-95.5/2/0_0.webp")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/webp"

    def test_overlay_query_params_ignored(self, client):
        """arrows/cells query params are silently ignored in window mode
        (no overlays can be drawn on a stitched canvas)."""
        c, ts, _ = client
        plain = c.get(f"/v2/radar/{ts}/256/5/37.0/-95.5/2/0_0.png")
        with_arrows = c.get(f"/v2/radar/{ts}/256/5/37.0/-95.5/2/0_0.png?arrows=1&cells=1")
        assert plain.status_code == 200
        assert with_arrows.status_code == 200
        assert with_arrows.content == plain.content


class TestRadarWindowCaching:
    def test_second_request_serves_from_cache_same_etag(self, client):
        c, ts, _ = client
        url = f"/v2/radar/{ts}/256/5/37.0/-95.5/2/0_0.png"
        first = c.get(url)
        assert first.status_code == 200
        second = c.get(url)
        assert second.status_code == 200
        assert second.headers["etag"] == first.headers["etag"]
        assert second.content == first.content

    def test_if_none_match_304(self, client):
        c, ts, _ = client
        url = f"/v2/radar/{ts}/256/5/37.0/-95.5/2/0_0.png"
        first = c.get(url)
        assert first.status_code == 200
        etag = first.headers["etag"]
        resp = c.get(url, headers={"If-None-Match": etag})
        assert resp.status_code == 304
        assert resp.content == b""
        assert resp.headers.get("etag") == etag
        assert resp.headers.get("cache-control", "").startswith("public, max-age=")

    def test_latest_frame_max_age_300(self, client, monkeypatch):
        c, ts, _ = client
        monkeypatch.setattr(routes, "_latest_ts_cache", None)
        resp = c.get(f"/v2/radar/{ts}/256/5/37.0/-95.5/2/0_0.png")
        assert resp.status_code == 200
        assert "max-age=300" in resp.headers["cache-control"]

    def test_historical_frame_max_age_7200(self, client, monkeypatch):
        c, _, ts_prev = client
        monkeypatch.setattr(routes, "_latest_ts_cache", None)
        resp = c.get(f"/v2/radar/{ts_prev}/256/5/37.0/-95.5/2/0_0.png")
        assert resp.status_code == 200
        assert "max-age=7200" in resp.headers["cache-control"]


class TestCoverageWindow:
    def test_window_mode_200(self, client):
        c, _, _ = client
        resp = c.get("/v2/coverage/0/256/5/37.0/-95.5/0/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert _png_size(resp.content) == (256, 256)

    def test_window_mode_512(self, client):
        c, _, _ = client
        resp = c.get("/v2/coverage/0/512/5/37.0/-95.5/0/0_0.png")
        assert resp.status_code == 200
        assert _png_size(resp.content) == (512, 512)

    def test_tile_mode_unchanged(self, client):
        c, _, _ = client
        resp = c.get("/v2/coverage/0/256/4/3/5/0/0_0.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_discrimination_tile_int_400_out_of_range(self, client):
        c, _, _ = client
        resp = c.get("/v2/coverage/0/256/4/16/5/0/0_0.png")
        assert resp.status_code == 400

    def test_discrimination_mixed_400(self, client):
        c, _, _ = client
        resp = c.get("/v2/coverage/0/256/4/3/4.5/0/0_0.png")
        assert resp.status_code == 400

    def test_discrimination_lat_out_of_range_400(self, client):
        c, _, _ = client
        resp = c.get("/v2/coverage/0/256/5/95.0/-95.5/0/0_0.png")
        assert resp.status_code == 400

    def test_window_no_data_transparent(self, client):
        c, _, _ = client
        resp = c.get("/v2/coverage/0/256/5/-45.0/120.0/0/0_0.png")
        assert resp.status_code == 200
        assert not _png_alpha(resp.content).any()

    def test_503_when_no_frame(self, client, monkeypatch):
        c, _, _ = client
        monkeypatch.setattr(routes, "frame_store", FrameStore(max_frames=12))
        resp = c.get("/v2/coverage/0/256/5/37.0/-95.5/0/0_0.png")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "No radar data available"


class TestHealthWindowBucket:
    def test_window_entries_reported(self, client):
        """After a window request the /health tile-cache section counts
        the present entry under the window bucket."""
        c, ts, _ = client
        resp = c.get(f"/v2/radar/{ts}/256/5/37.0/-95.5/2/0_0.png")
        assert resp.status_code == 200
        health = c.get("/health")
        assert health.status_code == 200
        tile_cache = health.json()["tile_cache"]
        assert "window_entries" in tile_cache
        assert "window_bytes" in tile_cache
        assert tile_cache["window_entries"] >= 1
        # Existing buckets stay present.
        for key in (
            "geometry_entries", "geometry_bytes",
            "present_entries", "present_bytes",
            "overlay_entries", "overlay_bytes",
            "satellite_entries", "satellite_bytes",
        ):
            assert key in tile_cache, f"missing pre-existing health key {key}"