# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Route tests for the ``?lightning=`` GOES GLM flash-point tile overlay.

Mirrors tests/test_api.py's cells-overlay idioms (a duck-typed store on the
``routes`` singleton, a warm geometry cache, and byte comparisons against
the plain tile) plus tests/test_window_routes.py's lazily-wired, restored
route state so this module never clobbers another test module's globals.
"""

import math
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.lightning

from librewxr.api import routes
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH

_POINT_DTYPE = np.dtype([
    ("time_s", "int64"),
    ("lat", "float32"),
    ("lon", "float32"),
    ("energy", "float32"),
    ("satellite", "int8"),
])


def _tile_center_latlon(z: int, x: int, y: int) -> tuple[float, float]:
    """Inverse Web-Mercator of the tile center.

    The returned point forward-projects back to tile-local pixel
    ``(tile_size/2, tile_size/2)``, so a flash placed there always paints.
    """
    n = 1 << z
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 0.5) / n))))
    return lat, lon


def _make_points(coords) -> np.ndarray:
    """Build a structured flash-point array from (lat, lon, energy) tuples."""
    arr = np.zeros(len(coords), dtype=_POINT_DTYPE)
    for i, (lat, lon, energy) in enumerate(coords):
        arr["lat"][i] = lat
        arr["lon"][i] = lon
        arr["energy"][i] = energy
        arr["time_s"][i] = 1_700_000_000 + i * 10
        arr["satellite"][i] = 19
    return arr


class _StubLightningStore:
    """Duck-typed LightningStore for the routes + /health surface.

    Mirrors the read-only property surface the route + health handler
    touch, with a mutable ``version`` so tests can simulate a pipeline
    artifact regeneration re-keying the overlay cache.
    """

    def __init__(self, points: np.ndarray | None = None, *, version: int = 1) -> None:
        self._points = (
            points if points is not None else np.empty(0, dtype=_POINT_DTYPE)
        )
        self.version = version
        self.last_updated = 1_700_000_000.0
        self.ceiling_trimmed_total = 0
        self.meta = {"satellites": {"18": {"age_s": 12}, "19": {"age_s": 9}}}
        self.reload_calls = 0

    @property
    def total_count(self) -> int:
        return int(self._points.shape[0])

    @property
    def window_start_s(self) -> int | None:
        if self._points.shape[0] == 0:
            return None
        return int(self._points["time_s"].min())

    @property
    def window_end_s(self) -> int | None:
        if self._points.shape[0] == 0:
            return None
        return int(self._points["time_s"].max())

    async def maybe_reload(self) -> bool:
        self.reload_calls += 1
        return True

    def points_in(
        self, lat0: float, lat1: float, lon0: float, lon1: float, *,
        since_s: int | None = None,
    ) -> np.ndarray:
        points = self._points
        if points.shape[0] == 0:
            return np.empty(0, dtype=_POINT_DTYPE)
        mask = (
            (points["lat"] >= lat0)
            & (points["lat"] <= lat1)
            & (points["lon"] >= lon0)
            & (points["lon"] <= lon1)
        )
        if since_s is not None:
            mask &= points["time_s"] >= since_s
        return points[mask]


_ROUTE_STATE_NAMES = (
    "frame_store", "tile_cache", "ecmwf_grid", "nwp_chain",
    "precip_mask", "nowcast_store", "shared_tile_store",
    "tile_request_tracker", "storm_cell_store", "satellite_grids",
    "start_time", "enabled_regions", "_latest_ts_cache",
    "lightning_store", "lightning_enabled",
)


@pytest.fixture(scope="module")
def client():
    """Module-scoped app + state wiring for this file's tests.

    The routes-module singletons are set lazily on first use (NOT at
    import time) and restored on teardown: other test modules build their
    own apps from the same module-level singletons, so clobbering them
    during collection would break their later tests.
    """
    import asyncio

    store = FrameStore(max_frames=12)
    cache = TileCache(max_mb=10)
    ts = int(time.time() // 300) * 300
    ts_prev = ts - 600

    data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
    data[2500:2700, 6000:6200] = 128

    asyncio.run(store.add_frame(RadarFrame(timestamp=ts, regions={"USCOMP": data})))
    asyncio.run(store.add_frame(RadarFrame(timestamp=ts_prev, regions={"USCOMP": data})))

    prev = {name: getattr(routes, name) for name in _ROUTE_STATE_NAMES}
    try:
        # Wire shared state directly - same as main.py does after lifespan init
        routes.frame_store = store
        routes.tile_cache = cache
        routes.ecmwf_grid = None
        routes.nwp_chain = None
        routes.precip_mask = None
        routes.nowcast_store = None
        routes.shared_tile_store = None
        routes.tile_request_tracker = None
        routes.storm_cell_store = None
        routes.satellite_grids = {}
        routes.start_time = time.time()
        routes.enabled_regions = ["USCOMP"]
        routes._latest_ts_cache = None
        routes.lightning_store = None
        routes.lightning_enabled = False

        test_app = FastAPI()
        test_app.include_router(routes.router)
        with TestClient(test_app, raise_server_exceptions=False) as c:
            yield c, ts, ts_prev
    finally:
        for name, value in prev.items():
            setattr(routes, name, value)


@pytest.fixture(autouse=True)
def _restore_lightning_state():
    """Snapshot the lightning singletons and reset the present cache.

    Monkeypatch already restores per-test patches; this guards any test
    that assigns the globals directly and keeps tile bytes from leaking
    across cases (the warm-geometry trick warms the cache inside a test).
    """
    saved = (routes.lightning_store, routes.lightning_enabled)
    if routes.tile_cache is not None:
        routes.tile_cache.clear()
    yield
    routes.lightning_store, routes.lightning_enabled = saved
    if routes.tile_cache is not None:
        routes.tile_cache.clear()


def _tile_url(ts: int) -> str:
    # 4/3/6 is the z=4 tile that actually covers the fixture's radar block
    # - a transparent tile short-circuits before any overlay.
    return f"/v2/radar/{ts}/256/4/3/6/2/0_0.png"


# ---------------------------------------------------------------------------
# Case 1: param parse / styles
# ---------------------------------------------------------------------------

def test_lightning_param_styles(client, monkeypatch):
    c, ts, _ = client
    url = _tile_url(ts)
    plain = c.get(url)
    assert plain.status_code == 200

    lat, lon = _tile_center_latlon(4, 3, 6)
    stub = _StubLightningStore(_make_points([(lat, lon, 1e-12)]))
    monkeypatch.setattr(routes, "lightning_store", stub)

    for value in ("1", "true", "dots"):
        resp = c.get(f"{url}?lightning={value}")
        assert resp.status_code == 200
        assert resp.content != plain.content, f"?lightning={value} did not paint"

    dots = c.get(f"{url}?lightning=dots")
    bolts = c.get(f"{url}?lightning=bolts")
    assert bolts.status_code == 200
    assert bolts.content != plain.content
    assert bolts.content != dots.content

    # Off and unknown values are byte-identical to the plain tile.
    assert c.get(f"{url}?lightning=").content == plain.content
    assert c.get(f"{url}?lightning=sparkles").content == plain.content


# ---------------------------------------------------------------------------
# Case 2: newest-analysis-frame only
# ---------------------------------------------------------------------------

def test_lightning_only_on_newest_frame(client, monkeypatch):
    c, ts, ts_prev = client
    lat, lon = _tile_center_latlon(4, 3, 6)
    stub = _StubLightningStore(_make_points([(lat, lon, 1e-12)]))
    monkeypatch.setattr(routes, "lightning_store", stub)

    newest = _tile_url(ts)
    past = _tile_url(ts_prev)

    assert c.get(f"{newest}?lightning=dots").content != c.get(newest).content
    assert c.get(f"{past}?lightning=dots").content == c.get(past).content


# ---------------------------------------------------------------------------
# Case 3: store unwired -> degrade
# ---------------------------------------------------------------------------

def test_lightning_store_none_degrades(client, monkeypatch):
    c, ts, _ = client
    url = _tile_url(ts)
    plain = c.get(url)

    monkeypatch.setattr(routes, "lightning_store", None)
    resp = c.get(f"{url}?lightning=dots")
    assert resp.status_code == 200
    assert resp.content == plain.content


# ---------------------------------------------------------------------------
# Case 4: no points in the tile -> degrade
# ---------------------------------------------------------------------------

def test_lightning_empty_points_degrades(client, monkeypatch):
    c, ts, _ = client
    url = _tile_url(ts)
    plain = c.get(url)

    stub = _StubLightningStore(np.empty(0, dtype=_POINT_DTYPE))
    monkeypatch.setattr(routes, "lightning_store", stub)
    assert c.get(f"{url}?lightning=dots").content == plain.content


# ---------------------------------------------------------------------------
# Case 5: version rekey
# ---------------------------------------------------------------------------

def test_lightning_version_bump_rekeys_overlay(client, monkeypatch):
    """Bumping the store version re-keys the overlay so the next request
    re-renders instead of serving the previous artifact's cached bytes."""
    c, ts, _ = client
    url = f"{_tile_url(ts)}?lightning=dots"
    lat, lon = _tile_center_latlon(4, 3, 6)
    stub = _StubLightningStore(_make_points([(lat, lon, 1e-12)]), version=1)
    monkeypatch.setattr(routes, "lightning_store", stub)
    routes.tile_cache.clear()

    orig_present = routes._present_tile_async
    calls = {"present": 0}

    async def counting_present(*args, **kwargs):
        calls["present"] += 1
        return await orig_present(*args, **kwargs)

    monkeypatch.setattr(routes, "_present_tile_async", counting_present)

    first = c.get(url)
    assert first.status_code == 200
    assert calls["present"] == 1

    # Repeat under the same version: in-worker overlay cache hit.
    repeat = c.get(url)
    assert repeat.status_code == 200
    assert repeat.content == first.content
    assert calls["present"] == 1

    # Pipeline re-saves the artifact: new version -> new key -> re-render.
    stub.version = 2
    again = c.get(url)
    assert again.status_code == 200
    assert again.content == first.content
    assert calls["present"] == 2


# ---------------------------------------------------------------------------
# Case 6: maybe_reload only when the param is active
# ---------------------------------------------------------------------------

def test_maybe_reload_only_when_active(client, monkeypatch):
    c, ts, _ = client
    url = _tile_url(ts)
    lat, lon = _tile_center_latlon(4, 3, 6)
    stub = _StubLightningStore(_make_points([(lat, lon, 1e-12)]))
    monkeypatch.setattr(routes, "lightning_store", stub)
    routes.tile_cache.clear()

    off = c.get(url)
    assert off.status_code == 200
    assert stub.reload_calls == 0

    on = c.get(f"{url}?lightning=dots")
    assert on.status_code == 200
    assert stub.reload_calls >= 1


# ---------------------------------------------------------------------------
# Case 7: /health lightning section
# ---------------------------------------------------------------------------

def test_health_lightning_disabled_when_store_absent(client, monkeypatch):
    c, _, _ = client
    monkeypatch.setattr(routes, "lightning_store", None)
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["lightning"] == {"enabled": False}


def test_health_lightning_reports_stub(client, monkeypatch):
    c, _, _ = client
    lat, lon = _tile_center_latlon(4, 3, 6)
    stub = _StubLightningStore(_make_points([(lat, lon, 1e-12)]), version=7)
    monkeypatch.setattr(routes, "lightning_store", stub)

    resp = c.get("/health")
    assert resp.status_code == 200
    section = resp.json()["lightning"]
    assert section["enabled"] is True
    assert section["points"] == 1
    assert section["version"] == 7
    assert section["satellites"] == stub.meta["satellites"]


# ---------------------------------------------------------------------------
# Case 8: /health overlay-kind accounting
# ---------------------------------------------------------------------------

def test_health_counts_overlay_entries(client, monkeypatch):
    """Overlay cache entries must land in the /health overlay bucket.

    The classifier keys off the overlay key length; a wrong length sent
    overlay entries into no bucket at all, silently undercounting
    ``overlay_entries``/``overlay_bytes``.
    """
    c, ts, _ = client
    url = _tile_url(ts)

    # Warm a geometry entry and a present entry with a plain request.
    assert c.get(url).status_code == 200

    lat, lon = _tile_center_latlon(4, 3, 6)
    stub = _StubLightningStore(_make_points([(lat, lon, 1e-12)]))
    monkeypatch.setattr(routes, "lightning_store", stub)

    # Warm an overlay entry.
    assert c.get(f"{url}?lightning=dots").status_code == 200

    resp = c.get("/health")
    assert resp.status_code == 200
    tile_cache = resp.json()["tile_cache"]
    assert tile_cache["overlay_entries"] >= 1
    assert tile_cache["overlay_bytes"] > 0
