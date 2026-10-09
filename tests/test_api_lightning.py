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
from datetime import datetime

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


def _make_points(coords, *, base_time: int | None = None) -> np.ndarray:
    """Build a structured flash-point array from (lat, lon, energy) tuples.

    ``base_time`` (epoch seconds) sets the first strike's ``time_s``; later
    entries follow at +10 s.  Callers that assert on paint should pass a
    time inside the target frame's ``(T-600, T]`` slot; the default is a
    fixed epoch far in the past so time-independent assertions (e.g.
    ``/health`` counts) stay stable.
    """
    arr = np.zeros(len(coords), dtype=_POINT_DTYPE)
    base = 1_700_000_000 if base_time is None else base_time
    for i, (lat, lon, energy) in enumerate(coords):
        arr["lat"][i] = lat
        arr["lon"][i] = lon
        arr["energy"][i] = energy
        arr["time_s"][i] = base + i * 10
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
    def points(self) -> np.ndarray:
        return self._points

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
        until_s: int | None = None,
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
        if until_s is not None:
            mask &= points["time_s"] <= until_s
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
    stub = _StubLightningStore(
        _make_points([(lat, lon, 1e-12)], base_time=ts - 60)
    )
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
# Case 2: per-frame slot windows (ruling 4)
# ---------------------------------------------------------------------------

def test_lightning_slot_windows_per_frame(client, monkeypatch):
    """Ruling 4: each frame draws its own 10-minute slot ``(T-600, T]``.

    A strike shows in the frames it happened in, so a past frame paints its
    own slot (Phase 1 asserted plain there).  The lower edge is exclusive
    and the upper edge inclusive.
    """
    c, ts, ts_prev = client
    lat, lon = _tile_center_latlon(4, 3, 6)
    newest = _tile_url(ts)
    past = _tile_url(ts_prev)

    newest_plain = c.get(newest).content
    past_plain = c.get(past).content

    def _strike(at: int) -> _StubLightningStore:
        routes.tile_cache.clear()
        stub = _StubLightningStore(
            _make_points([(lat, lon, 1e-12)], base_time=at)
        )
        monkeypatch.setattr(routes, "lightning_store", stub)
        return stub

    # Inside the newest frame's slot -> paints the newest frame.
    _strike(ts - 60)
    assert c.get(f"{newest}?lightning=dots").content != newest_plain

    # Inside the PAST frame's own slot -> paints the past frame, and does
    # not paint the newest frame.
    _strike(ts_prev - 60)
    assert c.get(f"{past}?lightning=dots").content != past_plain
    assert c.get(f"{newest}?lightning=dots").content == newest_plain

    # A strike after ``ts`` (future slot) does not paint frame ``ts``.
    _strike(ts + 60)
    assert c.get(f"{newest}?lightning=dots").content == newest_plain

    # A strike exactly at ``ts - 600`` belongs to the earlier frame (the
    # lower edge is exclusive): it paints the past frame, not the newest.
    _strike(ts - 600)
    assert c.get(f"{newest}?lightning=dots").content == newest_plain
    assert c.get(f"{past}?lightning=dots").content != past_plain

    # A strike older than every slot window paints nothing.
    _strike(ts - 2000)
    assert c.get(f"{newest}?lightning=dots").content == newest_plain
    assert c.get(f"{past}?lightning=dots").content == past_plain


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
# Case 5: content-fingerprint rekey
# ---------------------------------------------------------------------------

def test_lightning_fingerprint_rekeys_overlay(client, monkeypatch):
    """A change in the slot's strike content (count/newest time) re-keys the
    overlay so the next request re-renders; a bare store-version bump with
    identical points does NOT."""
    c, ts, _ = client
    url = f"{_tile_url(ts)}?lightning=dots"
    lat, lon = _tile_center_latlon(4, 3, 6)
    stub = _StubLightningStore(
        _make_points([(lat, lon, 1e-12)], base_time=ts - 60), version=1,
    )
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

    # Repeat with identical content: in-worker overlay cache hit.
    repeat = c.get(url)
    assert repeat.status_code == 200
    assert repeat.content == first.content
    assert calls["present"] == 1

    # A bare store-version bump with identical points must NOT re-render
    # (the fingerprint, not the version, keys the overlay).
    stub.version = 2
    version_bump = c.get(url)
    assert version_bump.status_code == 200
    assert version_bump.content == first.content
    assert calls["present"] == 1

    # A new strike (different count AND newest time) -> new fingerprint ->
    # re-render.
    stub._points = _make_points(
        [(lat, lon, 1e-12), (lat, lon, 2e-12)], base_time=ts - 60,
    )
    changed = c.get(url)
    assert changed.status_code == 200
    assert calls["present"] == 2

    # Same new content again -> cache hit.
    changed_repeat = c.get(url)
    assert changed_repeat.status_code == 200
    assert changed_repeat.content == changed.content
    assert calls["present"] == 2


def test_lightning_reload_identical_content_keeps_cache_hit(client, monkeypatch):
    """A reader reload returning identical content stays a cache hit.

    Phase 1 keyed the overlay on the live store version, so every reload
    re-rendered; the content fingerprint makes replay/scrubbing a hit once
    a slot's data settles."""
    c, ts, _ = client
    url = f"{_tile_url(ts)}?lightning=dots"
    lat, lon = _tile_center_latlon(4, 3, 6)
    stub = _StubLightningStore(
        _make_points([(lat, lon, 1e-12)], base_time=ts - 60),
    )
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
    reloads_after_first = stub.reload_calls
    assert reloads_after_first >= 1

    # Second request: maybe_reload runs again (counter bumps) but returns
    # the same points, so the overlay cache still serves the first bytes.
    second = c.get(url)
    assert second.status_code == 200
    assert second.content == first.content
    assert calls["present"] == 1
    assert stub.reload_calls > reloads_after_first


# ---------------------------------------------------------------------------
# Case 6: maybe_reload only when the param is active
# ---------------------------------------------------------------------------

def test_maybe_reload_only_when_active(client, monkeypatch):
    c, ts, _ = client
    url = _tile_url(ts)
    lat, lon = _tile_center_latlon(4, 3, 6)
    stub = _StubLightningStore(
        _make_points([(lat, lon, 1e-12)], base_time=ts - 60)
    )
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
    stub = _StubLightningStore(
        _make_points([(lat, lon, 1e-12)], base_time=ts - 60)
    )
    monkeypatch.setattr(routes, "lightning_store", stub)

    # Warm an overlay entry.
    assert c.get(f"{url}?lightning=dots").status_code == 200

    resp = c.get("/health")
    assert resp.status_code == 200
    tile_cache = resp.json()["tile_cache"]
    assert tile_cache["overlay_entries"] >= 1
    assert tile_cache["overlay_bytes"] > 0


# ---------------------------------------------------------------------------
# Case 9: /v2/lightning REST endpoint
# ---------------------------------------------------------------------------

def _recent_points(rows) -> np.ndarray:
    """Build recent flash points from (lat, lon, energy) tuples.

    ``time_s`` descends from now (i=0 newest) so strikes fall inside the
    default 30-minute window; satellite 19 -> ``"goes19"``.
    """
    now = int(time.time())
    arr = np.zeros(len(rows), dtype=_POINT_DTYPE)
    for i, (lat, lon, energy) in enumerate(rows):
        arr["lat"][i] = lat
        arr["lon"][i] = lon
        arr["energy"][i] = energy
        arr["time_s"][i] = now - i * 10
        arr["satellite"][i] = 19
    return arr


def test_lightning_rest_store_none_503(client, monkeypatch):
    c, _, _ = client
    monkeypatch.setattr(routes, "lightning_store", None)
    resp = c.get("/v2/lightning")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Lightning not available"


def test_lightning_rest_no_args_feature_shape(client, monkeypatch):
    c, _, _ = client
    now = int(time.time())
    stub = _StubLightningStore(_recent_points([(35.0, -95.0, 1.5e-5)]))
    monkeypatch.setattr(routes, "lightning_store", stub)

    resp = c.get("/v2/lightning")
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1

    feature = data["features"][0]
    assert feature["type"] == "Feature"
    assert feature["geometry"] == {"type": "Point", "coordinates": [-95.0, 35.0]}
    props = feature["properties"]
    assert props["satellite"] == "goes19"
    assert props["energy"] == pytest.approx(1.5e-5, rel=1e-6)
    # utc is ISO-8601 and round-trips to the point's epoch.
    assert int(datetime.fromisoformat(props["utc"]).timestamp()) == now


def test_lightning_rest_no_args_all_strikes(client, monkeypatch):
    c, _, _ = client
    stub = _StubLightningStore(
        _recent_points([(35.0, -95.0, 1.0e-5), (34.0, -94.0, 2.0e-5)]),
    )
    monkeypatch.setattr(routes, "lightning_store", stub)
    resp = c.get("/v2/lightning")
    assert resp.status_code == 200
    assert len(resp.json()["features"]) == 2


def test_lightning_rest_empty_store(client, monkeypatch):
    c, _, _ = client
    stub = _StubLightningStore(np.empty(0, dtype=_POINT_DTYPE))
    monkeypatch.setattr(routes, "lightning_store", stub)
    resp = c.get("/v2/lightning")
    assert resp.status_code == 200
    assert resp.json()["features"] == []


def test_lightning_rest_point_radius_default(client, monkeypatch):
    """Default radius_km=25 keeps the ~10 km strike, drops the ~60 km one."""
    c, _, _ = client
    near = (35.0, -95.0, 1.0e-5)
    far = (35.0 + 60.0 / 111.0, -95.0, 2.0e-5)
    stub = _StubLightningStore(_recent_points([near, far]))
    monkeypatch.setattr(routes, "lightning_store", stub)

    resp = c.get("/v2/lightning?lat=35.0&lon=-95.0")
    assert resp.status_code == 200
    features = resp.json()["features"]
    assert len(features) == 1
    assert features[0]["geometry"]["coordinates"] == [-95.0, 35.0]


def test_lightning_rest_lone_lat_400(client, monkeypatch):
    c, _, _ = client
    stub = _StubLightningStore(_recent_points([(35.0, -95.0, 1.0e-5)]))
    monkeypatch.setattr(routes, "lightning_store", stub)
    resp = c.get("/v2/lightning?lat=35.0")
    assert resp.status_code == 400
    assert resp.json()["detail"] == "lat and lon must be provided together"


def test_lightning_rest_lone_lon_400(client, monkeypatch):
    c, _, _ = client
    stub = _StubLightningStore(_recent_points([(35.0, -95.0, 1.0e-5)]))
    monkeypatch.setattr(routes, "lightning_store", stub)
    resp = c.get("/v2/lightning?lon=-95.0")
    assert resp.status_code == 400
    assert resp.json()["detail"] == "lat and lon must be provided together"


def test_lightning_rest_bbox_keeps_inside(client, monkeypatch):
    c, _, _ = client
    inside = (35.0, -95.0, 1.0e-5)
    outside = (10.0, 10.0, 2.0e-5)
    stub = _StubLightningStore(_recent_points([inside, outside]))
    monkeypatch.setattr(routes, "lightning_store", stub)

    resp = c.get("/v2/lightning?bbox=-100,30,-90,40")
    assert resp.status_code == 200
    features = resp.json()["features"]
    assert len(features) == 1
    assert features[0]["geometry"]["coordinates"] == [-95.0, 35.0]


@pytest.mark.parametrize(
    ("bbox", "detail"),
    [
        ("-100,30,-90", "bbox must be: west,south,east,north"),
        ("-100,xx,-90,40", "bbox values must be numeric"),
        ("-200,30,-90,40", "bbox values out of range"),
        ("-100,95,-90,40", "bbox values out of range"),
        ("-90,30,-100,40", "bbox values out of range"),
        ("-100,40,-90,30", "bbox values out of range"),
    ],
)
def test_lightning_rest_bbox_400(client, monkeypatch, bbox, detail):
    c, _, _ = client
    stub = _StubLightningStore(_recent_points([(35.0, -95.0, 1.0e-5)]))
    monkeypatch.setattr(routes, "lightning_store", stub)
    resp = c.get(f"/v2/lightning?bbox={bbox}")
    assert resp.status_code == 400
    assert resp.json()["detail"] == detail


def test_lightning_rest_minutes_clamped(client, monkeypatch):
    """minutes=120 clamps to the 30-min retention: the 45-min-old point drops."""
    c, _, _ = client
    now = int(time.time())
    arr = np.zeros(2, dtype=_POINT_DTYPE)
    arr["lat"] = [35.0, 36.0]
    arr["lon"] = [-95.0, -94.0]
    arr["energy"] = [1.0e-5, 2.0e-5]
    arr["time_s"] = [now - 10 * 60, now - 45 * 60]
    arr["satellite"] = 19
    stub = _StubLightningStore(arr)
    monkeypatch.setattr(routes, "lightning_store", stub)

    resp = c.get("/v2/lightning?minutes=120")
    assert resp.status_code == 200
    features = resp.json()["features"]
    assert len(features) == 1
    assert features[0]["geometry"]["coordinates"] == [-95.0, 35.0]


def test_lightning_rest_limit_newest_first(client, monkeypatch):
    c, _, _ = client
    now = int(time.time())
    arr = np.zeros(3, dtype=_POINT_DTYPE)
    # i=0 newest, i=2 oldest.
    arr["lat"] = [35.0, 34.0, 33.0]
    arr["lon"] = [-95.0, -94.0, -93.0]
    arr["energy"] = [1.0e-5, 2.0e-5, 3.0e-5]
    arr["time_s"] = [now, now - 60, now - 120]
    arr["satellite"] = 19
    stub = _StubLightningStore(arr)
    monkeypatch.setattr(routes, "lightning_store", stub)

    resp = c.get("/v2/lightning?limit=2")
    assert resp.status_code == 200
    features = resp.json()["features"]
    assert len(features) == 2
    coords = [f["geometry"]["coordinates"] for f in features]
    assert coords == [[-95.0, 35.0], [-94.0, 34.0]]
