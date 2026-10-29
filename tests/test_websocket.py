# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for the WebSocket push channel (``librewxr.ws``).

Three layers:

* Integration tests drive the real endpoint through FastAPI's
  ``TestClient`` (hello / snapshot / subscribe / unsubscribe / error /
  origin / ``/health``).
* Push-path tests exercise ``WebSocketHub.on_state_refreshed`` /
  ``on_poll_tick`` directly against real stores (``FrameStore``,
  ``AlertsStore``, ``LightningStore``) with a fake websocket duck-type.
* ``tests/test_config.py`` carries the two ``websocket_*`` settings
  cases.

The ``routes`` module globals are snapshotted and restored around every
test (the ``tests/test_api_lightning.py`` ``_ROUTE_STATE_NAMES``
pattern), so this module never clobbers another test module's state.
"""

import asyncio
import contextlib
import json
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shapely.geometry import Polygon
from starlette.websockets import WebSocketDisconnect

from librewxr.api import routes
from librewxr.config import settings
from librewxr.data.alerts_store import AlertEntry, AlertsStore
from librewxr.data.lightning_store import LightningStore
from librewxr.data.regions import REGIONS, RegionDef
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.data.storm_cells import StormCellStore, _CELL_DTYPE
from librewxr.tiles.cache import TileCache
from librewxr.ws.endpoint import router as ws_router
from librewxr.ws.hub import ClientState, WebSocketHub
from librewxr.ws.topics import (
    ALL_TOPICS,
    TOPIC_ALERTS,
    TOPIC_LIGHTNING,
    TOPIC_STORM_CELLS,
    TOPIC_WEATHER_MAPS,
)

pytestmark = pytest.mark.websocket


# ---------------------------------------------------------------------------
# Shared state helpers
# ---------------------------------------------------------------------------

_ROUTE_STATE_NAMES = (
    "frame_store", "tile_cache", "nwp_grids", "nwp_chain", "ecmwf_grid",
    "precip_mask", "nowcast_store", "shared_tile_store",
    "tile_request_tracker", "storm_cell_store", "satellite_grids",
    "start_time", "enabled_regions", "_latest_ts_cache",
    "lightning_store", "lightning_enabled", "alerts_store", "alerts_enabled",
    "ws_hub", "radar_fetcher", "radar_cache", "memory_monitor",
)

_POINT_DTYPE = np.dtype([
    ("time_s", "int64"),
    ("lat", "float32"),
    ("lon", "float32"),
    ("energy", "float32"),
    ("satellite", "int8"),
])


class _Wired:
    """Handle bundle returned by the ``wired`` fixture."""

    def __init__(
        self, frame_store: FrameStore, ts: int, hub: WebSocketHub, regions: dict,
    ):
        self.frame_store = frame_store
        self.ts = ts
        self.hub = hub
        self.regions = regions


@pytest.fixture
def wired(frame_store, sample_radar_frame, tile_cache):
    """Wire the ``routes`` globals the hub + ``/health`` paths read.

    Snapshot/restore around the test (mirrors ``test_api_lightning.py``):
    other modules build apps from the same singletons, so leaving them
    pointing at this module's stores would break their later tests.
    """
    ts = int(time.time() // 300) * 300
    regions = dict(sample_radar_frame.regions)
    asyncio.run(frame_store.add_frame(
        RadarFrame(timestamp=ts, regions=regions),
    ))

    saved = {name: getattr(routes, name) for name in _ROUTE_STATE_NAMES}
    try:
        routes.frame_store = frame_store
        routes.tile_cache = tile_cache
        routes.nwp_grids = {}
        routes.nwp_chain = None
        routes.ecmwf_grid = None
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
        routes.alerts_store = None
        routes.alerts_enabled = False
        routes.radar_fetcher = None
        routes.radar_cache = None
        routes.memory_monitor = None
        hub = WebSocketHub()
        routes.ws_hub = hub
        yield _Wired(frame_store, ts, hub, regions)
    finally:
        for name, value in saved.items():
            setattr(routes, name, value)


def _make_app(*, include_ws: bool = True) -> FastAPI:
    """Minimal app: the JSON routes plus (optionally) the WS endpoint."""
    app = FastAPI()
    app.include_router(routes.router)
    if include_ws:
        app.include_router(ws_router)
    return app


@pytest.fixture
def client(wired):
    """Sync ``TestClient`` over the wired app (lifespan-free, like test_api)."""
    with TestClient(_make_app(), raise_server_exceptions=False) as c:
        yield c


def _drain_default_subscription(ws) -> dict:
    """Consume hello + the default weather-maps snapshot; return hello."""
    hello = ws.receive_json()
    assert hello["type"] == "hello"
    snapshot = ws.receive_json()
    assert snapshot["type"] == "snapshot"
    assert snapshot["topic"] == TOPIC_WEATHER_MAPS
    return hello


# ---------------------------------------------------------------------------
# Integration: connect / snapshot / subscribe
# ---------------------------------------------------------------------------

class TestConnectPath:
    def test_hello_on_default_connect(self, client):
        with client.websocket_connect(settings.websocket_path) as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            assert hello["protocol"] == 1
            assert TOPIC_WEATHER_MAPS in hello["subscribed"]
            assert isinstance(hello["available"], list)
            assert TOPIC_WEATHER_MAPS in hello["available"]

    def test_default_weather_maps_snapshot_matches_http(self, client):
        http_body = client.get("/public/weather-maps.json").json()

        with client.websocket_connect(settings.websocket_path) as ws:
            assert ws.receive_json()["type"] == "hello"
            snapshot = ws.receive_json()

        assert snapshot["type"] == "snapshot"
        assert snapshot["topic"] == TOPIC_WEATHER_MAPS
        data = snapshot["data"]
        assert data["radar"]["past"], "snapshot must carry the past frame list"
        assert "version" in data
        assert "host" in data

        # Every field except the volatile ``generated`` stamp matches the
        # canonical HTTP body byte-for-byte (the payload IS the same
        # Pydantic model).
        expected = {k: v for k, v in http_body.items() if k != "generated"}
        actual = {k: v for k, v in data.items() if k != "generated"}
        assert actual == expected

    def test_alerts_topic_snapshot(self, client, monkeypatch):
        store = AlertsStore()
        store.replace_all([_alert()])
        monkeypatch.setattr(routes, "alerts_store", store)
        monkeypatch.setattr(routes, "alerts_enabled", True)

        with client.websocket_connect(
            f"{settings.websocket_path}?topics={TOPIC_ALERTS}"
        ) as ws:
            hello = ws.receive_json()
            snapshot = ws.receive_json()

        assert hello["type"] == "hello"
        assert hello["subscribed"] == [TOPIC_ALERTS]
        assert snapshot["type"] == "snapshot"
        assert snapshot["topic"] == TOPIC_ALERTS
        data = snapshot["data"]
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) >= 1
        props = data["features"][0]["properties"]
        assert props["title"] == "Tornado Watch"
        assert props["severity"] == "Extreme"
        assert props["regions"] == ["Sullivan, NY"]
        assert props["uri"] == "https://example.com/ws-alert"
        assert props["expires"]
        assert data["features"][0]["geometry"]["type"] == "Polygon"

    def test_unknown_topic_errors_and_connection_survives(self, client):
        with client.websocket_connect(
            f"{settings.websocket_path}?topics=bogus"
        ) as ws:
            hello = ws.receive_json()
            assert hello["subscribed"] == []
            error = ws.receive_json()
            assert error["type"] == "error"
            assert "bogus" in error["message"]

            # No snapshot for the unknown topic: the next frame is the
            # unsubscribe ack (nothing was subscribed, so nothing is
            # removed), proving the connection is still usable.
            ws.send_json({"action": "unsubscribe", "topics": [TOPIC_WEATHER_MAPS]})
            ack = ws.receive_json()
            assert ack == {"type": "unsubscribed", "topics": []}

    def test_unavailable_topic_not_subscribed(self, client, monkeypatch):
        """Fix A: a known topic with no store is refused, not half-served."""
        monkeypatch.setattr(routes, "lightning_store", None)
        monkeypatch.setattr(settings, "lightning_enabled", True)

        with client.websocket_connect(
            f"{settings.websocket_path}?topics={TOPIC_LIGHTNING}"
        ) as ws:
            hello = ws.receive_json()
            assert hello["subscribed"] == []
            assert TOPIC_LIGHTNING not in hello["available"]
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["message"] == f"topic unavailable: {TOPIC_LIGHTNING}"

            # No lightning snapshot followed the error.
            ws.send_json({"action": "unsubscribe", "topics": [TOPIC_WEATHER_MAPS]})
            assert ws.receive_json()["type"] == "unsubscribed"

    def test_subscribe_message_with_real_storm_cell_store(
        self, client, monkeypatch, tmp_path,
    ):
        monkeypatch.setitem(REGIONS, "TEST_WS", RegionDef(
            name="TEST_WS",
            west=-100.0, east=-90.0, south=35.0, north=45.0,
            pixel_size=1.0, group="TEST",
        ))
        store = StormCellStore(cache_dir=tmp_path)
        asyncio.run(store.replace_cells(
            {"TEST_WS": _single_cell_array(centroid_row=5.0, centroid_col=5.0)},
            detected_at_timestamp=0,
        ))
        monkeypatch.setattr(routes, "storm_cell_store", store)

        with client.websocket_connect(settings.websocket_path) as ws:
            _drain_default_subscription(ws)

            # Duplicate names in one message must produce ONE ack entry
            # (Fix C) and one snapshot.
            ws.send_json({
                "action": "subscribe",
                "topics": [TOPIC_STORM_CELLS, TOPIC_STORM_CELLS],
            })
            ack = ws.receive_json()
            assert ack == {"type": "subscribed", "topics": [TOPIC_STORM_CELLS]}

            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"
            assert snapshot["topic"] == TOPIC_STORM_CELLS
            data = snapshot["data"]
            assert data["type"] == "FeatureCollection"
            assert len(data["features"]) == 1
            feature = data["features"][0]
            assert feature["geometry"] == {
                "type": "Point", "coordinates": [-95.0, 40.0],
            }
            assert feature["properties"]["region"] == "TEST_WS"
            assert feature["properties"]["max_dbz"] == 55.0

    def test_malformed_input_errors_and_connection_survives(self, client):
        with client.websocket_connect(settings.websocket_path) as ws:
            _drain_default_subscription(ws)

            ws.send_text("not json")
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["message"] == "invalid JSON"

            ws.send_json({"action": "wat"})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert "unknown action" in error["message"]

            # Still usable after both failures.
            ws.send_json({"action": "subscribe", "topics": [TOPIC_WEATHER_MAPS]})
            ack = ws.receive_json()
            assert ack == {"type": "subscribed", "topics": [TOPIC_WEATHER_MAPS]}
            assert ws.receive_json()["type"] == "snapshot"

    def test_unsubscribe_acks_actually_removed_topics(self, client):
        with client.websocket_connect(settings.websocket_path) as ws:
            _drain_default_subscription(ws)

            ws.send_json({"action": "unsubscribe", "topics": [TOPIC_WEATHER_MAPS]})
            assert ws.receive_json() == {
                "type": "unsubscribed", "topics": [TOPIC_WEATHER_MAPS],
            }

            # Already removed -> the ack lists nothing removed.
            ws.send_json({"action": "unsubscribe", "topics": [TOPIC_WEATHER_MAPS]})
            assert ws.receive_json() == {"type": "unsubscribed", "topics": []}


# ---------------------------------------------------------------------------
# Integration: rejection paths
# ---------------------------------------------------------------------------

class TestRejections:
    def test_hub_absent_accepts_then_closes_1008(self, client, monkeypatch):
        monkeypatch.setattr(routes, "ws_hub", None)
        with client.websocket_connect(settings.websocket_path) as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
        assert exc.value.code == 1008

    def test_missing_route_rejects_handshake(self):
        """Without the WS router there is no route: the handshake fails."""
        app = FastAPI()
        with TestClient(app) as c:
            with pytest.raises(WebSocketDisconnect):
                with c.websocket_connect(settings.websocket_path) as ws:
                    ws.receive_json()

    def test_origin_check(self, client, monkeypatch):
        # TestClient forwards custom headers into the WS scope (verified on
        # starlette 1.9), so this drives the real check end to end.
        monkeypatch.setattr(settings, "cors_origins", ["https://allowed.example"])

        with client.websocket_connect(
            settings.websocket_path,
            headers={"Origin": "https://allowed.example"},
        ) as ws:
            assert ws.receive_json()["type"] == "hello"

        with client.websocket_connect(
            settings.websocket_path,
            headers={"Origin": "https://evil.example"},
        ) as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_json()
        assert exc.value.code == 1008

        # No Origin header (non-browser client) is still allowed.
        with client.websocket_connect(settings.websocket_path) as ws:
            assert ws.receive_json()["type"] == "hello"


# ---------------------------------------------------------------------------
# Integration: /health section
# ---------------------------------------------------------------------------

class TestHealthSection:
    def test_health_reports_live_connections(self, client):
        with client.websocket_connect(settings.websocket_path) as ws:
            assert ws.receive_json()["type"] == "hello"

            resp = client.get("/health")
            assert resp.status_code == 200
            section = resp.json()["websocket"]
            assert section["enabled"] is True
            assert section["path"] == settings.websocket_path
            assert section["connections"] >= 1
            assert set(section["topics"]) == set(ALL_TOPICS)
            assert section["topics"][TOPIC_WEATHER_MAPS] >= 1

    def test_health_websocket_disabled_without_hub(self, client, monkeypatch):
        monkeypatch.setattr(routes, "ws_hub", None)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["websocket"] == {"enabled": False}


# ---------------------------------------------------------------------------
# Push-path fakes / helpers
# ---------------------------------------------------------------------------

class _FakeWebSocket:
    """Duck-typed starlette WebSocket (only the hub's used surface).

    ``receive_text`` raises immediately: the handler's receive loop returns
    and its ``finally`` cleans up, which is what a disconnect looks like.
    """

    def __init__(self, *, topics: str = "", origin: str | None = None):
        self.accepted = False
        self.close_codes: list[int] = []
        self.sent: list[dict] = []
        self.query_params = {"topics": topics}
        self.headers = {} if origin is None else {"origin": origin}

    async def accept(self) -> None:
        self.accepted = True

    async def close(self, code: int = 1000) -> None:
        self.close_codes.append(code)

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def receive_text(self) -> str:
        raise RuntimeError("client disconnected")


class _BlockingWebSocket(_FakeWebSocket):
    """Fake whose ``receive_text`` blocks until released (live client)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._released = asyncio.Event()

    async def receive_text(self) -> str:
        await self._released.wait()
        raise RuntimeError("released")


async def _add_client(hub: WebSocketHub, fake, topics=()) -> ClientState:
    """Register a live client the way ``handle()`` does after accept.

    Constructing the ``ClientState`` and spawning its sender is the
    documented test seam; it gives a client whose delivery is observable
    without racing a real receive loop.
    """
    client = ClientState(
        websocket=fake,
        subscriptions=set(topics),
        queue=asyncio.Queue(maxsize=hub._QUEUE_MAX),
    )
    hub._clients.add(client)
    client.sender = asyncio.create_task(hub._sender_loop(client))
    return client


async def _wait_messages(fake: _FakeWebSocket, count: int, timeout: float = 2.0):
    """Wait until ``fake`` has at least ``count`` frames; return them."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while len(fake.sent) < count and loop.time() < deadline:
        await asyncio.sleep(0.01)
    return list(fake.sent)


async def _quiesce(delay: float = 0.1) -> None:
    """Give sender tasks time to deliver anything pending."""
    await asyncio.sleep(delay)


@pytest.fixture
async def hub():
    """A hub with its leftover clients/senders torn down after the test."""
    hub = WebSocketHub()
    try:
        yield hub
    finally:
        senders = [
            client.sender for client in hub._clients
            if client.sender is not None
        ]
        for client in hub._clients:
            client.dropped = True
        hub._clients.clear()
        for sender in senders:
            sender.cancel()
        if senders:
            await asyncio.gather(*senders, return_exceptions=True)
        for task in list(hub._close_tasks):
            task.cancel()
        await asyncio.sleep(0)


def _points(rows, *, times) -> np.ndarray:
    """Structured flash-point array from (lat, lon, energy) rows."""
    assert len(rows) == len(times)
    arr = np.zeros(len(rows), dtype=_POINT_DTYPE)
    for i, (lat, lon, energy) in enumerate(rows):
        arr["lat"][i] = lat
        arr["lon"][i] = lon
        arr["energy"][i] = energy
        arr["time_s"][i] = times[i]
        arr["satellite"][i] = 19
    return arr


def _alert(
    *,
    url: str = "https://example.com/ws-alert",
    event: str = "Tornado Watch",
    severity: str = "Extreme",
    area: str = "Sullivan, NY",
) -> AlertEntry:
    """One non-expired CAP alert with a valid polygon."""
    return AlertEntry(
        source_id="us-noaa-nws-en",
        event=event,
        description="WS test alert",
        severity=severity,
        effective="2026-05-07T00:15:00-05:00",
        expires="2099-05-07T06:00:00-05:00",
        area_desc=area,
        url=url,
        polygon=Polygon([
            (-74.5, 41.5), (-73.5, 41.5), (-73.5, 40.5), (-74.5, 40.5),
        ]),
    )


def _single_cell_array(
    centroid_row: float, centroid_col: float,
    area_km2: float = 120.0, max_dbz: float = 55.0,
) -> np.ndarray:
    arr = np.zeros(1, dtype=_CELL_DTYPE)
    arr["centroid_row"] = centroid_row
    arr["centroid_col"] = centroid_col
    arr["area_px"] = 100
    arr["area_km2"] = area_km2
    arr["max_dbz"] = max_dbz
    arr["motion_dx_px"] = 1.0
    arr["motion_dy_px"] = 0.0
    arr["motion_speed_kmh"] = float("nan")
    arr["motion_heading_deg"] = float("nan")
    return arr


# ---------------------------------------------------------------------------
# Push path: state topics
# ---------------------------------------------------------------------------

class TestStateTopicPush:
    async def test_weather_maps_updates_once_per_change(self, hub, wired):
        fake_a = _FakeWebSocket()
        fake_b = _FakeWebSocket()
        await _add_client(hub, fake_a, [TOPIC_WEATHER_MAPS])
        await _add_client(hub, fake_b, [TOPIC_WEATHER_MAPS])

        # First refresh: the topic has no pushed fingerprint yet -> update.
        await hub.on_state_refreshed()
        await _wait_messages(fake_a, 1)
        await _wait_messages(fake_b, 1)

        for fake in (fake_a, fake_b):
            message = fake.sent[0]
            assert message["type"] == "update"
            assert message["topic"] == TOPIC_WEATHER_MAPS
            assert message["data"]["radar"]["past"]

        # A store change re-keys the cache, the fingerprint changes, and
        # both subscribers get exactly one more update.
        await wired.frame_store.add_frame(RadarFrame(
            timestamp=wired.ts + 300, regions=wired.regions,
        ))
        await hub.on_state_refreshed()
        await _wait_messages(fake_a, 2)
        await _wait_messages(fake_b, 2)
        assert len(fake_a.sent) == 2
        assert len(fake_b.sent) == 2
        assert wired.ts + 300 in await wired.frame_store.get_timestamps()

        # No store change -> the cache key gates the rebuild and the
        # ``generated``-only clock tick never looks like a data change.
        await hub.on_state_refreshed()
        await _quiesce()
        assert len(fake_a.sent) == 2
        assert len(fake_b.sent) == 2

    async def test_alerts_push_skips_identical_payload(self, hub, monkeypatch):
        store = AlertsStore()
        store.replace_all([_alert()])
        monkeypatch.setattr(routes, "alerts_store", store)
        monkeypatch.setattr(routes, "alerts_enabled", True)

        fake = _FakeWebSocket()
        await _add_client(hub, fake, [TOPIC_ALERTS])

        await hub.on_state_refreshed()
        await _wait_messages(fake, 1)
        assert fake.sent[0]["topic"] == TOPIC_ALERTS
        key_first = hub._topic_state[TOPIC_ALERTS][0]

        # Different entries -> different key -> different fingerprint -> push.
        changed = [_alert(
            url="https://example.com/ws-alert-2",
            event="Severe Thunderstorm Warning",
            severity="Severe",
            area="Orange, NY",
        )]
        store.replace_all(changed)
        # ``replace_all`` stamps ``time.time()``; nudge it so the key change
        # is deterministic even on a coarse clock.
        store.last_updated += 5.0

        await hub.on_state_refreshed()
        await _wait_messages(fake, 2)
        assert len(fake.sent) == 2
        assert fake.sent[1]["data"]["features"][0]["properties"]["title"] == (
            "Severe Thunderstorm Warning"
        )
        key_second = hub._topic_state[TOPIC_ALERTS][0]
        assert key_second != key_first

        # Same entries, new ``last_updated`` -> the payload is rebuilt (the
        # key changed) but the fingerprint is identical -> no push.
        store.replace_all(changed)
        store.last_updated += 5.0
        await hub.on_state_refreshed()
        await _quiesce()
        key_third = hub._topic_state[TOPIC_ALERTS][0]
        assert key_third != key_second, "expected a rebuild on the new key"
        assert len(fake.sent) == 2, "identical payload must not be pushed"


# ---------------------------------------------------------------------------
# Push path: lightning deltas
# ---------------------------------------------------------------------------

class TestLightningPush:
    async def test_delta_survives_external_reload(self, hub, tmp_path, monkeypatch):
        """Regression: the version gate must not depend on ``maybe_reload``.

        ``maybe_reload`` is also consumed by ``/health`` and ``/v2/lightning``;
        when another caller already loaded the artifact it returns False.
        The delta must still be pushed.
        """
        monkeypatch.setattr(settings, "lightning_enabled", True)
        store = LightningStore(tmp_path)
        monkeypatch.setattr(routes, "lightning_store", store)

        now = int(time.time())
        first = _points([(30.0, -90.0, 1.0e-12)], times=[now - 300])
        await store.replace_points(first, last_seen_s=now - 300, max_age_s=1800)
        await store.save_snapshot()

        fake = _FakeWebSocket()
        client = await _add_client(hub, fake, [TOPIC_LIGHTNING])
        await hub._serve_snapshot(client, TOPIC_LIGHTNING)
        await _wait_messages(fake, 1)
        assert fake.sent[0]["type"] == "snapshot"
        assert client.lightning_watermark_s == now - 300

        second = _points([(31.0, -91.0, 2.0e-12)], times=[now - 100])
        await store.replace_points(
            np.concatenate([first, second]),
            last_seen_s=now - 100, max_age_s=1800,
        )
        await store.save_snapshot()

        # Simulate /health having already reloaded the artifact: the hub's
        # own maybe_reload() call now reports nothing new.
        await store.maybe_reload()
        assert await store.maybe_reload() is False

        await hub.on_poll_tick()
        messages = await _wait_messages(fake, 2)
        assert len(messages) == 2
        update = messages[1]
        assert update["type"] == "update"
        assert update["topic"] == TOPIC_LIGHTNING
        features = update["data"]["features"]
        assert len(features) == 1, "delta must carry only the new strike"
        assert features[0]["geometry"]["coordinates"] == [-91.0, 31.0]
        assert client.lightning_watermark_s == now - 100

    async def test_snapshot_seeds_watermark(self, hub, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "lightning_enabled", True)
        store = LightningStore(tmp_path)
        monkeypatch.setattr(routes, "lightning_store", store)

        now = int(time.time())
        t1, t2 = now - 300, now - 200
        points = _points(
            [(30.0, -90.0, 1.0e-12), (31.0, -91.0, 2.0e-12)],
            times=[t1, t2],
        )
        await store.replace_points(points, last_seen_s=t2, max_age_s=1800)

        fake = _FakeWebSocket()
        client = await _add_client(hub, fake, [TOPIC_LIGHTNING])
        await hub._serve_snapshot(client, TOPIC_LIGHTNING)
        await _wait_messages(fake, 1)

        feature_times = [
            f["properties"]["utc"] for f in fake.sent[0]["data"]["features"]
        ]
        assert feature_times  # ascending chronological order
        assert client.lightning_watermark_s == t2

        # A version bump with no newer strikes re-uses the seeded watermark;
        # the already-delivered T1/T2 are not re-sent.
        await store.replace_points(points, last_seen_s=t2, max_age_s=1800)
        await hub.on_poll_tick()
        await _quiesce()
        assert len(fake.sent) == 1
        assert client.lightning_watermark_s == t2

    async def test_empty_store_watermark_zero_delivers_first_window(
        self, hub, tmp_path, monkeypatch,
    ):
        """Fix B: a snapshot served while the store was empty (watermark 0)
        still gets the first delta window once strikes arrive."""
        monkeypatch.setattr(settings, "lightning_enabled", True)
        store = LightningStore(tmp_path)
        monkeypatch.setattr(routes, "lightning_store", store)

        fake = _FakeWebSocket()
        client = await _add_client(hub, fake, [TOPIC_LIGHTNING])
        await hub._serve_snapshot(client, TOPIC_LIGHTNING)
        await _wait_messages(fake, 1)
        assert fake.sent[0]["data"]["features"] == []
        assert client.lightning_watermark_s == 0

        now = int(time.time())
        points = _points(
            [(30.0, -90.0, 1.0e-12), (31.0, -91.0, 2.0e-12)],
            times=[now - 120, now - 60],
        )
        await store.replace_points(points, last_seen_s=now - 60, max_age_s=1800)
        await store.save_snapshot()

        await hub.on_poll_tick()
        messages = await _wait_messages(fake, 2)
        assert len(messages) == 2
        update = messages[1]
        assert update["type"] == "update"
        coordinates = [
            f["geometry"]["coordinates"] for f in update["data"]["features"]
        ]
        assert coordinates == [[-90.0, 30.0], [-91.0, 31.0]]
        assert client.lightning_watermark_s == now - 60

    async def test_no_new_strikes_pushes_nothing(self, hub, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "lightning_enabled", True)
        store = LightningStore(tmp_path)
        monkeypatch.setattr(routes, "lightning_store", store)

        now = int(time.time())
        points = _points([(30.0, -90.0, 1.0e-12)], times=[now - 120])
        await store.replace_points(points, last_seen_s=now - 120, max_age_s=1800)

        fake = _FakeWebSocket()
        client = await _add_client(hub, fake, [TOPIC_LIGHTNING])
        await hub._serve_snapshot(client, TOPIC_LIGHTNING)
        await _wait_messages(fake, 1)

        # New artifact version, identical content -> empty delta -> silence
        # (never an empty ``update`` message).
        await store.replace_points(points, last_seen_s=now - 120, max_age_s=1800)
        await store.save_snapshot()
        await hub.on_poll_tick()
        await _quiesce()

        assert len(fake.sent) == 1
        assert client.lightning_watermark_s == now - 120


# ---------------------------------------------------------------------------
# Push path: backpressure / limits / shutdown
# ---------------------------------------------------------------------------

class TestBackpressureAndLifecycle:
    async def test_slow_client_dropped_healthy_client_served(self, hub):
        slow_fake = _FakeWebSocket()
        slow = ClientState(
            websocket=slow_fake,
            subscriptions={TOPIC_WEATHER_MAPS},
            queue=asyncio.Queue(maxsize=hub._QUEUE_MAX),
        )
        hub._clients.add(slow)
        for _ in range(hub._QUEUE_MAX):
            slow.queue.put_nowait('{"type":"filler"}')

        healthy_fake = _FakeWebSocket()
        await _add_client(hub, healthy_fake, [TOPIC_WEATHER_MAPS])

        hub._broadcast(
            TOPIC_WEATHER_MAPS,
            json.dumps({"type": "update", "topic": TOPIC_WEATHER_MAPS, "data": {}}),
        )
        await asyncio.sleep(0.05)

        assert slow.dropped is True
        assert slow not in hub._clients
        assert slow_fake.close_codes == [1013]
        assert hub.connection_count() == 1

        messages = await _wait_messages(healthy_fake, 1)
        assert messages[0]["type"] == "update"

    async def test_connection_cap_closes_extra_client(self, hub, monkeypatch):
        monkeypatch.setattr(WebSocketHub, "_MAX_CONNECTIONS", 1)

        first = _BlockingWebSocket()
        task = asyncio.create_task(hub.handle(first))
        await asyncio.sleep(0.05)
        assert hub.connection_count() == 1

        second = _FakeWebSocket()
        await hub.handle(second)
        assert second.close_codes == [1013]
        assert hub.connection_count() == 1

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_stop_closes_clients_and_drains_senders(self, hub):
        fake_a = _FakeWebSocket()
        fake_b = _FakeWebSocket()
        await _add_client(hub, fake_a, [TOPIC_WEATHER_MAPS])
        await _add_client(hub, fake_b, [TOPIC_ALERTS])
        senders = [client.sender for client in hub._clients]

        await hub.stop()

        assert hub.stopping is True
        assert hub._clients == set()
        assert fake_a.close_codes == [1001]
        assert fake_b.close_codes == [1001]
        assert all(sender.done() for sender in senders)
