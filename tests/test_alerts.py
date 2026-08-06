# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

import json
import time
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from shapely.geometry import Polygon

from librewxr.api import routes
from librewxr.data.alerts_fetcher import _extract_polygons_from_cap, _parse_cap_time
from librewxr.data.alerts_store import AlertEntry, AlertsStore
from librewxr.data.store import FrameStore
from librewxr.tiles.cache import TileCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _save_restore_routes_state():
    """Save and restore routes module-level state to prevent cross-test pollution."""
    saved = {
        "alerts_store": routes.alerts_store,
        "alerts_fetcher": routes.alerts_fetcher,
        "alerts_enabled": routes.alerts_enabled,
        "frame_store": routes.frame_store,
        "tile_cache": routes.tile_cache,
        "ecmwf_grid": routes.ecmwf_grid,
        "nwp_grids": dict(routes.nwp_grids),
        "nwp_chain": routes.nwp_chain,
        "tile_warmer": routes.tile_warmer,
        "nowcast_store": routes.nowcast_store,
        "radar_cache": routes.radar_cache,
        "radar_fetcher": routes.radar_fetcher,
        "tile_request_tracker": routes.tile_request_tracker,
        "start_time": routes.start_time,
        "enabled_regions": routes.enabled_regions,
    }
    yield
    for key, val in saved.items():
        setattr(routes, key, val)


@pytest.fixture
def sample_alert_entries():
    """Three sample alerts for testing."""
    # Alert 1: polygon in New York area (complex enough to be simplified)
    poly1 = Polygon([
        (-74.5, 40.5), (-74.3, 40.55), (-74.1, 40.52), (-73.9, 40.58),
        (-73.7, 40.55), (-73.5, 40.5), (-73.5, 41.0), (-73.7, 41.1),
        (-73.9, 41.05), (-74.1, 41.1), (-74.3, 41.05), (-74.5, 41.5),
        (-74.5, 40.5),
    ])
    alert1 = AlertEntry(
        source_id="us-noaa-nws-en",
        event="Tornado Watch",
        description="TORNADO WATCH 189 REMAINS VALID...",
        severity="Extreme",
        effective="2026-05-07T00:15:00-05:00",
        expires="2099-05-07T06:00:00-05:00",
        area_desc="Sullivan, NY",
        url="https://example.com/alert1",
        polygon=poly1,
    )
    # Alert 2: polygon in California
    poly2 = Polygon([(-122.5, 37.0), (-121.5, 37.0), (-121.5, 38.0), (-122.5, 38.0), (-122.5, 37.0)])
    alert2 = AlertEntry(
        source_id="us-noaa-nws-en",
        event="High Wind Warning",
        description="Strong winds expected...",
        severity="Severe",
        effective="2026-05-07T01:00:00-08:00",
        expires="2099-05-07T18:00:00-08:00",
        area_desc="San Francisco, CA",
        url="https://example.com/alert2",
        polygon=poly2,
    )
    # Alert 3: no polygon (should be excluded from point/bbox lookups)
    alert3 = AlertEntry(
        source_id="fr-meteofrance-xx",
        event="Heavy Rain Warning",
        description="Heavy rain expected in Normandy...",
        severity="Moderate",
        effective="2026-05-07T06:00:00+02:00",
        expires="2099-05-07T12:00:00+02:00",
        area_desc="Normandy",
        url="https://example.com/alert3",
        polygon=None,
    )
    return [alert1, alert2, alert3]


@pytest.fixture
def alerts_store(sample_alert_entries):
    store = AlertsStore()
    store.replace_all(sample_alert_entries)
    return store


@pytest.fixture
def test_app(alerts_store):
    app = FastAPI()
    app.include_router(routes.router)
    routes.alerts_store = alerts_store
    routes.alerts_enabled = True
    routes.frame_store = FrameStore(max_frames=2)
    routes.tile_cache = TileCache(max_mb=10)
    routes.ecmwf_grid = None
    routes.hrrr_grid = None
    routes.icon_eu_grid = None
    routes.dmi_dini_grid = None
    routes.nwp_chain = None
    routes.tile_warmer = None
    routes.nowcast_store = None
    routes.radar_cache = None
    routes.radar_fetcher = None
    routes.tile_request_tracker = None
    routes.start_time = time.time()
    routes.enabled_regions = []
    return app


@pytest.fixture
async def client(test_app):
    # Clear NWS point cache and mock the fetcher so tests don't hit real API
    routes._nws_point_cache.clear()
    original_fetch = routes._fetch_nws_point_alerts

    async def _mock_fetch(lat, lon):
        return []

    routes._fetch_nws_point_alerts = _mock_fetch
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    routes._fetch_nws_point_alerts = original_fetch


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.alerts
class TestAlertsEndpoint:
    async def test_no_params_returns_all(self, client):
        resp = await client.get("/v2/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 3  # All alerts, including null geometry

    async def test_point_lookup_finds_alert(self, client):
        resp = await client.get("/v2/alerts?lat=40.7&lon=-74.0")
        assert resp.status_code == 200
        data = resp.json()
        # Alert 1 polygon matches; Alert 3 has no polygon
        assert len(data["features"]) == 1
        assert data["features"][0]["properties"]["title"] == "Tornado Watch"

    async def test_point_lookup_empty(self, client):
        resp = await client.get("/v2/alerts?lat=0.0&lon=0.0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["features"]) == 0

    async def test_point_lookup_boundary(self, client):
        # Point exactly on the polygon edge should match (intersects, not contains)
        resp = await client.get("/v2/alerts?lat=40.7&lon=-74.5")
        assert resp.status_code == 200
        data = resp.json()
        # Alert 1 polygon matches
        assert len(data["features"]) == 1

    async def test_bbox_filter_includes_intersecting(self, client):
        resp = await client.get("/v2/alerts?bbox=-125,35,-70,45")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["features"]) == 2  # Both NY and CA

    async def test_bbox_filter_excludes_non_intersecting(self, client):
        resp = await client.get("/v2/alerts?bbox=0,0,10,10")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["features"]) == 0

    async def test_bbox_bad_format(self, client):
        resp = await client.get("/v2/alerts?bbox=1,2,3")
        assert resp.status_code == 400

    async def test_bbox_out_of_range(self, client):
        resp = await client.get("/v2/alerts?bbox=-200,0,0,10")
        assert resp.status_code == 400

    async def test_simplify_reduces_vertices(self, client):
        # First get without simplify — find the feature that has geometry
        resp1 = await client.get("/v2/alerts?lat=40.7&lon=-74.0&simplify=0")
        data1 = resp1.json()
        geom_features1 = [f for f in data1["features"] if f["geometry"] is not None]
        assert len(geom_features1) >= 1
        vertices1 = len(geom_features1[0]["geometry"]["coordinates"][0])

        # Then with simplify
        resp2 = await client.get("/v2/alerts?lat=40.7&lon=-74.0&simplify=50000")
        data2 = resp2.json()
        geom_features2 = [f for f in data2["features"] if f["geometry"] is not None]
        assert len(geom_features2) >= 1
        vertices2 = len(geom_features2[0]["geometry"]["coordinates"][0])

        assert vertices2 < vertices1

    async def test_disabled_returns_503(self, client):
        routes.alerts_enabled = False
        resp = await client.get("/v2/alerts")
        assert resp.status_code == 503
        routes.alerts_enabled = True

    async def test_expired_alerts_filtered(self, alerts_store):
        expired_alert = AlertEntry(
            source_id="test",
            event="Expired",
            description="This alert has expired",
            severity="Minor",
            effective="2020-01-01T00:00:00+00:00",
            expires="2020-01-02T00:00:00+00:00",
            area_desc="Test",
            url="https://example.com/expired",
            polygon=Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        )
        alerts_store.replace_all([expired_alert])

        app = FastAPI()
        app.include_router(routes.router)
        routes.alerts_store = alerts_store
        routes.alerts_enabled = True
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/v2/alerts")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["features"]) == 0

    async def test_geojson_valid_structure(self, client):
        resp = await client.get("/v2/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert "type" in data
        assert "features" in data
        for feature in data["features"]:
            assert feature["type"] == "Feature"
            assert "properties" in feature
            assert "geometry" in feature
            props = feature["properties"]
            assert "title" in props
            assert "severity" in props
            assert "time" in props
            assert "expires" in props
            assert "description" in props
            assert "regions" in props
            assert "uri" in props

    async def test_health_includes_alerts(self, client, alerts_store):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data
        assert data["alerts"]["enabled"] is True
        assert data["alerts"]["count"] == 3
        assert data["alerts"]["ingest_ok"] is True


@pytest.mark.alerts
class TestAlertsStoreSnapshot:
    """Round-trip __getstate__ / __setstate__ via the multi-worker mechanism.

    The pipeline owns the WMO ingest; render-only workers see alerts only
    via the master_state snapshot.  Polygon serialisation goes through
    GeoJSON since shapely objects don't survive JSON round-trips natively.
    """

    def test_round_trip_preserves_alerts(self):
        producer = AlertsStore()
        poly = Polygon([(-105, 40), (-105, 41), (-104, 41), (-104, 40), (-105, 40)])
        producer.replace_all([
            AlertEntry(
                source_id="test-1",
                event="Severe Thunderstorm Warning",
                description="Hail to 1.5 inches",
                severity="Severe",
                effective="2026-05-08T20:00:00Z",
                expires="2026-05-08T21:00:00Z",
                area_desc="Boulder County",
                url="https://example.com/alerts/1",
                polygon=poly,
            ),
            AlertEntry(
                source_id="test-2",
                event="Flood Watch",
                description="Heavy rain expected",
                severity="Moderate",
                effective="2026-05-08T20:00:00Z",
                expires="2026-05-09T08:00:00Z",
                area_desc="Eastern Plains",
                url="https://example.com/alerts/2",
                polygon=None,  # alerts without geometry should round-trip too
            ),
        ])

        # JSON-roundtrip the snapshot to mirror what dump_state/load_state do.
        snapshot = json.loads(json.dumps(producer.__getstate__()))

        consumer = AlertsStore()
        consumer.__setstate__(snapshot)

        restored = consumer.alerts
        assert len(restored) == 2
        assert restored[0].event == "Severe Thunderstorm Warning"
        assert restored[0].polygon is not None
        # Polygon equality via centroid + area is enough — exact coord
        # ordering after GeoJSON round-trip can differ trivially.
        assert restored[0].polygon.equals(poly)
        assert restored[1].polygon is None
        assert consumer.fetch_success is True
        assert consumer.last_updated == producer.last_updated

    def test_empty_store_round_trips(self):
        producer = AlertsStore()
        snapshot = json.loads(json.dumps(producer.__getstate__()))
        consumer = AlertsStore()
        consumer.__setstate__(snapshot)
        assert consumer.alerts == []
        assert consumer.fetch_success is False


class TestCAPParsing:
    def test_parse_cap_time(self):
        assert _parse_cap_time("2026-05-07T00:15:00-05:00") is not None
        assert _parse_cap_time("") is None
        assert _parse_cap_time("invalid") is None

    def test_extract_polygons_from_cap(self):
        cap_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
          <info>
            <language>en-US</language>
            <event>Tornado Watch</event>
            <headline>Tornado Watch issued</headline>
            <description>Description here</description>
            <urgency>Future</urgency>
            <severity>Extreme</severity>
            <effective>2026-05-07T00:15:00-05:00</effective>
            <expires>2026-05-07T06:00:00-05:00</expires>
            <area>
              <areaDesc>Test Area</areaDesc>
              <polygon>32.5,-85.2 32.6,-85.1 32.5,-85.0 32.4,-85.1 32.5,-85.2</polygon>
            </area>
          </info>
        </alert>"""
        entries = _extract_polygons_from_cap(cap_xml, "test-source", "https://example.com/cap")
        assert len(entries) == 1
        assert entries[0].event == "Tornado Watch issued"
        assert entries[0].severity == "Extreme"
        assert entries[0].polygon is not None
        assert entries[0].polygon.is_valid

    def test_extract_polygons_past_urgency_skipped(self):
        cap_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
          <info>
            <language>en-US</language>
            <event>Past Event</event>
            <urgency>Past</urgency>
            <severity>Minor</severity>
            <area>
              <areaDesc>Test</areaDesc>
              <polygon>0,0 1,0 1,1 0,1 0,0</polygon>
            </area>
          </info>
        </alert>"""
        entries = _extract_polygons_from_cap(cap_xml, "test", "https://example.com")
        assert len(entries) == 0

    def test_extract_polygons_closed_if_needed(self):
        cap_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
          <info>
            <language>en-US</language>
            <event>Test</event>
            <severity>Minor</severity>
            <area>
              <areaDesc>Test</areaDesc>
              <polygon>0,0 1,0 1,1 0,1</polygon>
            </area>
          </info>
        </alert>"""
        entries = _extract_polygons_from_cap(cap_xml, "test", "https://example.com")
        assert len(entries) == 1
        # First and last should be same after auto-close
        coords = list(entries[0].polygon.exterior.coords)
        assert coords[0] == coords[-1]


@pytest.mark.alerts
class TestAlertsStore:
    def test_replace_all(self):
        store = AlertsStore()
        assert store.count == 0
        store.replace_all([AlertEntry("s", "e", "d", "sev", "eff", "exp", "area", "url")])
        assert store.count == 1
        assert store.fetch_success is True
        assert store.last_updated > 0

    def test_mark_failed(self):
        store = AlertsStore()
        store.mark_failed()
        assert store.fetch_success is False

    def test_alerts_copy(self):
        store = AlertsStore()
        store.replace_all([AlertEntry("s", "e", "d", "sev", "eff", "exp", "area", "url")])
        a1 = store.alerts
        a2 = store.alerts
        assert a1 is not a2  # Should be copies


@pytest.mark.alerts
class TestNwsPointCacheBucketing:
    """NWS point-lookup cache buckets keys to 0.1 degrees (~11 km cells).

    Two point queries whose coordinates differ but fall in the same bucket
    must trigger only ONE upstream NWS fetch; the second is served from the
    process-local cache.
    """

    async def test_same_bucket_single_upstream_fetch(self, monkeypatch):
        upstream_calls: list[str] = []

        class _FakeResponse:
            status_code = 200

            def json(self):
                return {"features": []}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, headers=None):
                upstream_calls.append(url)
                return _FakeResponse()

        monkeypatch.setattr(routes.httpx, "AsyncClient", _FakeClient)
        routes._nws_point_cache.clear()
        try:
            await routes._fetch_nws_point_alerts(39.7392, -104.9903)
            await routes._fetch_nws_point_alerts(39.74, -104.99)
        finally:
            routes._nws_point_cache.clear()

        assert len(upstream_calls) == 1
