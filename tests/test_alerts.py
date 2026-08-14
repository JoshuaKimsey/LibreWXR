# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from shapely.geometry import Polygon

from librewxr.api import routes
from librewxr.data.alerts_fetcher import (
    WMOAlertsFetcher,
    _ZONE_FAILURE_TTL,
    _extract_polygons_from_cap,
    _parse_cap_time,
)
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
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


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


# ---------------------------------------------------------------------------
# NWS zone resolution (ingest-time)
# ---------------------------------------------------------------------------

class _FakeNwsResponse:
    def __init__(self, status_code=200, payload=None, content=b"", text=""):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = text

    def json(self):
        return self._payload


class _FakeNwsClient:
    """Records requested URLs and serves canned payloads per URL."""

    def __init__(self, responses):
        self.responses = responses
        self.calls: list[str] = []

    async def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        resp = self.responses.get(url)
        if resp is None:
            return _FakeNwsResponse(404)
        return resp


def _zone_geojson(bounds):
    """Axis-aligned zone polygon GeoJSON for a (minx, miny, maxx, maxy) box."""
    minx, miny, maxx, maxy = bounds
    return {
        "type": "Polygon",
        "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]],
    }


def _zone_geometrycollection():
    """GeometryCollection wrapping two disjoint triangle polygons (area 0.5
    each) - mirrors how the NWS API serves some zone geometries (e.g. FLZ011)."""
    return {
        "type": "GeometryCollection",
        "geometries": [
            {
                "type": "Polygon",
                "coordinates": [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
            },
            {
                "type": "Polygon",
                "coordinates": [[[2.0, 0.0], [3.0, 0.0], [2.0, 1.0], [2.0, 0.0]]],
            },
        ],
    }


def _nws_feature(zone_urls, alert_id="urn:oid:test"):
    """Geometry-less NWS feature carrying ``affectedZones``."""
    return {
        "type": "Feature",
        "id": f"https://api.weather.gov/alerts/{alert_id}",
        "geometry": None,
        "properties": {
            "status": "Actual",
            "messageType": "Alert",
            "affectedZones": list(zone_urls),
            "headline": "Tornado Watch issued",
            "event": "Tornado Watch",
            "description": "TORNADO WATCH ...",
            "severity": "Extreme",
            "effective": "2026-05-07T00:15:00-05:00",
            "expires": "2099-05-07T06:00:00-05:00",
            "areaDesc": "Test County",
            "id": f"https://api.weather.gov/alerts/{alert_id}",
        },
    }


@pytest.mark.alerts
class TestNwsZoneResolution:
    """Geometry-less NWS alerts get union polygons resolved from zones at ingest."""

    async def _fetch(self, tmp_path, client):
        fetcher = WMOAlertsFetcher(store=AlertsStore(), cache_dir=str(tmp_path))

        async def _get_client():
            return client

        fetcher._get_client = _get_client  # type: ignore[method-assign]
        return await fetcher._fetch_nws_alerts()

    async def test_geometryless_alert_gets_zone_union(self, tmp_path):
        zone_a = "https://api.weather.gov/zones/forecast/COZ041"
        zone_b = "https://api.weather.gov/zones/forecast/COZ042"
        alerts_url = "https://api.weather.gov/alerts/active"
        client = _FakeNwsClient({
            alerts_url: _FakeNwsResponse(200, {
                "features": [_nws_feature([zone_a, zone_b])],
            }),
            zone_a: _FakeNwsResponse(200, {"geometry": _zone_geojson((-105.5, 39.0, -105.0, 39.5))}),
            zone_b: _FakeNwsResponse(200, {"geometry": _zone_geojson((-104.5, 39.0, -104.0, 39.5))}),
        })

        entries = await self._fetch(tmp_path, client)

        assert len(entries) == 1
        assert entries[0].event == "Tornado Watch issued"
        assert entries[0].polygon is not None
        # Union of two disjoint zone boxes -> MultiPolygon covering both
        assert entries[0].polygon.geom_type == "MultiPolygon"
        assert entries[0].polygon.area == pytest.approx(0.5)
        # Zone geometries written to the disk cache
        zones_dir = Path(tmp_path) / "alerts" / "zones"
        assert (zones_dir / "COZ041.json").exists()
        assert (zones_dir / "COZ042.json").exists()

    async def test_second_cycle_reuses_zone_disk_cache(self, tmp_path):
        zone_a = "https://api.weather.gov/zones/forecast/COZ041"
        alerts_url = "https://api.weather.gov/alerts/active"
        client = _FakeNwsClient({
            alerts_url: _FakeNwsResponse(200, {
                "features": [_nws_feature([zone_a])],
            }),
            zone_a: _FakeNwsResponse(200, {"geometry": _zone_geojson((-105.5, 39.0, -105.0, 39.5))}),
        })

        await self._fetch(tmp_path, client)
        assert [u for u in client.calls if u != alerts_url] == [zone_a]

        client.calls.clear()
        await self._fetch(tmp_path, client)
        # Second cycle: alerts feed refetched, zone geometry from disk cache
        assert client.calls == [alerts_url]

    async def test_zone_fetch_error_keeps_alert_polygon_none(self, tmp_path):
        zone_a = "https://api.weather.gov/zones/forecast/COZ041"
        alerts_url = "https://api.weather.gov/alerts/active"
        client = _FakeNwsClient({
            alerts_url: _FakeNwsResponse(200, {
                "features": [_nws_feature([zone_a])],
            }),
            zone_a: _FakeNwsResponse(500),
        })

        entries = await self._fetch(tmp_path, client)

        # Fail-soft: alert kept with polygon None, no raise
        assert len(entries) == 1
        assert entries[0].polygon is None

    async def test_shared_zone_fetched_once_across_alerts(self, tmp_path):
        zone_a = "https://api.weather.gov/zones/forecast/COZ041"
        zone_b = "https://api.weather.gov/zones/forecast/COZ042"
        alerts_url = "https://api.weather.gov/alerts/active"
        client = _FakeNwsClient({
            alerts_url: _FakeNwsResponse(200, {
                "features": [
                    _nws_feature([zone_a, zone_b], alert_id="urn:oid:watch1"),
                    _nws_feature([zone_a], alert_id="urn:oid:warning1"),
                ],
            }),
            zone_a: _FakeNwsResponse(200, {"geometry": _zone_geojson((-105.5, 39.0, -105.0, 39.5))}),
            zone_b: _FakeNwsResponse(200, {"geometry": _zone_geojson((-104.5, 39.0, -104.0, 39.5))}),
        })

        entries = await self._fetch(tmp_path, client)

        assert len(entries) == 2
        zone_calls = [u for u in client.calls if u != alerts_url]
        # Two unique zones across the alerts -> each fetched exactly once
        assert len(zone_calls) == 2
        assert set(zone_calls) == {zone_a, zone_b}
        # Both alerts resolved (each references the resolved zone_a)
        assert entries[0].polygon is not None
        assert entries[1].polygon is not None

    async def test_geocode_ugc_fallback_resolves_zones(self, tmp_path):
        alerts_url = "https://api.weather.gov/alerts/active"
        # No affectedZones: falls back to geocode.UGC codes (Z=forecast, C=county)
        zone_a = "https://api.weather.gov/zones/forecast/COZ041"
        zone_c = "https://api.weather.gov/zones/county/COC013"
        feature = {
            "type": "Feature",
            "id": "https://api.weather.gov/alerts/urn:oid:ugc",
            "geometry": None,
            "properties": {
                "status": "Actual",
                "messageType": "Alert",
                "geocode": {"SAME": ["008041"], "UGC": ["COZ041", "COC013"]},
                "headline": "Special Weather Statement",
                "event": "Special Weather Statement",
                "severity": "Moderate",
                "effective": "2026-05-07T00:15:00-05:00",
                "expires": "2099-05-07T06:00:00-05:00",
                "areaDesc": "COZ041;COC013",
                "id": "https://api.weather.gov/alerts/urn:oid:ugc",
            },
        }
        client = _FakeNwsClient({
            alerts_url: _FakeNwsResponse(200, {"features": [feature]}),
            zone_a: _FakeNwsResponse(200, {"geometry": _zone_geojson((-105.5, 39.0, -105.0, 39.5))}),
            zone_c: _FakeNwsResponse(200, {"geometry": _zone_geojson((-104.5, 39.0, -104.0, 39.5))}),
        })

        entries = await self._fetch(tmp_path, client)

        assert len(entries) == 1
        assert entries[0].polygon is not None
        assert entries[0].polygon.area == pytest.approx(0.5)
        assert set(u for u in client.calls if u != alerts_url) == {zone_a, zone_c}

    async def test_geometrycollection_zone_resolves_to_union(self, tmp_path):
        zone_a = "https://api.weather.gov/zones/forecast/FLZ011"
        alerts_url = "https://api.weather.gov/alerts/active"
        client = _FakeNwsClient({
            alerts_url: _FakeNwsResponse(200, {
                "features": [_nws_feature([zone_a])],
            }),
            zone_a: _FakeNwsResponse(200, {"geometry": _zone_geometrycollection()}),
        })

        entries = await self._fetch(tmp_path, client)

        # GeometryCollection of two disjoint triangles unions to a MultiPolygon
        assert len(entries) == 1
        assert entries[0].polygon is not None
        assert entries[0].polygon.geom_type == "MultiPolygon"
        assert entries[0].polygon.area == pytest.approx(1.0)
        # Resolved polygon written to the disk cache
        zones_dir = Path(tmp_path) / "alerts" / "zones"
        assert (zones_dir / "FLZ011.json").exists()

    async def test_alert_feature_with_geometrycollection_polygon(self, tmp_path):
        alerts_url = "https://api.weather.gov/alerts/active"
        feature = _nws_poly_feature("urn:oid:gc-alert")
        feature["geometry"] = {
            "type": "GeometryCollection",
            "geometries": [
                {
                    "type": "Polygon",
                    "coordinates": [[[-105.5, 39.0], [-105.0, 39.0], [-105.25, 39.5], [-105.5, 39.0]]],
                },
            ],
        }
        client = _FakeNwsClient({
            alerts_url: _FakeNwsResponse(200, {"features": [feature]}),
        })

        entries = await self._fetch(tmp_path, client)

        assert len(entries) == 1
        assert entries[0].polygon is not None
        assert entries[0].polygon.area == pytest.approx(0.125)

    async def test_null_geometry_writes_failure_marker_and_suppresses_retry(
        self, tmp_path, caplog
    ):
        zone_a = "https://api.weather.gov/zones/forecast/FLZ011"
        client = _FakeNwsClient({zone_a: _FakeNwsResponse(200, {"geometry": None})})
        fetcher = WMOAlertsFetcher(store=AlertsStore(), cache_dir=str(tmp_path))

        async def _get_client():
            return client

        fetcher._get_client = _get_client  # type: ignore[method-assign]

        with caplog.at_level("WARNING"):
            result = await fetcher._fetch_zone_polygons({zone_a})

        assert result == {}
        assert "NWS zone FLZ011 has no usable geometry" in caplog.text

        # Failure marker written to the zone cache path
        marker = Path(tmp_path) / "alerts" / "zones" / "FLZ011.json"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["failed"] is True
        assert data["geometry"] is None

        # Fresh marker: second call makes no HTTP request and logs nothing
        client.calls.clear()
        caplog.clear()
        with caplog.at_level("WARNING"):
            result2 = await fetcher._fetch_zone_polygons({zone_a})

        assert result2 == {}
        assert client.calls == []
        assert caplog.records == []

    async def test_stale_failure_marker_triggers_refetch(self, tmp_path):
        zone_a = "https://api.weather.gov/zones/forecast/FLZ011"
        marker = Path(tmp_path) / "alerts" / "zones" / "FLZ011.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "fetched_at": time.time() - _ZONE_FAILURE_TTL - 60,
            "geometry": None,
            "failed": True,
        }))

        client = _FakeNwsClient({
            zone_a: _FakeNwsResponse(200, {"geometry": _zone_geojson((-105.5, 39.0, -105.0, 39.5))}),
        })
        fetcher = WMOAlertsFetcher(store=AlertsStore(), cache_dir=str(tmp_path))

        async def _get_client():
            return client

        fetcher._get_client = _get_client  # type: ignore[method-assign]

        result = await fetcher._fetch_zone_polygons({zone_a})

        # Stale marker ignored -> zone refetched and now resolved
        assert zone_a in result
        assert result[zone_a] is not None
        assert client.calls == [zone_a]
        # Success now cached in place of the failure marker
        data = json.loads(marker.read_text())
        assert data.get("failed") is not True
        assert data.get("geometry") is not None


# ---------------------------------------------------------------------------
# Ingest cycle: WMO degradation must not orphan/freeze the parallel NWS fetch
# ---------------------------------------------------------------------------

_WMO_SOURCES_URL = "https://severeweather.wmo.int/json/sources.json"
_WMO_ALL_URL = "https://severeweather.wmo.int/v2/json/wmo_all.json"
_NWS_ACTIVE_URL = "https://api.weather.gov/alerts/active"


def _nws_poly_feature(alert_id="urn:oid:poly"):
    """NWS feature carrying inline geometry (no zone resolution needed)."""
    return {
        "type": "Feature",
        "id": f"https://api.weather.gov/alerts/{alert_id}",
        "geometry": _zone_geojson((-105.5, 39.0, -105.0, 39.5)),
        "properties": {
            "status": "Actual",
            "messageType": "Alert",
            "headline": "Severe Thunderstorm Warning issued",
            "event": "Severe Thunderstorm Warning",
            "description": "Severe thunderstorms expected",
            "severity": "Severe",
            "effective": "2026-05-07T00:15:00-05:00",
            "expires": "2099-05-07T06:00:00-05:00",
            "areaDesc": "Test County",
            "id": f"https://api.weather.gov/alerts/{alert_id}",
        },
    }


def _cap_xml_with_polygon():
    """Minimal CAP 1.2 XML with a usable polygon."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
      <info>
        <language>en-US</language>
        <event>Severe Thunderstorm Warning</event>
        <headline>Severe Thunderstorm Warning issued</headline>
        <description>Severe thunderstorms expected</description>
        <urgency>Immediate</urgency>
        <severity>Severe</severity>
        <effective>2026-05-07T00:15:00-05:00</effective>
        <expires>2099-05-07T06:00:00-05:00</expires>
        <area>
          <areaDesc>Test Area</areaDesc>
          <polygon>32.5,-85.2 32.6,-85.1 32.5,-85.0 32.4,-85.1 32.5,-85.2</polygon>
        </area>
      </info>
    </alert>"""


def _rss_with_item():
    """Minimal RSS 2.0 with one item whose guid matches a wmo_all id."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<rss version=\"2.0\"><channel>"
        "<item><link>https://severeweather.wmo.int/cap/fr-meteofrance-xx/1.xml</link>"
        "<guid>wmo-alert-1</guid></item>"
        "</channel></rss>"
    )


def _non_nws_alert():
    """A pre-existing non-NWS alert used to check preservation on degradation."""
    return AlertEntry(
        source_id="fr-meteofrance-xx",
        event="Heavy Rain Warning",
        description="Heavy rain expected in Normandy",
        severity="Moderate",
        effective="2026-05-07T06:00:00+02:00",
        expires="2099-05-07T12:00:00+02:00",
        area_desc="Normandy",
        url="https://example.com/alert",
        polygon=None,
    )


@pytest.mark.alerts
class TestFetchOnceDegradedPath:
    """WMO sources.json / wmo_all.json failures must not freeze the US slice."""

    def _fetcher(self, store, client, tmp_path):
        fetcher = WMOAlertsFetcher(store=store, cache_dir=str(tmp_path))

        async def _get_client():
            return client

        fetcher._get_client = _get_client  # type: ignore[method-assign]
        return fetcher

    def _sources_fail_client(self):
        return _FakeNwsClient({
            _WMO_SOURCES_URL: _FakeNwsResponse(500),
            _NWS_ACTIVE_URL: _FakeNwsResponse(
                200, {"features": [_nws_poly_feature("urn:oid:degraded")]}
            ),
        })

    def _wmo_all_fail_client(self):
        return _FakeNwsClient({
            _WMO_SOURCES_URL: _FakeNwsResponse(200, {"sources": []}),
            _WMO_ALL_URL: _FakeNwsResponse(500),
            _NWS_ACTIVE_URL: _FakeNwsResponse(
                200, {"features": [_nws_poly_feature("urn:oid:degraded")]}
            ),
        })

    async def test_sources_failure_refreshes_nws_and_preserves_others(self, tmp_path):
        store = AlertsStore()
        store.replace_all([_non_nws_alert()])

        fetcher = self._fetcher(store, self._sources_fail_client(), tmp_path)
        await fetcher._fetch_once()

        assert store.fetch_success is False
        alerts = store.alerts
        nws = [a for a in alerts if a.source_id == "nws-api"]
        others = [a for a in alerts if a.source_id != "nws-api"]
        assert len(nws) == 1
        assert nws[0].event == "Severe Thunderstorm Warning issued"
        assert nws[0].polygon is not None
        assert len(others) == 1
        assert others[0].event == "Heavy Rain Warning"

    async def test_wmo_all_failure_refreshes_nws_and_preserves_others(self, tmp_path):
        store = AlertsStore()
        store.replace_all([_non_nws_alert()])

        fetcher = self._fetcher(store, self._wmo_all_fail_client(), tmp_path)
        await fetcher._fetch_once()

        assert store.fetch_success is False
        alerts = store.alerts
        nws = [a for a in alerts if a.source_id == "nws-api"]
        others = [a for a in alerts if a.source_id != "nws-api"]
        assert len(nws) == 1
        assert nws[0].event == "Severe Thunderstorm Warning issued"
        assert len(others) == 1
        assert others[0].event == "Heavy Rain Warning"

    async def test_degraded_path_bumps_last_updated(self, tmp_path):
        store = AlertsStore()
        store.replace_all([_non_nws_alert()])
        old_ts = store.last_updated

        fetcher = self._fetcher(store, self._sources_fail_client(), tmp_path)
        await fetcher._fetch_once()

        assert store.last_updated >= old_ts

    def _wmo_ok_client(self):
        rss = _rss_with_item()
        cap_xml = _cap_xml_with_polygon()
        return _FakeNwsClient({
            _WMO_SOURCES_URL: _FakeNwsResponse(200, {
                "sources": [
                    {"source": {
                        "sourceId": "fr-meteofrance-xx",
                        "capAlertFeedStatus": "operating",
                    }},
                ],
            }),
            _WMO_ALL_URL: _FakeNwsResponse(200, {
                "items": [{"id": "wmo-alert-1", "capURL": "fr-meteofrance-xx", "url": ""}],
            }),
            "https://severeweather.wmo.int/v2/cap-alerts/fr-meteofrance-xx/rss.xml": (
                _FakeNwsResponse(200, content=rss.encode("utf-8"))
            ),
            "https://severeweather.wmo.int/cap/fr-meteofrance-xx/1.xml": (
                _FakeNwsResponse(200, text=cap_xml)
            ),
        })

    async def test_nws_failure_does_not_block_wmo_alerts(self, tmp_path):
        store = AlertsStore()
        fetcher = self._fetcher(store, self._wmo_ok_client(), tmp_path)

        async def _boom():
            raise RuntimeError("NWS API down")

        fetcher._fetch_nws_alerts = _boom  # type: ignore[method-assign]

        await fetcher._fetch_once()

        assert store.fetch_success is True
        alerts = store.alerts
        assert len(alerts) == 1
        assert alerts[0].source_id == "fr-meteofrance-xx"
        assert alerts[0].event == "Severe Thunderstorm Warning issued"
        assert alerts[0].polygon is not None
