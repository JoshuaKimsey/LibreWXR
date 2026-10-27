# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

import json
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Point, Polygon, shape

from librewxr.api import routes
from librewxr.data.alert_geocodes import (
    GeocodeProvider,
    _DWD_WFS_URL,
    _apply_meteoalarm_aliases,
    _bom_url_candidates,
    _download_dwd_layer,
    _load_bom_amoc,
    _load_dwd_warncell,
    _parse_bom_zip,
    _parse_dwd_geojson,
    _parse_meteoalarm_geojson,
    build_default_providers,
    ensure_loaded,
    resolve,
)
from librewxr.data.alerts_fetcher import (
    WMOAlertsFetcher,
    _ZONE_FAILURE_TTL,
    _extract_polygons_from_cap,
    _normalize_alert_id,
    _parse_cap_time,
    _rss_item_links,
)
from librewxr.data.alerts_store import AlertEntry, AlertsStore, ensure_valid_polygon
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
class TestMultiAreaAlertGrouping:
    """CAP alerts covering many regions (Bulgaria, Romania, France, ...) arrive
    as several AlertEntry objects sharing one url. The endpoint must merge each
    group into a single feature carrying the full polygon footprint."""

    @staticmethod
    def _make_multi_area_entries():
        poly_a = Polygon([
            (20.0, 41.0), (22.0, 41.0), (22.0, 43.0), (20.0, 43.0), (20.0, 41.0),
        ])
        poly_b = Polygon([
            (26.0, 43.0), (28.0, 43.0), (28.0, 45.0), (26.0, 45.0), (26.0, 43.0),
        ])
        return [
            AlertEntry(
                source_id="bg-plovdiv-xx",
                event="Heat Wave",
                description="Extreme heat across Bulgaria",
                severity="Severe",
                effective="2026-05-07T00:00:00+03:00",
                expires="2099-05-07T23:00:00+03:00",
                area_desc="Region A",
                url="https://example.com/multi",
                polygon=poly_a,
            ),
            AlertEntry(
                source_id="bg-plovdiv-xx",
                event="Heat Wave",
                description="Extreme heat across Bulgaria",
                severity="Severe",
                effective="2026-05-07T00:00:00+03:00",
                expires="2099-05-07T23:00:00+03:00",
                area_desc="Region B",
                url="https://example.com/multi",
                polygon=poly_b,
            ),
        ]

    async def _get(self, entries, path):
        store = AlertsStore()
        store.replace_all(entries)
        app = FastAPI()
        app.include_router(routes.router)
        routes.alerts_store = store
        routes.alerts_enabled = True
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            return await ac.get(path)

    async def test_global_groups_by_url_and_unions_polygons(self):
        resp = await self._get(self._make_multi_area_entries(), "/v2/alerts")
        assert resp.status_code == 200
        data = resp.json()
        # Two regions sharing one url -> exactly one feature
        assert len(data["features"]) == 1
        feature = data["features"][0]
        props = feature["properties"]
        assert props["uri"] == "https://example.com/multi"
        assert props["title"] == "Heat Wave"
        assert props["regions"] == ["Region A", "Region B"]
        # Disjoint polygons union to a MultiPolygon covering both regions
        assert feature["geometry"] is not None
        assert feature["geometry"]["type"] == "MultiPolygon"
        merged = shape(feature["geometry"])
        assert merged.contains(Point(21.0, 42.0))  # inside Region A
        assert merged.contains(Point(27.0, 44.0))  # inside Region B

    async def test_point_lookup_returns_only_matching_region(self):
        resp = await self._get(
            self._make_multi_area_entries(), "/v2/alerts?lat=42.0&lon=21.0"
        )
        assert resp.status_code == 200
        data = resp.json()
        # Point hits only Region A -> single feature, single region
        assert len(data["features"]) == 1
        feature = data["features"][0]
        props = feature["properties"]
        assert props["uri"] == "https://example.com/multi"
        assert props["regions"] == ["Region A"]
        # 1-element group unions to the polygon itself
        assert feature["geometry"] is not None
        assert feature["geometry"]["type"] == "Polygon"

    async def test_null_polygon_group_collapses_to_one_null_feature(self):
        entries = [
            AlertEntry(
                source_id="fr-meteofrance-xx",
                event="Heavy Rain",
                description="Heavy rain expected",
                severity="Moderate",
                effective="2026-05-07T06:00:00+02:00",
                expires="2099-05-07T12:00:00+02:00",
                area_desc="Region A",
                url="https://example.com/nullgeom",
                polygon=None,
            ),
            AlertEntry(
                source_id="fr-meteofrance-xx",
                event="Heavy Rain",
                description="Heavy rain expected",
                severity="Moderate",
                effective="2026-05-07T06:00:00+02:00",
                expires="2099-05-07T12:00:00+02:00",
                area_desc="Region B",
                url="https://example.com/nullgeom",
                polygon=None,
            ),
        ]
        resp = await self._get(entries, "/v2/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["features"]) == 1
        feature = data["features"][0]
        assert feature["properties"]["uri"] == "https://example.com/nullgeom"
        assert feature["properties"]["regions"] == ["Region A", "Region B"]
        assert feature["geometry"] is None


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


@pytest.mark.alerts
class TestInvalidPolygonHandling:
    """Self-intersecting (bow-tie) CAP polygons previously 500ed /v2/alerts.

    An invalid polygon raises ``GEOSException: TopologyException: side location
    conflict`` at the group ``unary_union`` in ``get_alerts``, taking the whole
    endpoint down while the offending alert sat in the store.  Repair happens
    at ingest, the store enforces "valid or None" on both entry points, and the
    request paths keep a defensive fallback.
    """

    @staticmethod
    def _bow_tie():
        """Self-intersecting ring: two lobes meeting at a point."""
        poly = Polygon([(0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0), (0.0, 0.0)])
        assert poly.is_valid is False  # fixture sanity: this is the bad input
        return poly

    @staticmethod
    def _entry(url, polygon, event="Heat Wave", area_desc="Region"):
        return AlertEntry(
            source_id="bg-plovdiv-xx",
            event=event,
            description="Extreme heat",
            severity="Severe",
            effective="2026-05-07T00:00:00+03:00",
            expires="2099-05-07T23:00:00+03:00",
            area_desc=area_desc,
            url=url,
            polygon=polygon,
        )

    @staticmethod
    async def _get(entries, path):
        store = AlertsStore()
        store.replace_all(entries)
        app = FastAPI()
        app.include_router(routes.router)
        routes.alerts_store = store
        routes.alerts_enabled = True
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            return await ac.get(path)

    def test_ensure_valid_polygon_repairs_bowtie(self):
        bow_tie = self._bow_tie()
        repaired = ensure_valid_polygon(bow_tie)
        assert repaired is not None
        assert repaired.is_valid is True
        assert repaired.is_empty is False
        assert repaired.geom_type in ("Polygon", "MultiPolygon")
        # The two lobes survive the repair (area 1 each).
        assert repaired.area == pytest.approx(2.0)

        # Fast path: already-valid input is returned unchanged, no copy.
        square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
        assert ensure_valid_polygon(square) is square

        assert ensure_valid_polygon(None) is None

    async def test_endpoint_renders_invalid_polygon_alert(self):
        """Regression: a bow-tie alert in the store must not 500 /v2/alerts."""
        resp = await self._get(
            [self._entry("https://example.com/bowtie", self._bow_tie())],
            "/v2/alerts",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["features"]) == 1
        geometry = data["features"][0]["geometry"]
        assert geometry is not None
        assert shape(geometry).is_valid is True

    async def test_union_failure_falls_back_to_first_polygon(self, monkeypatch):
        """A union that still raises must degrade to the first polygon, not 500."""
        poly_a = Polygon([
            (20.0, 41.0), (22.0, 41.0), (22.0, 43.0), (20.0, 43.0), (20.0, 41.0),
        ])
        poly_b = Polygon([
            (26.0, 43.0), (28.0, 43.0), (28.0, 45.0), (26.0, 45.0), (26.0, 43.0),
        ])
        entries = [
            self._entry("https://example.com/union", poly_a, area_desc="Region A"),
            self._entry("https://example.com/union", poly_b, area_desc="Region B"),
        ]

        def _explode(*args, **kwargs):
            raise GEOSException("TopologyException: side location conflict")

        # get_alerts imports unary_union lazily inside the function, so the
        # attribute lookup happens at call time and this patch takes effect.
        monkeypatch.setattr("shapely.ops.unary_union", _explode)

        resp = await self._get(entries, "/v2/alerts?simplify=0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["features"]) == 1
        geometry = data["features"][0]["geometry"]
        assert geometry is not None
        assert shape(geometry).equals(poly_a)

    def test_extract_polygons_repairs_invalid_ring(self):
        """A self-intersecting CAP <polygon> is repaired at ingest."""
        cap_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
          <info>
            <language>en-US</language>
            <event>Severe Heat</event>
            <severity>Severe</severity>
            <effective>2026-05-07T00:15:00-05:00</effective>
            <expires>2099-05-07T06:00:00-05:00</expires>
            <area>
              <areaDesc>Bow Tie Area</areaDesc>
              <polygon>0,0 0,2 2,0 2,2 0,0</polygon>
            </area>
          </info>
        </alert>"""
        entries = _extract_polygons_from_cap(
            cap_xml, "test-source", "https://example.com/cap", []
        )
        assert len(entries) == 1
        assert entries[0].polygon is not None
        assert entries[0].polygon.is_valid is True
        assert entries[0].polygon.is_empty is False

    def test_replace_all_repairs_invalid_polygon(self):
        store = AlertsStore()
        store.replace_all([self._entry("https://example.com/repair", self._bow_tie())])
        assert len(store.alerts) == 1
        assert store.alerts[0].polygon is not None
        assert store.alerts[0].polygon.is_valid is True

    def test_setstate_repairs_invalid_polygon(self):
        """Snapshots written by older revisions are repaired on load."""
        bow_tie = self._bow_tie()
        snapshot = {
            "last_updated": 123.0,
            "fetch_success": True,
            "alerts": [
                {
                    "source_id": "legacy-1",
                    "event": "Legacy Alert",
                    "description": "Snapshot from an older revision",
                    "severity": "Severe",
                    "effective": "2026-05-08T20:00:00Z",
                    "expires": "2099-05-08T21:00:00Z",
                    "area_desc": "Legacy Area",
                    "url": "https://example.com/legacy",
                    "polygon": bow_tie.__geo_interface__,
                }
            ],
        }
        # Mirror what dump_state/load_state do (JSON round-trip).
        snapshot = json.loads(json.dumps(snapshot))

        store = AlertsStore()
        store.__setstate__(snapshot)

        assert len(store.alerts) == 1
        restored = store.alerts[0].polygon
        assert restored is not None
        assert restored.is_valid is True
        assert restored.is_empty is False
        assert restored.area == pytest.approx(2.0)


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
        entries = _extract_polygons_from_cap(
            cap_xml, "test-source", "https://example.com/cap", []
        )
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
        entries = _extract_polygons_from_cap(cap_xml, "test", "https://example.com", [])
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
        entries = _extract_polygons_from_cap(cap_xml, "test", "https://example.com", [])
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

# An agency-declared CAP feed (the shape sources.json now ships) and the
# legacy synthesized URL the fetcher must NOT touch when a feed is declared.
_CAP_FEED_URL = "https://cap-sources.s3.amazonaws.com/fr-meteofrance-xx/rss.xml"
_LEGACY_FEED_URL = "https://severeweather.wmo.int/v2/cap-alerts/fr-meteofrance-xx/rss.xml"


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
                        "capAlertFeed": _CAP_FEED_URL,
                    }},
                ],
            }),
            _WMO_ALL_URL: _FakeNwsResponse(200, {
                "items": [{"id": "wmo-alert-1", "capURL": "fr-meteofrance-xx", "url": ""}],
            }),
            _CAP_FEED_URL: _FakeNwsResponse(200, content=rss.encode("utf-8")),
            "https://severeweather.wmo.int/cap/fr-meteofrance-xx/1.xml": (
                _FakeNwsResponse(200, text=cap_xml)
            ),
        })

    async def test_nws_failure_does_not_block_wmo_alerts(self, tmp_path):
        store = AlertsStore()
        client = self._wmo_ok_client()
        fetcher = self._fetcher(store, client, tmp_path)

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
        # Declared capAlertFeed wins; the legacy synthesized URL is not probed.
        assert _CAP_FEED_URL in client.calls
        assert _LEGACY_FEED_URL not in client.calls


# ---------------------------------------------------------------------------
# WMO feed ingest: per-agency capAlertFeed URLs, RSS 2.0 + Atom, id matching
# ---------------------------------------------------------------------------

_ATOM_FEED_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<feed xmlns="http://www.w3.org/2005/Atom" '
    'xmlns:cap="urn:oasis:names:tc:emergency:cap:1.2">'
)


def _atom_entry(href, *, identifier=None, entry_id=None):
    """MeteoAlarm-shaped Atom entry, decoy geocode link first on purpose."""
    body = "<entry>"
    if entry_id is not None:
        body += f"<id>{entry_id}</id>"
    body += (
        '<link title="MeteoAlarm" '
        'href="https://meteoalarm.org?geocode=EMMA_ID=FR802"/>'
    )
    body += f'<link type="application/cap+xml" href="{href}"/>'
    if identifier is not None:
        body += f"<cap:identifier>{identifier}</cap:identifier>"
    body += "</entry>"
    return body


def _atom_feed(entries):
    return _ATOM_FEED_HEADER + "".join(entries) + "</feed>"


def _sources_doc(*sources):
    return _FakeNwsResponse(200, {"sources": [{"source": s} for s in sources]})


def _wmo_all_doc(*items):
    return _FakeNwsResponse(200, {"items": list(items)})


@pytest.mark.alerts
class TestFeedItemLinks:
    """``_rss_item_links`` handles RSS 2.0 and Atom, namespace-robustly."""

    def test_rss_item_link_and_guid(self):
        rss = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<rss version="2.0"><channel>'
            "<item><link>https://example.com/cap/1.xml</link>"
            "<guid>urn:oid:2.49.0.0.76.0.1</guid></item>"
            "<item><link>https://example.com/cap/2.xml</link></item>"
            "<item><guid>urn:oid:nolink</guid></item>"
            "</channel></rss>"
        )
        assert _rss_item_links(rss.encode("utf-8")) == [
            ("https://example.com/cap/1.xml", "urn:oid:2.49.0.0.76.0.1"),
            ("https://example.com/cap/2.xml", None),
        ]

    def test_atom_prefixed_feed_prefers_cap_link_and_identifier(self):
        feed = _atom_feed([
            _atom_entry(
                "https://feeds.meteoalarm.org/cap/1.xml",
                identifier="2.49.0.0.250.0.FR.1",
                entry_id="https://feeds.meteoalarm.org/cap/1.xml?index_info=1",
            ),
            # Decoy-only entry with no id: nothing to fetch, skipped.
            '<entry><link href="https://meteoalarm.org?geocode=EMMA_ID=FR802"/></entry>',
        ])
        assert _rss_item_links(feed.encode("utf-8")) == [
            ("https://feeds.meteoalarm.org/cap/1.xml", "2.49.0.0.250.0.FR.1"),
        ]

    def test_atom_bare_feed_without_namespace(self):
        feed = (
            "<feed><entry>"
            '<link href="https://example.com/cap/9.xml"/>'
            "<identifier>2.49.0.0.724.0.ES.1</identifier>"
            "</entry></feed>"
        )
        assert _rss_item_links(feed.encode("utf-8")) == [
            ("https://example.com/cap/9.xml", "2.49.0.0.724.0.ES.1"),
        ]

    def test_atom_default_namespace_without_cap_identifier_uses_entry_id(self):
        feed = (
            '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
            "<id>https://example.com/cap/7.xml?index_info=abc</id>"
            '<link href="https://example.com/cap/7.xml"/>'
            "</entry></feed>"
        )
        assert _rss_item_links(feed.encode("utf-8")) == [
            ("https://example.com/cap/7.xml", "https://example.com/cap/7.xml?index_info=abc"),
        ]

    def test_atom_alternate_cap_prefix(self):
        feed = (
            '<feed xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:c="urn:oasis:names:tc:emergency:cap:1.2"><entry>'
            '<link type="application/cap+xml; charset=utf-8" '
            'href="https://example.com/cap/3.xml"/>'
            "<c:identifier>2.49.0.0.756.0.CH.3</c:identifier>"
            "</entry></feed>"
        )
        assert _rss_item_links(feed.encode("utf-8")) == [
            ("https://example.com/cap/3.xml", "2.49.0.0.756.0.CH.3"),
        ]

    def test_malformed_feed_returns_empty(self):
        assert _rss_item_links(b"not xml at all") == []

    def test_normalize_alert_id(self):
        assert _normalize_alert_id("  urn:oid:2.49.0.0.76.0.1 ") == "2.49.0.0.76.0.1"
        assert _normalize_alert_id("2.49.0.0.76.0.1") == "2.49.0.0.76.0.1"


@pytest.mark.alerts
class TestWmoFeedIngest:
    """Per-agency ``capAlertFeed`` ingest, Atom parsing, id + family matching."""

    def _fetcher(self, store, client, tmp_path):
        fetcher = WMOAlertsFetcher(store=store, cache_dir=str(tmp_path))

        async def _get_client():
            return client

        fetcher._get_client = _get_client  # type: ignore[method-assign]
        return fetcher

    async def test_client_follows_redirects(self, tmp_path):
        """Feeds that 301 to a www-canonical host must resolve, not fail."""
        fetcher = WMOAlertsFetcher(store=AlertsStore(), cache_dir=str(tmp_path))
        client = await fetcher._get_client()
        assert client.follow_redirects is True
        await fetcher.close()

    async def test_atom_feed_with_cap_identifier_is_parsed(self, tmp_path):
        feed_url = "https://feeds.meteoalarm.org/feeds/meteoalarm-legacy-atom-france"
        cap_url = "https://feeds.meteoalarm.org/cap/fr-meteofrance-xx/1.xml"
        hub_id = "2.49.0.0.250.0.FR.261023113903fbff72bf485c4fc6bbca"
        atom = _atom_feed([
            _atom_entry(cap_url, identifier=hub_id, entry_id=f"{cap_url}?index_info=1"),
        ])
        client = _FakeNwsClient({
            _WMO_SOURCES_URL: _sources_doc({
                "sourceId": "fr-meteofrance-xx",
                "capAlertFeedStatus": "operating",
                "capAlertFeed": feed_url,
            }),
            _WMO_ALL_URL: _wmo_all_doc({
                "id": f"urn:oid:{hub_id}",
                "capURL": "fr-meteofrance-xx/2026/10/23/11/39/3.xml",
            }),
            feed_url: _FakeNwsResponse(200, content=atom.encode("utf-8")),
            cap_url: _FakeNwsResponse(200, text=_cap_xml_with_polygon()),
        })
        store = AlertsStore()
        await self._fetcher(store, client, tmp_path)._fetch_once()

        assert store.fetch_success is True
        assert [a.source_id for a in store.alerts] == ["fr-meteofrance-xx"]
        assert cap_url in client.calls
        # The decoy MeteoAlarm geocode link is never fetched.
        assert not any("meteoalarm.org?" in u for u in client.calls)

    async def test_atom_href_embedded_hub_id_matches(self, tmp_path):
        """No cap:identifier (Jamaica): the hub id lives inside the CAP href."""
        feed_url = "https://jms.gov.jm/feeds/atom.xml"
        hub_id = "2.49.0.0.388.0.JM.261023155005.h6xdg95k9yZOmOo8G75n"
        cap_url = f"https://jms.gov.jm/cap/{hub_id}.xml"
        atom = _atom_feed([
            _atom_entry(cap_url, entry_id=f"{cap_url}?index_info=1"),
        ])
        client = _FakeNwsClient({
            _WMO_SOURCES_URL: _sources_doc({
                "sourceId": "jm-jms-en",
                "capAlertFeedStatus": "operating",
                "capAlertFeed": feed_url,
            }),
            _WMO_ALL_URL: _wmo_all_doc({
                "id": hub_id,
                "capURL": "jm-jms-en/2026/10/23/15/54/58-948-605.xml",
            }),
            feed_url: _FakeNwsResponse(200, content=atom.encode("utf-8")),
            cap_url: _FakeNwsResponse(200, text=_cap_xml_with_polygon()),
        })
        store = AlertsStore()
        await self._fetcher(store, client, tmp_path)._fetch_once()

        assert [a.source_id for a in store.alerts] == ["jm-jms-en"]
        assert cap_url in client.calls

    async def test_comma_separated_feeds_merged(self, tmp_path):
        feed_a = "https://cap-sources.s3.amazonaws.com/xx-feed/rss.xml"
        feed_b = "https://www.dwd.de/DWD/warnungen/cap-feed/de/rss.xml"
        cap_a = "https://example.com/cap/a.xml"
        cap_b = "https://example.com/cap/b.xml"
        hub_a = "2.49.0.0.276.0.DE.aaa"
        hub_b = "2.49.0.0.276.0.DE.bbb"
        rss_a = (
            '<rss version="2.0"><channel><item>'
            f"<link>{cap_a}</link><guid>urn:oid:{hub_a}</guid>"
            "</item></channel></rss>"
        )
        rss_b = (
            '<rss version="2.0"><channel><item>'
            f"<link>{cap_b}</link><guid>{hub_b}</guid>"
            "</item></channel></rss>"
        )
        client = _FakeNwsClient({
            _WMO_SOURCES_URL: _sources_doc({
                "sourceId": "de-dwd-en",
                "capAlertFeedStatus": "operating",
                "capAlertFeed": f"{feed_a}, {feed_b}",
            }),
            _WMO_ALL_URL: _wmo_all_doc(
                {"id": f"urn:oid:{hub_a}", "capURL": "de-dwd-en/a.xml"},
                {"id": hub_b, "capURL": "de-dwd-en/b.xml"},
            ),
            feed_a: _FakeNwsResponse(200, content=rss_a.encode("utf-8")),
            feed_b: _FakeNwsResponse(200, content=rss_b.encode("utf-8")),
            cap_a: _FakeNwsResponse(200, text=_cap_xml_with_polygon()),
            cap_b: _FakeNwsResponse(200, text=_cap_xml_with_polygon()),
        })
        store = AlertsStore()
        await self._fetcher(store, client, tmp_path)._fetch_once()

        assert feed_a in client.calls and feed_b in client.calls
        assert {a.url for a in store.alerts} == {cap_a, cap_b}
        assert store.fetch_success is True

    async def test_family_match_recovers_language_drift(self, tmp_path):
        """sources.json says de-dwd-en; the hub path segment is de-dwd-de."""
        feed_url = "https://cap-sources.s3.amazonaws.com/de-dwd-en/rss.xml"
        cap_url = "https://example.com/cap/dwd.xml"
        hub_id = "2.49.0.0.276.0.DE.2610231139"
        rss = (
            '<rss version="2.0"><channel><item>'
            f"<link>{cap_url}</link><guid>urn:oid:{hub_id}</guid>"
            "</item></channel></rss>"
        )
        client = _FakeNwsClient({
            _WMO_SOURCES_URL: _sources_doc({
                "sourceId": "de-dwd-en",
                "capAlertFeedStatus": "operating",
                "capAlertFeed": feed_url,
                "authorityCountry": "de",
                "authorityAbbrev": "dwd",
            }),
            _WMO_ALL_URL: _wmo_all_doc({
                "id": f"urn:oid:{hub_id}",
                "capURL": "de-dwd-de/2026/10/23/11/39/3.xml",
            }),
            feed_url: _FakeNwsResponse(200, content=rss.encode("utf-8")),
            cap_url: _FakeNwsResponse(200, text=_cap_xml_with_polygon()),
        })
        store = AlertsStore()
        await self._fetcher(store, client, tmp_path)._fetch_once()

        assert feed_url in client.calls
        assert [a.source_id for a in store.alerts] == ["de-dwd-en"]

    async def test_unmatched_agency_without_family_is_not_fetched(self, tmp_path):
        feed_url = "https://cap-sources.s3.amazonaws.com/zz-other/rss.xml"
        client = _FakeNwsClient({
            _WMO_SOURCES_URL: _sources_doc({
                "sourceId": "zz-other-en",
                "capAlertFeedStatus": "operating",
                "capAlertFeed": feed_url,
            }),
            _WMO_ALL_URL: _wmo_all_doc({
                "id": "urn:oid:2.49.0.0.276.0.DE.1",
                "capURL": "de-dwd-de/2026/10/23/11/39/3.xml",
            }),
            feed_url: _FakeNwsResponse(200, content=b"<rss/>"),
        })
        store = AlertsStore()
        await self._fetcher(store, client, tmp_path)._fetch_once()

        assert feed_url not in client.calls
        assert store.alerts == []
        # Nothing attempted -> not a degraded cycle.
        assert store.fetch_success is True

    async def test_all_feeds_failing_marks_fetch_unsuccessful(self, tmp_path, caplog):
        feed_a = "https://cap-sources.s3.amazonaws.com/aa-one/rss.xml"
        feed_b = "https://cap-sources.s3.amazonaws.com/bb-two/rss.xml"
        client = _FakeNwsClient({
            _WMO_SOURCES_URL: _sources_doc(
                {"sourceId": "aa-one-en", "capAlertFeedStatus": "operating",
                 "capAlertFeed": feed_a},
                {"sourceId": "bb-two-en", "capAlertFeedStatus": "operating",
                 "capAlertFeed": feed_b},
            ),
            _WMO_ALL_URL: _wmo_all_doc(
                {"id": "urn:oid:2.49.0.0.1.0.X.1", "capURL": "aa-one-en/a.xml"},
                {"id": "urn:oid:2.49.0.0.2.0.Y.1", "capURL": "bb-two-en/b.xml"},
            ),
            feed_a: _FakeNwsResponse(500),
            feed_b: _FakeNwsResponse(500),
        })
        store = AlertsStore()
        with caplog.at_level("WARNING"):
            await self._fetcher(store, client, tmp_path)._fetch_once()

        assert store.alerts == []
        assert store.fetch_success is False
        assert "WMO CAP feeds failing: 2 of 2 feed URLs unreachable" in caplog.text

    async def test_partial_feed_failure_keeps_success(self, tmp_path, caplog):
        feed_a = "https://cap-sources.s3.amazonaws.com/aa-one/rss.xml"
        feed_b = "https://cap-sources.s3.amazonaws.com/bb-two/rss.xml"
        cap_a = "https://example.com/cap/a.xml"
        hub_a = "2.49.0.0.1.0.X.1"
        rss_a = (
            '<rss version="2.0"><channel><item>'
            f"<link>{cap_a}</link><guid>urn:oid:{hub_a}</guid>"
            "</item></channel></rss>"
        )
        client = _FakeNwsClient({
            _WMO_SOURCES_URL: _sources_doc(
                {"sourceId": "aa-one-en", "capAlertFeedStatus": "operating",
                 "capAlertFeed": feed_a},
                {"sourceId": "bb-two-en", "capAlertFeedStatus": "operating",
                 "capAlertFeed": feed_b},
            ),
            _WMO_ALL_URL: _wmo_all_doc(
                {"id": f"urn:oid:{hub_a}", "capURL": "aa-one-en/a.xml"},
                {"id": "urn:oid:2.49.0.0.2.0.Y.1", "capURL": "bb-two-en/b.xml"},
            ),
            feed_a: _FakeNwsResponse(200, content=rss_a.encode("utf-8")),
            feed_b: _FakeNwsResponse(500),
            cap_a: _FakeNwsResponse(200, text=_cap_xml_with_polygon()),
        })
        store = AlertsStore()
        with caplog.at_level("WARNING"):
            await self._fetcher(store, client, tmp_path)._fetch_once()

        assert [a.source_id for a in store.alerts] == ["aa-one-en"]
        assert store.fetch_success is True
        assert "WMO CAP feeds failing: 1 of 2 feed URLs unreachable" in caplog.text

    async def test_tier2_fallback_fetches_unmatched_feed_items(self, tmp_path):
        """INMET-style URL guids match no hub id -> newest items still parsed."""
        feed_url = "https://avisos.inmet.gov.br/rss.xml"
        cap_1 = "https://avisos.inmet.gov.br/cap/1.xml"
        cap_2 = "https://avisos.inmet.gov.br/cap/2.xml"
        rss = (
            '<rss version="2.0"><channel>'
            "<item><link>%s</link><guid>https://avisos.inmet.gov.br/#1</guid></item>"
            "<item><link>%s</link><guid>https://avisos.inmet.gov.br/#2</guid></item>"
            "</channel></rss>"
        ) % (cap_1, cap_2)
        client = _FakeNwsClient({
            _WMO_SOURCES_URL: _sources_doc({
                "sourceId": "br-inmet-pt",
                "capAlertFeedStatus": "operating",
                "capAlertFeed": feed_url,
            }),
            _WMO_ALL_URL: _wmo_all_doc({
                "id": "urn:oid:2.49.0.0.76.0.2026.28553.1",
                "capURL": "br-inmet-pt/2026/10/23/x.xml",
            }),
            feed_url: _FakeNwsResponse(200, content=rss.encode("utf-8")),
            cap_1: _FakeNwsResponse(200, text=_cap_xml_with_polygon()),
            cap_2: _FakeNwsResponse(200, text=_cap_xml_with_polygon()),
        })
        store = AlertsStore()
        await self._fetcher(store, client, tmp_path)._fetch_once()

        assert {a.url for a in store.alerts} == {cap_1, cap_2}

    async def test_missing_cap_feed_falls_back_to_legacy_url(self, tmp_path):
        cap_url = "https://severeweather.wmo.int/cap/fr-meteofrance-xx/1.xml"
        client = _FakeNwsClient({
            _WMO_SOURCES_URL: _sources_doc({
                "sourceId": "fr-meteofrance-xx",
                "capAlertFeedStatus": "operating",
                "capAlertFeed": "",
            }),
            _WMO_ALL_URL: _wmo_all_doc({
                "id": "wmo-alert-1",
                "capURL": "fr-meteofrance-xx/2026/10/23/x.xml",
            }),
            _LEGACY_FEED_URL: _FakeNwsResponse(
                200, content=_rss_with_item().encode("utf-8")),
            cap_url: _FakeNwsResponse(200, text=_cap_xml_with_polygon()),
        })
        store = AlertsStore()
        await self._fetcher(store, client, tmp_path)._fetch_once()

        assert _LEGACY_FEED_URL in client.calls
        assert [a.source_id for a in store.alerts] == ["fr-meteofrance-xx"]
        assert store.fetch_success is True


# ---------------------------------------------------------------------------
# Geocode resolver registry (pluggable area-code tables)
# ---------------------------------------------------------------------------

_GEOCODE_BOX_A = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
_GEOCODE_BOX_B = [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0), (5.0, 5.0)]


def _geocode_feature(code, geometry):
    return {
        "type": "Feature",
        "properties": {"code": code},
        "geometry": geometry,
    }


def _geocode_polygon(offset=0.0, size=1.0):
    x = offset
    return {
        "type": "Polygon",
        "coordinates": [
            [[x, 0.0], [x + size, 0.0], [x + size, size], [x, size], [x, 0.0]]
        ],
    }


def _write_geocode_tables(tmp_path, features, alias_rows=()):
    """Write MeteoAlarm-shaped geocode + alias fixtures, return their paths."""
    geojson = tmp_path / "meteoalarm_geocodes.json"
    geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": list(features)})
    )
    aliases = tmp_path / "meteoalarm_aliases.csv"
    aliases.write_text(
        "CODE,ALIAS_CODE,ALIAS_TYPE\n"
        + "".join(f"{code},{alias},NUTS2\n" for code, alias in alias_rows)
    )
    return geojson, aliases


def _provider(name, value_names, polygons=None, loader=None, normalize=None):
    """Build a GeocodeProvider with an inert loader unless one is supplied."""

    async def _unloadable(client, cache_base):  # pragma: no cover - guard
        raise AssertionError("loader should not be called")

    kwargs = {"normalize": normalize} if normalize is not None else {}
    return GeocodeProvider(
        name=name,
        value_names=frozenset(value_names),
        loader=loader or _unloadable,
        polygons=dict(polygons or {}),
        **kwargs,
    )


def _cap_xml_with_geocode(value_name, value):
    """CAP 1.2 alert with no <polygon>, only one <geocode> reference."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
      <info>
        <language>en-US</language>
        <event>Wind Warning</event>
        <headline>Wind Warning issued</headline>
        <description>Strong winds expected</description>
        <urgency>Immediate</urgency>
        <severity>Severe</severity>
        <effective>2026-05-07T00:15:00-05:00</effective>
        <expires>2099-05-07T06:00:00-05:00</expires>
        <area>
          <areaDesc>Test Area</areaDesc>
          <geocode>
            <valueName>{value_name}</valueName>
            <value>{value}</value>
          </geocode>
        </area>
      </info>
    </alert>"""


@pytest.mark.alerts
class TestGeocodeResolve:
    def test_first_claiming_provider_wins(self):
        poly_a = Polygon(_GEOCODE_BOX_A)
        poly_b = Polygon(_GEOCODE_BOX_B)
        a = _provider("a", {"EMMA_ID"}, {"1": poly_a})
        b = _provider("b", {"EMMA_ID"}, {"1": poly_b})

        assert resolve([a, b], "EMMA_ID", "1") is poly_a
        assert resolve([b, a], "EMMA_ID", "1") is poly_b

    def test_value_name_is_case_insensitive(self):
        poly = Polygon(_GEOCODE_BOX_A)
        provider = _provider("p", {"EMMA_ID"}, {"AT001": poly})

        assert resolve([provider], "emma_id", "AT001") is poly
        assert resolve([provider], "  EMMA_ID  ", "AT001") is poly

    def test_normalize_is_applied(self):
        poly = Polygon(_GEOCODE_BOX_A)
        provider = _provider("p", {"EMMA_ID"}, {"AT001": poly})

        # Default normalize strips + upper-cases.
        assert resolve([provider], "EMMA_ID", " at001 ") is poly

    def test_custom_normalize_is_used(self):
        poly = Polygon(_GEOCODE_BOX_A)
        provider = _provider(
            "p", {"WARNCELLID"}, {"105754000": poly}, normalize=lambda v: v.strip()
        )

        assert resolve([provider], "WARNCELLID", " 105754000") is poly
        assert resolve([provider], "WARNCELLID", "10575400X") is None

    def test_miss_falls_through_to_next_claiming_provider(self):
        poly = Polygon(_GEOCODE_BOX_B)
        empty = _provider("empty", {"EMMA_ID"}, {})
        hit = _provider("hit", {"EMMA_ID"}, {"HR806": poly})

        assert resolve([empty, hit], "EMMA_ID", "HR806") is poly

    def test_non_claiming_provider_is_skipped(self):
        poly_b = Polygon(_GEOCODE_BOX_B)
        wrong = _provider("wrong", {"WARNCELLID"}, {"1": Polygon(_GEOCODE_BOX_A)})
        right = _provider("right", {"EMMA_ID"}, {"1": poly_b})

        assert resolve([wrong, right], "EMMA_ID", "1") is poly_b

    def test_unserved_or_empty_inputs_return_none(self):
        poly = Polygon(_GEOCODE_BOX_A)
        provider = _provider("p", {"EMMA_ID"}, {"1": poly})

        assert resolve([provider], "WARNCELLID", "1") is None
        assert resolve([provider], "", "1") is None
        assert resolve([provider], "EMMA_ID", "") is None
        assert resolve([], "EMMA_ID", "1") is None

    def test_default_registry_claims_nuts2(self):
        providers = build_default_providers()

        # MeteoAlarm stays first; the national data drops follow.
        assert providers[0].name == "meteoalarm"
        assert providers[0].value_names == frozenset({"EMMA_ID", "NUTS2", "NUTS3"})
        assert providers[0].polygons == {}


@pytest.mark.alerts
class TestGeocodeCapResolution:
    def test_nuts2_geocode_resolves_via_alias(self, tmp_path):
        """Belgium publishes NUTS2 (BE23); the table only carries an alias."""
        geojson, aliases = _write_geocode_tables(
            tmp_path,
            [_geocode_feature("BE003", _geocode_polygon())],
            [("BE003", "BE23")],
        )
        table = _apply_meteoalarm_aliases(
            _parse_meteoalarm_geojson(str(geojson)), str(aliases)
        )
        assert "BE23" in table
        provider = _provider("meteoalarm", {"EMMA_ID", "NUTS2", "NUTS3"}, table)

        entries = _extract_polygons_from_cap(
            _cap_xml_with_geocode("NUTS2", "BE23"),
            "be-irm-en",
            "https://example.com/cap",
            [provider],
        )

        assert len(entries) == 1
        assert entries[0].source_id == "be-irm-en"
        assert entries[0].polygon is not None
        assert entries[0].polygon.area == pytest.approx(1.0)

    def test_emma_id_geocode_still_resolves(self):
        poly = Polygon(_GEOCODE_BOX_A)
        provider = _provider("meteoalarm", {"EMMA_ID", "NUTS2", "NUTS3"}, {"HR804": poly})

        entries = _extract_polygons_from_cap(
            _cap_xml_with_geocode("EMMA_ID", "HR804"),
            "hr-meteo-hr",
            "https://example.com/cap",
            [provider],
        )

        assert len(entries) == 1
        assert entries[0].polygon is not None

    def test_unserved_value_name_is_still_dropped(self, caplog):
        providers = build_default_providers()

        with caplog.at_level("DEBUG"):
            entries = _extract_polygons_from_cap(
                _cap_xml_with_geocode("WARNCELLID", "105754000"),
                "de-dwd-en",
                "https://example.com/cap",
                providers,
            )

        assert entries == []
        assert "Dropped CAP alert for de-dwd-en" in caplog.text

    def test_invalid_table_polygon_repaired(self, tmp_path):
        """Self-intersecting table polygon (HR806 shape) is repaired, not lost."""
        bowtie = {
            "type": "Polygon",
            "coordinates": [
                [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
            ],
        }
        geojson, _aliases = _write_geocode_tables(
            tmp_path,
            [
                _geocode_feature("HR806", bowtie),
                _geocode_feature("X1", _geocode_polygon(2.0)),
                _geocode_feature("NULLGEOM", None),
            ],
        )

        table = _parse_meteoalarm_geojson(str(geojson))

        assert set(table) == {"HR806", "X1"}
        assert table["HR806"].is_valid
        assert not table["HR806"].is_empty
        assert table["HR806"].area > 0

    def test_repaired_polygon_serves_a_cap(self, tmp_path):
        bowtie = {
            "type": "Polygon",
            "coordinates": [
                [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
            ],
        }
        geojson, _aliases = _write_geocode_tables(
            tmp_path, [_geocode_feature("HR806", bowtie)]
        )
        provider = _provider(
            "meteoalarm", {"EMMA_ID"}, _parse_meteoalarm_geojson(str(geojson))
        )

        entries = _extract_polygons_from_cap(
            _cap_xml_with_geocode("EMMA_ID", "HR806"),
            "hr-meteo-hr",
            "https://example.com/cap",
            [provider],
        )

        assert len(entries) == 1
        assert entries[0].polygon.is_valid


@pytest.mark.alerts
class TestGeocodeTableLoading:
    async def test_ensure_loaded_is_idempotent(self, tmp_path):
        calls: list[object] = []
        poly = Polygon(_GEOCODE_BOX_A)

        async def _loader(client, cache_base):
            calls.append(cache_base)
            return {"AT001": poly}

        provider = _provider("meteoalarm", {"EMMA_ID"}, loader=_loader)

        await ensure_loaded(provider, None, tmp_path)
        await ensure_loaded(provider, None, tmp_path)

        assert calls == [tmp_path]
        assert provider.polygons == {"AT001": poly}

    async def test_provider_failure_does_not_stop_others(self, tmp_path, caplog):
        poly = Polygon(_GEOCODE_BOX_A)

        async def _boom(client, cache_base):
            raise RuntimeError("table down")

        async def _ok(client, cache_base):
            return {"AT001": poly}

        fetcher = WMOAlertsFetcher(store=AlertsStore(), cache_dir=str(tmp_path))
        fetcher._geocode_providers = [
            _provider("broken", {"WARNCELLID"}, loader=_boom),
            _provider("good", {"EMMA_ID"}, loader=_ok),
        ]

        with caplog.at_level("INFO"):
            await fetcher._ensure_geocode_tables()
        await fetcher.close()

        assert "Geocode table broken failed to load: table down" in caplog.text
        assert fetcher._tables_ready.is_set()
        assert fetcher._geocode_providers[0].polygons == {}
        assert fetcher._geocode_providers[1].polygons == {"AT001": poly}
        assert "Geocode table good ready: 1 polygons" in caplog.text

    async def test_default_tables_load_from_cache_dir(self, tmp_path):
        """Real MeteoAlarm loader against pre-seeded cache files (no HTTP)."""
        cache_alerts = tmp_path / "alerts"
        cache_alerts.mkdir(parents=True, exist_ok=True)
        _write_geocode_tables(
            cache_alerts,
            [_geocode_feature("BE003", _geocode_polygon())],
            [("BE003", "BE23")],
        )

        fetcher = WMOAlertsFetcher(store=AlertsStore(), cache_dir=str(tmp_path))
        # Only the pre-seeded table: the DWD/BoM providers would download.
        fetcher._geocode_providers = build_default_providers()[:1]
        await fetcher._ensure_geocode_tables()
        await fetcher.close()

        provider = fetcher._geocode_providers[0]
        assert provider.name == "meteoalarm"
        assert "BE003" in provider.polygons
        assert "BE23" in provider.polygons
        assert fetcher._tables_ready.is_set()


# ---------------------------------------------------------------------------
# Geocode providers: DWD WARNCELLID (Germany) and BoM AMOC (Australia)
# ---------------------------------------------------------------------------

_DWD_BOWTIE = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
}


def _write_dwd_geojson(tmp_path, features, name="dwd_warncell_gemeinden.json"):
    """Write a DWD-Warngebiete-shaped GeoJSON layer file."""
    path = tmp_path / name
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": list(features)})
    )
    return path


def _dwd_feature(warncellid, geometry=None):
    return {
        "type": "Feature",
        "properties": {"WARNCELLID": warncellid, "NAME": "test"},
        "geometry": geometry,
    }


def _clockwise_ring(offset=0.0, size=1.0):
    """ESRI exterior winding (clockwise -> negative shoelace area)."""
    x = offset
    return [
        [x, 0.0], [x, size], [x + size, size], [x + size, 0.0], [x, 0.0]
    ]


def _hole_ring(offset=1.0, size=2.0):
    """ESRI interior/hole winding (anticlockwise), nested inside a shell."""
    x = offset
    return [
        [x, x], [x + size, x], [x + size, x + size], [x, x + size], [x, x]
    ]


def _write_bom_zip(path, field_name, records):
    """Write a real ESRI polygon shapefile zip (via pyshp) for one product."""
    import shapefile

    src_dir = path.parent / (path.stem + "_src")
    src_dir.mkdir(parents=True, exist_ok=True)
    base = src_dir / "IDM00001"
    writer = shapefile.Writer(str(base))
    try:
        writer.field(field_name, "C", size=12)
        for code, rings in records:
            writer.poly(rings)
            writer.record(code)
    finally:
        writer.close()
    with zipfile.ZipFile(path, "w") as archive:
        for member in sorted(base.parent.glob(base.name + ".*")):
            archive.write(member, member.name)
    return path


@pytest.mark.alerts
class TestGeocodeProvidersDwd:
    def test_parse_pads_warncell_ids_to_nine_digits(self, tmp_path):
        """WARNCELLID is a JSON number; keys are the 9-digit CAP form."""
        path = _write_dwd_geojson(tmp_path, [
            _dwd_feature(801051118, _geocode_polygon()),
            _dwd_feature(501000004, _geocode_polygon(2.0)),
            _dwd_feature(51000004, _geocode_polygon(4.0)),
        ])

        cells = _parse_dwd_geojson(path)

        assert set(cells) == {"801051118", "501000004", "051000004"}
        assert all(isinstance(code, str) for code in cells)
        assert cells["801051118"].area == pytest.approx(1.0)

    def test_invalid_ring_repaired_and_null_geometry_skipped(self, tmp_path):
        path = _write_dwd_geojson(tmp_path, [
            _dwd_feature(801051118, _DWD_BOWTIE),
            _dwd_feature(999999999, None),
        ])

        cells = _parse_dwd_geojson(path)

        assert set(cells) == {"801051118"}
        assert cells["801051118"].is_valid
        assert cells["801051118"].area > 0

    async def test_loader_merges_cached_layers(self, tmp_path):
        _write_dwd_geojson(
            tmp_path,
            [_dwd_feature(801051118, _geocode_polygon())],
            "dwd_warncell_gemeinden.json",
        )
        _write_dwd_geojson(
            tmp_path,
            [_dwd_feature(501000004, _geocode_polygon(2.0))],
            "dwd_warncell_kueste.json",
        )
        _write_dwd_geojson(tmp_path, [], "dwd_warncell_kreise.json")
        _write_dwd_geojson(tmp_path, [], "dwd_warncell_binnenseen.json")

        cells = await _load_dwd_warncell(None, tmp_path)

        assert set(cells) == {"801051118", "501000004"}

    def test_resolve_with_warncell_provider(self):
        poly = Polygon(_GEOCODE_BOX_A)
        provider = _provider("dwd_warncell", {"WARNCELLID"}, {"801051118": poly})

        assert resolve([provider], "WARNCELLID", "801051118") is poly
        # valueName is matched case-insensitively, code is strip+upper'd.
        assert resolve([provider], "WarnCellId", " 801051118 ") is poly
        assert resolve([provider], "WARNCELLID", "801051119") is None


@pytest.mark.alerts
class TestGeocodeProvidersBom:
    def test_parse_extracts_aac_polygon(self, tmp_path):
        zip_path = _write_bom_zip(
            tmp_path / "bom_amoc_IDM00001.zip",
            "AAC",
            [("WA_PW010", [_clockwise_ring()])],
        )

        cells = _parse_bom_zip(zip_path, "PW public weather districts")

        assert set(cells) == {"WA_PW010"}
        assert isinstance(cells["WA_PW010"], Polygon)
        assert cells["WA_PW010"].area == pytest.approx(1.0)

    def test_alternate_aac_field_name(self, tmp_path):
        """Per-product field names vary: AAC_MW, AAC_FW, AAC_ME, AAC_RC."""
        zip_path = _write_bom_zip(
            tmp_path / "bom_amoc_IDM00003.zip",
            "AAC_MW",
            [("VIC_ME005", [_clockwise_ring()])],
        )

        assert set(_parse_bom_zip(zip_path, "MW marine waters")) == {"VIC_ME005"}

    def test_hole_ring_becomes_interior(self, tmp_path):
        zip_path = _write_bom_zip(
            tmp_path / "bom_amoc_IDM00017.zip",
            "AAC",
            [(
                "QLD_RC030",
                [_clockwise_ring(0.0, 4.0), _hole_ring(1.0, 2.0)],
            )],
        )

        cells = _parse_bom_zip(zip_path, "RC flood catchments")

        polygon = cells["QLD_RC030"]
        assert len(polygon.interiors) == 1
        assert polygon.area == pytest.approx(16.0 - 4.0)

    def test_missing_aac_field_is_skipped(self, tmp_path, caplog):
        zip_path = _write_bom_zip(
            tmp_path / "bom_amoc_IDM00001.zip",
            "DIST_NAME",
            [("WA_PW010", [_clockwise_ring()])],
        )

        with caplog.at_level("DEBUG"):
            cells = _parse_bom_zip(zip_path, "PW")

        assert cells == {}
        assert "no AAC field" in caplog.text

    async def test_loader_merges_cached_products(self, tmp_path):
        products = (
            ("IDM00001", "AAC", "WA_PW010"),
            ("IDM00003", "AAC_MW", "VIC_ME005"),
            ("IDM00007", "AAC_FW", "NSW_FW021"),
            ("IDM00014", "AAC_ME", "QLD_ME001"),
            ("IDM00017", "AAC_RC", "SA_RC009"),
        )
        for index, (product, field, code) in enumerate(products):
            _write_bom_zip(
                tmp_path / f"bom_amoc_{product}.zip",
                field,
                [(code, [_clockwise_ring(offset=float(index) * 10.0)])],
            )

        cells = await _load_bom_amoc(None, tmp_path)

        assert set(cells) == {code for _p, _f, code in products}

    def test_url_candidates_try_https_then_ftp(self):
        candidates = _bom_url_candidates("IDM00001")

        assert candidates[0].startswith("https://www.bom.gov.au/")
        assert candidates[-1].startswith("ftp://ftp.bom.gov.au/")
        assert candidates[0].endswith("/IDM00001.zip")


@pytest.mark.alerts
class TestProviderRouting:
    def test_default_registry_order_and_claims(self):
        providers = build_default_providers()

        assert [p.name for p in providers] == [
            "meteoalarm",
            "dwd_warncell",
            "bom_amoc",
        ]
        claims = {p.name: p.value_names for p in providers}
        assert claims["meteoalarm"] == frozenset({"EMMA_ID", "NUTS2", "NUTS3"})
        assert claims["dwd_warncell"] == frozenset({"WARNCELLID"})
        assert claims["bom_amoc"] == frozenset({"AMOC-AREACODE"})

    def test_cap_value_names_route_to_their_tables(self):
        emma = Polygon(_GEOCODE_BOX_A)
        dwd = Polygon(_GEOCODE_BOX_B)
        bom = Polygon([(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0)])
        providers = [
            _provider("meteoalarm", {"EMMA_ID", "NUTS2", "NUTS3"}, {"BE23": emma}),
            _provider("dwd_warncell", {"WARNCELLID"}, {"801051118": dwd}),
            _provider("bom_amoc", {"AMOC-AREACODE"}, {"WA_PW010": bom}),
        ]

        assert resolve(providers, "EMMA_ID", "BE23") is emma
        assert resolve(providers, "WARNCELLID", "801051118") is dwd
        assert resolve(providers, "AMOC-AreaCode", "WA_PW010") is bom
        assert resolve(providers, "AMOC-AreaCode", "XX_PW999") is None

    def test_cap_geocode_resolves_through_bom_provider(self):
        bom = Polygon(_GEOCODE_BOX_A)
        provider = _provider("bom_amoc", {"AMOC-AREACODE"}, {"WA_PW010": bom})

        entries = _extract_polygons_from_cap(
            _cap_xml_with_geocode("AMOC-AreaCode", "WA_PW010"),
            "au-bom-en",
            "https://example.com/cap",
            [provider],
        )

        assert len(entries) == 1
        assert entries[0].source_id == "au-bom-en"
        assert entries[0].polygon is not None


# ---------------------------------------------------------------------------
# One AlertEntry per CAP <info>: merged geometry + containment pruning
# ---------------------------------------------------------------------------

def _cap_polygon_text(*lat_lon_pairs):
    """CAP <polygon> text: space-separated ``lat,lon`` pairs."""
    return " ".join(f"{lat},{lon}" for lat, lon in lat_lon_pairs)


def _cap_box_text(min_lat, min_lon, max_lat, max_lon):
    """CAP <polygon> text for an axis-aligned box."""
    return _cap_polygon_text(
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, max_lon),
        (max_lat, min_lon),
    )


def _cap_xml_with_areas(areas, event="Wind Warning"):
    """CAP 1.2 alert with several <area> blocks under one <info>.

    Each area is ``{"desc": str, "polygon": cap_text | None,
    "geocodes": [(value_name, value), ...]}``.
    """
    bodies = []
    for area in areas:
        body = "<area>"
        if area.get("desc"):
            body += f"<areaDesc>{area['desc']}</areaDesc>"
        if area.get("polygon"):
            body += f"<polygon>{area['polygon']}</polygon>"
        for value_name, value in area.get("geocodes", ()):
            body += (
                "<geocode>"
                f"<valueName>{value_name}</valueName>"
                f"<value>{value}</value>"
                "</geocode>"
            )
        body += "</area>"
        bodies.append(body)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">'
        "<info>"
        "<language>en-US</language>"
        f"<event>{event}</event>"
        f"<headline>{event} issued</headline>"
        "<description>Strong winds expected</description>"
        "<urgency>Immediate</urgency>"
        "<severity>Severe</severity>"
        "<effective>2026-05-07T00:15:00-05:00</effective>"
        "<expires>2099-05-07T06:00:00-05:00</expires>"
        + "".join(bodies)
        + "</info></alert>"
    )


def _box(min_lon, min_lat, max_lon, max_lat):
    return Polygon([
        (min_lon, min_lat), (max_lon, min_lat),
        (max_lon, max_lat), (min_lon, max_lat), (min_lon, min_lat),
    ])


@pytest.mark.alerts
class TestPerInfoMerging:
    def test_contained_geocodes_pruned_against_aggregate_polygon(self):
        """DWD shape: aggregate <polygon> + per-municipality WARNCELLIDs."""
        aggregate = _cap_box_text(0, 0, 10, 10)
        provider = _provider(
            "dwd_warncell",
            {"WARNCELLID"},
            {
                "801051118": _box(1, 1, 2, 2),
                "501000004": _box(3, 3, 4, 4),
            },
        )
        cap = _cap_xml_with_areas([
            {"desc": "Germany", "polygon": aggregate},
            {"desc": "Municipality A", "geocodes": [("WARNCELLID", "801051118")]},
            {"desc": "Municipality B", "geocodes": [("WARNCELLID", "501000004")]},
        ])

        entries = _extract_polygons_from_cap(
            cap, "de-dwd-en", "https://example.com/cap", [provider]
        )

        assert len(entries) == 1
        # Every warn cell sat inside the aggregate, so only it remains.
        assert entries[0].polygon.geom_type == "Polygon"
        assert entries[0].polygon.equals(_box(0, 0, 10, 10))
        assert entries[0].area_desc == "Germany; Municipality A; Municipality B"

    def test_mixed_polygon_and_outside_geocode_merge(self):
        provider = _provider(
            "dwd_warncell", {"WARNCELLID"}, {"801051118": _box(5, 5, 6, 6)}
        )
        cap = _cap_xml_with_areas([
            {"desc": "North", "polygon": _cap_box_text(0, 0, 2, 2)},
            {"desc": "South", "geocodes": [("WARNCELLID", "801051118")]},
        ])

        entries = _extract_polygons_from_cap(
            cap, "de-dwd-en", "https://example.com/cap", [provider]
        )

        assert len(entries) == 1
        polygon = entries[0].polygon
        assert polygon.geom_type == "MultiPolygon"
        assert len(polygon.geoms) == 2
        # Coverage of both parts: the explicit box and the outside cell.
        assert polygon.contains(Point(1.0, 1.0))
        assert polygon.contains(Point(5.5, 5.5))
        assert entries[0].area_desc == "North; South"

    def test_repeated_geocode_in_one_info_is_not_duplicated(self):
        cell = _box(1, 1, 2, 2)
        provider = _provider("dwd_warncell", {"WARNCELLID"}, {"801051118": cell})
        cap = _cap_xml_with_areas([
            {"desc": "A", "geocodes": [("WARNCELLID", "801051118")]},
            {"desc": "B", "geocodes": [("WARNCELLID", "801051118")]},
        ])

        entries = _extract_polygons_from_cap(
            cap, "de-dwd-en", "https://example.com/cap", [provider]
        )

        assert len(entries) == 1
        # One shared table object -> a single plain Polygon, not a MultiPolygon.
        assert entries[0].polygon.geom_type == "Polygon"
        assert entries[0].polygon.equals(cell)

    def test_two_distinct_geocodes_build_multipolygon(self):
        provider = _provider(
            "dwd_warncell",
            {"WARNCELLID"},
            {"801051118": _box(1, 1, 2, 2), "501000004": _box(5, 5, 6, 6)},
        )
        cap = _cap_xml_with_areas([
            {"desc": "A", "geocodes": [("WARNCELLID", "801051118")]},
            {"desc": "B", "geocodes": [("WARNCELLID", "501000004")]},
        ])

        entries = _extract_polygons_from_cap(
            cap, "de-dwd-en", "https://example.com/cap", [provider]
        )

        assert len(entries) == 1
        assert entries[0].polygon.geom_type == "MultiPolygon"
        assert len(entries[0].polygon.geoms) == 2

    def test_single_area_yields_plain_polygon(self):
        cap = _cap_xml_with_areas([
            {"desc": "Test Area", "polygon": _cap_box_text(0, 0, 1, 1)},
        ])

        entries = _extract_polygons_from_cap(
            cap, "fr-meteofrance-xx", "https://example.com/cap", []
        )

        assert len(entries) == 1
        assert isinstance(entries[0].polygon, Polygon)
        assert not isinstance(entries[0].polygon, MultiPolygon)
        assert entries[0].area_desc == "Test Area"

    def test_info_without_any_geometry_is_still_dropped(self, caplog):
        cap = _cap_xml_with_areas([
            {"desc": "Nowhere", "geocodes": [("WARNCELLID", "105754000")]},
        ])

        with caplog.at_level("DEBUG"):
            entries = _extract_polygons_from_cap(
                cap, "de-dwd-en", "https://example.com/cap", build_default_providers()
            )

        assert entries == []
        assert "Dropped CAP alert for de-dwd-en" in caplog.text


# ---------------------------------------------------------------------------
# Geocode table robustness: corrupt cache eviction + WFS truncation guard
# ---------------------------------------------------------------------------

@pytest.mark.alerts
class TestGeocodeCacheRobustness:
    async def test_corrupt_dwd_layer_is_evicted_and_reraised(self, tmp_path):
        for suffix in ("gemeinden", "kreise", "kueste", "binnenseen"):
            _write_dwd_geojson(tmp_path, [], f"dwd_warncell_{suffix}.json")
        corrupt = tmp_path / "dwd_warncell_kreise.json"
        corrupt.write_text("{not valid json")

        with pytest.raises(json.JSONDecodeError):
            await _load_dwd_warncell(None, tmp_path)

        # The bad file is dropped so the next cycle re-downloads that layer;
        # the healthy layers stay cached.
        assert not corrupt.exists()
        assert (tmp_path / "dwd_warncell_gemeinden.json").exists()

    async def test_corrupt_bom_product_is_evicted_and_reraised(self, tmp_path):
        products = (
            ("IDM00001", "AAC", "WA_PW010"),
            ("IDM00003", "AAC_MW", "VIC_ME005"),
            ("IDM00007", "AAC_FW", "NSW_FW021"),
            ("IDM00014", "AAC_ME", "QLD_ME001"),
            ("IDM00017", "AAC_RC", "SA_RC009"),
        )
        for index, (product, field, code) in enumerate(products):
            _write_bom_zip(
                tmp_path / f"bom_amoc_{product}.zip",
                field,
                [(code, [_clockwise_ring(offset=float(index) * 10.0)])],
            )
        corrupt = tmp_path / "bom_amoc_IDM00007.zip"
        corrupt.write_bytes(b"not a zip archive")

        with pytest.raises(zipfile.BadZipFile):
            await _load_bom_amoc(None, tmp_path)

        assert not corrupt.exists()
        assert (tmp_path / "bom_amoc_IDM00001.zip").exists()

    async def test_truncated_wfs_layer_is_rejected(self, tmp_path):
        """numberReturned < numberMatched must fail loudly, not under-cover."""
        payload = json.dumps({
            "type": "FeatureCollection",
            "numberMatched": 11214,
            "numberReturned": 100,
            "features": [],
        }).encode()
        layer = "Warngebiete_Gemeinden"
        client = _FakeNwsClient({
            _DWD_WFS_URL.format(layer=layer): _FakeNwsResponse(200, content=payload),
        })
        dest = tmp_path / "dwd_warncell_gemeinden.json"

        with pytest.raises(RuntimeError, match="truncated"):
            await _download_dwd_layer(client, layer, dest)

        # Nothing was cached: the next cycle retries the download.
        assert not dest.exists()

    async def test_wfs_layer_without_counts_is_accepted(self, tmp_path):
        payload = json.dumps({
            "type": "FeatureCollection",
            "features": [],
        }).encode()
        layer = "Warngebiete_Binnenseen"
        client = _FakeNwsClient({
            _DWD_WFS_URL.format(layer=layer): _FakeNwsResponse(200, content=payload),
        })
        dest = tmp_path / "dwd_warncell_binnenseen.json"

        await _download_dwd_layer(client, layer, dest)

        assert dest.exists()

    async def test_ftp_candidate_uses_bounded_downloader(self, tmp_path, monkeypatch):
        """The FTP fallback runs through the timeout-bounded helper."""
        import librewxr.data.alert_geocodes as geocodes

        calls = []

        def _fake_ftp(url, dest, timeout):
            calls.append((url, timeout))
            served = tmp_path / "served.zip"
            _write_bom_zip(served, "AAC", [("WA_PW010", [_clockwise_ring()])])
            shutil.copyfile(served, dest)

        monkeypatch.setattr(geocodes, "_ftp_download", _fake_ftp)
        client = _FakeNwsClient({
            url: _FakeNwsResponse(403)
            for url in _bom_url_candidates("IDM00001")
            if url.startswith("https")
        })
        dest = tmp_path / "bom_amoc_IDM00001.zip"

        await geocodes._download_bom_product(client, "IDM00001", dest)

        assert dest.exists()
        assert len(calls) == 1
        assert calls[0][0].startswith("ftp://")
        assert calls[0][1] == 60.0

    async def test_wfs_layer_with_string_counts_accepted(self, tmp_path):
        payload = json.dumps({
            "type": "FeatureCollection",
            "numberMatched": "23",
            "numberReturned": "23",
            "features": [],
        }).encode()
        layer = "Warngebiete_Binnenseen"
        client = _FakeNwsClient({
            _DWD_WFS_URL.format(layer=layer): _FakeNwsResponse(200, content=payload),
        })
        dest = tmp_path / "dwd_warncell_binnenseen.json"

        await _download_dwd_layer(client, layer, dest)

        assert dest.exists()
