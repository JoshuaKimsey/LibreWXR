# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for MCP alerts query: alerts_within_radius with radius/severity/expiry
filtering against the merged WMO + NWS store."""

import pytest
from shapely.geometry import MultiPolygon, Polygon

from librewxr.api.models import AlertsResponse
from librewxr.data.alerts_store import AlertEntry
from librewxr.mcp.alerts_query import alerts_within_radius


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockAlertsStore:
    """Minimal object satisfying the ``alerts``-property interface."""
    def __init__(self, alerts):
        self.alerts = alerts
_NEAR_POLY = Polygon([
    (-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1),
    (-0.1, -0.1),
])
_FAR_POLY = Polygon([
    (0.9, -0.1), (1.1, -0.1), (1.1, 0.1), (0.9, 0.1),
    (0.9, -0.1),
])
_QUERY_LAT = 0.0
_QUERY_LON = 0.0
_FUTURE_EXPIRES = "2099-01-01T00:00:00+00:00"


def _make_alert(event, severity, polygon, url="https://wmo.example.com/a",
                expires=_FUTURE_EXPIRES, effective="2026-01-01T00:00:00+00:00"):
    return AlertEntry(
        source_id="test",
        event=event,
        description=f"{event} description",
        severity=severity,
        effective=effective,
        expires=expires,
        area_desc="Test Area",
        url=url,
        polygon=polygon,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.mcp
async def test_alerts_within_radius_none_store():
    """None store returns empty FeatureCollection."""
    result = await alerts_within_radius(None, _QUERY_LAT, _QUERY_LON, 25.0)
    assert isinstance(result, AlertsResponse)
    assert result.type == "FeatureCollection"
    assert result.features == []


@pytest.mark.mcp
async def test_alerts_within_radius_empty_store():
    """Store with no alerts returns empty FeatureCollection."""
    store = _MockAlertsStore([])
    result = await alerts_within_radius(store, _QUERY_LAT, _QUERY_LON, 25.0)
    assert len(result.features) == 0


@pytest.mark.mcp
async def test_alerts_within_radius_filter_by_radius():
    """Only alerts whose polygon is within the radius are returned."""
    near_alert = _make_alert("Near", "Severe", _NEAR_POLY, url="https://wmo.example.com/near")
    far_alert = _make_alert("Far", "Severe", _FAR_POLY, url="https://wmo.example.com/far")
    store = _MockAlertsStore([near_alert, far_alert])

    result = await alerts_within_radius(store, _QUERY_LAT, _QUERY_LON, 25.0)
    assert len(result.features) == 1
    assert result.features[0].properties.title == "Near"


@pytest.mark.mcp
async def test_alerts_within_radius_severity_filter():
    """severity parameter excludes alerts with a different severity."""
    severe_alert = _make_alert("Severe Alert", "Severe", _NEAR_POLY, url="https://wmo.example.com/severe")
    minor_alert = _make_alert("Minor Alert", "Minor", _NEAR_POLY, url="https://wmo.example.com/minor")
    store = _MockAlertsStore([severe_alert, minor_alert])

    result = await alerts_within_radius(store, _QUERY_LAT, _QUERY_LON, 25.0, severity="Severe")
    assert len(result.features) == 1
    assert result.features[0].properties.title == "Severe Alert"


@pytest.mark.mcp
async def test_alerts_within_radius_expiry():
    """An alert with a past expires timestamp is excluded."""
    active_alert = _make_alert(
        "Active", "Severe", _NEAR_POLY,
        url="https://wmo.example.com/active",
    )
    expired_alert = _make_alert(
        "Expired", "Severe", _NEAR_POLY,
        url="https://wmo.example.com/expired",
        expires="2020-01-01T00:00:00+00:00",
    )
    store = _MockAlertsStore([active_alert, expired_alert])

    result = await alerts_within_radius(store, _QUERY_LAT, _QUERY_LON, 25.0)
    assert len(result.features) == 1
    assert result.features[0].properties.title == "Active"


@pytest.mark.mcp
async def test_alerts_within_radius_no_polygon_skipped():
    """An alert with polygon=None is not included even if nominally in range."""
    poly_alert = _make_alert("HasPoly", "Severe", _NEAR_POLY, url="https://wmo.example.com/haspoly")
    no_poly_alert = _make_alert(
        "NoPoly", "Severe", None,
        url="https://wmo.example.com/nopoly",
    )
    store = _MockAlertsStore([poly_alert, no_poly_alert])

    result = await alerts_within_radius(store, _QUERY_LAT, _QUERY_LON, 25.0)
    assert len(result.features) == 1
    assert result.features[0].properties.title == "HasPoly"


@pytest.mark.mcp
async def test_alerts_within_radius_us_point_store_only():
    """US-point lookups are store-only: zone-based alerts arrive with polygons
    resolved at ingest, so no query-time NWS calls exist.  Never raises."""
    us_poly = Polygon([
        (-100.1, 39.9), (-99.9, 39.9), (-99.9, 40.1), (-100.1, 40.1),
        (-100.1, 39.9),
    ])
    wmo_alert = _make_alert(
        "US Alert", "Severe", us_poly,
        url="https://wmo.example.com/us",
    )
    store = _MockAlertsStore([wmo_alert])

    result = await alerts_within_radius(store, 40.0, -100.0, 25.0)
    assert len(result.features) == 1
    assert result.features[0].properties.title == "US Alert"


@pytest.mark.mcp
async def test_alerts_within_radius_multipolygon():
    """A ``MultiPolygon`` alert (e.g. multi-county warning) must not crash the tool.

    Regression test for the live production bug where the radius filter
    accessed ``alert.polygon.exterior.coords`` directly, which only exists
    on Polygon — MultiPolygon raises AttributeError. The fix reprojects
    each constituent Polygon; the alert is included when ANY constituent
    is within radius.
    """
    # Query point: near equator, lon=0 (same as other tests).
    lat, lon = 0.0, 0.0

    # Build a MultiPolygon: one part within 25 km of origin, one ~167 km away.
    near_poly = Polygon([
        (-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1),
        (-0.1, -0.1),
    ])
    far_poly = Polygon([
        (-0.1, 1.5), (0.1, 1.5), (0.1, 1.7), (-0.1, 1.7),
        (-0.1, 1.5),
    ])
    multi = MultiPolygon([near_poly, far_poly])

    alert = AlertEntry(
        source_id="wmo-multipolygon-test",
        event="Severe Thunderstorm Warning",
        description="Multi-county warning (test fixture)",
        severity="Severe",
        effective="2025-01-01T00:00:00+00:00",
        expires="2099-01-01T00:00:00+00:00",
        area_desc="Near area + far area",
        url="https://example.com/wmo-multipolygon",
        polygon=multi,
    )

    store = _MockAlertsStore([alert])

    result = await alerts_within_radius(store, lat, lon, radius_km=25.0)

    assert len(result.features) == 1, (
        f"Expected the MultiPolygon alert to be included (one part in radius), "
        f"got {len(result.features)} features"
    )
    assert result.features[0].properties.title == "Severe Thunderstorm Warning"
    assert result.features[0].properties.severity == "Severe"
    # The geometry should be the original MultiPolygon (mapping() handles both)
    assert result.features[0].geometry is not None
    assert result.features[0].geometry["type"] == "MultiPolygon"
