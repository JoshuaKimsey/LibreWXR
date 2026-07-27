# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for MCP alerts query: alerts_within_radius with radius/severity/expiry
filtering and NWS enrichment."""

import pytest
from shapely.geometry import Polygon

from librewxr.api.models import AlertProperties, AlertsResponse, GeoJSONFeature
from librewxr.data.alerts_store import AlertEntry
from librewxr.mcp.alerts_query import alerts_within_radius


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockAlertsStore:
    """Minimal object satisfying the ``alerts``-property interface."""
    def __init__(self, alerts):
        self.alerts = alerts


# Use a non-US query point (lat=0, lon=0) to avoid real NWS API calls in
# tests that do not explicitly test NWS enrichment.  NWS enrichment only
# triggers when lat in [20, 55] and lon in [-130, -60].
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
async def test_alerts_within_radius_nws_enrichment(monkeypatch):
    """US-point enriches WMO alerts with NWS point-alert features, deduped by URI."""
    us_poly = Polygon([
        (-100.1, 39.9), (-99.9, 39.9), (-99.9, 40.1), (-100.1, 40.1),
        (-100.1, 39.9),
    ])
    wmo_alert = _make_alert(
        "WMO Alert", "Severe", us_poly,
        url="https://wmo.example.com/wmo",
    )
    store = _MockAlertsStore([wmo_alert])

    nws_feature = GeoJSONFeature(
        type="Feature",
        properties=AlertProperties(
            title="NWS Alert",
            severity="Severe",
            time=1_700_000_000,
            expires=1_700_086_400,
            description="NWS test",
            regions=["NWS Area"],
            uri="https://nws.example.com/nws",
        ),
        geometry={"type": "Polygon", "coordinates": [[[-101, 39], [-99, 39], [-99, 41], [-101, 41], [-101, 39]]]},
    )

    async def mock_nws_fetch(_lat, _lon):
        return [nws_feature]

    monkeypatch.setattr("librewxr.mcp.alerts_query._fetch_nws_point_alerts", mock_nws_fetch)

    # Use a US point so the enrichment path is exercised
    result = await alerts_within_radius(store, 40.0, -100.0, 25.0)
    # One WMO + one NWS = 2 features
    assert len(result.features) == 2
    uris = {f.properties.uri for f in result.features}
    assert "https://wmo.example.com/wmo" in uris
    assert "https://nws.example.com/nws" in uris


@pytest.mark.mcp
async def test_alerts_within_radius_nws_failure_degrades(monkeypatch):
    """When the NWS fetch raises, the function degrades gracefully (does not raise)."""
    us_poly = Polygon([
        (-100.1, 39.9), (-99.9, 39.9), (-99.9, 40.1), (-100.1, 40.1),
        (-100.1, 39.9),
    ])
    wmo_alert = _make_alert(
        "WMO Only", "Severe", us_poly,
        url="https://wmo.example.com/wmo-only",
    )
    store = _MockAlertsStore([wmo_alert])

    async def mock_nws_failure(_lat, _lon):
        raise RuntimeError("NWS API unavailable")

    monkeypatch.setattr("librewxr.mcp.alerts_query._fetch_nws_point_alerts", mock_nws_failure)

    # Use a US point so the enrichment path is exercised and raises
    result = await alerts_within_radius(store, 40.0, -100.0, 25.0)
    assert len(result.features) == 1
    assert result.features[0].properties.title == "WMO Only"
