# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for MCP alerts query: alerts_within_radius with radius/severity/expiry
filtering against the merged WMO + NWS store."""

import pytest
from shapely.geometry import MultiPolygon, Polygon, shape

from librewxr.api.models import AlertsResponse
from librewxr.data.alerts_artifact import (
    AlertsArtifactStore,
    write_alerts_artifacts,
)
from librewxr.data.alerts_store import AlertEntry, AlertsStore
from librewxr.mcp.alerts_query import alerts_within_radius


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockAlertsStore:
    """Minimal object satisfying the artifact-store query interface.

    ``alerts_near_candidates`` returns the whole list: the conservative
    superset the artifact store promises (the query stays the geometric
    authority), so this doubles as the brute-force reference.
    """
    def __init__(self, alerts):
        self.alerts = alerts

    def alerts_near_candidates(self, lat, lon, radius_km):
        return self.alerts

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


@pytest.mark.mcp
async def test_invalid_polygon_repaired_not_raised():
    """A self-intersecting CAP polygon must not raise out of the tool.

    Regression: intersects/contains on invalid geometry raise
    ``GEOSException: TopologyException``.  The query repairs the alert polygon
    first, so the tool still returns the matching alert with valid geometry.
    """
    lat, lon = 0.0, 0.0
    # Bow-tie around the query point: the two lobes meet at (0, 0), so the
    # repaired MultiPolygon still touches the query circle.
    bow_tie = Polygon([
        (-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2), (0.2, 0.2), (-0.2, -0.2),
    ])
    assert bow_tie.is_valid is False  # fixture sanity: this is the bad input

    store = _MockAlertsStore([_make_alert("Bow Tie", "Severe", bow_tie)])
    result = await alerts_within_radius(store, lat, lon, radius_km=25.0)

    assert len(result.features) == 1
    feature = result.features[0]
    assert feature.properties.title == "Bow Tie"
    assert feature.geometry is not None
    assert shape(feature.geometry).is_valid is True


# ---------------------------------------------------------------------------
# Artifact-store parity: conservative candidates vs. a brute-force list
# ---------------------------------------------------------------------------

def _box(x: float, y: float, size: float = 0.2) -> Polygon:
    return Polygon([
        (x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y),
    ])


def _parity_entries() -> list[AlertEntry]:
    """A fixture with near/far, severity, expiry, polar and dateline entries."""
    return [
        _make_alert(
            "New York Warning", "Severe", _box(-74.1, 40.6),
            url="https://wmo.example.com/ny",
        ),
        _make_alert(
            "Berlin Warning", "Severe", _box(13.3, 52.4),
            url="https://wmo.example.com/berlin",
        ),
        _make_alert(
            "Multi Area Warning", "Severe",
            MultiPolygon([_box(-74.2, 40.5), _box(20.0, 40.0)]),
            url="https://wmo.example.com/multi",
        ),
        _make_alert(
            "Far Pacific Advisory", "Minor", _box(-160.0, -20.0),
            url="https://wmo.example.com/pacific",
        ),
        _make_alert(
            "Expired New York", "Severe", _box(-74.0, 40.7),
            url="https://wmo.example.com/expired",
            expires="2020-01-01T00:00:00+00:00",
        ),
        _make_alert(
            "Minor New York", "Minor", _box(-74.05, 40.65),
            url="https://wmo.example.com/minor-ny",
        ),
        _make_alert(
            "Polar Warning", "Extreme", _box(-0.1, 89.35),
            url="https://wmo.example.com/polar",
        ),
        _make_alert(
            "Dateline Warning", "Severe",
            MultiPolygon([_box(179.8, 0.0), _box(-179.95, 0.0)]),
            url="https://wmo.example.com/dateline",
        ),
    ]


def _titles(response: AlertsResponse) -> list[str]:
    return [feature.properties.title for feature in response.features]


@pytest.mark.mcp
async def test_radius_query_parity_with_bruteforce(tmp_path):
    """Artifact candidates + the shared query logic == a brute-force list.

    The artifact store only narrows the candidate set (SQL bbox prefilter, or
    a full scan where the envelope cannot be trusted); every filter, repair
    and geometric test must produce identical output either way.
    """
    store = AlertsStore()
    store.replace_all(_parity_entries())
    assert write_alerts_artifacts(store, tmp_path) is True

    artifact = AlertsArtifactStore(tmp_path)
    assert artifact.maybe_reload() is True
    brute = _MockAlertsStore(artifact.alerts())
    assert brute.alerts, "fixture must survive the artifact round-trip"

    queries = [
        ("mid-latitude", 40.7, -74.0, 25.0, None),
        ("europe", 52.5, 13.4, 50.0, None),
        ("severity", 40.7, -74.0, 25.0, "Minor"),
        ("polar", 89.0, 0.0, 100.0, None),
        ("antimeridian", 0.0, 179.9, 5000.0, None),
    ]
    for label, lat, lon, radius_km, severity in queries:
        expected = await alerts_within_radius(
            brute, lat, lon, radius_km, severity,
        )
        actual = await alerts_within_radius(
            artifact, lat, lon, radius_km, severity,
        )
        assert actual.model_dump() == expected.model_dump(), label

    # The prefilter is genuinely narrowing on a mid-latitude query (so the
    # parity above is not just two full scans).
    assert len(artifact.alerts_near_candidates(40.7, -74.0, 25.0)) < artifact.count

    mid_lat = await alerts_within_radius(artifact, 40.7, -74.0, 25.0)
    assert "New York Warning" in _titles(mid_lat)
    assert "Expired New York" not in _titles(mid_lat)

    severity = await alerts_within_radius(artifact, 40.7, -74.0, 25.0, "Minor")
    assert _titles(severity) == ["Minor New York"]

    # Polar (< 85 deg latitude trust floor) -> the store full-scans, and the
    # query still filters correctly.
    polar = await alerts_within_radius(artifact, 89.0, 0.0, 100.0)
    assert _titles(polar) == ["Polar Warning"]

    # Antimeridian-crossing envelope -> full scan fallback, still correct.
    dateline = await alerts_within_radius(artifact, 0.0, 179.9, 5000.0)
    assert "Dateline Warning" in _titles(dateline)
    assert "New York Warning" not in _titles(dateline)


@pytest.mark.mcp
async def test_radius_query_cold_artifact_is_empty(tmp_path):
    """A cold (missing) artifact degrades to an empty collection, no raise."""
    artifact = AlertsArtifactStore(tmp_path / "missing")
    assert artifact.maybe_reload() is False

    result = await alerts_within_radius(artifact, 40.7, -74.0, 25.0)
    assert result.type == "FeatureCollection"
    assert result.features == []
