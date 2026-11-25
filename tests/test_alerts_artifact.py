# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for the standalone alerts artifact (writer + lazy reader)."""

import json
import math
import os
import time
from pathlib import Path

import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

from librewxr.data.alerts_artifact import (
    ARTIFACT_NAME,
    AlertsArtifactStore,
    write_alerts_artifacts,
)
from librewxr.data.alerts_response import build_alerts_response
from librewxr.data.alerts_store import AlertEntry, AlertsStore, entry_to_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_entry(
    source_id: str,
    *,
    url: str | None = None,
    polygon=None,
    event: str = "Severe Thunderstorm Warning",
    description: str = "A description",
    severity: str = "Severe",
    effective: str = "2026-05-07T00:00:00+00:00",
    expires: str = "2099-05-07T06:00:00+00:00",
    area_desc: str = "Test area",
) -> AlertEntry:
    return AlertEntry(
        source_id=source_id,
        event=event,
        description=description,
        severity=severity,
        effective=effective,
        expires=expires,
        area_desc=area_desc,
        url=url if url is not None else f"https://example.test/{source_id}",
        polygon=polygon,
    )


def square(x: float, y: float, size: float = 0.5) -> Polygon:
    return Polygon(
        [
            (x, y),
            (x + size, y),
            (x + size, y + size),
            (x, y + size),
        ]
    )


def mixed_alerts() -> list[AlertEntry]:
    """Polygon, MultiPolygon, polygon-less and unicode entries."""
    poly = square(-74.5, 40.5)
    multipoly = MultiPolygon([square(10.0, 45.0, 0.4), square(12.0, 46.0, 0.4)])
    return [
        make_entry("poly-1", polygon=poly, area_desc="New York County"),
        make_entry(
            "multi-1",
            polygon=multipoly,
            event="Wetterwarnung",
            description="Sturmboen mit Orkanstarke \u2014 \u00e4\u00f6\u00fc\u00df",
            area_desc="Kreis Garmisch",
        ),
        make_entry("nogeom-1", polygon=None, event="Coastal Flood Advisory"),
        make_entry(
            "poly-2",
            polygon=square(-73.0, 41.0),
            url="https://example.test/shared-multi-area",
            area_desc="Region A",
        ),
        make_entry(
            "poly-3",
            polygon=square(-72.0, 42.0),
            url="https://example.test/shared-multi-area",
            area_desc="Region B",
        ),
    ]


def assert_entry_equal(actual: AlertEntry, expected: AlertEntry) -> None:
    assert actual.source_id == expected.source_id
    assert actual.event == expected.event
    assert actual.description == expected.description
    assert actual.severity == expected.severity
    assert actual.effective == expected.effective
    assert actual.expires == expected.expires
    assert actual.area_desc == expected.area_desc
    assert actual.url == expected.url
    if expected.polygon is None:
        assert actual.polygon is None
    else:
        assert actual.polygon is not None
        assert actual.polygon.geom_type == expected.polygon.geom_type
        assert actual.polygon.equals(expected.polygon)


def km_plane_candidates(
    alerts: list[AlertEntry], lat: float, lon: float, radius_km: float,
) -> set[str]:
    """Brute-force radius candidate set, mirroring mcp/alerts_query.py."""
    cos_lat = math.cos(math.radians(lat))
    deg_to_km_lat = 111.0
    deg_to_km_lon = 111.0 * cos_lat
    circle = Point(0.0, 0.0).buffer(radius_km)
    query_point = Point(lon, lat)
    found: set[str] = set()
    for alert in alerts:
        polygon = alert.polygon
        if polygon is None:
            continue
        if isinstance(polygon, MultiPolygon):
            reprojected = MultiPolygon(
                [
                    Polygon(
                        [
                            ((px - lon) * deg_to_km_lon, (py - lat) * deg_to_km_lat)
                            for px, py in sub.exterior.coords
                        ]
                    )
                    for sub in polygon.geoms
                ]
            )
        else:
            reprojected = Polygon(
                [
                    ((px - lon) * deg_to_km_lon, (py - lat) * deg_to_km_lat)
                    for px, py in polygon.exterior.coords
                ]
            )
        if circle.intersects(reprojected) or polygon.contains(query_point):
            found.add(alert.source_id)
    return found


@pytest.fixture
def populated_store() -> AlertsStore:
    store = AlertsStore()
    store.replace_all(mixed_alerts())
    return store


# ---------------------------------------------------------------------------
# 1. Write / read round-trip parity
# ---------------------------------------------------------------------------

def test_round_trip_parity(tmp_path: Path, populated_store: AlertsStore):
    original = populated_store.alerts
    assert write_alerts_artifacts(populated_store, tmp_path) is True

    artifact = AlertsArtifactStore(tmp_path)
    assert artifact.maybe_reload() is True

    assert artifact.count == len(original)
    assert artifact.last_updated == pytest.approx(populated_store.last_updated)
    assert artifact.fetch_success is True

    loaded = artifact.alerts()
    assert len(loaded) == len(original)
    for actual, expected in zip(loaded, original):
        assert_entry_equal(actual, expected)


# ---------------------------------------------------------------------------
# 2. Empty store + fetch_success=False
# ---------------------------------------------------------------------------

def test_empty_store_and_fetch_failure(tmp_path: Path):
    store = AlertsStore()
    store.replace_all([], fetch_success=False)

    assert write_alerts_artifacts(store, tmp_path) is True

    artifact = AlertsArtifactStore(tmp_path)
    assert artifact.maybe_reload() is True
    assert artifact.count == 0
    assert artifact.fetch_success is False
    assert artifact.last_updated == pytest.approx(store.last_updated)
    assert artifact.alerts() == []
    assert artifact.response_bytes() == (
        build_alerts_response([]).model_dump_json().encode()
    )


# ---------------------------------------------------------------------------
# 3. Packaged response byte equality
# ---------------------------------------------------------------------------

def test_response_bytes_match_direct_build(tmp_path: Path, populated_store: AlertsStore):
    assert write_alerts_artifacts(populated_store, tmp_path) is True
    artifact = AlertsArtifactStore(tmp_path)
    assert artifact.maybe_reload() is True

    expected = build_alerts_response(populated_store.alerts).model_dump_json()
    assert artifact.response_bytes() == expected.encode()

    # The payload is real JSON, not just matching bytes.
    payload = json.loads(artifact.response_bytes())
    assert payload["type"] == "FeatureCollection"
    # Five entries, four distinct urls (two entries share one).
    assert len(payload["features"]) == 4


# ---------------------------------------------------------------------------
# 4. Atomicity and one-shot reload
# ---------------------------------------------------------------------------

def test_atomic_write_and_single_reload(tmp_path: Path, populated_store: AlertsStore):
    assert write_alerts_artifacts(populated_store, tmp_path) is True
    alerts_dir = tmp_path / "alerts"
    assert (tmp_path / ARTIFACT_NAME).exists()
    names = sorted(p.name for p in alerts_dir.iterdir())
    assert names == ["current.db"]
    assert [p for p in alerts_dir.iterdir() if p.name.endswith(".tmp")] == []

    artifact = AlertsArtifactStore(tmp_path)
    assert artifact.maybe_reload() is True
    assert artifact.maybe_reload() is False

    # Rewrite with a different store: replaced artifact, reload fires once.
    smaller = AlertsStore()
    smaller.replace_all([make_entry("poly-1", polygon=square(-74.5, 40.5))])
    assert write_alerts_artifacts(smaller, tmp_path) is True
    assert sorted(p.name for p in alerts_dir.iterdir()) == ["current.db"]

    assert artifact.maybe_reload() is True
    assert artifact.count == 1
    assert artifact.maybe_reload() is False


# ---------------------------------------------------------------------------
# 5. mtime gating and corrupt artifact
# ---------------------------------------------------------------------------

def test_maybe_reload_gating_and_corruption(tmp_path: Path, populated_store: AlertsStore):
    assert write_alerts_artifacts(populated_store, tmp_path) is True
    artifact = AlertsArtifactStore(tmp_path)

    assert artifact.maybe_reload() is True
    good_count = artifact.count
    good_last_updated = artifact.last_updated
    good_success = artifact.fetch_success
    assert good_count > 0

    # Unchanged stat: no reopen, no reload.
    assert artifact.maybe_reload() is False
    assert artifact.maybe_reload() is False

    # Corrupt the file and bump mtime so the stamp changes.
    path = tmp_path / ARTIFACT_NAME
    path.write_bytes(b"this is not a sqlite database at all")
    future = time.time() + 60
    os.utime(path, (future, future))

    assert artifact.maybe_reload() is False
    # Last good scalars retained, old stamp retained so the next call retries.
    assert artifact.count == good_count
    assert artifact.last_updated == good_last_updated
    assert artifact.fetch_success is good_success
    assert artifact.maybe_reload() is False

    # A fresh good write is picked up afterwards.
    assert write_alerts_artifacts(populated_store, tmp_path) is True
    assert artifact.maybe_reload() is True
    assert artifact.count == good_count


# ---------------------------------------------------------------------------
# 6. Missing artifact
# ---------------------------------------------------------------------------

def test_missing_artifact_degrades(tmp_path: Path):
    artifact = AlertsArtifactStore(tmp_path / "does-not-exist")
    assert artifact.count == 0
    assert artifact.last_updated == 0.0
    assert artifact.fetch_success is False
    assert artifact.alerts() == []
    assert artifact.response_bytes() == b""

    assert artifact.maybe_reload() is False
    assert artifact.count == 0
    assert artifact.last_updated == 0.0
    assert artifact.fetch_success is False


# ---------------------------------------------------------------------------
# 7. Spatial queries
# ---------------------------------------------------------------------------

def test_spatial_queries_match_bruteforce(tmp_path: Path):
    entries = [
        make_entry("a", polygon=square(-74.5, 40.5)),
        make_entry("b", polygon=square(-73.0, 41.0)),
        make_entry("c", polygon=MultiPolygon([square(10.0, 45.0), square(30.0, 55.0)])),
        make_entry("nogeom", polygon=None),
    ]
    store = AlertsStore()
    store.replace_all(entries)
    assert write_alerts_artifacts(store, tmp_path) is True

    artifact = AlertsArtifactStore(tmp_path)
    assert artifact.maybe_reload() is True
    loaded = artifact.alerts()
    assert [e.source_id for e in loaded] == ["a", "b", "c", "nogeom"]

    # bbox queries
    for query in [
        (-75.0, 40.0, -72.5, 41.5),
        (9.0, 44.0, 11.0, 46.0),
        (-180.0, -90.0, 180.0, 90.0),
        (0.0, 0.0, 1.0, 1.0),
    ]:
        expected = [
            a.source_id
            for a in loaded
            if a.polygon is not None and box(*query).intersects(a.polygon)
        ]
        got = [a.source_id for a in artifact.alerts_in_bbox(*query)]
        assert got == expected, query

    # point queries
    for lon, lat in [(-74.4, 40.6), (-73.0, 41.0), (10.5, 45.5), (0.0, 0.0)]:
        expected = [
            a.source_id
            for a in loaded
            if a.polygon is not None and Point(lon, lat).intersects(a.polygon)
        ]
        got = [a.source_id for a in artifact.alerts_at_point(lon, lat)]
        assert got == expected, (lon, lat)

    # polygon-less entries never come back from spatial queries
    assert "nogeom" not in {a.source_id for a in artifact.alerts_in_bbox(-180, -90, 180, 90)}
    assert "nogeom" not in {a.source_id for a in artifact.alerts_at_point(-74.4, 40.6)}


# ---------------------------------------------------------------------------
# 8. Radius prefilter conservativeness
# ---------------------------------------------------------------------------

def test_near_candidates_conservative(tmp_path: Path):
    entries = [
        make_entry("near", polygon=square(13.40, 52.50, 0.1)),
        make_entry("edge", polygon=square(13.60, 52.50, 0.05)),
        make_entry("far", polygon=square(14.50, 52.50, 0.1)),
        make_entry("multi", polygon=MultiPolygon([
            square(13.38, 52.45, 0.05), square(20.0, 40.0, 0.1),
        ])),
        make_entry("nogeom", polygon=None),
    ]
    store = AlertsStore()
    store.replace_all(entries)
    assert write_alerts_artifacts(store, tmp_path) is True

    artifact = AlertsArtifactStore(tmp_path)
    assert artifact.maybe_reload() is True
    everything = artifact.alerts()

    # Mid-latitude query: every brute-force candidate must survive the prefilter.
    lat, lon, radius_km = 52.52, 13.40, 30.0
    expected = km_plane_candidates(everything, lat, lon, radius_km)
    assert expected, "fixture should put at least one alert in range"
    candidates = artifact.alerts_near_candidates(lat, lon, radius_km)
    got = {a.source_id for a in candidates}
    assert expected <= got
    assert "nogeom" not in got
    # Order preserved against the full scan.
    order = [a.source_id for a in everything if a.source_id in got]
    assert [a.source_id for a in candidates] == order

    # Polar query: full scan.
    polar = artifact.alerts_near_candidates(89.0, 0.0, 100.0)
    assert [a.source_id for a in polar] == [a.source_id for a in everything]

    # Antimeridian-crossing query: full scan.
    crossing = artifact.alerts_near_candidates(0.0, 179.9, 5000.0)
    assert [a.source_id for a in crossing] == [a.source_id for a in everything]

    # Negative radius: full scan (conservative, caller decides).
    negative = artifact.alerts_near_candidates(lat, lon, -1.0)
    assert [a.source_id for a in negative] == [a.source_id for a in everything]


# ---------------------------------------------------------------------------
# 9. entry_to_dict parity with the store snapshot
# ---------------------------------------------------------------------------

def test_entry_to_dict_matches_getstate():
    entry = make_entry("parity-1", polygon=square(-74.5, 40.5), area_desc="Area")
    store = AlertsStore()
    store.replace_all([entry])

    state = store.__getstate__()
    assert state["alerts"] == [entry_to_dict(store.alerts[0])]
    assert entry_to_dict(store.alerts[0]) == state["alerts"][0]

    # Polygon-less entries serialize with a None polygon.
    plain = AlertsStore()
    plain.replace_all([make_entry("parity-2", polygon=None)])
    assert plain.__getstate__()["alerts"][0]["polygon"] is None
    assert entry_to_dict(plain.alerts[0])["polygon"] is None
