# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.api

from librewxr.api import routes
from librewxr.api.tile_history import router as tile_history_router
from librewxr.config import settings
from librewxr.tiles.request_tracker import TileRequestTracker


def _tracker() -> TileRequestTracker:
    """A real tracker with two captured snapshots."""
    tracker = TileRequestTracker(min_zoom=7)
    for _ in range(4):
        tracker.record(7, 12, 34)
    tracker.record_request(8, 1, 2, cache_hit=True)
    tracker.record_latency(1_500_000, 800_000, 400_000)
    tracker.capture_snapshot()
    tracker.record(7, 12, 34)
    tracker.capture_snapshot()
    return tracker


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(tile_history_router)
    return TestClient(app)


def test_tile_history_endpoint_serves_snapshots(monkeypatch):
    tracker = _tracker()
    monkeypatch.setattr(routes, "tile_request_tracker", tracker)

    with _client() as client:
        resp = client.get("/health/tile-history")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert body["enabled"] is True
    assert body["tracking_enabled"] is True
    assert body["interval_s"] == settings.tile_history_interval_s
    assert body["max_entries"] == settings.tile_history_max_entries
    # Oldest first, byte-for-byte the tracker's ring through JSON.
    assert body["snapshots"] == tracker.history()
    ts = [snap["t"] for snap in body["snapshots"]]
    assert ts == sorted(ts)
    assert len(body["snapshots"]) == 2
    # Full per-snapshot shape.
    for snap in body["snapshots"]:
        for key in (
            "t", "total_requests", "cache_hits", "cache_misses", "hit_rate",
            "fast_path_total", "tracked_tiles", "hot_tiles",
            "avg_request_ms", "avg_compute_ms", "avg_present_ms",
            "by_zoom", "top",
        ):
            assert key in snap
        assert all(isinstance(z, str) for z in snap["by_zoom"])


def test_tile_history_reports_tracking_disabled(monkeypatch):
    monkeypatch.setattr(routes, "tile_request_tracker", None)

    with _client() as client:
        resp = client.get("/health/tile-history")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "enabled": True,
        "tracking_enabled": False,
        "interval_s": settings.tile_history_interval_s,
        "max_entries": settings.tile_history_max_entries,
        "snapshots": [],
    }
