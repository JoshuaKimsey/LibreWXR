# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for the get_recent_lightning MCP tool function with mocked stores.

Mirrors ``test_mcp_storm_cells.py``: the store is a minimal object exposing
the tool's surface (a ``points`` structured array plus an async
``maybe_reload``), so the pure store-passing function is exercised without
any artifact I/O.
"""

import time
from datetime import datetime

import numpy as np
import pytest

from librewxr.config import settings
from librewxr.data.lightning_store import _POINT_DTYPE
from librewxr.mcp.tools import get_recent_lightning

pytestmark = pytest.mark.mcp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockLightningStore:
    """Minimal async lightning store for testing."""

    def __init__(self, points):
        self._points = points
        self.reload_calls = 0

    async def maybe_reload(self):
        self.reload_calls += 1
        return False

    @property
    def points(self):
        return self._points


def _build_points(rows) -> np.ndarray:
    """Build a structured point array from (time_s, lat, lon, energy, sat)."""
    arr = np.zeros(len(rows), dtype=_POINT_DTYPE)
    for i, (time_s, lat, lon, energy, satellite) in enumerate(rows):
        arr[i] = (time_s, lat, lon, energy, satellite)
    return arr


def _epoch(iso: str) -> int:
    """Parse an ISO-8601 timestamp back to its epoch seconds."""
    return int(datetime.fromisoformat(iso).timestamp())


# ---------------------------------------------------------------------------
# Degraded-empty
# ---------------------------------------------------------------------------


async def test_get_recent_lightning_none_store():
    """None store -> empty list (degraded empty)."""
    result = await get_recent_lightning(None)
    assert result == []


async def test_get_recent_lightning_empty_store():
    """Empty store -> empty list."""
    store = _MockLightningStore(np.empty(0, dtype=_POINT_DTYPE))
    result = await get_recent_lightning(store)
    assert result == []


# ---------------------------------------------------------------------------
# Window filter + per-strike shape
# ---------------------------------------------------------------------------


async def test_get_recent_lightning_window_filter():
    """Only strikes inside the requested window are returned."""
    now = int(time.time())
    points = _build_points([
        (now - 10 * 60, 35.0, -95.0, 1.5e-5, 19),
        (now - 20 * 60, 34.0, -94.0, 2.0e-5, 18),
        (now - 40 * 60, 33.0, -93.0, 3.0e-5, 19),
    ])
    store = _MockLightningStore(points)
    result = await get_recent_lightning(store, minutes=15)

    assert len(result) == 1
    r = result[0]
    assert set(r.keys()) == {"lat", "lon", "utc", "energy", "satellite"}
    assert r["lat"] == pytest.approx(35.0, abs=1e-4)
    assert r["lon"] == pytest.approx(-95.0, abs=1e-4)
    assert r["satellite"] == "goes19"
    assert isinstance(r["energy"], float)
    assert r["energy"] == pytest.approx(1.5e-5, rel=1e-6)
    assert _epoch(r["utc"]) == now - 10 * 60


async def test_get_recent_lightning_satellite_label_fallback():
    """An unknown bird number falls back to ``goes<N>``."""
    now = int(time.time())
    points = _build_points([(now - 30, 35.0, -95.0, 1.0e-5, 20)])
    store = _MockLightningStore(points)
    result = await get_recent_lightning(store, minutes=30)

    assert len(result) == 1
    assert result[0]["satellite"] == "goes20"


# ---------------------------------------------------------------------------
# minutes clamp
# ---------------------------------------------------------------------------


async def test_get_recent_lightning_minutes_clamp(monkeypatch):
    """Requested minutes are clamped down to settings.lightning_max_age."""
    monkeypatch.setattr(settings, "lightning_max_age", 1800)
    now = int(time.time())
    points = _build_points([(now - 45 * 60, 35.0, -95.0, 1.0e-5, 19)])
    store = _MockLightningStore(points)

    # 120 requested but retention is 30 min -> the 45-min-old strike is out.
    result = await get_recent_lightning(store, minutes=120)
    assert result == []


async def test_get_recent_lightning_negative_minutes():
    """A negative window floors at zero -> no strikes."""
    now = int(time.time())
    points = _build_points([(now - 60, 35.0, -95.0, 1.0e-5, 19)])
    store = _MockLightningStore(points)
    result = await get_recent_lightning(store, minutes=-5)
    assert result == []


# ---------------------------------------------------------------------------
# limit
# ---------------------------------------------------------------------------


async def test_get_recent_lightning_limit(monkeypatch):
    """limit keeps the newest strikes, sorted descending by time."""
    monkeypatch.setattr(settings, "lightning_max_age", 1800)
    now = int(time.time())
    # i=0 is newest, i=4 is oldest.
    rows = [(now - i * 60, 35.0, -95.0, float(i), 19) for i in range(5)]
    store = _MockLightningStore(_build_points(rows))
    result = await get_recent_lightning(store, minutes=60, limit=2)

    assert len(result) == 2
    assert [_epoch(r["utc"]) for r in result] == [now, now - 60]


async def test_get_recent_lightning_nonpositive_limit(monkeypatch):
    """limit <= 0 returns an empty list."""
    monkeypatch.setattr(settings, "lightning_max_age", 1800)
    now = int(time.time())
    points = _build_points([(now - 60, 35.0, -95.0, 1.0e-5, 19)])
    store = _MockLightningStore(points)
    assert await get_recent_lightning(store, minutes=30, limit=0) == []


# ---------------------------------------------------------------------------
# Radius filter
# ---------------------------------------------------------------------------


async def test_get_recent_lightning_radius_filter(monkeypatch):
    """Near strike kept, far strike dropped; filter skipped without lat/lon."""
    monkeypatch.setattr(settings, "lightning_max_age", 1800)
    now = int(time.time())
    near = (now - 30, 35.0, -95.0, 1.0e-5, 19)
    # Same longitude, ~60 km north at lat 35 (111 km/deg).
    far = (now - 60, 35.0 + 60.0 / 111.0, -95.0, 2.0e-5, 18)
    store = _MockLightningStore(_build_points([near, far]))

    within = await get_recent_lightning(
        store, lat=35.0, lon=-95.0, radius_km=50.0, minutes=30,
    )
    assert len(within) == 1
    assert within[0]["satellite"] == "goes19"

    # Radius alone (no lat/lon) does no spatial filtering -> both return.
    all_strikes = await get_recent_lightning(store, radius_km=50.0, minutes=30)
    assert len(all_strikes) == 2


async def test_get_recent_lightning_lone_lat_skips_filter(monkeypatch):
    """A lone lat (no lon) skips the radius filter."""
    monkeypatch.setattr(settings, "lightning_max_age", 1800)
    now = int(time.time())
    far = (now - 60, 80.0, 10.0, 2.0e-5, 18)
    store = _MockLightningStore(_build_points([far]))
    result = await get_recent_lightning(store, lat=35.0, radius_km=50.0)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# maybe_reload
# ---------------------------------------------------------------------------


async def test_get_recent_lightning_reload_once():
    """maybe_reload is awaited exactly once per call."""
    store = _MockLightningStore(np.empty(0, dtype=_POINT_DTYPE))
    await get_recent_lightning(store)
    assert store.reload_calls == 1
    await get_recent_lightning(store)
    assert store.reload_calls == 2


# ---------------------------------------------------------------------------
# bbox filter
# ---------------------------------------------------------------------------


async def test_get_recent_lightning_bbox_filter():
    """bbox keeps in-box strikes and drops those outside."""
    now = int(time.time())
    inside = (now - 30, 35.0, -95.0, 1.0e-5, 19)
    outside = (now - 60, 10.0, 10.0, 2.0e-5, 18)
    store = _MockLightningStore(_build_points([inside, outside]))

    result = await get_recent_lightning(
        store, bbox=(-100.0, 30.0, -90.0, 40.0), minutes=30,
    )
    assert len(result) == 1
    assert result[0]["satellite"] == "goes19"


async def test_get_recent_lightning_bbox_without_latlon():
    """bbox applies on its own (no lat/lon): both in-box strikes return."""
    now = int(time.time())
    rows = [
        (now - 30, 35.0, -95.0, 1.0e-5, 19),
        (now - 60, 36.0, -94.0, 2.0e-5, 18),
        (now - 90, 60.0, 10.0, 3.0e-5, 19),
    ]
    store = _MockLightningStore(_build_points(rows))
    result = await get_recent_lightning(
        store, bbox=(-100.0, 30.0, -90.0, 40.0), minutes=30,
    )
    assert len(result) == 2


async def test_get_recent_lightning_point_wins_over_bbox():
    """lat+lon radius filter wins; bbox is silently ignored."""
    now = int(time.time())
    near = (now - 30, 35.0, -95.0, 1.0e-5, 19)
    # ~60 km north -- inside the huge bbox but outside the 50 km radius.
    far = (now - 60, 35.0 + 60.0 / 111.0, -95.0, 2.0e-5, 18)
    store = _MockLightningStore(_build_points([near, far]))

    result = await get_recent_lightning(
        store,
        lat=35.0,
        lon=-95.0,
        radius_km=50.0,
        bbox=(-120.0, 10.0, -70.0, 60.0),
        minutes=30,
    )
    assert len(result) == 1
    assert result[0]["satellite"] == "goes19"


async def test_get_recent_lightning_degenerate_bbox_empty():
    """An inverted (degenerate) box naturally selects nothing."""
    now = int(time.time())
    point = (now - 30, 35.0, -95.0, 1.0e-5, 19)
    store = _MockLightningStore(_build_points([point]))
    result = await get_recent_lightning(
        store, bbox=(-90.0, 30.0, -100.0, 40.0), minutes=30,
    )
    assert result == []


# ---------------------------------------------------------------------------
# limit=None (uncapped) / limit=0
# ---------------------------------------------------------------------------


async def test_get_recent_lightning_limit_none_uncapped(monkeypatch):
    """limit=None returns every selected strike (no cap)."""
    monkeypatch.setattr(settings, "lightning_max_age", 1800)
    now = int(time.time())
    rows = [(now - i * 60, 35.0, -95.0, float(i), 19) for i in range(5)]
    store = _MockLightningStore(_build_points(rows))
    result = await get_recent_lightning(store, minutes=60, limit=None)

    assert len(result) == 5
    assert [_epoch(r["utc"]) for r in result] == [now - i * 60 for i in range(5)]


async def test_get_recent_lightning_zero_limit_multiple():
    """limit=0 returns an empty list even when strikes are selected."""
    now = int(time.time())
    rows = [(now - i * 60, 35.0, -95.0, float(i), 19) for i in range(3)]
    store = _MockLightningStore(_build_points(rows))
    assert await get_recent_lightning(store, minutes=30, limit=0) == []
