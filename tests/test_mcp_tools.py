# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for MCP tool functions: get_precip_nowcast and get_active_alerts
with mocked stores."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from librewxr.api.models import AlertProperties, AlertsResponse, GeoJSONFeature
from librewxr.config import settings
from librewxr.data.alerts_artifact import (
    AlertsArtifactStore,
    write_alerts_artifacts,
)
from librewxr.data.alerts_store import AlertEntry, AlertsStore
from librewxr.data.nowcast import NowcastFrame
from librewxr.data.regions import REGIONS, RegionDef
from librewxr.mcp.sampling import dbz_to_rate_mmh
from librewxr.mcp.tools import get_active_alerts, get_precip_nowcast


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXED_NOW = 1_700_000_000
TS_10 = FIXED_NOW + 600   # +10 minutes
TS_20 = FIXED_NOW + 1200  # +20 minutes

# A synthetic latlon region that contains (40, -95):
#   width  = (east - west) / 1.0 = 10
#   height = (north - south) / 1.0 = 10
#   (lat=40, lon=-95)
#   col_f = (-95 - (-100)) / 1.0 = 5.0  -> col=5
#   row_f = (45 - 40) / 1.0 = 5.0      -> row=5
_TEST_REGION = RegionDef(
    name="TEST_US",
    west=-100.0, east=-90.0, south=35.0, north=45.0,
    pixel_size=1.0, group="TEST",
)

# Pre-build a 10x10 frame with a known pixel at (5,5) = 64 (0.0 dBZ).
_FRAME_DATA = np.zeros((10, 10), dtype=np.uint8)
_FRAME_DATA[5, 5] = 64

# A synthetic latlon region registered as the RRQPE region name, so the
# display-calibration undo path is exercised.  Same geometry as TEST_US.
_RRQPE_REGION = RegionDef(
    name="RRQPE",
    west=-100.0, east=-90.0, south=35.0, north=45.0,
    pixel_size=1.0, group="TEST",
)

# 10x10 frame with pixel (5,5) = 100 -> stored dBZ = 100/2 - 32 = 18.0.
_FRAME_DATA_HIGH = np.zeros((10, 10), dtype=np.uint8)
_FRAME_DATA_HIGH[5, 5] = 100


# ---------------------------------------------------------------------------
# Mock stores
# ---------------------------------------------------------------------------

class MockNowcastStore:
    """Minimal async nowcast store returning pre-built frames."""

    def __init__(self, timestamps, frames):
        self._timestamps = list(timestamps)
        self._frames = {ts: f for ts, f in zip(timestamps, frames)}

    async def get_timestamps(self):
        return list(self._timestamps)

    async def get_frame(self, ts):
        frame = self._frames.get(ts)
        if frame is None:
            return None, 0.0
        return frame, frame.blend_weight


class MockNWPSource:
    def has_data_at(self, ts):
        return True


class MockNWPChain:
    """Minimal NWP chain that returns a configurable single-pixel array."""

    def __init__(self, pixel_value=80):
        self.sources = [MockNWPSource()]
        self._pixel_value = pixel_value

    def sample(self, lat, lon, timestamp=None, bilinear=False):
        return np.array([self._pixel_value], dtype=np.uint8)


class MockAlertsStore:
    def __init__(self, alerts):
        self.alerts = alerts

    def alerts_near_candidates(self, lat, lon, radius_km):
        return self.alerts


# ---------------------------------------------------------------------------
# get_precip_nowcast
# ---------------------------------------------------------------------------

@pytest.mark.mcp
async def test_get_precip_nowcast_none_nowcast_store():
    """No nowcast store -> empty list."""
    result = await get_precip_nowcast(None, None, ["TEST_US"], 40.0, -95.0, 60)
    assert result == []


@pytest.mark.mcp
async def test_get_precip_nowcast_with_mock_store(monkeypatch):
    """Two future frames within horizon -> both returned with expected keys."""
    frame_10 = NowcastFrame(timestamp=TS_10, regions={"TEST_US": _FRAME_DATA}, blend_weight=0.7)
    frame_20 = NowcastFrame(timestamp=TS_20, regions={"TEST_US": _FRAME_DATA}, blend_weight=0.3)

    store = MockNowcastStore([TS_10, TS_20], [frame_10, frame_20])
    monkeypatch.setitem(REGIONS, "TEST_US", _TEST_REGION)

    # Pin time.time so the minutes_offset in get_precip_nowcast uses FIXED_NOW
    monkeypatch.setattr("librewxr.mcp.tools.time.time", lambda: FIXED_NOW)

    def _mock_sample(_name, _lat, _lon):
        return np.array([True])

    monkeypatch.setattr("librewxr.mcp.sampling.sample_coverage", _mock_sample)

    result = await get_precip_nowcast(None, store, ["TEST_US"], 40.0, -95.0, 30)

    assert len(result) == 2
    # First frame = +10 min
    r0 = result[0]
    assert set(r0.keys()) == {"time", "minutes_offset", "dbz", "rate_mmh",
                               "source", "blend_weight", "coverage"}
    assert r0["source"] == "radar"
    assert r0["coverage"] == "in_range"
    assert r0["minutes_offset"] == 10
    assert r0["dbz"] == 0.0
    assert r0["blend_weight"] == 0.7

    # Second frame = +20 min
    r1 = result[1]
    assert r1["source"] == "radar"
    assert r1["coverage"] == "in_range"
    assert r1["minutes_offset"] == 20
    assert r1["dbz"] == 0.0
    assert r1["blend_weight"] == 0.3


@pytest.mark.mcp
async def test_get_precip_nowcast_minutes_filter(monkeypatch):
    """Only frames with minutes_offset <= minutes are returned."""
    frame_10 = NowcastFrame(timestamp=TS_10, regions={"TEST_US": _FRAME_DATA}, blend_weight=0.7)
    frame_20 = NowcastFrame(timestamp=TS_20, regions={"TEST_US": _FRAME_DATA}, blend_weight=0.3)

    store = MockNowcastStore([TS_10, TS_20], [frame_10, frame_20])
    monkeypatch.setitem(REGIONS, "TEST_US", _TEST_REGION)

    monkeypatch.setattr("librewxr.mcp.tools.time.time", lambda: FIXED_NOW)

    def _mock_sample(_name, _lat, _lon):
        return np.array([True])

    monkeypatch.setattr("librewxr.mcp.sampling.sample_coverage", _mock_sample)

    result = await get_precip_nowcast(None, store, ["TEST_US"], 40.0, -95.0, 15)

    assert len(result) == 1
    assert result[0]["minutes_offset"] == 10


@pytest.mark.mcp
async def test_get_precip_nowcast_nwp_fallback(monkeypatch):
    """When the nowcast frame does not cover the point, NWP chain is consulted."""
    # Frame's regions dict does NOT contain "TEST_US" -> sample_nowcast_at_point
    # returns out_of_range -> triggers NWP fallback.
    frame = NowcastFrame(timestamp=TS_10, regions={"OTHER": _FRAME_DATA}, blend_weight=0.0)
    store = MockNowcastStore([TS_10], [frame])
    nwp = MockNWPChain(pixel_value=80)  # 80 -> (80/2) - 32 = 8.0 dBZ -> positive rate

    monkeypatch.setitem(REGIONS, "TEST_US", _TEST_REGION)

    def _mock_sample(_name, _lat, _lon):
        return np.array([True])

    monkeypatch.setattr("librewxr.mcp.sampling.sample_coverage", _mock_sample)

    result = await get_precip_nowcast(nwp, store, ["TEST_US"], 40.0, -95.0, 60)

    assert len(result) == 1
    r = result[0]
    assert r["source"] == "nwp"
    assert r["coverage"] == "in_range"
    assert r["dbz"] == pytest.approx(8.0, abs=0.5)
    assert r["rate_mmh"] > 0.0
    assert r["blend_weight"] == 0.0


@pytest.mark.mcp
async def test_get_precip_nowcast_source_none(monkeypatch):
    """When coverage is out-of-range and NWP is disabled, source='none'."""
    frame = NowcastFrame(timestamp=TS_10, regions={"TEST_US": _FRAME_DATA}, blend_weight=0.0)
    store = MockNowcastStore([TS_10], [frame])

    monkeypatch.setitem(REGIONS, "TEST_US", _TEST_REGION)

    def _mock_sample(_name, _lat, _lon):
        return np.array([False])

    monkeypatch.setattr("librewxr.mcp.sampling.sample_coverage", _mock_sample)

    # nwp_chain=None -> NWP disabled
    result = await get_precip_nowcast(None, store, ["TEST_US"], 40.0, -95.0, 60)

    assert len(result) == 1
    r = result[0]
    assert r["source"] == "none"
    assert r["coverage"] == "out_of_range"
    assert r["dbz"] is None
    assert r["rate_mmh"] == 0.0


@pytest.mark.mcp
async def test_get_precip_nowcast_rrqpe_undoes_display_offset(monkeypatch):
    """RRQPE stored dBZ carries the display shift -> MCP reports true values."""
    frame = NowcastFrame(timestamp=TS_10, regions={"RRQPE": _FRAME_DATA_HIGH}, blend_weight=1.0)
    store = MockNowcastStore([TS_10], [frame])

    monkeypatch.setitem(REGIONS, "RRQPE", _RRQPE_REGION)
    monkeypatch.setattr(settings, "rrqpe_dbz_offset", -7.0)
    monkeypatch.setattr("librewxr.mcp.tools.time.time", lambda: FIXED_NOW)

    def _mock_sample(_name, _lat, _lon):
        return np.array([True])

    monkeypatch.setattr("librewxr.mcp.sampling.sample_coverage", _mock_sample)

    result = await get_precip_nowcast(None, store, ["RRQPE"], 40.0, -95.0, 60)

    assert len(result) == 1
    r = result[0]
    assert r["source"] == "radar"
    assert r["blend_weight"] == 1.0
    # Stored pixel 100 -> 18.0 dBZ; rrqpe_dbz_offset=-7.0 -> true 25.0 dBZ.
    assert r["dbz"] == pytest.approx(25.0)
    assert r["rate_mmh"] == pytest.approx(dbz_to_rate_mmh(25.0))
    assert r["rate_mmh"] > dbz_to_rate_mmh(18.0)


@pytest.mark.mcp
async def test_get_precip_nowcast_non_rrqpe_not_corrected(monkeypatch):
    """Regions without a display offset pass their stored dBZ through as-is."""
    frame = NowcastFrame(timestamp=TS_10, regions={"TEST_US": _FRAME_DATA_HIGH}, blend_weight=0.7)
    store = MockNowcastStore([TS_10], [frame])

    monkeypatch.setitem(REGIONS, "TEST_US", _TEST_REGION)
    monkeypatch.setattr(settings, "rrqpe_dbz_offset", -7.0)
    monkeypatch.setattr("librewxr.mcp.tools.time.time", lambda: FIXED_NOW)

    def _mock_sample(_name, _lat, _lon):
        return np.array([True])

    monkeypatch.setattr("librewxr.mcp.sampling.sample_coverage", _mock_sample)

    result = await get_precip_nowcast(None, store, ["TEST_US"], 40.0, -95.0, 60)

    assert len(result) == 1
    r = result[0]
    assert r["source"] == "radar"
    assert r["dbz"] == pytest.approx(18.0)
    assert r["rate_mmh"] == pytest.approx(dbz_to_rate_mmh(18.0))


@pytest.mark.mcp
async def test_get_precip_nowcast_nwp_fallback_not_corrected(monkeypatch):
    """The NWP fallback branch keeps the chain's decoded dBZ uncorrected."""
    frame = NowcastFrame(timestamp=TS_10, regions={"OTHER": _FRAME_DATA}, blend_weight=0.0)
    store = MockNowcastStore([TS_10], [frame])
    nwp = MockNWPChain(pixel_value=100)  # 100 -> (100/2) - 32 = 18.0 dBZ

    monkeypatch.setitem(REGIONS, "RRQPE", _RRQPE_REGION)
    monkeypatch.setattr(settings, "rrqpe_dbz_offset", -7.0)

    def _mock_sample(_name, _lat, _lon):
        return np.array([True])

    monkeypatch.setattr("librewxr.mcp.sampling.sample_coverage", _mock_sample)

    result = await get_precip_nowcast(nwp, store, ["RRQPE"], 40.0, -95.0, 60)

    assert len(result) == 1
    r = result[0]
    assert r["source"] == "nwp"
    assert r["coverage"] == "in_range"
    assert r["dbz"] == pytest.approx(18.0)
    assert r["rate_mmh"] == pytest.approx(dbz_to_rate_mmh(18.0))


# ---------------------------------------------------------------------------
# get_active_alerts
# ---------------------------------------------------------------------------

@pytest.mark.mcp
async def test_get_active_alerts_disabled():
    """alerts_enabled=False returns empty FeatureCollection."""
    result = await get_active_alerts(None, False, 40.0, -100.0)
    assert isinstance(result, AlertsResponse)
    assert result.type == "FeatureCollection"
    assert result.features == []


@pytest.mark.mcp
async def test_get_active_alerts_enabled_but_none_store():
    """alerts_enabled=True but store=None -> empty (gate checks both)."""
    result = await get_active_alerts(None, True, 40.0, -100.0)
    assert len(result.features) == 0


@pytest.mark.mcp
async def test_get_active_alerts_delegates(monkeypatch):
    """get_active_alerts delegates to alerts_within_radius when both gates pass."""
    expected_feature = GeoJSONFeature(
        type="Feature",
        properties=AlertProperties(
            title="Delegated Alert",
            severity="Severe",
            time=1_700_000_000,
            expires=1_700_086_400,
            description="Delegated",
            regions=["Test"],
            uri="test:delegated",
        ),
        geometry=None,
    )
    expected_response = AlertsResponse(type="FeatureCollection", features=[expected_feature])

    async def mock_alerts_within_radius(store, lat, lon, radius_km=25.0, severity=None):
        return expected_response

    monkeypatch.setattr("librewxr.mcp.tools.alerts_within_radius", mock_alerts_within_radius)

    store = MockAlertsStore([])
    result = await get_active_alerts(store, True, 40.0, -100.0)

    assert len(result.features) == 1
    assert result.features[0].properties.title == "Delegated Alert"
    assert result.features[0].properties.uri == "test:delegated"


# ---------------------------------------------------------------------------
# get_active_alerts against a real alerts artifact store
# ---------------------------------------------------------------------------

def _alert_entry(title, polygon, url, severity="Severe"):
    return AlertEntry(
        source_id=title,
        event=title,
        description=f"{title} description",
        severity=severity,
        effective="2026-01-01T00:00:00+00:00",
        expires="2099-01-01T00:00:00+00:00",
        area_desc="Test area",
        url=url,
        polygon=polygon,
    )


@pytest.mark.mcp
async def test_get_active_alerts_artifact_store(tmp_path):
    """The artifact store feeds the tool end to end (near in, far out)."""
    near = _alert_entry(
        "Near", Polygon([
            (-100.1, 39.9), (-99.9, 39.9), (-99.9, 40.1), (-100.1, 40.1),
            (-100.1, 39.9),
        ]),
        "https://example.test/near",
    )
    far = _alert_entry(
        "Far", Polygon([
            (-99.1, 39.9), (-98.9, 39.9), (-98.9, 40.1), (-99.1, 40.1),
            (-99.1, 39.9),
        ]),
        "https://example.test/far",
    )
    store = AlertsStore()
    store.replace_all([near, far])
    assert write_alerts_artifacts(store, tmp_path) is True

    artifact = AlertsArtifactStore(tmp_path)
    assert artifact.maybe_reload() is True

    result = await get_active_alerts(artifact, True, 40.0, -100.0)
    assert [f.properties.title for f in result.features] == ["Near"]
    assert result.features[0].properties.uri == "https://example.test/near"


@pytest.mark.mcp
async def test_get_active_alerts_cold_artifact_store(tmp_path):
    """A cold (missing) artifact yields the empty collection, never a raise."""
    artifact = AlertsArtifactStore(tmp_path / "missing")
    assert artifact.maybe_reload() is False

    result = await get_active_alerts(artifact, True, 40.0, -100.0)
    assert isinstance(result, AlertsResponse)
    assert result.type == "FeatureCollection"
    assert result.features == []
