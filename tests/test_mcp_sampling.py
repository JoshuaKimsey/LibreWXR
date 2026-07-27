# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for pure MCP sampling functions: decode_dbz, dbz_to_rate_mmh,
resolve_region_for_point, sample_region_at_point, sample_nowcast_at_point.
"""

import numpy as np
import pytest

from librewxr.data.nowcast import NowcastFrame
from librewxr.data.regions import REGIONS, RegionDef
from librewxr.mcp.sampling import (
    dbz_to_rate_mmh,
    decode_dbz,
    resolve_region_for_point,
    sample_nowcast_at_point,
    sample_region_at_point,
)


# ---------------------------------------------------------------------------
# decode_dbz
# ---------------------------------------------------------------------------

@pytest.mark.mcp
def test_decode_dbz_nodata():
    """Pixel 0 maps to None (nodata/transparent)."""
    assert decode_dbz(0) is None


@pytest.mark.mcp
def test_decode_dbz_basic():
    """Known pixel-to-dBZ mappings from the encoder formula."""
    assert decode_dbz(64) == 0.0
    assert decode_dbz(255) == 95.5
    assert decode_dbz(1) == -31.5
    assert decode_dbz(128) == 32.0


# ---------------------------------------------------------------------------
# dbz_to_rate_mmh
# ---------------------------------------------------------------------------

@pytest.mark.mcp
def test_dbz_to_rate_mmh_none():
    """None dBZ returns 0.0 mm/h."""
    assert dbz_to_rate_mmh(None) == 0.0


@pytest.mark.mcp
def test_dbz_to_rate_mmh_zero():
    """dBZ <= 0 returns 0.0 mm/h."""
    assert dbz_to_rate_mmh(0) == 0.0
    assert dbz_to_rate_mmh(-5.0) == 0.0


@pytest.mark.mcp
def test_dbz_to_rate_mmh_rain():
    """Marshall-Palmer rain: 40 dBZ ~ 11.5 mm/h."""
    r = dbz_to_rate_mmh(40.0)
    expected = (10000.0 / 200.0) ** (1.0 / 1.6)
    assert r == pytest.approx(expected, abs=0.5)


@pytest.mark.mcp
def test_dbz_to_rate_mmh_snow():
    """Marshall-Palmer snow: 40 dBZ ~ 2.236 mm/h."""
    s = dbz_to_rate_mmh(40.0, is_snow=True)
    expected = (10000.0 / 2000.0) ** (1.0 / 2.0)
    assert s == pytest.approx(expected, abs=0.1)


# ---------------------------------------------------------------------------
# resolve_region_for_point
# ---------------------------------------------------------------------------

@pytest.mark.mcp
def test_resolve_region_for_point_empty_regions():
    """Empty enabled_regions list returns None."""
    assert resolve_region_for_point(40.0, -100.0, []) is None


@pytest.mark.mcp
def test_resolve_region_for_point_no_coverage(monkeypatch):
    """When sample_coverage returns False the point is unresolved."""
    test_region = RegionDef(
        name="TEST_COV",
        west=-180.0, east=180.0, south=-90.0, north=90.0,
        pixel_size=1.0, group="TEST",
    )
    monkeypatch.setitem(REGIONS, "TEST_COV", test_region)

    def _mock_sample(_name, _lat, _lon):
        return np.array([False])

    monkeypatch.setattr("librewxr.mcp.sampling.sample_coverage", _mock_sample)
    assert resolve_region_for_point(0.0, -160.0, ["TEST_COV"]) is None


@pytest.mark.mcp
def test_resolve_region_for_point_picks_finest(monkeypatch):
    """Among multiple covering regions, the one with smallest pixel_size wins."""
    coarse = RegionDef(
        name="COARSE", west=-100.0, east=-90.0,
        south=35.0, north=45.0, pixel_size=0.02, group="TEST",
    )
    fine = RegionDef(
        name="FINE", west=-100.0, east=-90.0,
        south=35.0, north=45.0, pixel_size=0.01, group="TEST",
    )
    monkeypatch.setitem(REGIONS, "COARSE", coarse)
    monkeypatch.setitem(REGIONS, "FINE", fine)

    def _mock_sample(_name, _lat, _lon):
        return np.array([True])

    monkeypatch.setattr("librewxr.mcp.sampling.sample_coverage", _mock_sample)

    result = resolve_region_for_point(40.0, -95.0, ["COARSE", "FINE"])
    assert result is not None
    assert result.name == "FINE"
    assert result.pixel_size == 0.01


# ---------------------------------------------------------------------------
# sample_region_at_point
# ---------------------------------------------------------------------------

@pytest.mark.mcp
def test_sample_region_at_point_latlon():
    """Sample a latlon region at a coordinate that maps into a known pixel."""
    region = RegionDef(
        name="TEST_LATLON",
        west=0.0, east=10.0, south=0.0, north=10.0,
        pixel_size=1.0, group="TEST",
    )
    # Array shape = (10, 10).  Set pixel at (row=5, col=5) to 64 (= 0.0 dBZ).
    frame = np.zeros((10, 10), dtype=np.uint8)
    frame[5, 5] = 64

    # Round-half-to-even in Python means (north - lat) / _ps_y must be exactly
    # 5.0 (not 5.5) to land on row 5.  lat=5.0 → (10-5)/1 = 5.0 → col=5.
    # lon=5.0 → (5-0)/1 = 5.0 → col=5.
    dbz, coverage = sample_region_at_point(region, 5.0, 5.0, frame)
    assert dbz == 0.0
    assert coverage == "in_range"


@pytest.mark.mcp
def test_sample_region_at_point_out_of_bounds():
    """A point outside the region's grid returns (None, 'out_of_range')."""
    region = RegionDef(
        name="TEST_LATLON",
        west=0.0, east=10.0, south=0.0, north=10.0,
        pixel_size=1.0, group="TEST",
    )
    frame = np.zeros((10, 10), dtype=np.uint8)

    # lat=20 -> row_f = (10 - 20) / 1.0 = -10.0  -> int(round(-10.0)) = -10 -> out of bounds
    # lon=20 -> col_f = (20 - 0) / 1.0 = 20.0    -> int(round(20.0)) = 20  -> out of bounds
    dbz, coverage = sample_region_at_point(region, 20.0, 20.0, frame)
    assert dbz is None
    assert coverage == "out_of_range"


# ---------------------------------------------------------------------------
# sample_nowcast_at_point
# ---------------------------------------------------------------------------

@pytest.mark.mcp
def test_sample_nowcast_at_point(monkeypatch):
    """Sample a nowcast frame at a point that lands within a known region."""
    region = RegionDef(
        name="TEST_LATLON",
        west=0.0, east=10.0, south=0.0, north=10.0,
        pixel_size=1.0, group="TEST",
    )
    monkeypatch.setitem(REGIONS, "TEST_LATLON", region)

    frame_arr = np.zeros((10, 10), dtype=np.uint8)
    frame_arr[5, 5] = 64  # 0.0 dBZ
    ncf = NowcastFrame(timestamp=1000, regions={"TEST_LATLON": frame_arr}, blend_weight=0.7)

    dbz, bw, coverage = sample_nowcast_at_point("TEST_LATLON", 5.0, 5.0, ncf)
    assert dbz == 0.0
    assert bw == 0.7
    assert coverage == "in_range"


@pytest.mark.mcp
def test_sample_nowcast_at_point_missing_region(monkeypatch):
    """When the frame does not contain the queried region, returns out_of_range."""
    region = RegionDef(
        name="TEST_LATLON",
        west=0.0, east=10.0, south=0.0, north=10.0,
        pixel_size=1.0, group="TEST",
    )
    monkeypatch.setitem(REGIONS, "TEST_LATLON", region)

    frame_arr = np.zeros((10, 10), dtype=np.uint8)
    ncf = NowcastFrame(timestamp=1000, regions={"OTHER": frame_arr}, blend_weight=0.5)

    dbz, bw, coverage = sample_nowcast_at_point("TEST_LATLON", 4.5, 5.5, ncf)
    assert dbz is None
    assert bw == 0.5
    assert coverage == "out_of_range"
