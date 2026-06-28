# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the Himawari-9 satellite source.

Covers auto-selection logic, provider registration, class metadata,
value mapping, and pickle round-trip.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from librewxr.sources.satellite.himawari.source import (
    HimawariIRSource,
    HimawariVISSource,
)

pytestmark = pytest.mark.sources


# ── Class metadata ──


def test_himawari_ir_metadata():
    assert HimawariIRSource.sat_lon == 140.7
    assert HimawariIRSource.s3_bucket == "noaa-himawari9"
    assert HimawariIRSource.channel == "IR"
    assert HimawariIRSource.band == 13
    assert HimawariIRSource.cadence_minutes == 10


def test_himawari_vis_metadata():
    assert HimawariVISSource.channel == "VIS"
    assert HimawariVISSource.band == 3


# ── Provider auto-selection ──


def test_provider_returns_himawari_for_tokyo():
    from librewxr.sources.satellite.himawari import satellite_provider

    settings = MagicMock()
    settings.get_bbox.return_value = (35.0, 139.0, 36.0, 140.0)
    settings.himawari_enabled = True
    settings.himawari_ir_enabled = True
    settings.himawari_vis_enabled = True
    settings.satellite_max_frames = 12

    contribs = satellite_provider(settings, cache_dir=None)
    assert len(contribs) == 2
    assert contribs[0].slug == "himawari9_ir_grid"
    assert contribs[1].slug == "himawari9_vis_grid"


def test_provider_returns_himawari_for_sydney():
    from librewxr.sources.satellite.himawari import satellite_provider

    settings = MagicMock()
    settings.get_bbox.return_value = (-34.0, 150.5, -33.5, 151.5)
    settings.himawari_enabled = True
    settings.himawari_ir_enabled = True
    settings.himawari_vis_enabled = True
    settings.satellite_max_frames = 12

    contribs = satellite_provider(settings, cache_dir=None)
    assert len(contribs) == 2
    assert contribs[0].slug == "himawari9_ir_grid"


def test_provider_returns_empty_for_london():
    """London (lon ~0°) is outside Himawari coverage."""
    from librewxr.sources.satellite.himawari import satellite_provider

    settings = MagicMock()
    settings.get_bbox.return_value = (51.0, -0.5, 52.0, 0.5)
    settings.himawari_enabled = True
    settings.satellite_max_frames = 12

    contribs = satellite_provider(settings, cache_dir=None)
    assert contribs == []


def test_provider_returns_empty_for_nyc():
    """NYC (lon ~-74) is outside Himawari coverage."""
    from librewxr.sources.satellite.himawari import satellite_provider

    settings = MagicMock()
    settings.get_bbox.return_value = (40.0, -74.5, 41.0, -73.5)
    settings.himawari_enabled = True
    settings.satellite_max_frames = 12

    contribs = satellite_provider(settings, cache_dir=None)
    assert contribs == []


def test_provider_returns_empty_when_disabled():
    from librewxr.sources.satellite.himawari import satellite_provider

    settings = MagicMock()
    settings.himawari_enabled = False
    contribs = satellite_provider(settings, cache_dir=None)
    assert contribs == []


def test_provider_uses_station_lon():
    from librewxr.sources.satellite.himawari import satellite_provider

    settings = MagicMock()
    settings.get_bbox.return_value = None
    settings.station_lon = 139.7  # Tokyo
    settings.himawari_enabled = True
    settings.himawari_ir_enabled = True
    settings.himawari_vis_enabled = False
    settings.satellite_max_frames = 12

    contribs = satellite_provider(settings, cache_dir=None)
    assert len(contribs) == 1
    assert contribs[0].slug == "himawari9_ir_grid"


# ── IR value mapping ──


def test_ir_cold_maps_to_high():
    src = HimawariIRSource(cache_dir=None)
    cold = np.array([[170.0]])
    encoded = src._map_to_uint8(cold)
    assert encoded[0, 0] == 255


def test_ir_warm_maps_to_low():
    src = HimawariIRSource(cache_dir=None)
    warm = np.array([[320.0]])
    encoded = src._map_to_uint8(warm)
    assert encoded[0, 0] == 0


# ── VIS value mapping ──


def test_vis_full_reflectance():
    src = HimawariVISSource(cache_dir=None)
    data = np.array([[1.0]])
    assert src._map_to_uint8(data)[0, 0] == 255


def test_vis_zero_reflectance():
    src = HimawariVISSource(cache_dir=None)
    data = np.array([[0.0]])
    assert src._map_to_uint8(data)[0, 0] == 0


# ── sample() with synthetic grid ──


def test_sample_empty_store():
    src = HimawariIRSource(cache_dir=None, max_frames=3)
    lat = np.array([[35.68]], dtype=np.float32)
    lon = np.array([[139.69]], dtype=np.float32)
    out = src.sample(lat, lon, timestamp=None)
    assert out.shape == (1, 1)
    assert out[0, 0] == 0


def test_sample_invisible_point():
    """New York is not visible from Himawari-9."""
    src = HimawariIRSource(cache_dir=None, max_frames=3)
    src._x_vec = np.linspace(-0.15, 0.15, 100, dtype=np.float64)
    src._y_vec = np.linspace(0.15, -0.15, 100, dtype=np.float64)
    src._grid_width = 100
    src._grid_height = 100

    grid = np.full((100, 100), 200, dtype=np.uint8)
    src._frames[12345] = grid
    src._sorted_timestamps = [12345]

    lat = np.array([[40.71]], dtype=np.float64)
    lon = np.array([[-74.01]], dtype=np.float64)  # NYC
    out = src.sample(lat, lon, timestamp=12345)
    assert out[0, 0] == 0  # not visible


# ── Pickle round-trip ──


def test_pickle_round_trip(tmp_path: Path):
    src = HimawariIRSource(cache_dir=tmp_path, max_frames=5)
    src._x_vec = np.linspace(-0.15, 0.15, 50, dtype=np.float64)
    src._y_vec = np.linspace(0.15, -0.15, 40, dtype=np.float64)
    src._grid_width = 50
    src._grid_height = 40

    grid = np.full((40, 50), 120, dtype=np.uint8)
    ts = 88888
    src._frames[ts] = grid
    src._sorted_timestamps = [ts]
    src._write_cache(ts, grid)

    state = src.__getstate__()
    render_src = HimawariIRSource.__new__(HimawariIRSource)
    render_src.__setstate__(state)
    assert render_src.timestamps == [ts]
    np.testing.assert_array_equal(render_src._frames[ts], grid)
