# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the geostationary projection module.

Verifies forward/inverse transforms against known GOES-18 coordinates,
round-trip accuracy, and correct NaN handling for off-disk points.
"""
from __future__ import annotations

import numpy as np
import pytest

from librewxr.tiles.geostationary import forward, inverse

pytestmark = pytest.mark.tiles

# GOES-18 parameters
SAT_LON = -137.0
SAT_HEIGHT = 35786023.0

# GOES-19 parameters
SAT_LON_EAST = -75.2
SAT_HEIGHT_EAST = 35786023.0


class TestForwardProjection:
    """Forward: lat/lon → scan angles."""

    def test_subsatellite_point_maps_to_origin(self):
        """The sub-satellite point (0°N, sat_lon) should map to (0, 0)."""
        lat = np.array([0.0])
        lon = np.array([SAT_LON])
        x, y = forward(lat, lon, SAT_LON, SAT_HEIGHT)
        assert abs(float(x[0])) < 1e-10
        assert abs(float(y[0])) < 1e-10

    def test_north_of_subsatellite_gives_positive_y(self):
        """A point north of the sub-satellite should have y > 0."""
        lat = np.array([45.0])
        lon = np.array([SAT_LON])
        x, y = forward(lat, lon, SAT_LON, SAT_HEIGHT)
        assert float(y[0]) > 0.0
        assert abs(float(x[0])) < 1e-6  # same longitude → x ≈ 0

    def test_east_of_subsatellite_gives_negative_x(self):
        """A point east of the sub-satellite should have x < 0 (GOES convention)."""
        lat = np.array([0.0])
        lon = np.array([SAT_LON + 10.0])
        x, y = forward(lat, lon, SAT_LON, SAT_HEIGHT)
        # x = arcsin(-sy/r), and sy is negative for east → x < 0
        assert float(x[0]) < 0.0

    def test_far_side_of_earth_returns_nan(self):
        """Points on the opposite side of the earth (behind it) return NaN."""
        lat = np.array([0.0])
        lon = np.array([SAT_LON + 180.0])  # antipodal
        x, y = forward(lat, lon, SAT_LON, SAT_HEIGHT)
        assert np.isnan(x[0])
        assert np.isnan(y[0])

    def test_conus_points_are_visible(self):
        """Major CONUS cities should be visible from GOES-18."""
        lats = np.array([34.05, 40.71, 47.61, 25.76])  # LA, NYC, Seattle, Miami
        lons = np.array([-118.24, -74.01, -122.33, -80.19])
        x, y = forward(lats, lons, SAT_LON, SAT_HEIGHT)
        assert not np.any(np.isnan(x))
        assert not np.any(np.isnan(y))

    def test_vectorized_output_shape(self):
        """Forward projection preserves input array shape."""
        lat = np.random.uniform(20, 50, (10, 10))
        lon = np.random.uniform(-130, -70, (10, 10))
        x, y = forward(lat, lon, SAT_LON, SAT_HEIGHT)
        assert x.shape == (10, 10)
        assert y.shape == (10, 10)


class TestInverseProjection:
    """Inverse: scan angles → lat/lon."""

    def test_origin_maps_to_subsatellite(self):
        """Scan angle (0, 0) maps to the sub-satellite point."""
        x = np.array([0.0])
        y = np.array([0.0])
        lat, lon = inverse(x, y, SAT_LON, SAT_HEIGHT)
        assert abs(float(lat[0])) < 0.01  # within ~1 km
        assert abs(float(lon[0]) - SAT_LON) < 0.01

    def test_outside_disk_returns_nan(self):
        """Scan angles outside the earth disk return NaN."""
        x = np.array([1.0])  # way outside valid range (~0.15 rad max)
        y = np.array([0.0])
        lat, lon = inverse(x, y, SAT_LON, SAT_HEIGHT)
        assert np.isnan(lat[0])
        assert np.isnan(lon[0])


class TestRoundTrip:
    """Forward → inverse round-trip accuracy."""

    def test_round_trip_conus(self):
        """CONUS points round-trip within 0.001° (~111 m)."""
        np.random.seed(42)
        lat_in = np.random.uniform(25, 50, 100)
        lon_in = np.random.uniform(-125, -70, 100)
        x, y = forward(lat_in, lon_in, SAT_LON, SAT_HEIGHT)
        lat_out, lon_out = inverse(x, y, SAT_LON, SAT_HEIGHT)
        np.testing.assert_allclose(lat_in, lat_out, atol=0.001)
        np.testing.assert_allclose(lon_in, lon_out, atol=0.001)

    def test_round_trip_goes_east(self):
        """GOES-19 (East) round-trip for eastern US."""
        np.random.seed(43)
        lat_in = np.random.uniform(25, 50, 50)
        lon_in = np.random.uniform(-90, -60, 50)
        x, y = forward(lat_in, lon_in, SAT_LON_EAST, SAT_HEIGHT_EAST)
        lat_out, lon_out = inverse(x, y, SAT_LON_EAST, SAT_HEIGHT_EAST)
        np.testing.assert_allclose(lat_in, lat_out, atol=0.001)
        np.testing.assert_allclose(lon_in, lon_out, atol=0.001)

    def test_round_trip_near_poles(self):
        """High-latitude points visible from GOES also round-trip."""
        lat_in = np.array([60.0, -60.0, 65.0])
        lon_in = np.array([SAT_LON, SAT_LON, SAT_LON - 10])
        x, y = forward(lat_in, lon_in, SAT_LON, SAT_HEIGHT)
        visible = ~np.isnan(x)
        if visible.any():
            lat_out, lon_out = inverse(x[visible], y[visible], SAT_LON, SAT_HEIGHT)
            np.testing.assert_allclose(lat_in[visible], lat_out, atol=0.001)
            np.testing.assert_allclose(lon_in[visible], lon_out, atol=0.001)

    def test_round_trip_himawari(self):
        """Round-trip for Himawari-9 (140.7°E) over Japan."""
        him_lon = 140.7
        him_height = 35785831.0
        lat_in = np.array([35.68, -33.87, 1.35])  # Tokyo, Sydney, Singapore
        lon_in = np.array([139.69, 151.21, 103.82])
        x, y = forward(lat_in, lon_in, him_lon, him_height)
        visible = ~np.isnan(x)
        lat_out, lon_out = inverse(x[visible], y[visible], him_lon, him_height)
        np.testing.assert_allclose(lat_in[visible], lat_out, atol=0.001)
        np.testing.assert_allclose(lon_in[visible], lon_out, atol=0.001)
