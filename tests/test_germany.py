# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import math

import numpy as np
import pytest

from librewrx.data.regions import REGIONS, resolve_regions
from librewrx.data.sources import _parse_dwd_hdf5, _parse_dwd_listing
from librewrx.tiles.coordinates import tile_overlaps_region


class TestGermanyRegion:
    def test_germany_in_regions(self):
        assert "GERMANY" in REGIONS

    def test_germany_dimensions(self):
        r = REGIONS["GERMANY"]
        assert r.width == 4400
        assert r.height == 4800
        assert r.grid_scale == 250.0

    def test_germany_projection(self):
        r = REGIONS["GERMANY"]
        assert r.proj == "stere"
        assert r.stere_lat_ts == 60.0
        assert r.stere_lon0 == 10.0
        assert r.stere_x0 != 0.0
        assert r.stere_y0 != 0.0

    def test_germany_group_resolution(self):
        assert resolve_regions("GERMANY") == ["GERMANY"]

    def test_all_includes_germany(self):
        all_regions = resolve_regions("ALL")
        assert "GERMANY" in all_regions

    def test_mixed_regions(self):
        result = resolve_regions("CONUS,GERMANY")
        assert "USCOMP" in result
        assert "GERMANY" in result

    def test_germany_no_iem_dirs(self):
        r = REGIONS["GERMANY"]
        assert r.live_dir == ""
        assert r.archive_dir == ""


class TestStereProjection:
    """Verify polar stereographic projection matches known DWD reference point."""

    def test_known_reference_point(self):
        """(9°E, 51°N) should map to approximately col=2174, row=4222."""
        from librewrx.tiles.coordinates import _stere_forward

        region = REGIONS["GERMANY"]
        lon = np.array([9.0])
        lat = np.array([51.0])
        x, y = _stere_forward(lon, lat, region)

        # Known projection coords for this point
        col = (x[0] - region.grid_x_min) / region.grid_scale
        row = (region.grid_y_max - y[0]) / region.grid_scale

        # Should be roughly in the center of the grid
        assert 1800 < col < 1950, f"col={col}"
        assert 2300 < row < 2500, f"row={row}"

    def test_central_meridian(self):
        """Point on central meridian (lon=10) should have x close to x_0."""
        from librewrx.tiles.coordinates import _stere_forward

        region = REGIONS["GERMANY"]
        lon = np.array([10.0])
        lat = np.array([51.0])
        x, y = _stere_forward(lon, lat, region)

        # At central meridian, x should equal false easting (x_0)
        expected_col = region.stere_x0 / region.grid_scale
        actual_col = x[0] / region.grid_scale
        assert abs(actual_col - expected_col) < 1.0

    def test_bounds_coverage(self):
        """Corner points should be within grid bounds."""
        from librewrx.tiles.coordinates import _stere_forward

        region = REGIONS["GERMANY"]
        # Test point well inside Germany
        lon = np.array([10.0])
        lat = np.array([51.0])
        x, y = _stere_forward(lon, lat, region)

        col = (x[0] - region.grid_x_min) / region.grid_scale
        row = (region.grid_y_max - y[0]) / region.grid_scale

        assert 0 <= col < region.width
        assert 0 <= row < region.height


class TestDWDListingParsing:
    def test_parse_valid_listing(self):
        html = """
        <html><body>
        <a href="composite_hx_20260310_1200-hd5">composite_hx_20260310_1200-hd5</a>
        <a href="composite_hx_20260310_1205-hd5">composite_hx_20260310_1205-hd5</a>
        <a href="composite_hx_20260310_1210-hd5">composite_hx_20260310_1210-hd5</a>
        </body></html>
        """
        entries = _parse_dwd_listing(html)
        assert len(entries) == 3
        # Should be sorted newest-first
        assert entries[0][0] > entries[1][0]
        assert entries[1][0] > entries[2][0]
        # Filenames should be preserved
        assert entries[0][1] == "composite_hx_20260310_1210-hd5"

    def test_parse_empty_listing(self):
        html = "<html><body>No files here</body></html>"
        entries = _parse_dwd_listing(html)
        assert entries == []

    def test_parse_ignores_latest(self):
        html = """
        <a href="composite_hx_LATEST-hd5">composite_hx_LATEST-hd5</a>
        <a href="composite_hx_20260310_1200-hd5">composite_hx_20260310_1200-hd5</a>
        """
        entries = _parse_dwd_listing(html)
        assert len(entries) == 1
        assert "1200" in entries[0][1]


class TestGermanyTileOverlap:
    def test_tile_over_germany_overlaps(self):
        region = REGIONS["GERMANY"]
        # z=4, x=8, y=5 is roughly over central Europe
        assert tile_overlaps_region(region, z=4, x=8, y=5)

    def test_tile_over_us_does_not_overlap(self):
        region = REGIONS["GERMANY"]
        # z=4, x=3, y=5 is over the US
        assert not tile_overlaps_region(region, z=4, x=3, y=5)

    def test_tile_over_scandinavia_does_not_overlap(self):
        region = REGIONS["GERMANY"]
        # z=6, x=35, y=18 is over Norway (north of Germany coverage)
        assert not tile_overlaps_region(region, z=6, x=35, y=18)
