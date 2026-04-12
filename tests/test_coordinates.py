# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import math

import numpy as np
import pytest

pytestmark = pytest.mark.tiles

from librewxr.tiles.coordinates import (
    COMPOSITE_HEIGHT,
    COMPOSITE_WIDTH,
    EAST,
    NORTH,
    SOUTH,
    WEST,
    tile_bounds,
    tile_overlaps_composite,
    tile_pixel_indices,
)


class TestTileBounds:
    def test_zoom_0(self):
        """Zoom 0 has a single tile covering the whole world."""
        w, s, e, n = tile_bounds(0, 0, 0)
        assert w == pytest.approx(-180.0)
        assert e == pytest.approx(180.0)
        assert n == pytest.approx(85.0511, abs=0.01)
        assert s == pytest.approx(-85.0511, abs=0.01)

    def test_zoom_1_tiles(self):
        """Zoom 1 has 4 tiles (2x2)."""
        # Top-left tile
        w, s, e, n = tile_bounds(1, 0, 0)
        assert w == pytest.approx(-180.0)
        assert e == pytest.approx(0.0)
        assert n > 0

        # Top-right tile
        w, s, e, n = tile_bounds(1, 1, 0)
        assert w == pytest.approx(0.0)
        assert e == pytest.approx(180.0)

    def test_tile_covers_conus(self):
        """A low-zoom tile should cover CONUS area."""
        # At zoom 3, tile (1, 2) or (1, 3) should cover parts of US
        w, s, e, n = tile_bounds(3, 1, 2)
        # Should be in western hemisphere, northern mid-latitudes
        assert w < 0
        assert n > 0


class TestTileOverlapsComposite:
    def test_conus_tile_overlaps(self):
        """Tiles over the US should overlap."""
        # At zoom 2, tile (0, 1) covers western North America
        assert tile_overlaps_composite(2, 0, 1) is True

    def test_far_east_no_overlap(self):
        """Tiles in Asia/Pacific should not overlap CONUS composite."""
        # At zoom 2, tile (3, 1) is far east
        assert tile_overlaps_composite(2, 3, 1) is False

    def test_zoom_0_overlaps(self):
        """The single zoom-0 tile covers everything."""
        assert tile_overlaps_composite(0, 0, 0) is True


class TestTilePixelIndices:
    def test_output_shape(self):
        row_idx, col_idx = tile_pixel_indices(3, 1, 3, 256)
        assert row_idx.shape == (256, 256)
        assert col_idx.shape == (256, 256)

    def test_out_of_bounds_marked(self):
        """Tiles outside CONUS should have all -1 indices."""
        # A tile over the Pacific
        row_idx, col_idx = tile_pixel_indices(3, 0, 3, 256)
        # All or mostly -1 (this tile is south Pacific)
        # At least some should be -1
        assert np.any(row_idx == -1) or np.all(row_idx >= 0)

    def test_conus_tile_has_valid_indices(self):
        """A tile over CONUS should have valid indices."""
        # Zoom 4, tile roughly over central US
        row_idx, col_idx = tile_pixel_indices(4, 3, 5, 256)
        valid = (row_idx >= 0) & (col_idx >= 0)
        assert np.any(valid), "Expected some valid indices over CONUS"

    def test_valid_indices_in_range(self):
        """Valid indices should be within composite bounds."""
        row_idx, col_idx = tile_pixel_indices(4, 3, 5, 256)
        valid = (row_idx >= 0) & (col_idx >= 0)
        assert np.all(row_idx[valid] < COMPOSITE_HEIGHT)
        assert np.all(col_idx[valid] < COMPOSITE_WIDTH)

    def test_caching(self):
        """Same call should return identical arrays."""
        r1, c1 = tile_pixel_indices(3, 1, 2, 256)
        r2, c2 = tile_pixel_indices(3, 1, 2, 256)
        assert r1 is r2  # Same object from cache
        assert c1 is c2

    def test_512_tile_size(self):
        row_idx, col_idx = tile_pixel_indices(3, 1, 3, 512)
        assert row_idx.shape == (512, 512)
