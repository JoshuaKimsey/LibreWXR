# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import numpy as np
import pytest

from librewrx.data.gfs_reflectivity import (
    GRID_HEIGHT,
    GRID_WIDTH,
    GFSReflectivityGrid,
)


class TestGFSReflectivityGrid:
    """Tests for the GFS simulated reflectivity grid."""

    def test_initial_state(self):
        grid = GFSReflectivityGrid()
        assert grid.data is None

    def test_sample_returns_zeros_when_no_data(self):
        grid = GFSReflectivityGrid()
        lat = np.array([40.0, 50.0])
        lon = np.array([-90.0, 10.0])
        result = grid.sample(lat, lon)
        assert result.shape == (2,)
        assert (result == 0).all()

    def test_sample_with_data(self):
        grid = GFSReflectivityGrid()
        # Create a synthetic grid: 20 dBZ everywhere
        # pixel = (20 + 32) * 2 = 104
        grid.data = np.full((GRID_HEIGHT, GRID_WIDTH), 104, dtype=np.uint8)

        lat = np.array([40.0, 0.0, -30.0])
        lon = np.array([-90.0, 0.0, 120.0])
        result = grid.sample(lat, lon)
        assert result.shape == (3,)
        assert (result == 104).all()

    def test_sample_2d_array(self):
        grid = GFSReflectivityGrid()
        grid.data = np.full((GRID_HEIGHT, GRID_WIDTH), 80, dtype=np.uint8)

        lat = np.ones((256, 256)) * 45.0
        lon = np.ones((256, 256)) * -75.0
        result = grid.sample(lat, lon)
        assert result.shape == (256, 256)
        assert (result == 80).all()

    def test_sample_clamps_coordinates(self):
        """Coordinates outside -90/90 or -180/180 should clamp, not crash."""
        grid = GFSReflectivityGrid()
        grid.data = np.full((GRID_HEIGHT, GRID_WIDTH), 64, dtype=np.uint8)

        lat = np.array([91.0, -91.0])
        lon = np.array([181.0, -181.0])
        result = grid.sample(lat, lon)
        assert result.shape == (2,)
        # Should not crash, values should be valid uint8
        assert result.dtype == np.uint8

    def test_grid_dimensions(self):
        assert GRID_WIDTH == 1440
        assert GRID_HEIGHT == 720


class TestGFSFallbackRendering:
    """Tests for GFS fallback in the tile renderer."""

    def test_gfs_only_tile_with_data(self):
        """A tile with no radar regions should render from GFS if available."""
        from librewrx.tiles.renderer import render_tile

        grid = GFSReflectivityGrid()
        # 15 dBZ everywhere → pixel = (15 + 32) * 2 = 94
        grid.data = np.full((GRID_HEIGHT, GRID_WIDTH), 94, dtype=np.uint8)

        # Tile over the Atlantic Ocean (no radar regions)
        tile_bytes = render_tile(
            frame_regions={},
            z=3, x=3, y=3,
            tile_size=256,
            color_scheme=2,
            fmt="png",
            enabled_regions=["USCOMP"],
            reflectivity_grid=grid,
        )
        # Should produce a non-transparent tile (has GFS data)
        assert len(tile_bytes) > 0
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(tile_bytes)).convert("RGBA")
        arr = np.array(img)
        # At least some pixels should be non-transparent
        assert arr[:, :, 3].max() > 0

    def test_gfs_only_tile_without_data(self):
        """Without GFS data, tiles outside radar coverage should be transparent."""
        from librewrx.tiles.renderer import render_tile

        tile_bytes = render_tile(
            frame_regions={},
            z=3, x=3, y=3,
            tile_size=256,
            color_scheme=2,
            fmt="png",
            enabled_regions=["USCOMP"],
            reflectivity_grid=None,
        )
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(tile_bytes)).convert("RGBA")
        arr = np.array(img)
        # All transparent
        assert arr[:, :, 3].max() == 0

    def test_coverage_mask_excludes_radar_areas(self):
        """GFS should NOT fill pixels that are within radar region bounds."""
        from librewrx.tiles.renderer import _build_coverage_mask
        from librewrx.data.regions import REGIONS

        # Tile over CONUS (z=4, x=4, y=5 is roughly central US)
        mask = _build_coverage_mask(
            [REGIONS["USCOMP"]], z=4, x=4, y=5, tile_size=256, pad=0,
        )
        # Most pixels should be covered (True)
        assert mask.sum() > 0.5 * 256 * 256

    def test_gfs_fills_uncovered_pixels(self):
        """GFS data should fill pixels outside radar coverage."""
        from librewrx.tiles.renderer import _fill_gfs_fallback
        from librewrx.data.regions import REGIONS

        grid = GFSReflectivityGrid()
        grid.data = np.full((GRID_HEIGHT, GRID_WIDTH), 94, dtype=np.uint8)

        # Tile at edge of USCOMP — some pixels in, some out
        # z=4, x=3, y=5 is at the western edge of USCOMP
        values = np.zeros((256, 256), dtype=np.uint8)
        result = _fill_gfs_fallback(
            values, [REGIONS["USCOMP"]], z=4, x=3, y=5, tile_size=256,
            pad=0, reflectivity_grid=grid,
        )
        # Some pixels should be filled with GFS value (94),
        # covered pixels should remain 0
        has_gfs = (result == 94).any()
        has_zero = (result == 0).any()
        assert has_gfs, "GFS should fill some uncovered pixels"
        assert has_zero, "Covered pixels should remain as radar values"
