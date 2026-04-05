# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io

import numpy as np
import pytest
from PIL import Image

from librewxr.data.ecmwf_grid import (
    GRID_HEIGHT,
    GRID_WIDTH,
    PIXEL_SIZE,
    ECMWFGrid,
    ZR_A_RAIN,
    ZR_B_RAIN,
    ZR_A_SNOW,
    ZR_B_SNOW,
)


class TestECMWFGrid:
    """Tests for the ECMWF IFS precipitation grid."""

    def test_initial_state(self):
        grid = ECMWFGrid()
        assert grid.data is None
        assert grid.reference_time is None

    def test_sample_returns_zeros_when_no_data(self):
        grid = ECMWFGrid()
        lat = np.array([40.0, 50.0])
        lon = np.array([-90.0, 10.0])
        result = grid.sample(lat, lon)
        assert result.shape == (2,)
        assert (result == 0).all()

    def test_get_snow_mask_returns_false_when_no_data(self):
        grid = ECMWFGrid()
        lat = np.array([40.0, 50.0])
        lon = np.array([-90.0, 10.0])
        result = grid.get_snow_mask(lat, lon)
        assert result.shape == (2,)
        assert not result.any()

    def test_sample_with_data(self):
        grid = ECMWFGrid()
        # 20 dBZ everywhere → pixel = (20 + 32) * 2 = 104
        grid._precip_dbz = np.full((GRID_HEIGHT, GRID_WIDTH), 104, dtype=np.uint8)

        lat = np.array([40.0, 0.0, -30.0])
        lon = np.array([-90.0, 0.0, 120.0])
        result = grid.sample(lat, lon)
        assert result.shape == (3,)
        assert (result == 104).all()

    def test_sample_2d_array(self):
        grid = ECMWFGrid()
        grid._precip_dbz = np.full((GRID_HEIGHT, GRID_WIDTH), 80, dtype=np.uint8)

        lat = np.ones((256, 256)) * 45.0
        lon = np.ones((256, 256)) * -75.0
        result = grid.sample(lat, lon)
        assert result.shape == (256, 256)
        assert (result == 80).all()

    def test_sample_clamps_coordinates(self):
        """Coordinates outside -90/90 or -180/180 should clamp, not crash."""
        grid = ECMWFGrid()
        grid._precip_dbz = np.full((GRID_HEIGHT, GRID_WIDTH), 64, dtype=np.uint8)

        lat = np.array([91.0, -91.0])
        lon = np.array([181.0, -181.0])
        result = grid.sample(lat, lon)
        assert result.shape == (2,)
        assert result.dtype == np.uint8

    def test_get_snow_mask_with_data(self):
        grid = ECMWFGrid()
        grid._snow_mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
        # Mark northern hemisphere as snow
        grid._snow_mask[:GRID_HEIGHT // 2, :] = True

        # Northern point should be snow
        lat_n = np.array([60.0])
        lon_n = np.array([0.0])
        assert grid.get_snow_mask(lat_n, lon_n)[0] == True

        # Southern point should not be snow
        lat_s = np.array([-30.0])
        lon_s = np.array([0.0])
        assert grid.get_snow_mask(lat_s, lon_s)[0] == False

    def test_grid_dimensions(self):
        assert GRID_WIDTH == 3600
        assert GRID_HEIGHT == 1801
        assert PIXEL_SIZE == 0.1


class TestZRConversion:
    """Tests for the Z-R relationship math."""

    def test_zero_precip_gives_zero_dbz(self):
        """No precipitation should produce 0 dBZ (clear sky)."""
        rate = 0.0
        z = ZR_A_RAIN * (rate ** ZR_B_RAIN)
        assert z == 0.0

    def test_rain_1mm_hr(self):
        """1 mm/h rain should produce ~23 dBZ (Marshall-Palmer)."""
        rate = 1.0
        z = ZR_A_RAIN * (rate ** ZR_B_RAIN)  # Z = 200 * 1^1.6 = 200
        dbz = 10.0 * np.log10(z)  # 10 * log10(200) ≈ 23
        assert 22.5 < dbz < 23.5

    def test_heavy_rain_50mm_hr(self):
        """50 mm/h rain should produce ~50 dBZ."""
        rate = 50.0
        z = ZR_A_RAIN * (rate ** ZR_B_RAIN)
        dbz = 10.0 * np.log10(z)
        assert 49.0 < dbz < 52.0

    def test_snow_zr_differs_from_rain(self):
        """Snow Z-R should give different values than rain for the same rate."""
        rate = 5.0
        z_rain = ZR_A_RAIN * (rate ** ZR_B_RAIN)
        z_snow = ZR_A_SNOW * (rate ** ZR_B_SNOW)
        assert z_rain != z_snow

    def test_uint8_encoding(self):
        """Verify the dBZ to uint8 encoding: pixel = clamp((dBZ + 32) * 2, 0, 255)."""
        # 20 dBZ → (20 + 32) * 2 = 104
        assert int(np.clip((20.0 + 32.0) * 2.0, 0, 255)) == 104
        # 0 dBZ → (0 + 32) * 2 = 64
        assert int(np.clip((0.0 + 32.0) * 2.0, 0, 255)) == 64
        # -32 dBZ → 0
        assert int(np.clip((-32.0 + 32.0) * 2.0, 0, 255)) == 0
        # 95 dBZ → 254 (capped)
        assert int(np.clip((95.0 + 32.0) * 2.0, 0, 255)) == 254


class TestECMWFFallbackRendering:
    """Tests for ECMWF fallback in the tile renderer."""

    def test_ecmwf_only_tile_with_data(self):
        """A tile with no radar regions should render from ECMWF if available."""
        from librewxr.tiles.renderer import render_tile

        grid = ECMWFGrid()
        # 15 dBZ everywhere → pixel = (15 + 32) * 2 = 94
        grid._precip_dbz = np.full((GRID_HEIGHT, GRID_WIDTH), 94, dtype=np.uint8)

        # Tile over the Atlantic Ocean (no radar regions)
        tile_bytes = render_tile(
            frame_regions={},
            z=3, x=3, y=3,
            tile_size=256,
            color_scheme=2,
            fmt="png",
            enabled_regions=["USCOMP"],
            ecmwf_grid=grid,
        )
        assert len(tile_bytes) > 0
        img = Image.open(io.BytesIO(tile_bytes)).convert("RGBA")
        arr = np.array(img)
        # At least some pixels should be non-transparent
        assert arr[:, :, 3].max() > 0

    def test_ecmwf_only_tile_without_data(self):
        """Without ECMWF data, tiles outside radar coverage should be transparent."""
        from librewxr.tiles.renderer import render_tile

        tile_bytes = render_tile(
            frame_regions={},
            z=3, x=3, y=3,
            tile_size=256,
            color_scheme=2,
            fmt="png",
            enabled_regions=["USCOMP"],
            ecmwf_grid=None,
        )
        img = Image.open(io.BytesIO(tile_bytes)).convert("RGBA")
        arr = np.array(img)
        assert arr[:, :, 3].max() == 0

    def test_coverage_mask_excludes_radar_areas(self):
        """ECMWF should NOT fill pixels that are within radar region bounds."""
        from librewxr.tiles.renderer import _build_coverage_mask
        from librewxr.data.regions import REGIONS

        mask = _build_coverage_mask(
            [REGIONS["USCOMP"]], z=4, x=4, y=5, tile_size=256, pad=0,
        )
        assert mask.sum() > 0.5 * 256 * 256

    def test_ecmwf_fills_uncovered_pixels(self):
        """ECMWF data should fill pixels outside radar coverage."""
        from librewxr.tiles.renderer import _fill_ecmwf_fallback
        from librewxr.data.regions import REGIONS

        grid = ECMWFGrid()
        grid._precip_dbz = np.full((GRID_HEIGHT, GRID_WIDTH), 94, dtype=np.uint8)

        values = np.zeros((256, 256), dtype=np.uint8)
        result = _fill_ecmwf_fallback(
            values, [REGIONS["USCOMP"]], z=4, x=3, y=5, tile_size=256,
            pad=0, ecmwf_grid=grid,
        )
        has_ecmwf = (result == 94).any()
        has_zero = (result == 0).any()
        assert has_ecmwf, "ECMWF should fill some uncovered pixels"
        assert has_zero, "Covered pixels should remain as radar values"

    def test_snow_mask_used_for_coloring(self):
        """When snow=True, the renderer should use ECMWF snow mask."""
        from librewxr.tiles.renderer import render_tile

        grid = ECMWFGrid()
        grid._precip_dbz = np.full((GRID_HEIGHT, GRID_WIDTH), 94, dtype=np.uint8)
        grid._snow_mask = np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=bool)

        tile_snow = render_tile(
            frame_regions={},
            z=3, x=3, y=3,
            tile_size=256,
            color_scheme=2,
            snow=True,
            fmt="png",
            enabled_regions=["USCOMP"],
            ecmwf_grid=grid,
        )

        tile_rain = render_tile(
            frame_regions={},
            z=3, x=3, y=3,
            tile_size=256,
            color_scheme=2,
            snow=False,
            fmt="png",
            enabled_regions=["USCOMP"],
            ecmwf_grid=grid,
        )

        # Snow and rain tiles should differ in color
        assert tile_snow != tile_rain
