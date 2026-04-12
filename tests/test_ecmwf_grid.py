# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.ecmwf

from librewxr.data.ecmwf_grid import (
    GRID_HEIGHT,
    GRID_WIDTH,
    PIXEL_SIZE,
    WEST,
    ECMWFGrid,
    ZR_A_RAIN,
    ZR_B_RAIN,
    ZR_A_SNOW,
    ZR_B_SNOW,
)


def _inject_timestep(grid, precip_dbz, snow_mask=None, timestamp=1000000):
    """Helper to inject test data into an ECMWFGrid's multi-timestep store."""
    if snow_mask is None:
        snow_mask = np.zeros_like(precip_dbz, dtype=bool)
    grid._timesteps[timestamp] = (precip_dbz, snow_mask)
    grid._sorted_timestamps = sorted(grid._timesteps.keys())


class TestECMWFGrid:
    """Tests for the ECMWF IFS precipitation grid."""

    def test_initial_state(self):
        grid = ECMWFGrid()
        assert grid.data is None
        assert grid.reference_time is None
        assert grid.timestep_count == 0

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
        _inject_timestep(grid, np.full((GRID_HEIGHT, GRID_WIDTH), 104, dtype=np.uint8))

        lat = np.array([40.0, 0.0, -30.0])
        lon = np.array([-90.0, 0.0, 120.0])
        result = grid.sample(lat, lon)
        assert result.shape == (3,)
        assert (result == 104).all()

    def test_sample_2d_array(self):
        grid = ECMWFGrid()
        _inject_timestep(grid, np.full((GRID_HEIGHT, GRID_WIDTH), 80, dtype=np.uint8))

        lat = np.ones((256, 256)) * 45.0
        lon = np.ones((256, 256)) * -75.0
        result = grid.sample(lat, lon)
        assert result.shape == (256, 256)
        assert (result == 80).all()

    def test_sample_clamps_coordinates(self):
        """Coordinates outside -90/90 or -180/180 should clamp, not crash."""
        grid = ECMWFGrid()
        _inject_timestep(grid, np.full((GRID_HEIGHT, GRID_WIDTH), 64, dtype=np.uint8))

        lat = np.array([91.0, -91.0])
        lon = np.array([181.0, -181.0])
        result = grid.sample(lat, lon)
        assert result.shape == (2,)
        assert result.dtype == np.uint8

    def test_get_snow_mask_with_data(self):
        grid = ECMWFGrid()
        snow_mask = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
        # Mark northern hemisphere as snow
        snow_mask[:GRID_HEIGHT // 2, :] = True
        _inject_timestep(
            grid,
            np.full((GRID_HEIGHT, GRID_WIDTH), 94, dtype=np.uint8),
            snow_mask=snow_mask,
        )

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

    def test_nearest_timestamp(self):
        """Binary search should find the closest stored timestep."""
        grid = ECMWFGrid()
        precip = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        _inject_timestep(grid, precip, timestamp=1000)
        _inject_timestep(grid, precip, timestamp=4600)
        _inject_timestep(grid, precip, timestamp=8200)

        assert grid._nearest_timestamp(1000) == 1000
        assert grid._nearest_timestamp(2000) == 1000
        assert grid._nearest_timestamp(3000) == 4600
        assert grid._nearest_timestamp(7000) == 8200
        assert grid._nearest_timestamp(None) == 8200

    def test_sample_selects_correct_timestep(self):
        """sample() should return data from the nearest timestep."""
        grid = ECMWFGrid()
        _inject_timestep(grid, np.full((GRID_HEIGHT, GRID_WIDTH), 50, dtype=np.uint8), timestamp=1000)
        _inject_timestep(grid, np.full((GRID_HEIGHT, GRID_WIDTH), 100, dtype=np.uint8), timestamp=4600)

        lat = np.array([40.0])
        lon = np.array([-90.0])

        # Close to ts=1000
        assert grid.sample(lat, lon, timestamp=1200)[0] == 50
        # Close to ts=4600
        assert grid.sample(lat, lon, timestamp=4000)[0] == 100
        # None → latest (4600)
        assert grid.sample(lat, lon, timestamp=None)[0] == 100

    def test_bilinear_sample_smooths_between_pixels(self):
        """Bilinear sampling should produce intermediate values between source cells."""
        grid = ECMWFGrid()
        # Build a grid with a simple gradient: every column has a different value
        precip = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        precip[:, :] = 100  # uniform non-zero so no zero-fallback kicks in
        precip[:, GRID_WIDTH // 2] = 200
        _inject_timestep(grid, precip)

        # Sample at the lon corresponding to the boundary between 100 and 200
        # (col index slightly under WIDTH/2)
        lat = np.array([0.0])
        lon_center = WEST + (GRID_WIDTH // 2) * PIXEL_SIZE - PIXEL_SIZE * 0.5
        lon = np.array([lon_center])

        nearest = grid.sample(lat, lon, bilinear=False)
        bilinear = grid.sample(lat, lon, bilinear=True)

        # Bilinear should give an intermediate value (~150), nearest should be 100 or 200
        assert bilinear[0] != nearest[0]
        assert 100 < bilinear[0] < 200

    def test_bilinear_sample_avoids_zero_bleed(self):
        """Bilinear should fall back to nearest at the precip/clear-sky boundary."""
        grid = ECMWFGrid()
        precip = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        precip[:, GRID_WIDTH // 2:] = 200  # right half non-zero, left half zero
        _inject_timestep(grid, precip)

        # Sample lots of points across the boundary
        lats = np.zeros(100)
        lons = np.linspace(-1.0, 1.0, 100)
        result = grid.sample(lats, lons, bilinear=True)

        # No intermediate ghost values: every pixel is either 0 or 200
        unique = set(result.tolist())
        assert unique <= {0, 200}, f"Bilinear bled into clear sky: {unique}"

    def test_data_property_returns_latest(self):
        """The data property should return the latest timestep's precip grid."""
        grid = ECMWFGrid()
        early = np.full((GRID_HEIGHT, GRID_WIDTH), 50, dtype=np.uint8)
        late = np.full((GRID_HEIGHT, GRID_WIDTH), 100, dtype=np.uint8)
        _inject_timestep(grid, early, timestamp=1000)
        _inject_timestep(grid, late, timestamp=4600)

        assert (grid.data == 100).all()

    def test_select_valid_times_brackets_now(self):
        """Window should bracket current time, not end before it."""
        from datetime import datetime, timezone
        from unittest.mock import patch

        vt_list = [f"2026-04-05T{h:02d}:00Z" for h in range(1, 13)]

        # now=06:30Z, max_ts=3 → first vt >= now is 07Z → window [05, 06, 07]
        mock_now = datetime(2026, 4, 5, 6, 30, tzinfo=timezone.utc)
        with patch("librewxr.data.ecmwf_grid.datetime") as mock_dt, \
             patch("librewxr.data.ecmwf_grid.settings") as mock_settings:
            mock_dt.now.return_value = mock_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_settings.nowcast_enabled = False
            result = ECMWFGrid._select_valid_times(vt_list, max_ts=3)

        assert result == [
            "2026-04-05T05:00Z",
            "2026-04-05T06:00Z",
            "2026-04-05T07:00Z",
        ]

    def test_select_valid_times_nowcast_shifts_anchor(self):
        """When nowcast is enabled, anchor shifts forward by nowcast duration."""
        from datetime import datetime, timezone
        from unittest.mock import patch

        vt_list = [f"2026-04-05T{h:02d}:00Z" for h in range(1, 13)]

        # now=06:30Z, nowcast=6×600s=3600s → anchor at 07:30Z
        # first vt >= 07:30 is 08Z → window [06, 07, 08]
        mock_now = datetime(2026, 4, 5, 6, 30, tzinfo=timezone.utc)
        with patch("librewxr.data.ecmwf_grid.datetime") as mock_dt, \
             patch("librewxr.data.ecmwf_grid.settings") as mock_settings:
            mock_dt.now.return_value = mock_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_settings.nowcast_enabled = True
            mock_settings.nowcast_frames = 6
            mock_settings.fetch_interval = 600
            result = ECMWFGrid._select_valid_times(vt_list, max_ts=3)

        assert result == [
            "2026-04-05T06:00Z",
            "2026-04-05T07:00Z",
            "2026-04-05T08:00Z",
        ]

    def test_select_valid_times_all_future(self):
        """When all IFS hours are in the future, take the earliest ones."""
        from datetime import datetime, timezone
        from unittest.mock import patch

        # IFS run just released — all valid_times are after now
        vt_list = [f"2026-04-05T{h:02d}:00Z" for h in range(7, 19)]
        mock_now = datetime(2026, 4, 5, 6, 30, tzinfo=timezone.utc)
        with patch("librewxr.data.ecmwf_grid.datetime") as mock_dt, \
             patch("librewxr.data.ecmwf_grid.settings") as mock_settings:
            mock_dt.now.return_value = mock_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_settings.nowcast_enabled = False
            result = ECMWFGrid._select_valid_times(vt_list, max_ts=3)

        # First vt >= now is 07Z (idx 0), window shifts forward to fill
        assert result == [
            "2026-04-05T07:00Z",
            "2026-04-05T08:00Z",
            "2026-04-05T09:00Z",
        ]

    def test_select_valid_times_all_past(self):
        """When all IFS hours are in the past, take the latest ones."""
        from datetime import datetime, timezone
        from unittest.mock import patch

        vt_list = [f"2026-04-05T{h:02d}:00Z" for h in range(1, 5)]
        mock_now = datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc)
        with patch("librewxr.data.ecmwf_grid.datetime") as mock_dt, \
             patch("librewxr.data.ecmwf_grid.settings") as mock_settings:
            mock_dt.now.return_value = mock_now
            mock_dt.fromisoformat = datetime.fromisoformat
            mock_settings.nowcast_enabled = False
            result = ECMWFGrid._select_valid_times(vt_list, max_ts=3)

        # No vt >= now → anchor at last index → window [02, 03, 04]
        assert result == [
            "2026-04-05T02:00Z",
            "2026-04-05T03:00Z",
            "2026-04-05T04:00Z",
        ]

    def test_select_valid_times_few_available(self):
        """When fewer valid_times than max_ts, return all of them."""
        vt_list = ["2026-04-05T01:00Z", "2026-04-05T02:00Z"]
        result = ECMWFGrid._select_valid_times(vt_list, max_ts=3)
        assert result == vt_list


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
        _inject_timestep(grid, np.full((GRID_HEIGHT, GRID_WIDTH), 94, dtype=np.uint8))

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

    def test_ecmwf_fills_uncovered_pixels(self, monkeypatch):
        """ECMWF should fill pixels outside radar station coverage."""
        from librewxr.tiles import renderer
        from librewxr.data.regions import REGIONS

        # Patch sample_coverage to report "nothing is covered" → every
        # zero-valued pixel is eligible for ECMWF fill.
        monkeypatch.setattr(
            renderer, "sample_coverage",
            lambda name, lat, lon: np.zeros(lat.shape, dtype=bool),
        )

        grid = ECMWFGrid()
        _inject_timestep(grid, np.full((GRID_HEIGHT, GRID_WIDTH), 94, dtype=np.uint8))

        values = np.zeros((256, 256), dtype=np.uint8)
        result = renderer._fill_ecmwf_fallback(
            values, [REGIONS["USCOMP"]], z=4, x=3, y=5, tile_size=256,
            pad=0, ecmwf_grid=grid,
        )
        assert (result == 94).all()

    def test_ecmwf_preserves_covered_pixels(self, monkeypatch):
        """ECMWF must not overwrite pixels inside radar coverage.

        Including clear-sky (value 0) pixels inside coverage — they are
        "known-dry", not "unknown".
        """
        from librewxr.tiles import renderer
        from librewxr.data.regions import REGIONS

        # Left half of the tile is inside radar coverage; right half is outside.
        def fake_coverage(name, lat, lon):
            mask = np.zeros(lat.shape, dtype=bool)
            mask[:, :128] = True
            return mask

        monkeypatch.setattr(renderer, "sample_coverage", fake_coverage)

        grid = ECMWFGrid()
        _inject_timestep(grid, np.full((GRID_HEIGHT, GRID_WIDTH), 200, dtype=np.uint8))

        # All-zero radar values. Left half is "covered clear sky"; right half
        # is "outside coverage" and should be filled.
        values = np.zeros((256, 256), dtype=np.uint8)
        result = renderer._fill_ecmwf_fallback(
            values, [REGIONS["USCOMP"]], z=4, x=3, y=5, tile_size=256,
            pad=0, ecmwf_grid=grid,
        )
        assert (result[:, :128] == 0).all()
        assert (result[:, 128:] == 200).all()

    def test_ecmwf_preserves_nonzero_radar_pixels(self, monkeypatch):
        """Any non-zero radar pixel is left alone, covered or not."""
        from librewxr.tiles import renderer
        from librewxr.data.regions import REGIONS

        monkeypatch.setattr(
            renderer, "sample_coverage",
            lambda name, lat, lon: np.zeros(lat.shape, dtype=bool),
        )

        grid = ECMWFGrid()
        _inject_timestep(grid, np.full((GRID_HEIGHT, GRID_WIDTH), 200, dtype=np.uint8))

        values = np.zeros((256, 256), dtype=np.uint8)
        values[:, :128] = 50
        result = renderer._fill_ecmwf_fallback(
            values, [REGIONS["USCOMP"]], z=4, x=3, y=5, tile_size=256,
            pad=0, ecmwf_grid=grid,
        )
        assert (result[:, :128] == 50).all()
        assert (result[:, 128:] == 200).all()

    def test_snow_mask_used_for_coloring(self):
        """When snow=True, the renderer should use ECMWF snow mask."""
        from librewxr.tiles.renderer import render_tile

        grid = ECMWFGrid()
        _inject_timestep(
            grid,
            np.full((GRID_HEIGHT, GRID_WIDTH), 94, dtype=np.uint8),
            snow_mask=np.ones((GRID_HEIGHT, GRID_WIDTH), dtype=bool),
        )

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
