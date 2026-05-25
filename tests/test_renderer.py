# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.tiles

from librewxr.data.regions import REGIONS
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH
from librewxr.tiles.renderer import (
    TileGeometry,
    _compute_blur_radius,
    compute_tile_geometry,
    present_tile,
    render_coverage_tile,
    render_tile,
)


class TestRenderTile:
    def test_transparent_outside_conus(self):
        """Tiles outside CONUS should be fully transparent."""
        data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
        regions = {"USCOMP": data}
        # Tile over Pacific Ocean (zoom 3, x=0, y=3)
        tile = render_tile(regions, z=3, x=0, y=3, tile_size=256, color_scheme=2)
        img = Image.open(io.BytesIO(tile))
        assert img.size == (256, 256)
        assert img.mode == "RGBA"

    def test_render_valid_tile(self, sample_frame_data):
        """A tile over CONUS with data should produce a valid image."""
        regions = {"USCOMP": sample_frame_data}
        tile = render_tile(
            regions, z=4, x=3, y=5,
            tile_size=256, color_scheme=2,
        )
        img = Image.open(io.BytesIO(tile))
        assert img.size == (256, 256)
        assert img.mode == "RGBA"
        assert len(tile) > 0

    def test_render_512_tile(self, sample_frame_data):
        regions = {"USCOMP": sample_frame_data}
        tile = render_tile(
            regions, z=4, x=3, y=5,
            tile_size=512, color_scheme=2,
        )
        img = Image.open(io.BytesIO(tile))
        assert img.size == (512, 512)

    def test_render_webp(self, sample_frame_data):
        regions = {"USCOMP": sample_frame_data}
        tile = render_tile(
            regions, z=4, x=3, y=5,
            tile_size=256, color_scheme=2, fmt="webp",
        )
        img = Image.open(io.BytesIO(tile))
        assert img.size == (256, 256)

    def test_render_with_smooth(self, sample_frame_data):
        regions = {"USCOMP": sample_frame_data}
        tile = render_tile(
            regions, z=4, x=3, y=5,
            tile_size=256, color_scheme=2, smooth=True,
        )
        img = Image.open(io.BytesIO(tile))
        assert img.size == (256, 256)

    def test_all_color_schemes(self, sample_frame_data):
        """All color schemes should produce valid tiles."""
        regions = {"USCOMP": sample_frame_data}
        for scheme in [0, 1, 2, 3, 4, 5, 6, 7, 8, 255]:
            tile = render_tile(
                regions, z=4, x=3, y=5,
                tile_size=256, color_scheme=scheme,
            )
            img = Image.open(io.BytesIO(tile))
            assert img.size == (256, 256), f"Scheme {scheme} failed"


class TestRenderCoverageTile:
    def test_coverage_empty_data(self):
        data = np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)
        regions = {"USCOMP": data}
        tile = render_coverage_tile(regions, z=4, x=3, y=5, tile_size=256)
        img = Image.open(io.BytesIO(tile))
        assert img.size == (256, 256)

    def test_coverage_with_data(self, sample_frame_data):
        regions = {"USCOMP": sample_frame_data}
        tile = render_coverage_tile(regions, z=4, x=3, y=5, tile_size=256)
        img = Image.open(io.BytesIO(tile))
        assert img.size == (256, 256)


class TestTileGeometryCache:
    """The compute/present split is what lets one cached tile serve every
    color scheme + format + arrow style.  These tests pin that contract."""

    def test_compute_returns_geometry(self, sample_frame_data):
        regions = {"USCOMP": sample_frame_data}
        geom = compute_tile_geometry(
            regions, z=4, x=3, y=5, tile_size=256,
        )
        assert isinstance(geom, TileGeometry)
        assert not geom.is_transparent
        assert geom.values.shape == (256, 256)
        assert geom.values.dtype == np.uint8
        assert geom.snow_mask is None  # snow defaults to False
        assert geom.blur_radius == 0.0
        assert geom.pad == 0

    def test_transparent_when_no_data(self):
        """Tiles with no radar AND no NWP return the transparent sentinel."""
        regions = {}  # no radar data
        geom = compute_tile_geometry(
            regions, z=3, x=0, y=3, tile_size=256, nwp_chain=None,
        )
        assert geom.is_transparent

    def test_present_handles_transparent(self):
        """Transparent geometry should encode to a transparent tile of the right size."""
        geom = TileGeometry.transparent(256)
        for fmt in ("png", "webp"):
            tile = present_tile(geom, color_scheme=2, fmt=fmt)
            img = Image.open(io.BytesIO(tile))
            assert img.size == (256, 256)
            assert img.mode == "RGBA"

    def test_one_geometry_serves_all_color_schemes(self, sample_frame_data):
        """The whole point of the refactor: compute once, present in any color."""
        regions = {"USCOMP": sample_frame_data}
        # (5, 7, 12) is the tile that overlaps the sample data block.
        geom = compute_tile_geometry(
            regions, z=5, x=7, y=12, tile_size=256,
        )
        assert geom.values.max() > 0, "test fixture has no data at this tile"
        rendered = {}
        for scheme in (0, 1, 2, 3, 4, 5, 6, 7, 8):
            tile = present_tile(geom, color_scheme=scheme, fmt="png")
            assert len(tile) > 0, f"scheme {scheme} produced no bytes"
            img = Image.open(io.BytesIO(tile))
            assert img.size == (256, 256)
            rendered[scheme] = tile
        # Different schemes should not all produce identical bytes —
        # otherwise the LUT isn't actually being applied.
        unique = len(set(rendered.values()))
        assert unique >= 2, "color schemes produced identical bytes"

    def test_one_geometry_serves_both_formats(self, sample_frame_data):
        """PNG and WebP from the same geometry should both decode cleanly."""
        regions = {"USCOMP": sample_frame_data}
        geom = compute_tile_geometry(
            regions, z=5, x=7, y=12, tile_size=256,
        )
        png_bytes = present_tile(geom, color_scheme=2, fmt="png")
        webp_bytes = present_tile(geom, color_scheme=2, fmt="webp")
        assert Image.open(io.BytesIO(png_bytes)).size == (256, 256)
        assert Image.open(io.BytesIO(webp_bytes)).size == (256, 256)
        assert png_bytes != webp_bytes

    def test_cache_accepts_geometry(self, sample_frame_data):
        """TileCache must size and store TileGeometry like it does bytes."""
        cache = TileCache(max_mb=10)
        regions = {"USCOMP": sample_frame_data}
        geom = compute_tile_geometry(
            regions, z=4, x=3, y=5, tile_size=256,
        )
        key = (1700000000, 4, 3, 5, 256, False, False)
        cache.put(key, geom)
        assert cache.get(key) is geom
        assert cache.total_bytes == geom.nbytes
        # Eviction by timestamp should free the right number of bytes.
        cache.invalidate_timestamp(1700000000)
        assert cache.total_bytes == 0
        assert cache.get(key) is None


class TestBlurRadius:
    """Blur radius must scale with how many tile pixels a region pixel covers."""

    @staticmethod
    def _lonlat_to_tile(lon, lat, z):
        import math
        n = 2 ** z
        x = int((lon + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    @pytest.fixture(autouse=True)
    def _pin_smooth_radius(self, monkeypatch):
        from librewxr.config import settings
        monkeypatch.setattr(settings, "smooth_radius", 1.0)

    def test_blur_grows_with_zoom(self):
        """At high zoom, more tile pixels per region pixel → larger blur."""
        uscomp = REGIONS["USCOMP"]
        radii = []
        for z in (5, 8, 11):
            x, y = self._lonlat_to_tile(-90.0, 35.0, z)  # Memphis-ish
            radii.append(_compute_blur_radius(uscomp, z, x, y, 256))
        assert radii[0] <= radii[1] < radii[2], (
            f"blur should grow monotonically with zoom, got {radii}"
        )

    def test_blur_larger_for_coarse_region(self):
        """At the same zoom, a coarser region should get more blur."""
        uscomp = REGIONS["USCOMP"]  # 0.005° (~500 m)
        opera = REGIONS["OPERA"]  # 2 km LAEA — 4× coarser
        z = 10
        us_x, us_y = self._lonlat_to_tile(-90.0, 35.0, z)
        eu_x, eu_y = self._lonlat_to_tile(10.0, 50.0, z)
        us_blur = _compute_blur_radius(uscomp, z, us_x, us_y, 256)
        eu_blur = _compute_blur_radius(opera, z, eu_x, eu_y, 256)
        assert eu_blur > us_blur, (
            f"coarser region should get more blur, USCOMP={us_blur:.2f} OPERA={eu_blur:.2f}"
        )

    def test_blur_capped_at_tile_eighth(self):
        """Blur must never exceed tile_size / 32 to avoid smearing cells."""
        opera = REGIONS["OPERA"]
        z = 12
        x, y = self._lonlat_to_tile(10.0, 50.0, z)
        r = _compute_blur_radius(opera, z, x, y, 256)
        assert r <= 256 / 32 + 1e-6, f"blur {r} exceeded safety cap"
