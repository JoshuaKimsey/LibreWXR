# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.tiles

from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH
from librewxr.tiles.renderer import render_coverage_tile, render_tile


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
