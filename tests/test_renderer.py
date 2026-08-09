# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.tiles

from librewxr.data.regions import REGIONS
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import (
    COMPOSITE_HEIGHT,
    COMPOSITE_WIDTH,
    compute_blur_radius,
)
from librewxr.tiles.png_palette import _PALETTE_MIN_COLORS, encode_png
from librewxr.tiles.renderer import (
    TileGeometry,
    _compute_nwp_only_geometry,
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
        # (5, 7, 12) is the tile that overlaps the sample data block
        # (tile (4, 3, 5) is empty there — see TestEmptyTileFastPath).
        regions = {"USCOMP": sample_frame_data}
        geom = compute_tile_geometry(
            regions, z=5, x=7, y=12, tile_size=256,
        )
        assert isinstance(geom, TileGeometry)
        assert not geom.is_transparent
        assert geom.values.max() > 0, "test fixture has no data at this tile"
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


class TestAdaptivePalettePng:
    """Adaptive lossless PNG8 encoding (see tiles/png_palette.py).

    Tiles with ``_PALETTE_MIN_COLORS``..256 unique RGBA colors are encoded
    as exact-palette P-mode PNGs (lossless, full 8-bit alpha via tRNS);
    everything else keeps the plain 32-bit RGBA encoding.  These tests
    prove losslessness (decode -> RGBA equals the input exactly) and the
    mode selection rules, both on ``encode_png`` directly and through the
    ``present_tile`` integration path.
    """

    @staticmethod
    def _banded_img(colors: int, width: int = 16) -> Image.Image:
        """One horizontal band per color: exactly ``colors`` unique pixels.

        ``(r, g)`` encodes the band index bijectively (r = i % 256, g =
        i // 256), so the fixture holds exactly ``colors`` unique colors
        even above 256.
        """
        arr = np.zeros((colors, width, 4), dtype=np.uint8)
        for i in range(colors):
            arr[i, :, :] = (
                i % 256,
                (i // 256) * 255,
                (i * 7) % 256,
                128 + (i * 3) % 128,
            )
        return Image.fromarray(arr, "RGBA")

    def test_single_color_stays_rgba(self):
        """1 unique color always takes the plain RGBA path."""
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        out = Image.open(io.BytesIO(encode_png(img)))
        assert out.mode == "RGBA"
        assert np.array_equal(np.asarray(out), np.asarray(img))

    def test_three_colors_with_partial_alpha(self):
        """3 colors incl. partial alpha -> palette path (threshold 2), and
        the decode round-trips to the exact input RGBA array."""
        arr = np.zeros((3, 16, 4), dtype=np.uint8)
        arr[0, :, :] = (255, 0, 0, 255)
        arr[1, :, :] = (0, 255, 0, 128)  # partial alpha
        arr[2, :, :] = (0, 0, 255, 64)   # more partial alpha
        img = Image.fromarray(arr, "RGBA")
        data = encode_png(img)
        out = Image.open(io.BytesIO(data))
        if _PALETTE_MIN_COLORS <= 3:
            assert out.mode == "P"
        else:
            assert out.mode == "RGBA"
        assert np.array_equal(np.asarray(out.convert("RGBA")), np.asarray(img))

    def test_forty_colors_round_trips(self):
        """~40 colors -> palette path, bit-exact round trip."""
        img = self._banded_img(40)
        data = encode_png(img)
        out = Image.open(io.BytesIO(data))
        assert out.mode == "P"  # 40 is well above any plausible threshold
        assert np.array_equal(np.asarray(out.convert("RGBA")), np.asarray(img))

    def test_300_colors_stays_rgba(self):
        """> 256 unique colors keep the plain RGBA path."""
        img = self._banded_img(300)
        uniq = len(np.unique(np.asarray(img).reshape(-1, 4), axis=0))
        assert uniq > 256, "fixture must exceed the 256-color palette cap"
        out = Image.open(io.BytesIO(encode_png(img)))
        assert out.mode == "RGBA"
        assert np.array_equal(np.asarray(out), np.asarray(img))

    def test_no_smooth_tile_encodes_as_palette(self, sample_frame_data, monkeypatch):
        """A no-smooth radar tile (few unique colors) must decode as mode P
        and match the lossless-WebP render pixel-for-pixel."""
        # Pin the shipped default (quality 100 = lossless) so this test does
        # not depend on ambient LIBREWXR_WEBP_QUALITY / .env values.
        from librewxr.config import settings
        monkeypatch.setattr(settings, "webp_quality", 100)
        regions = {"USCOMP": sample_frame_data}
        geom = compute_tile_geometry(
            regions, z=5, x=7, y=12, tile_size=256,
        )
        assert not geom.is_transparent
        assert geom.blur_radius == 0.0  # no smoothing -> palette-friendly
        png = present_tile(geom, color_scheme=2, fmt="png")
        png_img = Image.open(io.BytesIO(png))
        assert png_img.mode == "P"
        webp = present_tile(geom, color_scheme=2, fmt="webp")
        webp_img = Image.open(io.BytesIO(webp)).convert("RGBA")
        assert np.array_equal(
            np.asarray(png_img.convert("RGBA")), np.asarray(webp_img)
        )

    def test_smooth_tile_stays_rgba(self, sample_frame_data):
        """Blurred tiles have too many colors for a palette -> plain RGBA."""
        regions = {"USCOMP": sample_frame_data}
        geom = compute_tile_geometry(
            regions, z=5, x=7, y=12, tile_size=256, smooth=True,
        )
        png = present_tile(geom, color_scheme=2, fmt="png")
        assert Image.open(io.BytesIO(png)).mode == "RGBA"

    def test_png_present_is_deterministic(self, sample_frame_data):
        """Same geometry + params -> identical PNG bytes."""
        regions = {"USCOMP": sample_frame_data}
        geom = compute_tile_geometry(
            regions, z=5, x=7, y=12, tile_size=256,
        )
        first = present_tile(geom, color_scheme=2, fmt="png")
        second = present_tile(geom, color_scheme=2, fmt="png")
        assert first == second

    def test_transparent_tile_stays_rgba(self):
        """Fully transparent tile (1 color) stays on the RGBA path."""
        geom = TileGeometry.transparent(256)
        png = present_tile(geom, color_scheme=2, fmt="png")
        assert Image.open(io.BytesIO(png)).mode == "RGBA"

    def test_transparent_tile_is_compact(self):
        """A fully transparent 256x256 tile must stay small on any host.

        The RGBA fallback encodes at compress_level=6 (like the palette
        path) so its size does not depend on the host libz flavor: stock
        zlib and zlib-ng diverge ~4x at level 1 but converge at level 6
        (~250-350 B).  600 B is a safe host-independent bound.
        """
        geom = TileGeometry.transparent(256)
        png = present_tile(geom, color_scheme=2, fmt="png")
        assert len(png) < 600


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
            radii.append(compute_blur_radius(uscomp, z, x, y, 256))
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
        us_blur = compute_blur_radius(uscomp, z, us_x, us_y, 256)
        eu_blur = compute_blur_radius(opera, z, eu_x, eu_y, 256)
        assert eu_blur > us_blur, (
            f"coarser region should get more blur, USCOMP={us_blur:.2f} OPERA={eu_blur:.2f}"
        )

    def test_blur_capped_at_tile_eighth(self):
        """Blur must never exceed tile_size / 32 to avoid smearing cells."""
        opera = REGIONS["OPERA"]
        z = 12
        x, y = self._lonlat_to_tile(10.0, 50.0, z)
        r = compute_blur_radius(opera, z, x, y, 256)
        assert r <= 256 / 32 + 1e-6, f"blur {r} exceeded safety cap"


class TestEmptyTileFastPath:
    """The empty-tile fast path: transparent sentinel for precip-empty tiles.

    Tier 1 (post-sample) catches NWP-sampled-empty and nowcast-empty cases;
    Tier 2 (pre-sample) skips ``nwp_chain.sample`` entirely when the global
    precip mask (multi-mode only) reports no precip in the tile bbox — one
    mechanism for the past-radar, nowcast, and NWP-only paths.  Both must
    be display-exact: never transparent when the old code rendered a pixel.
    """

    # Tile (z=4, x=3, y=5) sits over empty composite space — the sample
    # frame fixture has no data there (verified: values.max() == 0 on the
    # pre-fast-path code), so it is the canonical "empty radar" tile.
    _EMPTY_TILE = (4, 3, 5)

    @staticmethod
    def _empty_uscomp() -> np.ndarray:
        return np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)

    @classmethod
    def _populated_uscomp(cls, value: int) -> np.ndarray:
        """USCOMP frame with ``value`` at every region pixel tile (4,3,5) samples."""
        from librewxr.data.regions import REGIONS
        from librewxr.tiles.coordinates import region_pixel_indices

        data = cls._empty_uscomp()
        row_idx, col_idx = region_pixel_indices(REGIONS["USCOMP"], 4, 3, 5, 256)
        in_bounds = (row_idx >= 0) & (col_idx >= 0)
        data[row_idx[in_bounds], col_idx[in_bounds]] = value
        return data

    @staticmethod
    def _gate_chain() -> MagicMock:
        chain = MagicMock()
        chain.has_data.return_value = True
        return chain

    @staticmethod
    def _gate_mask(has_precip: bool) -> MagicMock:
        mask = MagicMock()
        mask.has_precip_in_bbox.return_value = has_precip
        return mask

    def test_empty_frame_regions_no_nwp_is_transparent(self):
        """(a) No radar regions + no NWP -> transparent sentinel."""
        geom = compute_tile_geometry({}, z=3, x=0, y=3, tile_size=256, nwp_chain=None)
        assert geom.is_transparent is True

    def test_tier2_gate_skips_sample(self):
        """(b) Empty radar + no mask precip in bbox -> transparent, no sample."""
        z, x, y = self._EMPTY_TILE
        chain = self._gate_chain()
        mask = self._gate_mask(has_precip=False)
        geom = compute_tile_geometry(
            {"USCOMP": self._empty_uscomp()}, z, x, y, tile_size=256,
            nwp_chain=chain, frame_timestamp=1700000000, precip_mask=mask,
        )
        assert geom.is_transparent is True
        assert geom.fast_path == "tier2_mask_past"
        chain.sample.assert_not_called()
        mask.has_precip_in_bbox.assert_called_once()

    def test_tier2_gate_pass_through_fills(self, monkeypatch):
        """(c) Mask gate says precip possible -> falls through to the fill."""
        from librewxr.tiles import renderer as renderer_mod

        monkeypatch.setattr(
            renderer_mod, "sample_coverage",
            lambda name, lat, lon: np.zeros(lat.shape, dtype=bool),
        )
        z, x, y = self._EMPTY_TILE
        chain = self._gate_chain()
        chain.sample.return_value = np.full((256, 256), 100, dtype=np.uint8)
        mask = self._gate_mask(has_precip=True)
        geom = compute_tile_geometry(
            {"USCOMP": self._empty_uscomp()}, z, x, y, tile_size=256,
            nwp_chain=chain, frame_timestamp=1700000000, precip_mask=mask,
        )
        assert geom.is_transparent is False
        chain.sample.assert_called()
        assert (geom.values == 100).all()

    def test_tier2_nowcast_mask_skip(self):
        """(c2) Nowcast path (Tier 3) folded into the same mask gate."""
        z, x, y = self._EMPTY_TILE
        chain = self._gate_chain()
        mask = self._gate_mask(has_precip=False)
        geom = compute_tile_geometry(
            {"USCOMP": self._empty_uscomp()}, z, x, y, tile_size=256,
            nwp_chain=chain, frame_timestamp=1700000000, nowcast_blend=0.5,
            precip_mask=mask,
        )
        assert geom.is_transparent is True
        assert geom.fast_path == "tier2_mask_nowcast"
        chain.sample.assert_not_called()

    def test_radar_present_never_transparent(self):
        """(d) Any radar pixel >= threshold -> fast path never fires."""
        regions = {"USCOMP": self._populated_uscomp(255)}
        z, x, y = self._EMPTY_TILE

        # No NWP at all: Case A must not fire.
        geom = compute_tile_geometry(regions, z, x, y, tile_size=256, nwp_chain=None)
        assert geom.is_transparent is False
        assert (geom.values >= 84).any()

        # With NWP whose mask gate claims no precip: Tier 2 must not fire
        # either (the gate requires radar_empty).
        chain = self._gate_chain()
        chain.sample.return_value = np.zeros((256, 256), dtype=np.uint8)
        geom2 = compute_tile_geometry(
            regions, z, x, y, tile_size=256,
            nwp_chain=chain, frame_timestamp=1700000000,
        )
        assert geom2.is_transparent is False
        assert (geom2.values >= 84).any()

    def test_nowcast_empty_tier1_transparent(self):
        """(e) Nowcast blend all-zero (both_zero) -> Tier 1 transparent."""
        z, x, y = self._EMPTY_TILE
        chain = self._gate_chain()
        chain.sample.return_value = np.zeros((256, 256), dtype=np.uint8)
        geom = compute_tile_geometry(
            {"USCOMP": self._empty_uscomp()}, z, x, y, tile_size=256,
            nwp_chain=chain, frame_timestamp=1700000000, nowcast_blend=0.5,
        )
        assert geom.is_transparent is True

    def test_noise_floor_disabled_predicate(self, monkeypatch):
        """(f) Thresholding off -> low radar values are NOT "empty"."""
        from librewxr.config import settings

        # Values in [1, 84) — below threshold under default settings.
        regions = {"USCOMP": self._populated_uscomp(50)}
        z, x, y = self._EMPTY_TILE

        # Enabled (noise_floor_dbz=10.0 -> threshold 84): radar_empty,
        # no NWP -> Case A fast path.
        geom = compute_tile_geometry(regions, z, x, y, tile_size=256, nwp_chain=None)
        assert geom.is_transparent is True

        # Disabled (threshold 0): any pixel counts -> not empty.
        monkeypatch.setattr(settings, "noise_floor_dbz", -33.0)
        geom2 = compute_tile_geometry(regions, z, x, y, tile_size=256, nwp_chain=None)
        assert geom2.is_transparent is False
        assert (geom2.values == 50).any()

    def test_fast_path_bytes_match_transparent_sentinel(self):
        """(g) Fast-path geometries encode byte-identically to the sentinel."""
        z, x, y = self._EMPTY_TILE
        geom_a = compute_tile_geometry(
            {"USCOMP": self._empty_uscomp()}, z, x, y, tile_size=256, nwp_chain=None,
        )
        chain = self._gate_chain()
        mask = self._gate_mask(has_precip=False)
        geom_b = compute_tile_geometry(
            {"USCOMP": self._empty_uscomp()}, z, x, y, tile_size=256,
            nwp_chain=chain, frame_timestamp=1700000000, precip_mask=mask,
        )
        ref = TileGeometry.transparent(256)
        for fmt in ("png", "webp"):
            assert present_tile(geom_a, color_scheme=2, fmt=fmt) == present_tile(
                ref, color_scheme=2, fmt=fmt
            )
            assert present_tile(geom_b, color_scheme=2, fmt=fmt) == present_tile(
                ref, color_scheme=2, fmt=fmt
            )

    def test_noise_floor_still_zeroes_below_threshold_fill(self, monkeypatch):
        """(k) Regression: NWP fill below threshold is still zeroed post-refactor."""
        from librewxr.tiles import renderer as renderer_mod

        monkeypatch.setattr(
            renderer_mod, "sample_coverage",
            lambda name, lat, lon: np.zeros(lat.shape, dtype=bool),
        )
        z, x, y = self._EMPTY_TILE
        chain = self._gate_chain()
        arr = np.full((256, 256), 50, dtype=np.uint8)  # below threshold 84
        arr[0, 0] = 200  # above threshold: keeps the tile alive past Tier 1
        chain.sample.return_value = arr
        geom = compute_tile_geometry(
            {"USCOMP": self._empty_uscomp()}, z, x, y, tile_size=256,
            nwp_chain=chain, frame_timestamp=1700000000,
        )
        assert geom.is_transparent is False
        assert (geom.values == 50).sum() == 0, "below-threshold fill not zeroed"
        assert (geom.values == 200).sum() == 1

    def test_nwp_only_tier2_gate_skips_sample(self):
        """(l) _compute_nwp_only_geometry: no mask precip in bbox -> transparent."""
        chain = self._gate_chain()
        mask = self._gate_mask(has_precip=False)
        z, x, y = self._EMPTY_TILE
        geom = _compute_nwp_only_geometry(
            chain, z, x, y, tile_size=256, smooth=False, snow=False,
            frame_timestamp=1700000000, precip_mask=mask,
        )
        assert geom.is_transparent is True
        assert geom.fast_path == "tier2_mask_nwp_only"
        chain.sample.assert_not_called()

    def test_frame_regions_not_mutated(self):
        """(q) The radar-empty predicate never mutates the input frame."""
        regions = {"USCOMP": self._populated_uscomp(255)}
        data = regions["USCOMP"]
        snapshot = data.copy()
        z, x, y = self._EMPTY_TILE
        chain = self._gate_chain()
        chain.sample.return_value = np.zeros((256, 256), dtype=np.uint8)
        compute_tile_geometry(
            regions, z, x, y, tile_size=256,
            nwp_chain=chain, frame_timestamp=1700000000,
        )
        assert np.array_equal(data, snapshot), "frame_regions array was mutated"

    @pytest.mark.parametrize("scenario", [
        "no_regions_no_nwp",
        "tier2_mask_past",
        "case_a_no_nwp_empty_radar",
        "tier1_post_fill",
        "tier1_post_blend",
        "tier2_mask_nowcast",
        "tier2_mask_nwp_only",
        "tier1_nwp_only_post_sample",
    ])
    def test_fast_path_reason_propagation(self, scenario):
        """(m) Each fast-path return site labels its transparent geometry.

        The scenario name doubles as the expected ``fast_path`` reason so
        the parametrization is (input scenario, expected string) in one.
        """
        z, x, y = self._EMPTY_TILE
        ts = 1700000000
        empty = self._empty_uscomp()
        if scenario == "no_regions_no_nwp":
            geom = compute_tile_geometry(
                {}, z=3, x=0, y=3, tile_size=256, nwp_chain=None,
            )
        elif scenario == "tier2_mask_past":
            geom = compute_tile_geometry(
                {"USCOMP": empty}, z, x, y, tile_size=256,
                nwp_chain=self._gate_chain(), frame_timestamp=ts,
                precip_mask=self._gate_mask(has_precip=False),
            )
        elif scenario == "case_a_no_nwp_empty_radar":
            geom = compute_tile_geometry(
                {"USCOMP": empty}, z, x, y, tile_size=256, nwp_chain=None,
            )
        elif scenario == "tier1_post_fill":
            chain = self._gate_chain()
            chain.sample.return_value = np.zeros((256, 256), dtype=np.uint8)
            geom = compute_tile_geometry(
                {"USCOMP": empty}, z, x, y, tile_size=256,
                nwp_chain=chain, frame_timestamp=ts,
            )
        elif scenario == "tier1_post_blend":
            chain = self._gate_chain()
            chain.sample.return_value = np.zeros((256, 256), dtype=np.uint8)
            geom = compute_tile_geometry(
                {"USCOMP": empty}, z, x, y, tile_size=256,
                nwp_chain=chain, frame_timestamp=ts, nowcast_blend=0.5,
            )
        elif scenario == "tier2_mask_nowcast":
            geom = compute_tile_geometry(
                {"USCOMP": empty}, z, x, y, tile_size=256,
                nwp_chain=self._gate_chain(), frame_timestamp=ts,
                nowcast_blend=0.5,
                precip_mask=self._gate_mask(has_precip=False),
            )
        elif scenario == "tier2_mask_nwp_only":
            geom = _compute_nwp_only_geometry(
                self._gate_chain(), z, x, y, tile_size=256,
                smooth=False, snow=False, frame_timestamp=ts,
                precip_mask=self._gate_mask(has_precip=False),
            )
        else:  # tier1_nwp_only_post_sample
            chain = self._gate_chain()
            chain.sample.return_value = np.zeros((256, 256), dtype=np.uint8)
            geom = _compute_nwp_only_geometry(
                chain, z, x, y, tile_size=256,
                smooth=False, snow=False, frame_timestamp=ts,
            )
        assert geom.is_transparent is True
        assert geom.fast_path == scenario

    def test_non_transparent_geometry_fast_path_none(self, sample_frame_data):
        """(n) Real (non-transparent) geometries carry no fast-path label."""
        regions = {"USCOMP": sample_frame_data}
        geom = compute_tile_geometry(
            regions, z=5, x=7, y=12, tile_size=256,
        )
        assert geom.is_transparent is False
        assert geom.fast_path is None

    def test_transparent_default_fast_path_none(self):
        """(o) transparent() without fast_path leaves the field None."""
        assert TileGeometry.transparent(256).fast_path is None

    def test_fast_path_label_does_not_affect_present_bytes(self):
        """(p) fast_path is metadata only: labeled geometry presents identically."""
        z, x, y = self._EMPTY_TILE
        chain = self._gate_chain()
        mask = self._gate_mask(has_precip=False)
        geom_b = compute_tile_geometry(
            {"USCOMP": self._empty_uscomp()}, z, x, y, tile_size=256,
            nwp_chain=chain, frame_timestamp=1700000000, precip_mask=mask,
        )
        ref = TileGeometry.transparent(256)
        assert geom_b.fast_path is not None
        assert ref.fast_path is None
        for fmt in ("png", "webp"):
            assert present_tile(geom_b, color_scheme=2, fmt=fmt) == present_tile(
                ref, color_scheme=2, fmt=fmt
            )


class TestNowcastBlendGuard:
    """The nowcast blend must not let a dry model pixel erase real radar.

    Model pixel value 0 encodes -32 dBZ, the bottom of the encoding, NOT
    "no data" — so blending toward it drags real radar echoes below the
    display noise floor and the post-blend thresholding zeroes them
    (issue #24).  ``_blend_nowcast`` pins the radar weight to 1.0
    wherever the (blurred) model is below the floor.  These tests also
    pin that the NWP fill/blend only honors regions that actually
    delivered a frame this cycle (``regions_with_data``), so a down
    region can't suppress the model with its feather or block the NWP
    fill with its coverage mask.
    """

    # Tile (z=4, x=3, y=5) sits over empty composite space (same tile as
    # TestEmptyTileFastPath).  The exact location doesn't drive results
    # here — sample_feather / overlapping_regions are patched — but
    # keeping it consistent makes the fixtures deterministic.
    _Z, _X, _Y, _TILE = 4, 3, 5, 256

    @staticmethod
    def _chain(model_arr: np.ndarray) -> MagicMock:
        chain = MagicMock()
        chain.has_data.return_value = True
        chain.sample.return_value = model_arr
        return chain

    @pytest.fixture(autouse=True)
    def _pin_noise_floor(self, monkeypatch):
        from librewxr.config import settings
        monkeypatch.setattr(settings, "noise_floor_dbz", 10.0)  # threshold 84

    @staticmethod
    def _feather_all_ones() -> callable:
        return lambda name, lat, lon: np.ones(lat.shape, dtype=np.float32)

    def _blend(self, radar, model, blend_weight, monkeypatch):
        """Call _blend_nowcast with feather=1 everywhere, floor=10.0."""
        from librewxr.tiles import renderer as renderer_mod

        monkeypatch.setattr(renderer_mod, "sample_feather", self._feather_all_ones())
        return renderer_mod._blend_nowcast(
            radar, [REGIONS["USCOMP"]], self._Z, self._X, self._Y,
            self._TILE, 0, self._chain(model), blend_weight=blend_weight,
        )

    # --- Unit tests: _blend_nowcast directly --------------------------------

    def test_model_empty_preserves_radar(self, monkeypatch):
        """(1) Dry model (all 0) must not halve the radar value."""
        radar = np.full((256, 256), 104, dtype=np.uint8)
        model = np.zeros((256, 256), dtype=np.uint8)
        result = self._blend(radar, model, 0.5, monkeypatch)
        assert (result == 104).all()  # pre-fix: 52

    def test_model_above_floor_blends_normally(self, monkeypatch):
        """(2) Model with real precip still blends exactly as before."""
        radar = np.full((256, 256), 104, dtype=np.uint8)
        model = np.full((256, 256), 200, dtype=np.uint8)
        result = self._blend(radar, model, 0.5, monkeypatch)
        assert (result == 152).all()  # clip(0.5*104 + 0.5*200 + 0.5)

    def test_blur_fringe_does_not_erase_radar(self, monkeypatch):
        """(3) Gaussian fringe around a model echo must not eat radar."""
        radar = np.full((256, 256), 104, dtype=np.uint8)
        model = np.zeros((256, 256), dtype=np.uint8)
        model[128, 128] = 255
        result = self._blend(radar, model, 0.5, monkeypatch)
        # The 3x3 Gaussian leaves a ~32 fringe on the orthogonal
        # neighbours — below the 84 floor, so they must stay pure radar.
        for r, c in ((128, 127), (128, 129), (127, 128), (129, 128)):
            assert result[r, c] == 104  # pre-fix: 68

    def test_floor_disabled_keeps_old_behavior(self, monkeypatch):
        """(4) Noise floor disabled (-33) -> the guard is a no-op."""
        from librewxr.config import settings

        monkeypatch.setattr(settings, "noise_floor_dbz", -33.0)
        radar = np.full((256, 256), 104, dtype=np.uint8)
        model = np.zeros((256, 256), dtype=np.uint8)
        result = self._blend(radar, model, 0.5, monkeypatch)
        assert (result == 52).all()

    def test_model_mode_does_not_leak_radar(self, monkeypatch):
        """(5) blend_weight 0 ("model" mode) must stay pure model."""
        radar = np.full((256, 256), 104, dtype=np.uint8)
        model = np.zeros((256, 256), dtype=np.uint8)
        result = self._blend(radar, model, 0.0, monkeypatch)
        assert (result == 0).all()

    def test_both_zero_still_zero(self, monkeypatch):
        """(6) Radar and model both dry -> both_zero keeps the tile 0."""
        radar = np.zeros((256, 256), dtype=np.uint8)
        model = np.zeros((256, 256), dtype=np.uint8)
        result = self._blend(radar, model, 0.5, monkeypatch)
        assert (result == 0).all()

    # --- Compute-level regression tests for the regions_with_data fix -------

    @staticmethod
    def _patch_two_regions(monkeypatch):
        """Force the tile to overlap USCOMP + CACOMP; only USCOMP has a frame.

        ``sample_coverage`` and ``sample_feather`` are name-based: CACOMP
        reports full coverage / full feather (it "overlaps" this tile even
        though it delivered no frame this cycle), USCOMP reports nothing.
        """
        from librewxr.tiles import renderer as renderer_mod

        monkeypatch.setattr(
            renderer_mod, "overlapping_regions",
            lambda z, x, y, enabled=None: [
                REGIONS["USCOMP"], REGIONS["CACOMP"],
            ],
        )
        monkeypatch.setattr(
            renderer_mod, "sample_coverage",
            lambda name, lat, lon: (
                np.ones(lat.shape, dtype=bool)
                if name == "CACOMP" else np.zeros(lat.shape, dtype=bool)
            ),
        )
        monkeypatch.setattr(
            renderer_mod, "sample_feather",
            lambda name, lat, lon: (
                np.ones(lat.shape, dtype=np.float32)
                if name == "CACOMP" else np.zeros(lat.shape, dtype=np.float32)
            ),
        )

    @staticmethod
    def _empty_uscomp_frame() -> np.ndarray:
        return np.zeros((COMPOSITE_HEIGHT, COMPOSITE_WIDTH), dtype=np.uint8)

    def test_past_fill_ignores_data_less_region_coverage(self, monkeypatch):
        """(7) A down region's coverage must not block the NWP fallback fill."""
        from librewxr.tiles import renderer as renderer_mod

        self._patch_two_regions(monkeypatch)
        chain = self._chain(np.full((256, 256), 200, dtype=np.uint8))
        geom = renderer_mod.compute_tile_geometry(
            {"USCOMP": self._empty_uscomp_frame()},
            self._Z, self._X, self._Y, tile_size=256,
            nwp_chain=chain, frame_timestamp=1700000000,
        )
        assert geom.is_transparent is False  # pre-fix: tier1_post_fill
        assert (geom.values == 200).all()

    def test_nowcast_ignores_data_less_region_feather(self, monkeypatch):
        """(8) A down region's feather must not suppress the model blend."""
        from librewxr.tiles import renderer as renderer_mod

        self._patch_two_regions(monkeypatch)
        chain = self._chain(np.full((256, 256), 150, dtype=np.uint8))
        geom = renderer_mod.compute_tile_geometry(
            {"USCOMP": self._empty_uscomp_frame()},
            self._Z, self._X, self._Y, tile_size=256,
            nwp_chain=chain, frame_timestamp=1700000000, nowcast_blend=0.5,
        )
        assert geom.is_transparent is False  # pre-fix: tier1_post_blend
        assert (geom.values == 150).all()
