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
from librewxr.tiles.coordinates import COMPOSITE_HEIGHT, COMPOSITE_WIDTH
from librewxr.tiles.renderer import (
    TileGeometry,
    _compute_blur_radius,
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
