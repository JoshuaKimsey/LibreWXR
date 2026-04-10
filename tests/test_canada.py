# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io

import numpy as np
import pytest
from PIL import Image

from librewxr.data.regions import REGIONS, resolve_regions
from librewxr.data.sources import (
    _MSC_CANADA_PALETTE,
    _decode_msc_canada_png,
    _mmhr_to_dbz,
)
from librewxr.tiles.coordinates import tile_overlaps_region


class TestCanadaRegion:
    def test_cacomp_in_regions(self):
        assert "CACOMP" in REGIONS

    def test_cacomp_is_latlon(self):
        r = REGIONS["CACOMP"]
        assert r.proj == "latlon"
        assert r.group == "CANADA"

    def test_cacomp_bounds(self):
        r = REGIONS["CACOMP"]
        assert r.west == -141.0
        assert r.east == -52.0
        assert r.south == 41.0
        assert r.north == 84.0

    def test_cacomp_dimensions_reasonable(self):
        r = REGIONS["CACOMP"]
        # Must stay under typical WMS server size caps (~4096 on each axis)
        assert r.width < 4096
        assert r.height < 4096
        # Must be big enough to be useful
        assert r.width > 1000
        assert r.height > 500

    def test_canada_group_resolution(self):
        assert resolve_regions("CANADA") == ["CACOMP"]

    def test_all_includes_canada(self):
        assert "CACOMP" in resolve_regions("ALL")

    def test_mixed_regions(self):
        result = resolve_regions("CONUS,CANADA")
        assert "USCOMP" in result
        assert "CACOMP" in result


class TestMarshallPalmer:
    def test_known_values(self):
        # Z = 200 * R^1.6, dBZ = 10*log10(Z)
        # R=1 mm/h → Z=200 → dBZ=23.01
        assert _mmhr_to_dbz(np.array([1.0]))[0] == pytest.approx(23.01, abs=0.1)
        # R=100 mm/h → Z=200 * 100^1.6 = 200 * 10^3.2 ≈ 316228 → dBZ≈55.0
        assert _mmhr_to_dbz(np.array([100.0]))[0] == pytest.approx(55.0, abs=0.5)

    def test_monotonic(self):
        rates = np.array([0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0])
        dbz = _mmhr_to_dbz(rates)
        assert np.all(np.diff(dbz) > 0), "dBZ must be monotonic in rain rate"

    def test_nan_propagates(self):
        rates = np.array([1.0, np.nan, 5.0])
        dbz = _mmhr_to_dbz(rates)
        assert not np.isnan(dbz[0])
        assert np.isnan(dbz[1])
        assert not np.isnan(dbz[2])

    def test_zero_becomes_nan(self):
        # log(0) is undefined; zero rates must become no-data, not -inf
        dbz = _mmhr_to_dbz(np.array([0.0]))
        assert np.isnan(dbz[0])


class TestPaletteDecoder:
    def _make_png(self, pixels: list[tuple[int, int, int, int]]) -> bytes:
        """Build a tiny Nx1 RGBA PNG from a list of pixels."""
        arr = np.array([pixels], dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_transparent_becomes_zero(self):
        png = self._make_png([(0, 0, 0, 0), (0, 0, 0, 0)])
        out = _decode_msc_canada_png(png)
        assert out is not None
        assert out.shape == (1, 2)
        assert np.all(out == 0)

    def test_every_anchor_decodes(self):
        """Every palette anchor must round-trip to a non-zero uint8 dBZ."""
        pixels = [(r, g, b, 255) for r, g, b, _ in _MSC_CANADA_PALETTE]
        png = self._make_png(pixels)
        out = _decode_msc_canada_png(png)
        assert out is not None
        assert out.shape == (1, len(_MSC_CANADA_PALETTE))
        # All anchors encode rates ≥ 0.1 mm/h → dBZ ≥ 7 → uint8 ≥ 78
        assert np.all(out[0] > 0)
        # Higher rates must encode to higher uint8 values (monotonic palette)
        assert np.all(np.diff(out[0].astype(int)) > 0)

    def test_off_by_one_rgb_tolerance(self):
        """Composite pixels are within ±1 of legend colors — must still decode."""
        # Take anchor #1 (1.0 mm/h) and shift every channel by +1
        r, g, b, _ = _MSC_CANADA_PALETTE[1]
        png = self._make_png([(r + 1, g + 1, b + 1, 255)])
        out = _decode_msc_canada_png(png)
        # Should decode to the same value as the exact anchor
        exact_png = self._make_png([(r, g, b, 255)])
        exact = _decode_msc_canada_png(exact_png)
        assert out[0, 0] == exact[0, 0]

    def test_unknown_color_becomes_nodata(self):
        # Pure magenta (255, 0, 255) isn't in the palette and is far from
        # every anchor — should be treated as no-data.
        png = self._make_png([(255, 0, 255, 255)])
        out = _decode_msc_canada_png(png)
        assert out is not None
        assert out[0, 0] == 0

    def test_lowest_bucket_encoding(self):
        # Lowest bucket [0.1, 1.0) mm/h, geomean = 0.316 mm/h → ~15 dBZ
        # → uint8 = (15+32)*2 = 94.  Critically this is ABOVE the default
        # 10 dBZ noise floor (uint8 84), so light rain stays visible.
        r, g, b, _ = _MSC_CANADA_PALETTE[0]
        png = self._make_png([(r, g, b, 255)])
        out = _decode_msc_canada_png(png)
        assert out[0, 0] == pytest.approx(94, abs=2)

    def test_lowest_bucket_clears_noise_floor(self):
        # Explicit guard: the lowest bucket must round-trip to a dBZ
        # value clearly above the default noise_floor_dbz (10.0).
        r, g, b, _ = _MSC_CANADA_PALETTE[0]
        png = self._make_png([(r, g, b, 255)])
        out = _decode_msc_canada_png(png)
        dbz = (int(out[0, 0]) / 2) - 32
        assert dbz > 12.0, f"lowest bucket dBZ={dbz} too close to noise floor"

    def test_highest_bucket_encoding(self):
        # Highest bucket (≥200 mm/h, represented as 250) → ~61.3 dBZ
        # → uint8 = (61.3+32)*2 ≈ 187
        r, g, b, _ = _MSC_CANADA_PALETTE[-1]
        png = self._make_png([(r, g, b, 255)])
        out = _decode_msc_canada_png(png)
        assert out[0, 0] == pytest.approx(187, abs=2)

    def test_bad_png_returns_none(self):
        out = _decode_msc_canada_png(b"not a png")
        assert out is None


class TestCanadaTileOverlap:
    def test_tile_over_canada_overlaps(self):
        region = REGIONS["CACOMP"]
        # z=4, x=4, y=5 is over northern North America
        assert tile_overlaps_region(region, z=4, x=4, y=5)

    def test_tile_over_europe_does_not_overlap(self):
        region = REGIONS["CACOMP"]
        # z=4, x=8, y=5 is over central Europe
        assert not tile_overlaps_region(region, z=4, x=8, y=5)
