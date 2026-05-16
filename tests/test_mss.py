# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the simplified Singapore MSS 50 km radar source."""
import asyncio
import io
from datetime import datetime, timezone

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.sources

from librewxr.data.regions import REGIONS, RegionDef, resolve_regions
from librewxr.data.sources import (
    MSSSource,
    _MSS_PALETTE,
    _decode_mss_png,
)


# ─────────────────────────────────────────────────────────────────────
# Region definition
# ─────────────────────────────────────────────────────────────────────
class TestSgcompRegion:
    def test_sgcomp_in_regions(self):
        assert "SGCOMP" in REGIONS

    def test_sgcomp_group_and_proj(self):
        r = REGIONS["SGCOMP"]
        assert r.proj == "latlon"
        assert r.group == "SOUTHEAST_ASIA"

    def test_sgcomp_bounds_around_changi(self):
        # 50 km display product centred on MSS Changi (1.3521°N,
        # 103.8198°E).  Bounds derived from the image dimensions
        # (217×120 px at ~0.5 km/px) — must straddle the radar so the
        # rendered tiles align with Singapore's coastline.
        r = REGIONS["SGCOMP"]
        assert r.west < 103.8198 < r.east
        assert r.south < 1.3521 < r.north
        # Horizontal-leaning rectangle (~108 × 60 km).
        assert (r.east - r.west) > (r.north - r.south)

    def test_sgcomp_dimensions(self):
        r = REGIONS["SGCOMP"]
        assert r.width == 217
        assert r.height == 120

    def test_sgcomp_nonsquare_pixels(self):
        # The 50 km display crop uses square-ish pixels in km but not
        # in degrees — pixel_size_y must be set explicitly or the
        # renderer mis-aligns the latitude axis.
        r = REGIONS["SGCOMP"]
        assert r.pixel_size_y > 0

    def test_southeast_asia_group_resolution(self):
        # All three SE Asia regions must be listed by the group alias.
        assert resolve_regions("SOUTHEAST_ASIA") == [
            "SGCOMP", "MYPENINSULAR", "MYEAST",
        ]

    def test_all_includes_sgcomp(self):
        assert "SGCOMP" in resolve_regions("ALL")


# ─────────────────────────────────────────────────────────────────────
# URL construction
# ─────────────────────────────────────────────────────────────────────
class TestUrlForTimestamp:
    """MSS publishes filenames in Singapore local time (UTC+8) even
    though we request by UTC timestamp.  These tests pin the SGT
    conversion explicitly so the URL builder can't silently slip 8
    hours.
    """

    def _src(self) -> MSSSource:
        return MSSSource("https://example.test/files/rainarea/50km/v2")

    def test_rounds_down_to_5_min_boundary(self):
        # 2026-05-15 10:47:32 UTC → 10:45 UTC native → 18:45 SGT filename
        src = self._src()
        ts = int(datetime(
            2026, 5, 15, 10, 47, 32, tzinfo=timezone.utc,
        ).timestamp())
        url = src._url_for_timestamp(ts)
        assert url.endswith("dpsri_70km_2026051518450000dBR.dpsri.png"), url

    def test_exact_boundary_kept(self):
        # 10:30 UTC → 18:30 SGT
        src = self._src()
        ts = int(datetime(
            2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc,
        ).timestamp())
        url = src._url_for_timestamp(ts)
        assert url.endswith("dpsri_70km_2026051518300000dBR.dpsri.png")

    def test_top_of_hour_kept(self):
        # 11:00 UTC → 19:00 SGT
        src = self._src()
        ts = int(datetime(
            2026, 5, 15, 11, 0, 0, tzinfo=timezone.utc,
        ).timestamp())
        url = src._url_for_timestamp(ts)
        assert url.endswith("dpsri_70km_2026051519000000dBR.dpsri.png")

    def test_utc_evening_crosses_to_next_day_in_sgt(self):
        # 18:30 UTC = 02:30 SGT the NEXT day — date in filename must
        # roll over correctly.
        src = self._src()
        ts = int(datetime(
            2026, 5, 15, 18, 30, 0, tzinfo=timezone.utc,
        ).timestamp())
        url = src._url_for_timestamp(ts)
        assert url.endswith("dpsri_70km_2026051602300000dBR.dpsri.png"), url

    def test_base_url_is_preserved(self):
        src = self._src()
        ts = int(datetime(
            2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc,
        ).timestamp())
        assert src._url_for_timestamp(ts).startswith(
            "https://example.test/files/rainarea/50km/v2/"
        )

    def test_trailing_slash_stripped(self):
        src = MSSSource("https://example.test/dir/")
        ts = int(datetime(
            2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc,
        ).timestamp())
        assert "//" not in src._url_for_timestamp(ts).split("://", 1)[1]


# ─────────────────────────────────────────────────────────────────────
# Palette decoding
# ─────────────────────────────────────────────────────────────────────
class TestPaletteDecode:
    """The 50 km MSS PNG uses the same 31-stop discrete palette as the
    older 480 km product.  The decoder snaps each opaque pixel to its
    nearest anchor in RGB space and maps the rank to dBZ."""

    def _fake_region(self, w: int) -> RegionDef:
        # A tiny stand-in region — decode-correctness doesn't depend on
        # the real grid dimensions, but the decoder logs a shape
        # mismatch warning if w/h don't match.
        return RegionDef(
            name="SGCOMP", west=0.0, east=1.0, south=0.0, north=1.0,
            pixel_size=1.0, group="SOUTHEAST_ASIA",
            grid_width=w, grid_height=1,
        )

    def _make_png(
        self, pixels: list[tuple[int, int, int, int]],
    ) -> bytes:
        arr = np.array([pixels], dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _expected_uint8(self, dbz: float) -> int:
        return int(np.clip((dbz + 32.0) * 2.0, 0, 255).astype(np.uint8))

    def test_transparent_is_no_data(self):
        png = self._make_png([(0, 0, 0, 0)])
        out = _decode_mss_png(png, self._fake_region(1))
        assert out is not None
        assert out.shape == (1, 1)
        assert out[0, 0] == 0

    def test_each_palette_stop_decodes_to_expected_dbz(self):
        pixels = [(r, g, b, 255) for r, g, b, _ in _MSS_PALETTE]
        png = self._make_png(pixels)
        out = _decode_mss_png(png, self._fake_region(len(pixels)))
        assert out is not None
        for i, (_, _, _, dbz) in enumerate(_MSS_PALETTE):
            assert out[0, i] == self._expected_uint8(dbz), (
                f"stop {i} dBZ={dbz} decoded wrong"
            )

    def test_intensity_monotonic_across_palette(self):
        pixels = [(r, g, b, 255) for r, g, b, _ in _MSS_PALETTE]
        png = self._make_png(pixels)
        out = _decode_mss_png(png, self._fake_region(len(pixels)))
        diffs = np.diff(out[0].astype(int))
        assert np.all(diffs > 0), f"non-monotonic palette decode: {out[0]}"

    def test_near_anchor_within_tolerance_snaps(self):
        # ±1 channel perturbation should still snap to the nearest stop.
        r, g, b, dbz = _MSS_PALETTE[0]
        png = self._make_png([(r + 1, g - 1, b + 1, 255)])
        out = _decode_mss_png(png, self._fake_region(1))
        assert out[0, 0] == self._expected_uint8(dbz)

    def test_off_palette_color_is_no_data(self):
        # Pure black sits far from every anchor; should map to 0.
        png = self._make_png([(0, 0, 0, 255)])
        out = _decode_mss_png(png, self._fake_region(1))
        assert out[0, 0] == 0

    def test_bad_png_returns_none(self):
        assert _decode_mss_png(b"not a png", self._fake_region(1)) is None


# ─────────────────────────────────────────────────────────────────────
# Fetch + cache behaviour
# ─────────────────────────────────────────────────────────────────────
class _FakeResp:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content


def _make_native_png(value: int = 10) -> bytes:
    """Render a 217×120 RGBA PNG painted with one palette stop."""
    r, g, b, _ = _MSS_PALETTE[value]
    arr = np.zeros((120, 217, 4), dtype=np.uint8)
    arr[..., 0] = r
    arr[..., 1] = g
    arr[..., 2] = b
    arr[..., 3] = 255
    img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestMSSFetch:
    def test_returns_decoded_grid_on_200(self):
        src = MSSSource()
        png = _make_native_png(value=8)

        async def fake_retry_get(*args, **kwargs):
            return _FakeResp(200, content=png)

        import librewxr.data.mss_source as mss_mod
        original = mss_mod.retry_get
        mss_mod.retry_get = fake_retry_get
        try:
            grid = asyncio.run(
                src.fetch_frame(REGIONS["SGCOMP"], minutes_ago=0),
            )
        finally:
            mss_mod.retry_get = original

        assert grid is not None
        assert grid.shape == (120, 217)
        # All pixels carry the same palette stop → all the same uint8.
        expected = int(
            np.clip((_MSS_PALETTE[8][3] + 32.0) * 2.0, 0, 255),
        )
        assert (grid == expected).all()

    def test_404_returns_none_and_caches(self):
        src = MSSSource()
        call_count = {"n": 0}

        async def fake_retry_get(*args, **kwargs):
            call_count["n"] += 1
            return _FakeResp(404)

        import librewxr.data.mss_source as mss_mod
        original = mss_mod.retry_get
        mss_mod.retry_get = fake_retry_get
        try:
            async def run():
                a = await src.fetch_frame(
                    REGIONS["SGCOMP"], minutes_ago=0,
                )
                b = await src.fetch_frame(
                    REGIONS["SGCOMP"], minutes_ago=0,
                )
                return a, b
            results = asyncio.run(run())
        finally:
            mss_mod.retry_get = original

        assert results == (None, None)
        # Cached 404 must coalesce — second call hits the cache.
        assert call_count["n"] == 1

    def test_concurrent_requests_for_same_ts_coalesce(self):
        # Two concurrent calls for the same slot must share one HTTP
        # fetch via the per-ts lock.
        src = MSSSource()
        png = _make_native_png(value=4)
        call_count = {"n": 0}

        async def fake_retry_get(*args, **kwargs):
            call_count["n"] += 1
            # Tiny await yields to let both calls reach the lock first
            await asyncio.sleep(0.01)
            return _FakeResp(200, content=png)

        import librewxr.data.mss_source as mss_mod
        original = mss_mod.retry_get
        mss_mod.retry_get = fake_retry_get
        try:
            async def run():
                a_task = asyncio.create_task(
                    src.fetch_frame(REGIONS["SGCOMP"], minutes_ago=0),
                )
                b_task = asyncio.create_task(
                    src.fetch_frame(REGIONS["SGCOMP"], minutes_ago=0),
                )
                return await asyncio.gather(a_task, b_task)
            a, b = asyncio.run(run())
        finally:
            mss_mod.retry_get = original

        assert a is not None
        assert b is not None
        assert call_count["n"] == 1, (
            f"expected 1 fetch, got {call_count['n']}"
        )
