# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the MET Malaysia radar source (data/mmd_source.py)."""
import asyncio
import io
from datetime import datetime, timezone

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.sources

from librewxr.data.mmd_source import (
    MMDSource,
    _decode_mmd_frame,
    _decode_mmd_palette,
    _extract_region,
    _fill_boundary_gaps,
    _frame_timestamps,
    _MMD_EXPECTED_FRAMES,
    _MMD_EXPECTED_HEIGHT,
    _MMD_EXPECTED_WIDTH,
    _MMD_PALETTE,
    _MMD_SUBRECTS,
    _parse_last_modified,
)
from librewxr.data.regions import REGIONS, resolve_regions


# ─────────────────────────────────────────────────────────────────────
# Region definitions
# ─────────────────────────────────────────────────────────────────────
class TestMalaysiaRegions:
    def test_mypeninsular_in_regions(self):
        assert "MYPENINSULAR" in REGIONS

    def test_myeast_in_regions(self):
        assert "MYEAST" in REGIONS

    def test_mypeninsular_bounds_match_rainviewer_metadata(self):
        # Bounds come from data.rainviewer.com/images/MYCOMP71/0_products.json.
        # Off-by-much here would mis-register Peninsular Malaysia coastlines.
        r = REGIONS["MYPENINSULAR"]
        assert r.west == pytest.approx(96.92, abs=0.01)
        assert r.east == pytest.approx(106.28, abs=0.01)
        assert r.south == pytest.approx(-1.33, abs=0.01)
        assert r.north == pytest.approx(8.97, abs=0.01)
        assert r.group == "SOUTHEAST_ASIA"

    def test_myeast_bounds_match_rainviewer_metadata(self):
        # From MYCOMP72/0_products.json — covers Borneo + Brunei.
        r = REGIONS["MYEAST"]
        assert r.west == pytest.approx(107.08, abs=0.01)
        assert r.east == pytest.approx(121.19, abs=0.01)
        assert r.south == pytest.approx(-1.48, abs=0.01)
        assert r.north == pytest.approx(9.18, abs=0.01)
        assert r.group == "SOUTHEAST_ASIA"

    def test_mypeninsular_grid_matches_subrect(self):
        r = REGIONS["MYPENINSULAR"]
        y0, y1, x0, x1 = _MMD_SUBRECTS["MYPENINSULAR"]
        assert r.width == (x1 - x0)
        assert r.height == (y1 - y0)

    def test_myeast_grid_matches_subrect(self):
        r = REGIONS["MYEAST"]
        y0, y1, x0, x1 = _MMD_SUBRECTS["MYEAST"]
        assert r.width == (x1 - x0)
        assert r.height == (y1 - y0)

    def test_nonsquare_pixels(self):
        # The combined GIF renders both regions with finer latitude
        # resolution than longitude resolution; pixel_size_y must differ
        # from pixel_size or coastlines drift vertically.
        for name in ("MYPENINSULAR", "MYEAST"):
            r = REGIONS[name]
            assert r.pixel_size_y != 0.0
            assert r.pixel_size_y != r.pixel_size

    def test_southeast_asia_group_is_just_malaysia(self):
        # MET Malaysia is the sole source in this group after the MSS
        # Singapore removal.
        names = resolve_regions("SOUTHEAST_ASIA")
        assert names == ["MYPENINSULAR", "MYEAST"]

    def test_all_includes_malaysia(self):
        names = resolve_regions("ALL")
        assert "MYPENINSULAR" in names
        assert "MYEAST" in names


# ─────────────────────────────────────────────────────────────────────
# Palette decode
# ─────────────────────────────────────────────────────────────────────
class TestPaletteDecode:
    """The MMD GIF uses 18 discrete palette stops mapped to dBZ values
    by Marshall-Palmer.  The decoder snaps each pixel to the nearest
    stop by squared-RGB distance and emits the shared uint8 encoding."""

    def _expected_uint8(self, dbz: float) -> int:
        return int(np.clip((dbz + 32.0) * 2.0, 0, 255).astype(np.uint8))

    def test_each_palette_stop_decodes_to_expected_dbz(self):
        # Build a 1-row RGB image carrying every palette stop.
        rgb = np.array(
            [[(r, g, b) for r, g, b, _ in _MMD_PALETTE]], dtype=np.uint8,
        )
        decoded = _decode_mmd_palette(rgb)
        assert decoded.shape == (1, len(_MMD_PALETTE))
        for i, (_, _, _, dbz) in enumerate(_MMD_PALETTE):
            assert decoded[0, i] == self._expected_uint8(dbz), (
                f"palette stop {i} dBZ={dbz} decoded wrong (got {decoded[0, i]})"
            )

    def test_off_palette_pixel_is_nodata(self):
        # Mid-grey sits far from every palette stop (closest is white
        # at dist² ≈ 38000, well past the 64 tolerance).  Should map to 0.
        rgb = np.array([[(128, 128, 128)]], dtype=np.uint8)
        decoded = _decode_mmd_palette(rgb)
        assert decoded[0, 0] == 0

    def test_sea_blue_is_nodata(self):
        # The land/sea basemap colours must NOT match any precipitation
        # palette stop — otherwise the radar would falsely "rain" over
        # the ocean.  (100, 103, 175) is the sea blue observed in the
        # combined GIF.
        rgb = np.array([[(100, 103, 175)]], dtype=np.uint8)
        decoded = _decode_mmd_palette(rgb)
        assert decoded[0, 0] == 0

    def test_within_tolerance_snaps_to_nearest(self):
        # ±2 channel perturbation should still snap to the nearest stop.
        # Stop 12 is (0, 172, 0) — 1 mm/h green; channels have headroom
        # for both +2 and -2 perturbations.
        r, g, b, dbz = _MMD_PALETTE[12]
        rgb = np.array([[(r + 2, g - 1, b + 2)]], dtype=np.uint8)
        decoded = _decode_mmd_palette(rgb)
        assert decoded[0, 0] == self._expected_uint8(dbz)

    def test_decoded_palette_is_monotonic_in_intensity(self):
        # The palette is ordered top-to-bottom of the legend
        # (highest dBZ first).  Decoded uint8 values should strictly
        # decrease as we walk the table.
        rgb = np.array(
            [[(r, g, b) for r, g, b, _ in _MMD_PALETTE]], dtype=np.uint8,
        )
        decoded = _decode_mmd_palette(rgb)[0].astype(int)
        diffs = np.diff(decoded)
        assert np.all(diffs < 0), f"non-monotonic decode: {decoded.tolist()}"

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            _decode_mmd_palette(np.zeros((5, 5), dtype=np.uint8))


# ─────────────────────────────────────────────────────────────────────
# Sub-rectangle extraction
# ─────────────────────────────────────────────────────────────────────
class TestExtractRegion:
    def test_mypeninsular_shape(self):
        full = np.zeros((570, 1352, 3), dtype=np.uint8)
        sub = _extract_region(full, "MYPENINSULAR")
        r = REGIONS["MYPENINSULAR"]
        assert sub.shape == (r.height, r.width, 3)

    def test_myeast_shape(self):
        full = np.zeros((570, 1352, 3), dtype=np.uint8)
        sub = _extract_region(full, "MYEAST")
        r = REGIONS["MYEAST"]
        assert sub.shape == (r.height, r.width, 3)

    def test_sub_rectangles_dont_overlap(self):
        # MYPENINSULAR and MYEAST cover disjoint x-ranges separated by
        # the South China Sea gap — overlap would mean one region was
        # decoding the other's airspace.
        _, _, px0, px1 = _MMD_SUBRECTS["MYPENINSULAR"]
        _, _, ex0, ex1 = _MMD_SUBRECTS["MYEAST"]
        assert px1 <= ex0, "sub-rectangles overlap"

    def test_decode_frame_routes_to_region(self):
        # Fill the MYPENINSULAR sub-rect with one palette colour, the
        # MYEAST sub-rect with a different one.  Each region's decoded
        # grid should reflect ITS painted colour, not its peer's.
        full = np.zeros((570, 1352, 3), dtype=np.uint8)
        pen_color = np.array(_MMD_PALETTE[10][:3], dtype=np.uint8)   # 5 mm/h green
        east_color = np.array(_MMD_PALETTE[5][:3], dtype=np.uint8)   # 80 mm/h red-orange

        py0, py1, px0, px1 = _MMD_SUBRECTS["MYPENINSULAR"]
        ey0, ey1, ex0, ex1 = _MMD_SUBRECTS["MYEAST"]
        full[py0:py1, px0:px1] = pen_color
        full[ey0:ey1, ex0:ex1] = east_color

        pen_grid = _decode_mmd_frame(full, REGIONS["MYPENINSULAR"])
        east_grid = _decode_mmd_frame(full, REGIONS["MYEAST"])

        pen_expected = int(np.clip((_MMD_PALETTE[10][3] + 32.0) * 2.0, 0, 255))
        east_expected = int(np.clip((_MMD_PALETTE[5][3] + 32.0) * 2.0, 0, 255))
        assert (pen_grid == pen_expected).all()
        assert (east_grid == east_expected).all()

    def test_decode_unknown_region_raises(self):
        full = np.zeros((570, 1352, 3), dtype=np.uint8)
        from librewxr.data.regions import RegionDef
        bogus = RegionDef(
            name="BOGUS", west=0, east=1, south=0, north=1,
            pixel_size=1.0, group="X",
        )
        with pytest.raises(ValueError):
            _decode_mmd_frame(full, bogus)


# ─────────────────────────────────────────────────────────────────────
# Boundary-line gap fill
# ─────────────────────────────────────────────────────────────────────
class TestFillBoundaryGaps:
    """Burned-in state borders in the upstream GIF decode as no-data
    pixels.  Where a border crosses precipitation, this leaves a thin
    zero-stripe that the gap-fill is supposed to bridge."""

    def test_thin_line_through_precip_is_bridged(self):
        # 20×20 field of mid-intensity precip with a 1-px wide vertical
        # zero-stripe down the middle (simulating a state border line).
        grid = np.full((20, 20), 100, dtype=np.uint8)
        grid[:, 10] = 0
        filled = _fill_boundary_gaps(grid)
        # The stripe should be filled to roughly the surrounding value.
        assert (filled[:, 10] > 0).all()
        # Surrounding precipitation untouched.
        assert (filled[:, :10] == 100).all()
        assert (filled[:, 11:] == 100).all()

    def test_large_no_precip_area_untouched(self):
        # A 20×20 zero field with a small precip blob in one corner.
        # The close cannot bridge the large no-data area, so nothing
        # outside the blob's immediate neighbourhood should fill.
        grid = np.zeros((20, 20), dtype=np.uint8)
        grid[0:3, 0:3] = 80
        filled = _fill_boundary_gaps(grid)
        # Far-corner pixels stay zero — we don't want false rain.
        assert filled[19, 19] == 0
        assert filled[15, 15] == 0
        # Original blob preserved.
        assert (filled[0:3, 0:3] == 80).all()

    def test_no_precip_at_all_is_passthrough(self):
        grid = np.zeros((10, 10), dtype=np.uint8)
        filled = _fill_boundary_gaps(grid)
        assert (filled == 0).all()
        # And no exception when there are no gap pixels to fill.

    def test_isolated_precip_pixel_untouched(self):
        # A single precip pixel surrounded by zeros has no thin-gap
        # topology to bridge — should pass through unchanged.
        grid = np.zeros((10, 10), dtype=np.uint8)
        grid[5, 5] = 120
        filled = _fill_boundary_gaps(grid)
        assert filled[5, 5] == 120
        # Neighbouring zeros stay zero — close can't fill a 1-px blob's
        # holes because there are no holes.
        assert filled[0, 0] == 0


# ─────────────────────────────────────────────────────────────────────
# Last-Modified parsing
# ─────────────────────────────────────────────────────────────────────
class TestLastModified:
    def test_parses_rfc7231_format(self):
        ts = _parse_last_modified("Sat, 16 May 2026 06:21:05 GMT")
        assert ts == int(datetime(
            2026, 5, 16, 6, 21, 5, tzinfo=timezone.utc,
        ).timestamp())

    def test_none_returns_none(self):
        assert _parse_last_modified(None) is None

    def test_empty_returns_none(self):
        assert _parse_last_modified("") is None

    def test_malformed_returns_none(self):
        assert _parse_last_modified("not a date") is None


# ─────────────────────────────────────────────────────────────────────
# Frame timestamp derivation
# ─────────────────────────────────────────────────────────────────────
class TestFrameTimestamps:
    """Timestamps are labelled at the current wall-clock 10-min slot,
    not MET's real publish time.  MET publishes each slot ~11 min late,
    so anchoring on the real time would leave the renderer's current
    slot permanently empty.  Last-Modified is still consulted as a
    ceiling so a clearly-stale response doesn't get labelled as fresh."""

    def _utc(self, *args):
        return int(datetime(*args, tzinfo=timezone.utc).timestamp())

    def test_six_frames_returned(self):
        lm = self._utc(2026, 5, 16, 6, 21, 5)
        ts = _frame_timestamps(lm, 600, now_unix=lm)
        assert len(ts) == _MMD_EXPECTED_FRAMES

    def test_oldest_first_in_strict_10min_steps(self):
        lm = self._utc(2026, 5, 16, 6, 21, 5)
        ts = _frame_timestamps(lm, 600, now_unix=lm)
        diffs = np.diff(ts)
        assert all(d == 600 for d in diffs)

    def test_wall_clock_anchor_labels_newest_at_current_slot(self):
        # Production behaviour: at wall-clock 09:03 with a stale LM
        # from 09:01 (MET hasn't published 09:00 yet), the newest frame
        # gets labelled at 09:00 — the current wall-clock slot.
        now = self._utc(2026, 5, 16, 9, 3, 0)
        lm = self._utc(2026, 5, 16, 9, 1, 0)
        ts = _frame_timestamps(lm, 600, now_unix=now)
        assert ts[-1] == self._utc(2026, 5, 16, 9, 0, 0)
        assert ts[0] == self._utc(2026, 5, 16, 8, 10, 0)

    def test_stale_lm_does_not_label_as_fresh(self):
        # If LM is hours behind wall clock (which would happen with no
        # wall-clock shift), we'd anchor on it.  With wall-clock shift,
        # we relabel forward so the newest frame is always at "now".
        now = self._utc(2026, 5, 16, 12, 5, 0)
        lm = self._utc(2026, 5, 16, 6, 21, 5)
        ts = _frame_timestamps(lm, 600, now_unix=now)
        assert ts[-1] == self._utc(2026, 5, 16, 12, 0, 0)

    def test_future_lm_does_not_lie_about_freshness(self):
        # Reverse skew: if LM is from the future (client clock behind
        # server), trust LM rather than the wall-clock floor — otherwise
        # we'd label a fresh server response as older than it is.
        now = self._utc(2026, 5, 16, 9, 0, 0)
        lm = self._utc(2026, 5, 16, 9, 35, 0)
        ts = _frame_timestamps(lm, 0, now_unix=now)
        assert ts[-1] == self._utc(2026, 5, 16, 9, 30, 0)

    def test_publish_lag_zero_anchors_to_grid_floor(self):
        # With zero lag and LM == now, newest frame ts is just
        # floor(LM, 10min).
        lm = self._utc(2026, 5, 16, 6, 25, 30)
        ts = _frame_timestamps(lm, 0, now_unix=lm)
        assert ts[-1] == self._utc(2026, 5, 16, 6, 20, 0)


# ─────────────────────────────────────────────────────────────────────
# GIF decode pipeline
# ─────────────────────────────────────────────────────────────────────
def _make_test_gif(per_frame_colors: list[tuple]) -> bytes:
    """Build an animated GIF of expected MMD dimensions.

    ``per_frame_colors`` is a list of (peninsular_rgb, east_rgb) tuples
    — one per frame.  Each frame paints those colours into the relevant
    sub-rectangles so the decoder's region routing can be verified.

    A 1-pixel "frame index" marker in the chrome region of each frame
    keeps them byte-distinct so PIL's GIF encoder doesn't dedupe
    identical frames into a single frame on save.
    """
    frames = []
    for i, (pen_color, east_color) in enumerate(per_frame_colors):
        arr = np.zeros(
            (_MMD_EXPECTED_HEIGHT, _MMD_EXPECTED_WIDTH, 3), dtype=np.uint8,
        )
        py0, py1, px0, px1 = _MMD_SUBRECTS["MYPENINSULAR"]
        ey0, ey1, ex0, ex1 = _MMD_SUBRECTS["MYEAST"]
        arr[py0:py1, px0:px1] = pen_color
        arr[ey0:ey1, ex0:ex1] = east_color
        # Unique marker in the chrome panel (not in either sub-rect)
        # so this pixel never affects decoded region content.
        arr[5 + i, 1200] = (i * 30, i * 30, i * 30)
        frames.append(Image.fromarray(arr, mode="RGB"))

    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=1000, loop=0, disposal=2,
    )
    return buf.getvalue()


class _FakeResp:
    def __init__(self, status_code: int, content: bytes = b"",
                 headers: dict | None = None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class TestGifDecode:
    """End-to-end through the source's _decode_gif method — verifies
    timestamps, region routing, and palette decoding all hang together
    on a synthetic 1352×570×6 animated GIF."""

    def test_decode_populates_all_six_frame_timestamps(self):
        pen = _MMD_PALETTE[10][:3]   # green
        east = _MMD_PALETTE[5][:3]   # red-orange
        gif = _make_test_gif([(pen, east)] * 6)
        src = MMDSource()
        lm_unix = int(datetime(
            2026, 5, 16, 6, 21, 5, tzinfo=timezone.utc,
        ).timestamp())
        cache = src._decode_gif(gif, lm_unix)
        assert len(cache) == 6
        # Each frame should carry both regions decoded.
        for per_region in cache.values():
            assert set(per_region.keys()) == {"MYPENINSULAR", "MYEAST"}

    def test_decode_region_routing(self):
        # Different colours in the two sub-rects must come out as
        # different decoded dBZ values for the two regions.
        pen_color = _MMD_PALETTE[10][:3]    # 5 mm/h green
        east_color = _MMD_PALETTE[5][:3]    # 80 mm/h red-orange
        gif = _make_test_gif([(pen_color, east_color)] * 6)
        src = MMDSource()
        cache = src._decode_gif(gif, int(datetime(
            2026, 5, 16, 6, 21, 5, tzinfo=timezone.utc,
        ).timestamp()))
        pen_expected = int(np.clip((_MMD_PALETTE[10][3] + 32.0) * 2.0, 0, 255))
        east_expected = int(np.clip((_MMD_PALETTE[5][3] + 32.0) * 2.0, 0, 255))
        for per_region in cache.values():
            # Most pixels in each region are the painted colour; check
            # the dominant value to avoid edge-effect false negatives.
            pen_mode = np.bincount(per_region["MYPENINSULAR"].ravel()).argmax()
            east_mode = np.bincount(per_region["MYEAST"].ravel()).argmax()
            assert pen_mode == pen_expected
            assert east_mode == east_expected

    def test_wrong_size_gif_raises(self):
        # A 100×100 GIF must not be decoded silently — that means the
        # upstream endpoint changed.
        frame = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        buf = io.BytesIO()
        frame.save(buf, format="GIF")
        src = MMDSource()
        with pytest.raises(ValueError):
            src._decode_gif(buf.getvalue(), 0)


# ─────────────────────────────────────────────────────────────────────
# Source fetch + cache
# ─────────────────────────────────────────────────────────────────────
class TestMMDSource:
    def test_url_for_default_base(self):
        src = MMDSource()
        assert src._gif_url == (
            "https://api.met.gov.my/static/images/radar-latest.gif"
        )

    def test_url_strips_trailing_slash_in_base(self):
        src = MMDSource("https://example.test/")
        assert src._gif_url == (
            "https://example.test/static/images/radar-latest.gif"
        )

    def test_fetch_returns_none_when_no_refresh_yet(self):
        # Brand-new source, fetch_frame called with a future-leading ts:
        # cache is empty and the throttle blocks immediate refresh only
        # if _last_fetch_unix was just set — on a cold source it is 0,
        # so refresh runs.  We mock that path with a 404 response so
        # the cache stays empty and the call returns None.
        src = MMDSource()

        async def fake_retry_get(*args, **kwargs):
            return _FakeResp(404)

        import librewxr.data.mmd_source as mmd_mod
        original = mmd_mod.retry_get
        mmd_mod.retry_get = fake_retry_get
        try:
            result = asyncio.run(
                src.fetch_frame(REGIONS["MYPENINSULAR"], minutes_ago=0)
            )
        finally:
            mmd_mod.retry_get = original
        assert result is None

    def test_fetch_uses_cached_decode_after_first_refresh(self):
        # After one successful refresh, repeated fetch_frame calls
        # within the TTL must reuse the cache without re-fetching.
        src = MMDSource()
        pen = _MMD_PALETTE[10][:3]
        east = _MMD_PALETTE[5][:3]
        gif = _make_test_gif([(pen, east)] * 6)
        # Use a Last-Modified that's "now-ish" so the newest frame
        # timestamp lands at a slot fetch_frame will look up.
        now = int(datetime.now(timezone.utc).timestamp())
        lm_unix = (now // 600) * 600 + 60  # one min past the boundary
        from email.utils import formatdate
        lm_header = formatdate(lm_unix, usegmt=True)

        call_count = {"n": 0}

        async def fake_retry_get(*args, **kwargs):
            call_count["n"] += 1
            return _FakeResp(
                200, content=gif, headers={"Last-Modified": lm_header},
            )

        import librewxr.data.mmd_source as mmd_mod
        original = mmd_mod.retry_get
        mmd_mod.retry_get = fake_retry_get
        try:
            async def run():
                # First call: triggers refresh.
                a = await src.fetch_frame(
                    REGIONS["MYPENINSULAR"], minutes_ago=0,
                )
                b = await src.fetch_frame(
                    REGIONS["MYEAST"], minutes_ago=0,
                )
                c = await src.fetch_frame(
                    REGIONS["MYPENINSULAR"], minutes_ago=10,
                )
                return a, b, c

            results = asyncio.run(run())
        finally:
            mmd_mod.retry_get = original

        # All three calls should return frames (or None — but at least
        # one of them must be non-None: the slot matching the GIF's
        # newest frame timestamp).
        non_none = sum(r is not None for r in results)
        assert non_none >= 1, "expected at least one cache hit"
        # Critically, only ONE HTTP fetch must have happened.
        assert call_count["n"] == 1, (
            f"expected 1 HTTP fetch, got {call_count['n']}"
        )
