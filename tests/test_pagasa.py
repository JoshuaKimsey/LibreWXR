# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the PAGASA Philippines radar source (sources/regional/
southeast_asia/philippines/radar/pagasa/)."""
import asyncio
import io
import json
from datetime import datetime, timezone

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.sources

from librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.source import (
    PAGASASource,
    _MAX_NEAREST_OFFSET_SEC,
    _PAGASA_CLUTTER_STOPS,
    _PAGASA_EXPECTED_FRAMES,
    _PAGASA_EXPECTED_HEIGHT,
    _PAGASA_EXPECTED_WIDTH,
    _PAGASA_NATIVE_CADENCE_SEC,
    _PAGASA_PALETTE,
    _build_palette_arrays,
    _decode_pagasa_png,
)
from librewxr.data.regions import REGIONS, resolve_regions


# ─────────────────────────────────────────────────────────────────────
# Region definition
# ─────────────────────────────────────────────────────────────────────
class TestPhilippinesRegion:
    def test_phcomp_in_regions(self):
        assert "PHCOMP" in REGIONS

    def test_phcomp_bounds_match_panahon_js_bundle(self):
        # Bounds come from the PANAHON web app's JS bundle
        # (``leftBottom``/``rightTop`` for the composite view).  Off-by-
        # much here would mis-register Philippine coastlines.
        r = REGIONS["PHCOMP"]
        assert r.west == pytest.approx(115.4155, abs=0.001)
        assert r.east == pytest.approx(129.5173, abs=0.001)
        assert r.south == pytest.approx(3.8017, abs=0.001)
        assert r.north == pytest.approx(22.4585, abs=0.001)
        assert r.group == "SOUTHEAST_ASIA"

    def test_phcomp_native_grid_is_2048_square(self):
        # The PNG endpoint serves a 2048×2048 RGBA at the composite
        # bounds.  Region size must match exactly so tiles register
        # against PAGASA's pixel grid 1:1.
        r = REGIONS["PHCOMP"]
        assert r.width == 2048
        assert r.height == 2048

    def test_phcomp_pixel_size_is_anisotropic(self):
        # The 2048×2048 PNG covers a wider lat span (18.66°) than
        # lon span (14.10°), so pixels are non-square in geographic
        # units — pixel_size_y must be set explicitly.
        r = REGIONS["PHCOMP"]
        assert r._ps_y > r.pixel_size
        assert r.pixel_size == pytest.approx(0.006886, abs=1e-5)
        assert r._ps_y == pytest.approx(0.009110, abs=1e-5)

    def test_southeast_asia_includes_phcomp(self):
        names = resolve_regions("SOUTHEAST_ASIA")
        assert "PHCOMP" in names

    def test_all_includes_phcomp(self):
        names = resolve_regions("ALL")
        assert "PHCOMP" in names


# ─────────────────────────────────────────────────────────────────────
# Station inventory
# ─────────────────────────────────────────────────────────────────────
class TestStationInventory:
    def test_nine_stations_including_davao(self):
        # The app's current station list has NINE radars (Davao added
        # since May 2026): Baguio, Baler, Echague, Daet, Guiuan, Iloilo,
        # Davao, Kabacan, Panabo.
        from librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.stations import (
            PHCOMP_STATIONS,
        )
        assert len(PHCOMP_STATIONS) == 9

    def test_davao_station_present(self):
        from librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.stations import (
            PHCOMP_STATIONS,
        )
        names_and_coords = PHCOMP_STATIONS
        # Davao is the last Mindanao-side entry; check a coordinate
        # close to the PAGASA Davao radar site (~7.13, 125.65).
        davao = names_and_coords[6]
        assert davao[0] == pytest.approx(7.13, abs=0.05)
        assert davao[1] == pytest.approx(125.65, abs=0.05)


# ─────────────────────────────────────────────────────────────────────
# Palette decode
# ─────────────────────────────────────────────────────────────────────
class TestPaletteDecode:
    def test_palette_is_thirteen_stops(self):
        # The PANAHON JS bundle's
        # ``generateGradientColormap([...], 13, [0, 75])`` produces a
        # 13-stop linear ramp.  Adding/removing stops here will desync
        # the decoder from the upstream colour mapping.
        assert len(_PAGASA_PALETTE) == 13

    def test_palette_dbz_is_linear_zero_to_seventyfive(self):
        # 0–75 dBZ in 13 stops = 6.25 dBZ per stop.  This is the
        # spec encoded in the JS bundle; verify our table matches.
        for i, (_, _, _, dbz) in enumerate(_PAGASA_PALETTE):
            assert dbz == pytest.approx(i * 75.0 / 12.0, abs=1e-6)

    def test_clutter_stops_are_grays(self):
        # The first three stops are the weak-echo greys (#9c9c9c,
        # #b4b4b4, #c8c8c8) that we treat as no-data downstream.
        for i in range(_PAGASA_CLUTTER_STOPS):
            r, g, b, _ = _PAGASA_PALETTE[i]
            assert r == g == b

    def test_transparent_pixels_decode_to_nodata(self):
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[..., :3] = 0xff
        rgba[..., 3] = 0
        out = _decode_pagasa_png(rgba)
        assert out.dtype == np.uint8
        assert out.shape == (4, 4)
        assert (out == 0).all()

    def test_precip_pixels_decode_to_palette_dbz(self):
        # Paint a single yellow pixel (#ffff00, dBZ=37.5) at α=255.
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[1, 2] = (0xff, 0xff, 0x00, 0xff)
        out = _decode_pagasa_png(rgba)
        expected = int(np.clip((37.5 + 32.0) * 2.0, 0, 255))
        assert out[1, 2] == expected
        # Other pixels are transparent → no-data.
        assert out.sum() == expected

    def test_clutter_grays_decode_to_nodata(self):
        # All three clutter grays must decode to no-data even when
        # painted with non-zero alpha, matching the alpha-encoded
        # weak-echo flag from the source PNG.
        for i in range(_PAGASA_CLUTTER_STOPS):
            r, g, b, _ = _PAGASA_PALETTE[i]
            rgba = np.zeros((2, 2, 4), dtype=np.uint8)
            rgba[0, 0] = (r, g, b, 200)
            out = _decode_pagasa_png(rgba)
            assert out[0, 0] == 0, f"clutter stop {i} should decode to 0"

    def test_each_precip_stop_decodes_distinctly(self):
        # Every non-clutter palette stop must map to a unique uint8
        # encoded dBZ value — collisions would silently merge bins.
        encoded = set()
        for i in range(_PAGASA_CLUTTER_STOPS, len(_PAGASA_PALETTE)):
            r, g, b, dbz = _PAGASA_PALETTE[i]
            rgba = np.zeros((1, 1, 4), dtype=np.uint8)
            rgba[0, 0] = (r, g, b, 255)
            out = _decode_pagasa_png(rgba)
            encoded.add(int(out[0, 0]))
        assert len(encoded) == len(_PAGASA_PALETTE) - _PAGASA_CLUTTER_STOPS

    def test_wrong_channel_count_raises(self):
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            _decode_pagasa_png(rgb)


# ─────────────────────────────────────────────────────────────────────
# Timeline + image decode end-to-end
# ─────────────────────────────────────────────────────────────────────
def _make_test_png(palette_index: int) -> bytes:
    """Build a 2048×2048 RGBA PNG filled with one palette colour."""
    r, g, b, _ = _PAGASA_PALETTE[palette_index]
    rgba = np.zeros(
        (_PAGASA_EXPECTED_HEIGHT, _PAGASA_EXPECTED_WIDTH, 4), dtype=np.uint8,
    )
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = 255
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _make_timeline(start_unix: int, count: int = 6) -> dict:
    """Build a timeline payload mirroring the PANAHON API shape."""
    entries = []
    for i in range(count):
        ts = start_unix + i * _PAGASA_NATIVE_CADENCE_SEC
        entries.append({
            "observed_at": "2026-05-16 00:00:00",
            "observed_at_unix": ts,
            "image_url": f"http://panahon.gov.ph/api/v1/radar?id={i}",
        })
    return {"success": True, "data": {"timeline": entries}}


class _FakeResp:
    def __init__(
        self,
        status_code: int,
        content: bytes = b"",
        payload: dict | None = None,
        headers: dict | None = None,
    ):
        self.status_code = status_code
        self.content = content
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise json.JSONDecodeError("no body", "", 0)
        return self._payload


def _stub_retry_get(timeline_resp, image_resps):
    """Return a retry_get stand-in that dispatches by URL substring."""
    async def fake(client, url, **kwargs):
        if "/timeline" in url:
            return timeline_resp
        if "index=" in url:
            idx = int(url.rsplit("index=", 1)[1].split("&", 1)[0])
            return image_resps[idx]
        raise AssertionError(f"unexpected URL {url}")
    return fake


class TestPAGASARefresh:
    """End-to-end through the source's _refresh — verifies the
    timeline → image-fetch → cache pipeline on synthetic responses."""

    def test_refresh_populates_six_timestamps(self):
        # The timeline endpoint returns 6 frames at 15-min cadence.
        # After one refresh the cache must hold all six native ts.
        start = 1778962500
        timeline = _FakeResp(200, payload=_make_timeline(start))
        # Paint each frame with a different palette stop so a per-frame
        # decode mistake (frame swap, off-by-one) would show as a
        # different dominant uint8 value per timestamp.
        images = [
            _FakeResp(200, content=_make_test_png(_PAGASA_CLUTTER_STOPS + i))
            for i in range(_PAGASA_EXPECTED_FRAMES)
        ]
        src = PAGASASource()
        import librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.source as mod
        original = mod.retry_get
        mod.retry_get = _stub_retry_get(timeline, images)
        try:
            asyncio.run(src._refresh())
        finally:
            mod.retry_get = original

        assert len(src._cache_order) == _PAGASA_EXPECTED_FRAMES
        # Timestamps must come back in ascending order and at the
        # native cadence — verifies we don't deduplicate or drop any.
        deltas = [b - a for a, b in zip(src._cache_order, src._cache_order[1:])]
        assert all(d == _PAGASA_NATIVE_CADENCE_SEC for d in deltas)

    def test_refresh_handles_empty_timeline_gracefully(self):
        timeline = _FakeResp(200, payload={"success": True, "data": {"timeline": []}})
        src = PAGASASource()
        import librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.source as mod
        original = mod.retry_get
        mod.retry_get = _stub_retry_get(timeline, [])
        try:
            asyncio.run(src._refresh())
        finally:
            mod.retry_get = original
        assert src._cache_order == []

    def test_refresh_survives_failed_image_fetch(self):
        # If one of the per-index image fetches 404s, the rest must
        # still populate — a single dropped frame must not poison the
        # whole cache.
        start = 1778962500
        timeline = _FakeResp(200, payload=_make_timeline(start))
        images = []
        for i in range(_PAGASA_EXPECTED_FRAMES):
            if i == 2:
                images.append(_FakeResp(404))
            else:
                images.append(_FakeResp(
                    200, content=_make_test_png(_PAGASA_CLUTTER_STOPS),
                ))
        src = PAGASASource()
        import librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.source as mod
        original = mod.retry_get
        mod.retry_get = _stub_retry_get(timeline, images)
        try:
            asyncio.run(src._refresh())
        finally:
            mod.retry_get = original
        assert len(src._cache_order) == _PAGASA_EXPECTED_FRAMES - 1


# ─────────────────────────────────────────────────────────────────────
# Nearest-frame lookup (15-min native vs 10-min store grid)
# ─────────────────────────────────────────────────────────────────────
class TestNearestCached:
    def test_returns_none_on_empty_cache(self):
        src = PAGASASource()
        assert src._nearest_cached(1778962500) is None

    def test_exact_match_returns_exact_ts(self):
        src = PAGASASource()
        ts = 1778962500
        src._frame_cache[ts] = np.zeros((4, 4), dtype=np.uint8)
        src._cache_order = [ts]
        assert src._nearest_cached(ts) == ts

    def test_returns_nearest_within_threshold(self):
        # 10-min store slot 5 min off a 15-min native frame should
        # still hit — the threshold is half the native cadence (7.5 min).
        src = PAGASASource()
        native_ts = 1778962500     # 0
        store_ts = native_ts + 300  # +5 min
        src._frame_cache[native_ts] = np.zeros((4, 4), dtype=np.uint8)
        src._cache_order = [native_ts]
        assert src._nearest_cached(store_ts) == native_ts

    def test_returns_none_outside_threshold(self):
        # 8 min off must NOT match — that store slot belongs to a
        # different native frame which is not in cache yet.
        src = PAGASASource()
        native_ts = 1778962500
        store_ts = native_ts + _MAX_NEAREST_OFFSET_SEC + 1
        src._frame_cache[native_ts] = np.zeros((4, 4), dtype=np.uint8)
        src._cache_order = [native_ts]
        assert src._nearest_cached(store_ts) is None

    def test_returns_truly_nearest_with_multiple_candidates(self):
        src = PAGASASource()
        a = 1778962500
        b = a + _PAGASA_NATIVE_CADENCE_SEC   # +15 min
        src._frame_cache[a] = np.zeros((4, 4), dtype=np.uint8)
        src._frame_cache[b] = np.zeros((4, 4), dtype=np.uint8)
        src._cache_order = [a, b]
        # Midway between two native frames → ties go to the earlier
        # candidate via Python's min stability.  Either is acceptable.
        midpoint = a + _PAGASA_NATIVE_CADENCE_SEC // 2
        assert src._nearest_cached(midpoint) in (a, b)
        assert src._nearest_cached(a + 100) == a
        assert src._nearest_cached(b - 100) == b


# ─────────────────────────────────────────────────────────────────────
# Source fetch + cache
# ─────────────────────────────────────────────────────────────────────
class TestPAGASASource:
    def test_url_strips_trailing_slash_in_base(self):
        src = PAGASASource("https://example.test/")
        assert src._timeline_url == (
            "https://example.test/api/v1/radar/timeline"
        )

    def test_image_url_includes_sublayer_and_index(self):
        src = PAGASASource("https://example.test")
        url = src._image_url(3)
        assert "sublayer=hybrid-reflectivity" in url
        assert "index=3" in url

    def test_fetch_returns_none_when_timeline_fails(self):
        src = PAGASASource()
        timeline = _FakeResp(404)
        import librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.source as mod
        original = mod.retry_get
        mod.retry_get = _stub_retry_get(timeline, [])
        try:
            result = asyncio.run(
                src.fetch_frame(REGIONS["PHCOMP"], minutes_ago=0)
            )
        finally:
            mod.retry_get = original
        assert result is None

    def test_fetch_raises_on_wrong_region(self):
        src = PAGASASource()
        with pytest.raises(ValueError):
            asyncio.run(
                src.fetch_frame(REGIONS["MYPENINSULAR"], minutes_ago=0)
            )

    def test_fetch_for_archive_ts_outside_buffer_returns_none(self):
        # The timeline endpoint only carries ~75 min of backfill, so
        # archive lookups for last week must return None instead of
        # silently lying with the oldest cached frame.
        src = PAGASASource()
        start = 1778962500
        timeline = _FakeResp(200, payload=_make_timeline(start))
        images = [
            _FakeResp(200, content=_make_test_png(_PAGASA_CLUTTER_STOPS))
            for _ in range(_PAGASA_EXPECTED_FRAMES)
        ]
        import librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.source as mod
        original = mod.retry_get
        mod.retry_get = _stub_retry_get(timeline, images)
        try:
            stale_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
            result = asyncio.run(
                src.fetch_archive_frame(REGIONS["PHCOMP"], stale_dt)
            )
        finally:
            mod.retry_get = original
        assert result is None