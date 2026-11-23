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
    _PAGASA_EXPECTED_FRAMES,
    _PAGASA_EXPECTED_HEIGHT,
    _PAGASA_EXPECTED_WIDTH,
    _PAGASA_FALLBACK_COLORMAP,
    _PAGASA_FALLBACK_SCALE_MAX,
    _PAGASA_NATIVE_CADENCE_SEC,
    _build_palette_arrays,
    _decode_pagasa_png,
)
from librewxr.data.regions import REGIONS, resolve_regions


def _colormap_payload() -> list:
    """The fallback colormap in the timeline response's JSON shape."""
    return [
        [list(dbz_range), list(colour)]
        for dbz_range, colour in _PAGASA_FALLBACK_COLORMAP
    ]


def _clutter_stop_count() -> int:
    """Leading RGB-gray stops in the fallback table (the clutter ones)."""
    count = 0
    for _, (r, g, b, _) in _PAGASA_FALLBACK_COLORMAP:
        if r == g == b:
            count += 1
        else:
            break
    return count


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
    def test_palette_is_fourteen_stops(self):
        # The new timeline API reports 14 stops over 0-80 dBZ.  The
        # fallback table mirrors it verbatim; adding/removing stops here
        # would desync the decoder from the upstream colour mapping
        # whenever the response colormap is unavailable.
        assert len(_PAGASA_FALLBACK_COLORMAP) == 14

    def test_fallback_dbz_is_range_midpoint_clamped_to_scale(self):
        # Each stop decodes to the midpoint of its dBZ range, with the
        # upper bound clamped to the reported scale max.  So [35, 40)
        # is 37.5 and the open-ended [65, 999) is 72.5 at 80 dBZ.
        palette = _build_palette_arrays(
            _colormap_payload(), _PAGASA_FALLBACK_SCALE_MAX,
        )
        assert palette is not None
        _, dbz, _ = palette
        for i, (dbz_range, _) in enumerate(_PAGASA_FALLBACK_COLORMAP):
            lo, hi = dbz_range
            expected = (lo + min(hi, _PAGASA_FALLBACK_SCALE_MAX)) / 2.0
            assert dbz[i] == pytest.approx(expected, abs=1e-5)
        assert dbz[7] == pytest.approx(37.5)
        assert dbz[13] == pytest.approx(72.5)

    def test_clutter_stops_are_grays(self):
        # The first stops are the weak-echo greys (#555555, #7f7f7f,
        # #a8a8a8) that we treat as no-data downstream.
        clutter = _clutter_stop_count()
        assert clutter == 3
        for i in range(clutter):
            _, (r, g, b, _) = _PAGASA_FALLBACK_COLORMAP[i]
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
        # Paint a single amber pixel ((255,196,0), stop [35,40) → 37.5)
        # at alpha=255.
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[1, 2] = (255, 196, 0, 0xff)
        out = _decode_pagasa_png(rgba)
        expected = int(np.clip((37.5 + 32.0) * 2.0, 0, 255))
        assert out[1, 2] == expected
        # Other pixels are transparent → no-data.
        assert out.sum() == expected

    def test_clutter_grays_decode_to_nodata(self):
        # All three clutter grays must decode to no-data even when
        # painted with non-zero alpha, matching the alpha-encoded
        # weak-echo flag from the source PNG.
        for i in range(_clutter_stop_count()):
            _, (r, g, b, _) = _PAGASA_FALLBACK_COLORMAP[i]
            rgba = np.zeros((2, 2, 4), dtype=np.uint8)
            rgba[0, 0] = (r, g, b, 200)
            out = _decode_pagasa_png(rgba)
            assert out[0, 0] == 0, f"clutter stop {i} should decode to 0"

    def test_each_precip_stop_decodes_distinctly(self):
        # Every non-clutter palette stop must map to a unique uint8
        # encoded dBZ value — collisions would silently merge bins.
        clutter = _clutter_stop_count()
        encoded = set()
        for _, (r, g, b, _) in _PAGASA_FALLBACK_COLORMAP[clutter:]:
            rgba = np.zeros((1, 1, 4), dtype=np.uint8)
            rgba[0, 0] = (r, g, b, 255)
            out = _decode_pagasa_png(rgba)
            encoded.add(int(out[0, 0]))
        assert len(encoded) == len(_PAGASA_FALLBACK_COLORMAP) - clutter

    def test_wrong_channel_count_raises(self):
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            _decode_pagasa_png(rgb)


# ─────────────────────────────────────────────────────────────────────
# Timeline + image decode end-to-end
# ─────────────────────────────────────────────────────────────────────
def _png_from_rgba(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def _make_test_png(colormap_index: int) -> bytes:
    """Build a 2048×2048 RGBA PNG filled with one fallback stop colour."""
    _, (r, g, b, _) = _PAGASA_FALLBACK_COLORMAP[colormap_index]
    rgba = np.zeros(
        (_PAGASA_EXPECTED_HEIGHT, _PAGASA_EXPECTED_WIDTH, 4), dtype=np.uint8,
    )
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = 255
    return _png_from_rgba(rgba)


def _make_test_png_rgb(rgb: tuple[int, int, int]) -> bytes:
    """Build a 2048×2048 RGBA PNG filled with an arbitrary RGB colour."""
    rgba = np.zeros(
        (_PAGASA_EXPECTED_HEIGHT, _PAGASA_EXPECTED_WIDTH, 4), dtype=np.uint8,
    )
    rgba[..., 0] = rgb[0]
    rgba[..., 1] = rgb[1]
    rgba[..., 2] = rgb[2]
    rgba[..., 3] = 255
    return _png_from_rgba(rgba)


def _make_placeholder_png() -> bytes:
    """The 1x1 RGBA placeholder the upstream serves while its renderer
    is down."""
    return _png_from_rgba(np.zeros((1, 1, 4), dtype=np.uint8))


def _make_timeline(
    start_unix: int, count: int = 9, colormap: list | None = None,
) -> dict:
    """Build a timeline payload mirroring the new PANAHON API shape."""
    entries = []
    for i in range(count):
        ts = start_unix + i * _PAGASA_NATIVE_CADENCE_SEC
        entries.append({
            "observed_at": "2026-11-23 00:00:00",
            "observed_at_unix": ts,
            "image_url": "",
        })
    return {
        "success": True,
        "data": {
            "timeline": entries,
            "tile_version": 5,
            "scale": {
                "mode": "dbz",
                "max": _PAGASA_FALLBACK_SCALE_MAX,
                "sqrt": False,
                "unit": "dBZ",
                "label": "Reflectivity",
            },
            "colormap": (
                colormap if colormap is not None else _colormap_payload()
            ),
        },
    }


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
        if "/radar/timeline" in url:
            return timeline_resp
        if "t=" in url:
            ts = int(url.rsplit("t=", 1)[1].split("&", 1)[0])
            return image_resps[ts]
        raise AssertionError(f"unexpected URL {url}")
    return fake


class TestPAGASARefresh:
    """End-to-end through the source's _refresh — verifies the
    timeline → image-fetch → cache pipeline on synthetic responses."""

    def test_refresh_populates_nine_timestamps(self):
        # The timeline endpoint returns 9 frames at 10-min cadence.
        # After one refresh the cache must hold all nine native ts.
        start = 1778962500
        timeline = _FakeResp(200, payload=_make_timeline(start))
        # Paint each frame with a different palette stop so a per-frame
        # decode mistake (frame swap, off-by-one) would show as a
        # different dominant uint8 value per timestamp.
        images = {
            start + i * _PAGASA_NATIVE_CADENCE_SEC: _FakeResp(
                200,
                content=_make_test_png(_clutter_stop_count() + i),
            )
            for i in range(_PAGASA_EXPECTED_FRAMES)
        }
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

    def test_refresh_uses_response_colormap_over_fallback(self):
        # Upstream has retuned its palette before, so the response
        # colormap must win over the module fallback table.
        custom_colormap = [
            [[0, 10], [11, 22, 33, 255]],
            [[10, 20], [200, 10, 20, 255]],
        ]
        start = 1778962500
        timeline = _FakeResp(
            200, payload=_make_timeline(start, colormap=custom_colormap),
        )
        # (200,10,20) is not a fallback colour, so a fallback decode
        # would yield 0.
        images = {
            start + i * _PAGASA_NATIVE_CADENCE_SEC: _FakeResp(
                200, content=_make_test_png_rgb((200, 10, 20)),
            )
            for i in range(_PAGASA_EXPECTED_FRAMES)
        }
        src = PAGASASource()
        import librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.source as mod
        original = mod.retry_get
        mod.retry_get = _stub_retry_get(timeline, images)
        try:
            asyncio.run(src._refresh())
        finally:
            mod.retry_get = original

        assert len(src._cache_order) == _PAGASA_EXPECTED_FRAMES
        # Stop [10, 20) → midpoint 15 dBZ → clamp((15+32)*2, 0, 255) = 94.
        expected = int(np.clip((15.0 + 32.0) * 2.0, 0, 255))
        for ts in src._cache_order:
            grid = src._frame_cache[ts]
            assert grid.dtype == np.uint8
            assert (grid == expected).all()

    def test_refresh_survives_non_json_timeline_body(self):
        # The host answers non-browser user agents with a decoy PNG at
        # HTTP 200; the body is then undecodable as JSON and the cycle
        # must be skipped with a warning, not an exception.
        timeline = _FakeResp(200, content=_make_placeholder_png())
        src = PAGASASource()
        import librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.source as mod
        original = mod.retry_get
        mod.retry_get = _stub_retry_get(timeline, {})
        try:
            asyncio.run(src._refresh())
        finally:
            mod.retry_get = original
        assert src._cache_order == []

    def test_refresh_handles_empty_timeline_gracefully(self):
        timeline = _FakeResp(200, payload={"success": True, "data": {"timeline": []}})
        src = PAGASASource()
        import librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.source as mod
        original = mod.retry_get
        mod.retry_get = _stub_retry_get(timeline, {})
        try:
            asyncio.run(src._refresh())
        finally:
            mod.retry_get = original
        assert src._cache_order == []

    def test_refresh_skips_placeholder_and_keeps_the_rest(self, caplog):
        # While the upstream radar renderer is down every image request
        # returns HTTP 200 with a 1x1 placeholder; the size guard must
        # turn that into a skipped frame, not a corrupt cached one.
        start = 1778962500
        timeline = _FakeResp(200, payload=_make_timeline(start))
        placeholder_ts = start
        images = {
            start + i * _PAGASA_NATIVE_CADENCE_SEC: _FakeResp(
                200,
                content=(
                    _make_placeholder_png()
                    if start + i * _PAGASA_NATIVE_CADENCE_SEC == placeholder_ts
                    else _make_test_png(_clutter_stop_count())
                ),
            )
            for i in range(_PAGASA_EXPECTED_FRAMES)
        }
        src = PAGASASource()
        import librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.source as mod
        original = mod.retry_get
        mod.retry_get = _stub_retry_get(timeline, images)
        try:
            with caplog.at_level("WARNING"):
                asyncio.run(src._refresh())
        finally:
            mod.retry_get = original

        assert placeholder_ts not in src._cache_order
        assert len(src._cache_order) == _PAGASA_EXPECTED_FRAMES - 1
        assert "decode failed" in caplog.text
        with pytest.raises(ValueError):
            src._decode_png(_make_placeholder_png())

    def test_refresh_survives_failed_image_fetch(self):
        # If one of the per-timestamp image fetches 404s, the rest must
        # still populate — a single dropped frame must not poison the
        # whole cache.
        start = 1778962500
        timeline = _FakeResp(200, payload=_make_timeline(start))
        images = {}
        for i in range(_PAGASA_EXPECTED_FRAMES):
            ts = start + i * _PAGASA_NATIVE_CADENCE_SEC
            if i == 2:
                images[ts] = _FakeResp(404)
            else:
                images[ts] = _FakeResp(
                    200, content=_make_test_png(_clutter_stop_count()),
                )
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
# Nearest-frame lookup (10-min native vs 10-min store grid)
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
        # A store slot 5 min off a native frame should still hit — the
        # threshold is half the native cadence (5 min).
        src = PAGASASource()
        native_ts = 1778962500     # 0
        store_ts = native_ts + 300  # +5 min
        src._frame_cache[native_ts] = np.zeros((4, 4), dtype=np.uint8)
        src._cache_order = [native_ts]
        assert src._nearest_cached(store_ts) == native_ts

    def test_returns_none_outside_threshold(self):
        # Just past 5 min off must NOT match — that store slot belongs
        # to a different native frame which is not in cache yet.
        src = PAGASASource()
        native_ts = 1778962500
        store_ts = native_ts + _MAX_NEAREST_OFFSET_SEC + 1
        src._frame_cache[native_ts] = np.zeros((4, 4), dtype=np.uint8)
        src._cache_order = [native_ts]
        assert src._nearest_cached(store_ts) is None

    def test_returns_truly_nearest_with_multiple_candidates(self):
        src = PAGASASource()
        a = 1778962500
        b = a + _PAGASA_NATIVE_CADENCE_SEC   # +10 min
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
    def test_client_sends_browser_user_agent(self):
        # The host soft-blocks non-browser UAs with a decoy PNG, so the
        # shared client must carry a browser UA on every request.
        import librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.source as mod

        src = PAGASASource()

        async def get_then_close():
            client = await src._get_client()
            ua = client.headers.get("user-agent")
            await src.close()
            return ua

        assert asyncio.run(get_then_close()) == mod._PAGASA_USER_AGENT

    def test_url_strips_trailing_slash_in_base(self):
        src = PAGASASource("https://example.test/")
        assert src._timeline_url == (
            "https://example.test/api/v1/radar/timeline"
            "?sublayer=mosaic-reflectivity"
        )

    def test_image_url_includes_sublayer_timestamp_and_version(self):
        src = PAGASASource("https://example.test")
        url = src._image_url(1778962500)
        assert "sublayer=mosaic-reflectivity" in url
        assert "t=1778962500" in url
        assert "mode=dbz" in url
        assert "size=2048" in url
        assert "v=5" in url

    def test_fetch_returns_none_when_timeline_fails(self):
        src = PAGASASource()
        timeline = _FakeResp(404)
        import librewxr.sources.regional.southeast_asia.philippines.radar.pagasa.source as mod
        original = mod.retry_get
        mod.retry_get = _stub_retry_get(timeline, {})
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
        # The timeline endpoint only carries ~90 min of backfill, so
        # archive lookups for last year must return None instead of
        # silently lying with the oldest cached frame.
        src = PAGASASource()
        start = 1778962500
        timeline = _FakeResp(200, payload=_make_timeline(start))
        images = {
            start + i * _PAGASA_NATIVE_CADENCE_SEC: _FakeResp(
                200, content=_make_test_png(_clutter_stop_count()),
            )
            for i in range(_PAGASA_EXPECTED_FRAMES)
        }
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
