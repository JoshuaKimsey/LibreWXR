# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.sources

from librewxr.data.regions import REGIONS, RegionDef, resolve_regions
from librewxr.data.sources import (
    MSSSource,
    _decode_mss_png,
    _MSS_PALETTE,
)
from librewxr.tiles.coordinates import tile_overlaps_region


class TestSeacompRegion:
    def test_seacomp_in_regions(self):
        assert "SEACOMP" in REGIONS

    def test_seacomp_group_and_proj(self):
        r = REGIONS["SEACOMP"]
        assert r.proj == "latlon"
        assert r.group == "SOUTHEAST_ASIA"

    def test_seacomp_bounds_around_changi(self):
        # Bounds are derived from MSS Changi radar (1.3521°N, 103.8198°E)
        # ± 4.32° (~480 km).  These must straddle the radar; if the
        # values drift the renderer will mis-register coastlines.
        r = REGIONS["SEACOMP"]
        assert r.west < 103.8198 < r.east
        assert r.south < 1.3521 < r.north
        # ~8.64° span on each axis (480 km × 2).
        assert abs((r.east - r.west) - 8.64) < 0.1
        assert abs((r.north - r.south) - 8.64) < 0.1

    def test_seacomp_dimensions(self):
        r = REGIONS["SEACOMP"]
        # Native 480 km product is 480×480 — fixed grid dimensions avoid
        # pixel-size fencepost rounding ambiguity.
        assert r.width == 480
        assert r.height == 480

    def test_southeast_asia_group_resolution(self):
        assert resolve_regions("SOUTHEAST_ASIA") == ["SEACOMP"]

    def test_all_includes_seacomp(self):
        assert "SEACOMP" in resolve_regions("ALL")


class TestUrlForTimestamp:
    """MSS publishes filenames in Singapore local time (UTC+8) even
    though the request API is UTC.  These tests pin that conversion
    explicitly — the original implementation silently served 8-hour
    stale frames because it formatted the UTC timestamp directly into
    the filename instead of shifting to SGT first.
    """

    def _src(self) -> MSSSource:
        return MSSSource("https://example.test/files/rainarea/480km")

    def test_rounds_down_to_30_min_boundary(self):
        # 2026-05-15 10:47:32 UTC → 10:30 UTC native → 18:30 SGT filename
        src = self._src()
        ts = int(datetime(2026, 5, 15, 10, 47, 32, tzinfo=timezone.utc).timestamp())
        url = src._url_for_timestamp(ts)
        assert url.endswith("dpsri_480km_2026051518300000dBR.dpsri.png"), url

    def test_exact_boundary_kept(self):
        # 10:30 UTC → 18:30 SGT
        src = self._src()
        ts = int(datetime(2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp())
        url = src._url_for_timestamp(ts)
        assert url.endswith("dpsri_480km_2026051518300000dBR.dpsri.png")

    def test_top_of_hour_kept(self):
        # 11:00 UTC → 19:00 SGT
        src = self._src()
        ts = int(datetime(2026, 5, 15, 11, 0, 0, tzinfo=timezone.utc).timestamp())
        url = src._url_for_timestamp(ts)
        assert url.endswith("dpsri_480km_2026051519000000dBR.dpsri.png")

    def test_utc_evening_crosses_to_next_day_in_sgt(self):
        # 18:30 UTC = 02:30 SGT the NEXT day — verifies the date in the
        # filename rolls over correctly (any UTC after 16:00 lands in
        # tomorrow's SGT date).
        src = self._src()
        ts = int(datetime(2026, 5, 15, 18, 30, 0, tzinfo=timezone.utc).timestamp())
        url = src._url_for_timestamp(ts)
        assert url.endswith("dpsri_480km_2026051602300000dBR.dpsri.png"), url

    def test_base_url_is_preserved(self):
        src = self._src()
        ts = int(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        assert src._url_for_timestamp(ts).startswith(
            "https://example.test/files/rainarea/480km/"
        )

    def test_trailing_slash_stripped(self):
        src = MSSSource("https://example.test/dir/")
        ts = int(datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        # Should not produce a double-slash in the path.
        assert "//" not in src._url_for_timestamp(ts).split("://", 1)[1]


class TestPaletteDecode:
    """The 480 km MSS PNG uses a discrete 31-stop palette; the decoder
    snaps each opaque pixel to its nearest anchor in RGB space and maps
    the rank to dBZ via the ``_MSS_PALETTE`` table."""

    def _fake_region(self, w: int) -> RegionDef:
        # SEACOMP's full shape isn't needed for decode-correctness tests,
        # but the decoder logs a shape-mismatch warning.  Build a tiny
        # stand-in region matching the test PNG.
        return RegionDef(
            name="SEACOMP", west=0.0, east=1.0, south=0.0, north=1.0,
            pixel_size=1.0, group="SOUTHEAST_ASIA",
            grid_width=w, grid_height=1,
        )

    def _make_png(self, pixels: list[tuple[int, int, int, int]]) -> bytes:
        arr = np.array([pixels], dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _expected_uint8(self, dbz: float) -> int:
        # Match _dbz_float_to_uint8: clip then astype(uint8), which
        # truncates rather than rounding.
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
        # The palette is ordered by intensity; uint8 dBZ output should
        # rise monotonically through the table.
        pixels = [(r, g, b, 255) for r, g, b, _ in _MSS_PALETTE]
        png = self._make_png(pixels)
        out = _decode_mss_png(png, self._fake_region(len(pixels)))
        diffs = np.diff(out[0].astype(int))
        assert np.all(diffs > 0), f"non-monotonic palette decode: {out[0]}"

    def test_near_anchor_within_tolerance_snaps(self):
        # A pixel within the tolerance should snap to the nearest anchor.
        # First palette stop is (0, 239, 239, dBZ=5).  Perturb by ±1.
        r, g, b, dbz = _MSS_PALETTE[0]
        png = self._make_png([(r + 1, g - 1, b + 1, 255)])
        out = _decode_mss_png(png, self._fake_region(1))
        assert out[0, 0] == self._expected_uint8(dbz)

    def test_off_palette_color_is_no_data(self):
        # Pure black (0,0,0) sits far from every cyan/green/yellow/red/
        # magenta anchor (closest is (0,128,69) ≈ 21 RGB units, dist²
        # ≈ 21000 — well past the 64 tolerance).
        png = self._make_png([(0, 0, 0, 255)])
        out = _decode_mss_png(png, self._fake_region(1))
        assert out[0, 0] == 0

    def test_bad_png_returns_none(self):
        assert _decode_mss_png(b"not a png", self._fake_region(1)) is None


class _FakeResp:
    """Minimal stand-in for the httpx.Response shape MSSSource expects."""

    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content


def _make_native_png(value: int = 26) -> bytes:
    """Render a 480x480 RGBA PNG using one stop from the MSS palette.

    *value* indexes into _MSS_PALETTE so different natives can be
    visually distinct (which is what we need to verify interpolation
    actually warps between them rather than copying one side through).
    """
    r, g, b, _ = _MSS_PALETTE[value]
    arr = np.zeros((480, 480, 4), dtype=np.uint8)
    arr[..., 0] = r
    arr[..., 1] = g
    arr[..., 2] = b
    arr[..., 3] = 255
    # Add a small variation so the optical-flow pass has something to
    # actually track (uniform frames produce zero-flow).
    arr[200:280, 200:280, :3] = _MSS_PALETTE[value + 4][:3]
    img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestNativeTsRounding:
    """Bracket pair derivation — the math underneath fetch_for_ts."""

    def test_aligned_30_min_is_its_own_native(self):
        ts = int(datetime(2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp())
        assert MSSSource._native_ts_for(ts) == ts

    def test_10_min_offset_rounds_down(self):
        prev = datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc)
        for minute in (0, 10, 20):
            ts = int(prev.replace(minute=minute).timestamp())
            assert MSSSource._native_ts_for(ts) == int(prev.timestamp())

    def test_just_past_30_rounds_to_30(self):
        anchor = int(datetime(2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp())
        for minute in (30, 40, 50):
            ts = int(datetime(2026, 5, 15, 10, minute, 0, tzinfo=timezone.utc).timestamp())
            assert MSSSource._native_ts_for(ts) == anchor


class TestBracketPairFetch:
    """Verify fetch_archive_frame dispatches correctly to native vs interp.

    All HTTP calls are stubbed; the test exercises the routing + caching
    logic rather than the network or the optical-flow output quality.
    """

    @pytest.fixture
    def src(self, monkeypatch) -> MSSSource:
        src = MSSSource(interpolation=True)
        # Track per-native-ts fetch counts to assert deduplication.
        src._fetch_log: dict[int, int] = {}

        async def fake_get_client():
            return None  # never actually used — retry_get is stubbed

        async def fake_retry_get(client, url, log_name=None):
            # URL filename is in SGT (UTC+8) — see MSSSource._LOCAL_TZ_OFFSET.
            # Subtract 8h to recover the UTC native_ts the source asked for.
            fname = url.rsplit("/", 1)[-1]
            ts_str = fname.split("_")[2][:12]
            sgt = datetime.strptime(ts_str, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            utc_native_ts = int((sgt - timedelta(hours=8)).timestamp())
            src._fetch_log[utc_native_ts] = src._fetch_log.get(utc_native_ts, 0) + 1
            # Use the half-hour-of-day as a value index into the palette
            # so consecutive natives differ visually.
            idx = (sgt.minute // 30) + sgt.hour * 2
            return _FakeResp(200, _make_native_png(value=idx % 20))

        monkeypatch.setattr(src, "_get_client", fake_get_client)
        monkeypatch.setattr(
            "librewxr.data.sources.retry_get", fake_retry_get,
        )
        return src

    async def test_aligned_request_returns_native_only(self, src: MSSSource):
        region = REGIONS["SEACOMP"]
        dt = datetime(2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc)
        frame = await src.fetch_archive_frame(region, dt)
        assert frame is not None
        # Only the 10:30 native was fetched — no bracket pair work.
        assert list(src._fetch_log.keys()) == [int(dt.timestamp())]
        # No flow computation for an aligned request.
        assert len(src._flow_cache) == 0

    async def test_sub_interval_fetches_both_brackets(self, src: MSSSource):
        region = REGIONS["SEACOMP"]
        # 10:10 sits between native 10:00 and native 10:30.
        dt = datetime(2026, 5, 15, 10, 10, 0, tzinfo=timezone.utc)
        frame = await src.fetch_archive_frame(region, dt)
        assert frame is not None
        ts_prev = int(datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp())
        ts_next = int(datetime(2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp())
        assert set(src._fetch_log.keys()) == {ts_prev, ts_next}
        # Flow between this pair should be cached.
        assert (ts_prev, ts_next) in src._flow_cache

    async def test_two_sub_interval_slots_share_one_flow_compute(self, src: MSSSource):
        region = REGIONS["SEACOMP"]
        # 10:10 and 10:20 both bracket the same 10:00 → 10:30 pair.
        for minute in (10, 20):
            dt = datetime(2026, 5, 15, 10, minute, 0, tzinfo=timezone.utc)
            f = await src.fetch_archive_frame(region, dt)
            assert f is not None
        ts_prev = int(datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp())
        ts_next = int(datetime(2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc).timestamp())
        # Each native was fetched exactly once, not twice.
        assert src._fetch_log[ts_prev] == 1
        assert src._fetch_log[ts_next] == 1
        # Exactly one flow pair cached.
        assert len(src._flow_cache) == 1

    async def test_three_consecutive_slots_produce_three_distinct_frames(
        self, src: MSSSource,
    ):
        """The bug we're fixing: 10:00 / 10:10 / 10:20 used to be identical.

        After the refactor they must be three distinct frames — the
        native at 10:00 plus two genuinely interpolated frames.
        """
        region = REGIONS["SEACOMP"]
        frames = []
        for minute in (0, 10, 20):
            dt = datetime(2026, 5, 15, 10, minute, 0, tzinfo=timezone.utc)
            f = await src.fetch_archive_frame(region, dt)
            assert f is not None
            frames.append(f)
        # No pair should be byte-identical.
        assert not np.array_equal(frames[0], frames[1])
        assert not np.array_equal(frames[1], frames[2])
        assert not np.array_equal(frames[0], frames[2])

    async def test_interpolation_off_holds_last_frame(self, monkeypatch):
        """With interpolation disabled, sub-interval slots return the
        same earlier-native data — the explicit hold-last-frame
        fallback documented for the config knob."""
        src = MSSSource(interpolation=False)
        src._fetch_log = {}

        async def fake_get_client():
            return None

        async def fake_retry_get(client, url, log_name=None):
            # Filename is SGT; convert back to UTC native_ts for logging.
            fname = url.rsplit("/", 1)[-1]
            ts_str = fname.split("_")[2][:12]
            sgt = datetime.strptime(ts_str, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            utc_native_ts = int((sgt - timedelta(hours=8)).timestamp())
            src._fetch_log[utc_native_ts] = (
                src._fetch_log.get(utc_native_ts, 0) + 1
            )
            idx = (sgt.minute // 30) + sgt.hour * 2
            return _FakeResp(200, _make_native_png(value=idx % 20))

        monkeypatch.setattr(src, "_get_client", fake_get_client)
        monkeypatch.setattr(
            "librewxr.data.sources.retry_get", fake_retry_get,
        )

        region = REGIONS["SEACOMP"]
        frames = []
        for minute in (0, 10, 20):
            dt = datetime(2026, 5, 15, 10, minute, 0, tzinfo=timezone.utc)
            f = await src.fetch_archive_frame(region, dt)
            frames.append(f)
        # All three slots resolve to the native at 10:00, so the frames
        # are byte-identical AND only one native was fetched.
        assert np.array_equal(frames[0], frames[1])
        assert np.array_equal(frames[1], frames[2])
        ts_prev = int(datetime(2026, 5, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp())
        assert list(src._fetch_log.keys()) == [ts_prev]

    async def test_next_bracket_404_falls_back_to_prev_native(self, monkeypatch):
        """If the future bracket end isn't published yet (common for
        the most-recent slot), the source returns the earlier native
        rather than a hole in the timeline."""
        src = MSSSource(interpolation=True)

        async def fake_get_client():
            return None

        async def fake_retry_get(client, url, log_name=None):
            # Filename is SGT; convert back to UTC for the comparison.
            fname = url.rsplit("/", 1)[-1]
            ts_str = fname.split("_")[2][:12]
            sgt = datetime.strptime(ts_str, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            utc_native = sgt - timedelta(hours=8)
            # Pretend everything at or after the future bracket (10:30 UTC) is unpublished.
            if utc_native >= datetime(2026, 5, 15, 10, 30, 0, tzinfo=timezone.utc):
                return _FakeResp(404)
            return _FakeResp(200, _make_native_png(value=5))

        monkeypatch.setattr(src, "_get_client", fake_get_client)
        monkeypatch.setattr(
            "librewxr.data.sources.retry_get", fake_retry_get,
        )

        region = REGIONS["SEACOMP"]
        # Ask for 10:20: bracket pair is (10:00, 10:30). 10:30 is 404.
        # Expect 10:00 native returned, no flow computed.
        dt = datetime(2026, 5, 15, 10, 20, 0, tzinfo=timezone.utc)
        frame = await src.fetch_archive_frame(region, dt)
        assert frame is not None
        assert len(src._flow_cache) == 0  # no interp without both ends


class TestSeacompTileOverlap:
    def test_tile_over_strait_of_malacca_overlaps(self):
        region = REGIONS["SEACOMP"]
        # z=6, x=50, y=31 covers roughly Singapore / Sumatra.
        assert tile_overlaps_region(region, z=6, x=50, y=31)

    def test_tile_over_europe_does_not_overlap(self):
        region = REGIONS["SEACOMP"]
        # z=4, x=8, y=5 is over central Europe — no overlap.
        assert not tile_overlaps_region(region, z=4, x=8, y=5)
