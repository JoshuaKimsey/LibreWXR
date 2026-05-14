# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Unit tests for DMI HARMONIE DINI grid math, decode, and chain integration."""
from __future__ import annotations

import struct
from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.dmi_dini

from librewxr.data.dmi_dini_grid import (
    BRACKET_INTERVAL_SECONDS,
    CYCLE_INTERVAL_SECONDS,
    DMI_DINI_GRID_HEIGHT,
    DMI_DINI_GRID_WIDTH,
    DMI_DINI_LA1,
    DMI_DINI_LO1,
    SOURCE_STEP_SECONDS,
    STORED_INTERVAL_SECONDS,
    DMIDiniGrid,
    bracket_lead_seconds,
    decode_tp_message,
    decode_2t_message,
    domain_mask,
    feather_mask,
    file_url,
    find_tp_message_offset,
    find_2t_message_offset,
    floor_cycle,
    grid_indices,
    latest_published_run,
    lcc_forward,
    precip_rate_to_dbz_encoded,
)
from librewxr.data.nwp_source import NWPChain, NWPSource


# ── LCC projection + grid ─────────────────────────────────────────────


class TestLCCProjection:
    @pytest.mark.parametrize(
        "name,lat,lon,inside",
        [
            # Cities expected INSIDE the verified DINI extent
            ("London",     51.51,  -0.13, True),
            ("Paris",      48.86,   2.35, True),
            ("Berlin",     52.52,  13.41, True),
            ("Copenhagen", 55.68,  12.57, True),
            ("Munich",     48.14,  11.58, True),
            ("Warsaw",     52.23,  21.01, True),
            ("Reykjavik",  64.13, -21.82, True),
            ("Helsinki",   60.17,  24.94, True),
            ("Rome",       41.90,  12.50, True),
            ("Vienna",     48.21,  16.37, True),
            ("Stockholm",  59.33,  18.07, True),
            # Cities expected OUTSIDE the DINI domain
            ("Madrid",     40.42,  -3.70, False),   # south of grid
            ("Athens",     37.98,  23.73, False),   # south + east
            ("Moscow",     55.75,  37.62, False),   # east of grid
            ("Murmansk",   68.97,  33.08, False),   # north of grid
            ("New York",   40.71, -74.01, False),   # far west
            ("Tokyo",      35.68, 139.69, False),
            ("Cape Town", -33.92,  18.42, False),
        ],
    )
    def test_domain_mask_known_points(self, name, lat, lon, inside):
        m = domain_mask(np.array([lat]), np.array([lon]))
        assert bool(m[0]) is inside, name

    def test_grid_origin_at_south_west(self):
        # The native (un-flipped) GRIB scan puts the SW corner at row N-1,
        # col 0.  We check via the documented (La1, Lo1) corner.
        row, col = grid_indices(np.array([DMI_DINI_LA1]), np.array([DMI_DINI_LO1]))
        # SW corner in our flipped (north-up) grid → row = HEIGHT - 1, col = 0
        assert abs(row[0] - (DMI_DINI_GRID_HEIGHT - 1)) < 1e-3
        assert abs(col[0] - 0) < 1e-3

    def test_lcc_north_pole_bounded(self):
        # At the standard parallel + central meridian, projection should
        # land on the central meridian's y-axis (x ~ 0).
        x, y = lcc_forward(np.array([55.5]), np.array([-8.0]))
        assert abs(float(x[0])) < 1e-3

    def test_lcc_round_trip_at_central_lat(self):
        # On the standard parallel, LCC scaling is unity; a point at
        # (55.5°N, central meridian + 1°) should be ~111 km * cos(55.5°) east.
        x_at_lon0, _ = lcc_forward(np.array([55.5]), np.array([-8.0]))
        x_at_lon1, _ = lcc_forward(np.array([55.5]), np.array([-7.0]))
        dx = float(x_at_lon1[0] - x_at_lon0[0])
        # 1° lon at 55.5°N → ~63 km in true distance, scaled by R/Re ratio
        # of the conformal projection.  Sanity: 50-80 km range.
        assert 50_000 < dx < 80_000


# ── Feather ───────────────────────────────────────────────────────────


class TestFeatherMask:
    def test_inside_full_weight(self):
        # Berlin → 1.0 (well inside)
        f = feather_mask(np.array([52.52]), np.array([13.41]))
        assert f.dtype == np.float32
        assert f[0] == pytest.approx(1.0)

    def test_outside_zero(self):
        # NYC → 0
        f = feather_mask(np.array([40.71]), np.array([-74.01]))
        assert f[0] == 0.0

    def test_taper_monotonic_walking_off_north_edge(self):
        # Walk lat from inside (60°N) to outside (72°N) along the
        # central meridian.  Feather should be non-increasing.
        lats = np.linspace(60.0, 72.0, 25)
        lons = np.full_like(lats, -8.0)
        f = feather_mask(lats, lons)
        diffs = np.diff(f)
        assert (diffs <= 1e-6).all()


# ── Timing helpers ────────────────────────────────────────────────────


class TestTiming:
    def test_floor_cycle_3h(self):
        # 14:23 UTC → 12:00 UTC
        ts = int(datetime(2026, 5, 1, 14, 23, tzinfo=timezone.utc).timestamp())
        floored = floor_cycle(ts)
        expected = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        assert floored == expected

    def test_floor_cycle_at_boundary(self):
        ts = int(datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc).timestamp())
        assert floor_cycle(ts) == ts

    def test_latest_published_run(self):
        now = int(datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc).timestamp())
        # 3h delay → floor_cycle(11:00) = 09:00
        run = latest_published_run(now, 3 * 3600)
        expected = int(datetime(2026, 5, 1, 9, 0, tzinfo=timezone.utc).timestamp())
        assert run == expected

    @pytest.mark.parametrize(
        "lead_min,l0_min,l1_min,alpha",
        [
            (0,    0,   60, 0.0),
            (30,   0,   60, 0.5),
            (60,   60, 120, 0.0),
            (90,   60, 120, 0.5),
            (120, 120, 180, 0.0),
        ],
    )
    def test_bracket_lead_seconds(self, lead_min, l0_min, l1_min, alpha):
        l0, l1, a = bracket_lead_seconds(lead_min * 60)
        assert l0 == l0_min * 60
        assert l1 == l1_min * 60
        assert a == pytest.approx(alpha)

    def test_cycle_interval_constants(self):
        assert CYCLE_INTERVAL_SECONDS == 3 * 3600
        assert BRACKET_INTERVAL_SECONDS == 3600


# ── URL construction ──────────────────────────────────────────────────


class TestFileUrl:
    def test_format_matches_dmi_pattern(self):
        run = datetime(2026, 5, 3, 0, tzinfo=timezone.utc)
        url = file_url(run, 1)
        assert url.endswith(
            "/forecastdata/HARMONIE_DINI_SF/"
            "HARMONIE_DINI_SF_2026-05-03T000000Z_2026-05-03T010000Z.grib"
        )

    def test_step_zero_valid_equals_run(self):
        run = datetime(2026, 5, 3, 0, tzinfo=timezone.utc)
        url = file_url(run, 0)
        assert "2026-05-03T000000Z_2026-05-03T000000Z.grib" in url

    def test_step_crosses_day_boundary(self):
        run = datetime(2026, 5, 3, 21, tzinfo=timezone.utc)
        url = file_url(run, 6)
        # 21:00 + 6h = next day 03:00
        assert "2026-05-03T210000Z_2026-05-04T030000Z.grib" in url

    def test_uses_settings_bucket_and_region(self):
        run = datetime(2026, 5, 3, 0, tzinfo=timezone.utc)
        url = file_url(run, 1)
        assert "dmi-opendata" in url
        assert "eu-north-1" in url


# ── Z-R conversion ────────────────────────────────────────────────────


class TestZR:
    def test_zero_rate_zero_encoded(self):
        encoded = precip_rate_to_dbz_encoded(np.array([0.0, 0.0]))
        assert (encoded == 0).all()

    def test_higher_rate_higher_dbz(self):
        encoded = precip_rate_to_dbz_encoded(np.array([0.5, 5.0, 50.0]))
        assert encoded[0] < encoded[1] < encoded[2]
        # 50 mm/h → ~50 dBZ → encoded ≈ 164
        assert abs(int(encoded[2]) - 164) <= 2

    def test_handles_nan_and_negative(self):
        encoded = precip_rate_to_dbz_encoded(np.array([np.nan, -1.0, 1.0]))
        assert encoded[0] == 0
        assert encoded[1] == 0
        assert encoded[2] > 0

    def test_dbz_offset_shifts_uniformly(self):
        rates = np.array([1.0, 5.0, 25.0])
        base = precip_rate_to_dbz_encoded(rates, dbz_offset=0.0)
        shifted = precip_rate_to_dbz_encoded(rates, dbz_offset=6.0)
        # +6 dBZ at encoding scale (dBZ+32)*2 = +12 pixel units
        for b, s in zip(base, shifted):
            if b > 0:
                assert int(s) - int(b) == 12

    def test_zero_rate_offset_still_zero(self):
        encoded = precip_rate_to_dbz_encoded(
            np.array([0.0, 0.0]), dbz_offset=10.0,
        )
        assert (encoded == 0).all()


# ── Decode orientation ────────────────────────────────────────────────


class TestDecodeOrientation:
    def test_decode_flips_south_up_grib(self, monkeypatch):
        from contextlib import contextmanager
        from librewxr.data import dmi_dini_grid as dmi

        # Synthetic cfgrib output: row 0 at the SOUTHERN edge.
        tp = np.zeros((DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH), dtype=np.float32)
        tp[0, 100] = 5.0    # marker at south
        tp[-1, 200] = 8.0   # marker at north

        # Build latitude coord that increases with row index (south-up).
        lat = np.broadcast_to(
            np.linspace(40.0, 70.0, DMI_DINI_GRID_HEIGHT)[:, None],
            (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH),
        )
        lon = np.broadcast_to(
            np.linspace(-25.0, 30.0, DMI_DINI_GRID_WIDTH)[None, :],
            (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH),
        )

        import xarray as xr
        fake_ds = xr.Dataset(
            {"tp": (("y", "x"), tp)},
            coords={
                "latitude": (("y", "x"), lat),
                "longitude": (("y", "x"), lon),
            },
        )

        @contextmanager
        def _noop():
            yield

        monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: fake_ds)
        monkeypatch.setattr(dmi, "_suppress_eccodes_stderr", _noop)

        arr = dmi.decode_tp_message(b"ignored")
        assert arr is not None
        # After flip: south marker (cfgrib row 0) → our row -1 (south)
        assert arr[-1, 100] == 5.0
        assert arr[0, 200] == 8.0


# ── Byte-range header walk ────────────────────────────────────────────


def _build_synthetic_grib(
    messages: list[tuple[int, ...]],
) -> bytes:
    """Build a multi-message GRIB2 byte stream.

    Each tuple is ``(disc, cat, num)`` or
    ``(disc, cat, num, fs_type, fs_value)``.  The 5-tuple form fills in
    the section-4 fixed-surface type (cur+22) and scaled value
    (cur+24..27) so callers exercising ``find_2t_message_offset`` can
    distinguish 2-m TMP (type=103, value=2) from surface / soil / level
    TMP messages that share the same parameter triplet.
    """
    out = bytearray()
    for msg in messages:
        if len(msg) == 3:
            disc, cat, num = msg
            fs_type = 0
            fs_value = 0
        elif len(msg) == 5:
            disc, cat, num, fs_type, fs_value = msg
        else:
            raise ValueError(f"unexpected message tuple length: {len(msg)}")
        sec0 = bytearray(16)
        sec0[:4] = b"GRIB"
        sec0[6] = disc
        sec0[7] = 2  # edition
        # Length filled in below
        sec1 = bytearray(21)
        struct.pack_into(">I", sec1, 0, 21)
        sec1[4] = 1  # section number
        # Section 4 PDS — at least 28 bytes so byte 22 (fs_type) and
        # bytes 24..27 (fs_value) fit, plus a little padding for the
        # walker's section_4 boundary check.
        sec4 = bytearray(34)
        struct.pack_into(">I", sec4, 0, 34)
        sec4[4] = 4
        sec4[9] = cat
        sec4[10] = num
        sec4[22] = fs_type & 0xFF
        struct.pack_into(">i", sec4, 24, int(fs_value))
        # Section 8 end marker
        sec8 = b"7777"
        msg_body = bytes(sec1) + bytes(sec4) + sec8
        msg_len = len(sec0) + len(msg_body)
        struct.pack_into(">Q", sec0, 8, msg_len)
        out.extend(sec0)
        out.extend(msg_body)
    return bytes(out)


class FakeAsyncClient:
    """Tiny stub mimicking the slice of httpx.AsyncClient the walker uses."""
    def __init__(self, content: bytes):
        self._content = content

    async def get(self, url: str, headers: dict | None = None):
        rng = headers["Range"] if headers else "bytes=0-"
        # Parse "bytes=START-END"
        spec = rng.split("=")[1]
        start, end = spec.split("-")
        start = int(start)
        end = int(end) if end else len(self._content) - 1
        chunk = self._content[start:end + 1]
        return _FakeResp(chunk)


class _FakeResp:
    def __init__(self, content: bytes):
        self.content = content
    def raise_for_status(self):
        pass


class TestHeaderWalk:
    async def test_finds_tp_message(self):
        # cat=1 num=52 in the third message
        content = _build_synthetic_grib([
            (0, 6, 5),    # cloud cover
            (0, 0, 0),    # temperature
            (0, 1, 52),   # tp ← target
            (0, 1, 8),    # other moisture variable
        ])
        client = FakeAsyncClient(content)
        loc = await find_tp_message_offset("http://fake/", client)
        assert loc is not None
        offset, size = loc
        # Each synthetic message: 16 + 21 + 34 + 4 = 75 bytes
        assert offset == 75 * 2
        assert size == 75

    async def test_returns_none_when_missing(self):
        content = _build_synthetic_grib([
            (0, 6, 5),
            (0, 0, 0),
            (0, 1, 8),    # no num=52 anywhere
        ])
        client = FakeAsyncClient(content)
        loc = await find_tp_message_offset("http://fake/", client)
        assert loc is None

    async def test_handles_truncated_content(self):
        # Only "GRIB" prefix — no length, no sections.
        client = FakeAsyncClient(b"GRIB\x00\x00\x00")
        loc = await find_tp_message_offset("http://fake/", client)
        assert loc is None


# ── Protocol + sample ─────────────────────────────────────────────────


def _inject_frame(g: DMIDiniGrid, run_ts: int, lead_seconds: int, encoded_value: int):
    """Inject a uniform-value frame into the in-memory store."""
    arr = np.full(
        (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH),
        encoded_value, dtype=np.uint8,
    )
    g._frames[(run_ts, lead_seconds)] = arr
    if g._latest_run_ts is None or run_ts > g._latest_run_ts:
        g._latest_run_ts = run_ts


@pytest.fixture
def hourly_brackets(monkeypatch):
    """Force the legacy hourly bracket behaviour for tests that inject
    frames at hourly spacing only.  Post-interpolation behaviour gets
    its own dedicated test class.
    """
    from librewxr.config import settings as _settings
    monkeypatch.setattr(_settings, "regional_interpolation", False)


class TestProtocol:
    def test_satisfies_nwpsource(self):
        g = DMIDiniGrid()
        assert isinstance(g, NWPSource)
        assert g.name == "dmi_dini"

    def test_empty_grid_returns_zeros(self):
        g = DMIDiniGrid()
        out = g.sample(np.array([52.5]), np.array([13.4]), timestamp=12345)
        assert out.shape == (1,)
        assert out[0] == 0

    def test_sample_at_exact_bracket(self, hourly_brackets):
        g = DMIDiniGrid()
        run = int(datetime(2026, 5, 3, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 100)
        _inject_frame(g, run, 7200, 100)
        out = g.sample(np.array([52.52]), np.array([13.41]), timestamp=run + 3600)
        assert int(out[0]) == 100

    def test_sample_lerps_between_brackets(self, hourly_brackets):
        g = DMIDiniGrid()
        run = int(datetime(2026, 5, 3, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 80)
        _inject_frame(g, run, 7200, 160)
        out = g.sample(np.array([52.52]), np.array([13.41]), timestamp=run + 5400)
        assert abs(int(out[0]) - 120) <= 1

    def test_outside_domain_zero(self):
        g = DMIDiniGrid()
        run = int(datetime(2026, 5, 3, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 200)
        _inject_frame(g, run, 7200, 200)
        # Madrid is south of DINI; NYC is far west
        for lat, lon in [(40.42, -3.70), (40.71, -74.01)]:
            out = g.sample(np.array([lat]), np.array([lon]), timestamp=run + 3600)
            assert int(out[0]) == 0

    def test_has_data_at_within_horizon(self, hourly_brackets):
        # has_data_at requires BOTH bracketing frames loaded — same
        # convention as HRRRGrid / ICONEUGrid.  At an exact frame
        # boundary the bracket is (L, L+interval), so we need three
        # frames to assert "True" across the (L, L+interval) interior.
        g = DMIDiniGrid()
        run = int(datetime(2026, 5, 3, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 50)
        _inject_frame(g, run, 7200, 50)
        _inject_frame(g, run, 10800, 50)
        assert g.has_data_at(run + 3600) is True
        assert g.has_data_at(run + 5400) is True
        assert g.has_data_at(run + 7200) is True
        # Beyond the last loaded bracket (lead 10800 needs frame at 14400)
        assert g.has_data_at(run + 11000) is False


# ── Chain integration ────────────────────────────────────────────────


class TestChainOrdering:
    def test_chain_prefers_dini_inside_dini_falls_back_outside(self, hourly_brackets):
        # Build a minimal DINI in front of a global IFS fallback.  Inside
        # DINI the chain should return DINI's value; outside DINI it
        # should return the IFS value.
        from librewxr.data.ecmwf_grid import ECMWFGrid
        from librewxr.data.ecmwf_grid import (
            GRID_HEIGHT as IFS_H, GRID_WIDTH as IFS_W,
        )

        ifs = ECMWFGrid()
        ifs_dbz = np.full((IFS_H, IFS_W), 84, dtype=np.uint8)  # 10 dBZ
        ifs._timesteps[1000000] = (ifs_dbz, np.zeros_like(ifs_dbz, dtype=bool))
        ifs._sorted_timestamps = [1000000]

        dini = DMIDiniGrid()
        run = 1000000 - 1500
        _inject_frame(dini, run, 0, 164)      # 50 dBZ, exact bracket
        _inject_frame(dini, run, 3600, 164)

        chain = NWPChain([dini, ifs])
        # Berlin: inside DINI
        out_eu = chain.sample(np.array([52.52]), np.array([13.41]), timestamp=1000000)
        assert abs(int(out_eu[0]) - 164) <= 1
        # NYC: outside DINI → IFS fills
        out_us = chain.sample(np.array([40.71]), np.array([-74.01]), timestamp=1000000)
        assert int(out_us[0]) == 84


# ── Optical-flow interpolation ────────────────────────────────────────


def _make_blob(
    cy: int, cx: int, radius: int = 30, value: int = 150,
) -> np.ndarray:
    """Build a test precip grid with a circular blob at (cy, cx)."""
    grid = np.zeros((DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH), dtype=np.uint8)
    ys, xs = np.ogrid[0:DMI_DINI_GRID_HEIGHT, 0:DMI_DINI_GRID_WIDTH]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
    grid[mask] = value
    return grid


class TestInterpolateRunFrames:
    """``_interpolate_run_frames`` fills 10-min synthetics between hourly originals."""

    def test_fills_synthetic_leads_between_hourly_originals(self):
        # Inject two hourly originals; expect 5 synthetics at 600s steps.
        grid = DMIDiniGrid()
        run_ts = int(datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp())
        f0 = _make_blob(400, 600)
        f1 = _make_blob(400, 650)   # blob translated by 50 px east
        grid._frames[(run_ts, 0)] = f0
        grid._frames[(run_ts, 3600)] = f1
        grid._latest_run_ts = run_ts

        added = grid._interpolate_run_frames(run_ts)
        assert added == 5  # leads 600, 1200, 1800, 2400, 3000
        for lead in (600, 1200, 1800, 2400, 3000):
            assert (run_ts, lead) in grid._frames
            arr = grid._frames[(run_ts, lead)]
            assert arr.shape == (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH)
            assert arr.dtype == np.uint8

    def test_idempotent_on_second_call(self):
        grid = DMIDiniGrid()
        run_ts = int(datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp())
        grid._frames[(run_ts, 0)] = _make_blob(400, 600)
        grid._frames[(run_ts, 3600)] = _make_blob(400, 650)
        grid._latest_run_ts = run_ts

        first = grid._interpolate_run_frames(run_ts)
        second = grid._interpolate_run_frames(run_ts)
        assert first == 5
        assert second == 0

    def test_no_snow_masks_when_none_loaded(self):
        # When no snow masks are loaded for the run, the interpolator
        # produces precip frames only — no synthetic snow side appears.
        grid = DMIDiniGrid()
        run_ts = int(datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp())
        grid._frames[(run_ts, 0)] = _make_blob(400, 600)
        grid._frames[(run_ts, 3600)] = _make_blob(400, 650)
        grid._latest_run_ts = run_ts

        grid._interpolate_run_frames(run_ts)
        # Snow dict exists (DMI DINI supports snow now) but is empty
        # because no T2 data was decoded.
        assert grid.snow_mask_count == 0

    def test_returns_zero_when_run_has_one_or_fewer_frames(self):
        grid = DMIDiniGrid()
        run_ts = int(datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp())
        # No frames yet
        assert grid._interpolate_run_frames(run_ts) == 0
        # Only one frame
        grid._frames[(run_ts, 0)] = _make_blob(400, 600)
        assert grid._interpolate_run_frames(run_ts) == 0

    def test_skips_other_runs(self):
        grid = DMIDiniGrid()
        run_a = int(datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp())
        run_b = run_a + 3 * 3600  # DMI cycle interval is 3 h
        grid._frames[(run_a, 0)] = _make_blob(400, 600)
        grid._frames[(run_a, 3600)] = _make_blob(400, 650)
        grid._frames[(run_b, 0)] = _make_blob(400, 600)
        grid._frames[(run_b, 3600)] = _make_blob(400, 650)

        added_a = grid._interpolate_run_frames(run_a)
        assert added_a == 5
        # run_b untouched until its own _interpolate_run_frames call
        run_b_leads = [lead for (r, lead) in grid._frames if r == run_b]
        assert sorted(run_b_leads) == [0, 3600]


class TestPostInterpolationBracket:
    """Sample uses 10-min brackets when frames are interpolated."""

    @pytest.mark.asyncio
    async def test_sample_uses_10min_bracket_when_interpolation_enabled(self, tmp_path):
        grid = DMIDiniGrid(cache_dir=tmp_path)
        run_ts = int(datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp())
        f0 = _make_blob(400, 600)
        f1 = _make_blob(400, 650)
        mm0 = grid._to_memmap(f"r{run_ts}_l0", f0)
        mm1 = grid._to_memmap(f"r{run_ts}_l3600", f1)
        grid._frames[(run_ts, 0)] = mm0
        grid._frames[(run_ts, 3600)] = mm1
        grid._latest_run_ts = run_ts

        # Interpolate to populate 600s synthetics.
        grid._interpolate_run_frames(run_ts)

        # Bracket at 25 min in should be (1200, 1800), alpha=0.5.
        ts = run_ts + 25 * 60
        l0, l1, alpha = bracket_lead_seconds(ts - run_ts, 600)
        assert l0 == 1200
        assert l1 == 1800
        assert alpha == pytest.approx(0.5)
        assert (run_ts, 1200) in grid._frames
        assert (run_ts, 1800) in grid._frames

        # _pick_run finds the run via the 600s bracket lookup.
        assert grid._pick_run(ts) == run_ts
        await grid.close()


class TestRegionalInterpolationToggle:
    """The bracket interval follows ``LIBREWXR_REGIONAL_INTERPOLATION``."""

    def test_bracket_interval_is_hourly_when_disabled(self, hourly_brackets):
        grid = DMIDiniGrid()
        assert grid._bracket_interval() == SOURCE_STEP_SECONDS

    def test_bracket_interval_is_10min_when_enabled(self):
        # Default settings have it enabled.
        grid = DMIDiniGrid()
        assert grid._bracket_interval() == STORED_INTERVAL_SECONDS


# ── 2t header walk ───────────────────────────────────────────────────


class Test2THeaderWalk:
    """``find_2t_message_offset`` matches 2-m TMP via discipline + cat + num + fixed-surface."""

    async def test_finds_2t_message(self):
        # First message is surface TMP (type=1), second is 2-m TMP
        # (type=103, value=2) → target.  The walker should skip the
        # first and return the second.
        content = _build_synthetic_grib([
            (0, 0, 0, 1, 0),      # surface temperature — same triplet, different surface
            (0, 0, 0, 103, 2),    # 2-m TMP ← target
            (0, 1, 52, 0, 0),     # tp (irrelevant for this walker)
        ])
        client = FakeAsyncClient(content)
        loc = await find_2t_message_offset("http://fake/", client)
        assert loc is not None
        offset, size = loc
        # Each synthetic message: 16 + 21 + 34 + 4 = 75 bytes
        assert offset == 75
        assert size == 75

    async def test_returns_none_when_no_2m_temperature(self):
        # Temperature messages exist but none at the 2-m height.
        content = _build_synthetic_grib([
            (0, 0, 0, 1, 0),      # surface
            (0, 0, 0, 100, 850),  # 850 hPa
            (0, 1, 52, 0, 0),
        ])
        client = FakeAsyncClient(content)
        loc = await find_2t_message_offset("http://fake/", client)
        assert loc is None

    async def test_returns_none_when_truncated(self):
        client = FakeAsyncClient(b"GRIB\x00\x00\x00")
        loc = await find_2t_message_offset("http://fake/", client)
        assert loc is None


# ── 2t decode orientation ────────────────────────────────────────────


class TestDecode2TOrientation:
    """``decode_2t_message`` flips south-up GRIBs and converts Kelvin → Celsius."""

    def test_decode_flips_south_up_and_converts_kelvin(self, monkeypatch):
        from contextlib import contextmanager
        from librewxr.data import dmi_dini_grid as dmi

        # Synthetic cfgrib output in Kelvin: row 0 at southern edge.
        # 263.15 K = -10 °C (cold north), 283.15 K = +10 °C (warm south)
        t2 = np.full(
            (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH), 273.15, dtype=np.float32,
        )
        t2[0, 100] = 283.15   # marker at south (cfgrib row 0)
        t2[-1, 200] = 263.15  # marker at north (cfgrib row -1)

        lat = np.broadcast_to(
            np.linspace(40.0, 70.0, DMI_DINI_GRID_HEIGHT)[:, None],
            (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH),
        )
        lon = np.broadcast_to(
            np.linspace(-25.0, 30.0, DMI_DINI_GRID_WIDTH)[None, :],
            (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH),
        )

        import xarray as xr
        fake_ds = xr.Dataset(
            {"t2m": (("y", "x"), t2)},
            coords={
                "latitude": (("y", "x"), lat),
                "longitude": (("y", "x"), lon),
            },
        )

        @contextmanager
        def _noop():
            yield

        monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: fake_ds)
        monkeypatch.setattr(dmi, "_suppress_eccodes_stderr", _noop)

        arr = decode_2t_message(b"ignored")
        assert arr is not None
        # After flip: south marker (cfgrib row 0) → our row -1 (south)
        # After Kelvin → Celsius: 283.15 K - 273.15 = 10 °C
        assert arr[-1, 100] == pytest.approx(10.0)
        assert arr[0, 200] == pytest.approx(-10.0)


# ── Snow mask ────────────────────────────────────────────────────────


def _inject_frame_and_snow(
    grid: DMIDiniGrid,
    run_ts: int,
    lead_seconds: int,
    *,
    snow_value: int | None = None,
) -> None:
    """Inject a uniform precip frame, optionally with a parallel snow mask."""
    fake = np.zeros(
        (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH), dtype=np.uint8,
    )
    grid._frames[(run_ts, lead_seconds)] = fake
    if grid._latest_run_ts is None or run_ts > grid._latest_run_ts:
        grid._latest_run_ts = run_ts
    if snow_value is not None:
        snow = np.full(
            (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH),
            snow_value & 0x01,
            dtype=np.uint8,
        )
        grid._snow_masks[(run_ts, lead_seconds)] = snow


class TestSnowMask:
    """DMIDiniGrid.get_snow_mask end-to-end behaviour."""

    def test_supports_snow_is_true(self):
        grid = DMIDiniGrid()
        assert grid.supports_snow is True

    def test_no_loaded_masks_returns_all_false(self):
        # Snow masks empty → falls through so the chain dispatcher
        # reaches the next snow-capable source.
        grid = DMIDiniGrid()
        out = grid.get_snow_mask(np.array([55.68]), np.array([12.57]))
        assert out.dtype == np.bool_
        assert not out.any()

    def test_uniform_snow_returns_true_in_domain(self, hourly_brackets):
        grid = DMIDiniGrid()
        run = int(datetime(2026, 5, 3, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame_and_snow(grid, run, 3 * 3600, snow_value=1)
        _inject_frame_and_snow(grid, run, 4 * 3600, snow_value=1)

        # Copenhagen — inside DINI's domain
        out = grid.get_snow_mask(
            np.array([55.68]), np.array([12.57]),
            timestamp=run + 3 * 3600 + 1800,
        )
        assert out.tolist() == [True]

    def test_uniform_rain_returns_false_in_domain(self, hourly_brackets):
        grid = DMIDiniGrid()
        run = int(datetime(2026, 5, 3, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame_and_snow(grid, run, 3 * 3600, snow_value=0)
        _inject_frame_and_snow(grid, run, 4 * 3600, snow_value=0)

        out = grid.get_snow_mask(
            np.array([52.52]), np.array([13.41]),   # Berlin
            timestamp=run + 3 * 3600 + 1800,
        )
        assert out.tolist() == [False]

    def test_outside_domain_returns_false(self, hourly_brackets):
        grid = DMIDiniGrid()
        run = int(datetime(2026, 5, 3, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame_and_snow(grid, run, 3 * 3600, snow_value=1)
        _inject_frame_and_snow(grid, run, 4 * 3600, snow_value=1)

        # Madrid is south of DINI; should sample as False outside-domain
        out = grid.get_snow_mask(
            np.array([40.42]), np.array([-3.70]),
            timestamp=run + 3 * 3600 + 1800,
        )
        assert out.tolist() == [False]

    def test_lerp_bracket_majority_at_midpoint(self, hourly_brackets):
        grid = DMIDiniGrid()
        run = int(datetime(2026, 5, 3, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame_and_snow(grid, run, 3 * 3600, snow_value=0)  # L0: rain
        _inject_frame_and_snow(grid, run, 4 * 3600, snow_value=1)  # L1: snow

        # alpha=0.25 → L0 wins
        out_low = grid.get_snow_mask(
            np.array([55.68]), np.array([12.57]),
            timestamp=run + 3 * 3600 + 15 * 60,
        )
        assert out_low.tolist() == [False]

        # alpha=0.75 → L1 wins
        out_high = grid.get_snow_mask(
            np.array([55.68]), np.array([12.57]),
            timestamp=run + 3 * 3600 + 45 * 60,
        )
        assert out_high.tolist() == [True]

    def test_partial_bracket_returns_false(self, hourly_brackets):
        # Precip frames at both leads, but only L0 has a snow mask.
        grid = DMIDiniGrid()
        run = int(datetime(2026, 5, 3, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame_and_snow(grid, run, 3 * 3600, snow_value=1)
        _inject_frame_and_snow(grid, run, 4 * 3600, snow_value=None)

        out = grid.get_snow_mask(
            np.array([55.68]), np.array([12.57]),
            timestamp=run + 3 * 3600 + 1800,
        )
        assert not out.any()


class TestSnowMaskPersistence:
    """Snow masks are atomic-write parallel files alongside precip frames."""

    @pytest.mark.asyncio
    async def test_snow_mask_round_trips_through_disk(self, tmp_path, hourly_brackets):
        run_ts = int(datetime(2026, 5, 3, 6, tzinfo=timezone.utc).timestamp())

        g1 = DMIDiniGrid(cache_dir=tmp_path)
        fake_precip = np.zeros(
            (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH), dtype=np.uint8,
        )
        fake_snow = np.ones(
            (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH), dtype=np.uint8,
        )
        for lead in (3 * 3600, 4 * 3600):
            mm = g1._to_memmap(f"r{run_ts}_l{lead}", fake_precip)
            g1._frames[(run_ts, lead)] = mm
            mm_s = g1._to_memmap(f"r{run_ts}_l{lead}_snow", fake_snow)
            g1._snow_masks[(run_ts, lead)] = mm_s
        g1._latest_run_ts = run_ts
        await g1.close()

        cache_dir = tmp_path / "dmi_dini"
        assert (cache_dir / f"r{run_ts}_l{3*3600}.dat").exists()
        assert (cache_dir / f"r{run_ts}_l{3*3600}_snow.dat").exists()

        g2 = DMIDiniGrid(cache_dir=tmp_path)
        assert g2.frame_count == 2
        assert g2.snow_mask_count == 2
        assert (run_ts, 3 * 3600) in g2._snow_masks
        assert (run_ts, 4 * 3600) in g2._snow_masks

        sample_ts = run_ts + 3 * 3600 + 1800
        out = g2.get_snow_mask(
            np.array([55.68]), np.array([12.57]),
            timestamp=sample_ts,
        )
        assert out.tolist() == [True]
        await g2.close()

    @pytest.mark.asyncio
    async def test_orphan_snow_mask_is_removed(self, tmp_path):
        cache_dir = tmp_path / "dmi_dini"
        cache_dir.mkdir(parents=True)
        orphan = cache_dir / "r1234_l3600_snow.dat"
        size = DMI_DINI_GRID_HEIGHT * DMI_DINI_GRID_WIDTH
        orphan.write_bytes(b"\x00" * size)
        assert orphan.exists()

        g = DMIDiniGrid(cache_dir=tmp_path)
        assert not orphan.exists()
        assert g.snow_mask_count == 0
        await g.close()

    @pytest.mark.asyncio
    async def test_eviction_removes_snow_files_too(self, tmp_path):
        run_ts = int(datetime(2026, 5, 3, 6, tzinfo=timezone.utc).timestamp())
        g = DMIDiniGrid(cache_dir=tmp_path)
        fake_precip = np.zeros(
            (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH), dtype=np.uint8,
        )
        fake_snow = np.ones(
            (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH), dtype=np.uint8,
        )
        g._to_memmap(f"r{run_ts}_l3600", fake_precip)
        g._to_memmap(f"r{run_ts}_l3600_snow", fake_snow)
        mm = np.memmap(
            tmp_path / "dmi_dini" / f"r{run_ts}_l3600.dat",
            dtype=np.uint8, mode="r",
            shape=(DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH),
        )
        g._frames[(run_ts, 3600)] = mm
        mm_s = np.memmap(
            tmp_path / "dmi_dini" / f"r{run_ts}_l3600_snow.dat",
            dtype=np.uint8, mode="r",
            shape=(DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH),
        )
        g._snow_masks[(run_ts, 3600)] = mm_s

        far_future = run_ts + 7 * 24 * 3600
        g._evict_outside_window(far_future, far_future + 600)
        assert (run_ts, 3600) not in g._frames
        assert (run_ts, 3600) not in g._snow_masks
        assert not (tmp_path / "dmi_dini" / f"r{run_ts}_l3600.dat").exists()
        assert not (tmp_path / "dmi_dini" / f"r{run_ts}_l3600_snow.dat").exists()
        await g.close()


class TestChainSnowMaskWithDMIDini:
    def test_chain_prefers_dini_snow_inside_domain(self, hourly_brackets):
        from librewxr.data.ecmwf_grid import ECMWFGrid
        from librewxr.data.ecmwf_grid import GRID_HEIGHT as IFS_H, GRID_WIDTH as IFS_W

        # IFS says snow everywhere; DINI says rain inside its domain.
        # Inside DINI, DINI wins → rain.  Outside (NYC), IFS wins → snow.
        ifs = ECMWFGrid()
        ifs_dbz = np.full((IFS_H, IFS_W), int((10 + 32) * 2), dtype=np.uint8)
        ifs_snow = np.ones((IFS_H, IFS_W), dtype=bool)
        ifs._timesteps[1000000] = (ifs_dbz, ifs_snow)
        ifs._sorted_timestamps = [1000000]

        dini = DMIDiniGrid()
        run = 1000000 - 1800   # 30 min into the (0, 3600) bracket
        _inject_frame_and_snow(dini, run, 0, snow_value=0)
        _inject_frame_and_snow(dini, run, 3600, snow_value=0)

        chain = NWPChain([dini, ifs])

        # Copenhagen: DINI says rain → False
        out = chain.get_snow_mask(
            np.array([55.68]), np.array([12.57]), timestamp=1000000,
        )
        assert out.tolist() == [False]

        # New York: outside DINI → IFS wins → True
        out = chain.get_snow_mask(
            np.array([40.71]), np.array([-74.01]), timestamp=1000000,
        )
        assert out.tolist() == [True]
