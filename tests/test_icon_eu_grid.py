# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Unit tests for ICON-EU grid math, decode orientation, and chain integration."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.icon_eu

from librewxr.sources.regional.europe.nwp.icon_eu.grid import (
    BRACKET_INTERVAL_SECONDS,
    CYCLE_INTERVAL_SECONDS,
    ICON_EU_GRID_HEIGHT,
    ICON_EU_GRID_WIDTH,
    ICON_EU_LAT_MAX,
    ICON_EU_LAT_MIN,
    SOURCE_STEP_SECONDS,
    STORED_INTERVAL_SECONDS,
    ICONEUGrid,
    bracket_lead_seconds,
    decode_tp_message,
    decode_t_2m_message,
    domain_mask,
    feather_mask,
    file_url,
    floor_cycle,
    grid_indices,
    latest_published_run,
    precip_rate_to_dbz_encoded,
)
from librewxr.data.nwp_source import NWPChain, NWPSource


# ── Lat/lon grid ──────────────────────────────────────────────────────


class TestLatLonGrid:
    @pytest.mark.parametrize(
        "name,lat,lon,inside",
        [
            ("Berlin",     52.52,  13.40, True),
            ("London",     51.51,  -0.13, True),
            ("Madrid",     40.42,  -3.70, True),
            ("Stockholm",  59.33,  18.07, True),
            ("Athens",     37.98,  23.73, True),
            ("Iceland",    64.13, -21.94, True),
            ("NYC",        40.71, -74.01, False),
            ("Tokyo",      35.68, 139.69, False),
            ("Cape Town", -33.92,  18.42, False),
        ],
    )
    def test_domain_mask_known_points(self, name, lat, lon, inside):
        m = domain_mask(np.array([lat]), np.array([lon]))
        assert bool(m[0]) is inside, name

    def test_grid_origin_at_north_west(self):
        # Row 0 should be the NORTHERN edge (after our orientation flip).
        # Project a point at the NW corner and expect (row=0, col=0).
        row, col = grid_indices(np.array([ICON_EU_LAT_MAX]), np.array([-23.5]))
        assert abs(row[0] - 0) < 1e-6
        assert abs(col[0] - 0) < 1e-6

    def test_grid_origin_at_south_east(self):
        # Bottom-right corner.
        row, col = grid_indices(np.array([ICON_EU_LAT_MIN]), np.array([62.5]))
        assert abs(row[0] - (ICON_EU_GRID_HEIGHT - 1)) < 1e-6
        assert abs(col[0] - (ICON_EU_GRID_WIDTH - 1)) < 1e-6


# ── Feather ───────────────────────────────────────────────────────────


class TestFeatherMask:
    def test_inside_full_weight(self):
        # Far from edges → 1.0
        f = feather_mask(np.array([50.0]), np.array([10.0]))
        assert f.dtype == np.float32
        assert f[0] == pytest.approx(1.0)

    def test_outside_zero(self):
        # NYC → 0
        f = feather_mask(np.array([40.71]), np.array([-74.01]))
        assert f[0] == 0.0

    def test_taper_monotonic_at_north_edge(self):
        # Walk lat from inside to outside
        lats = np.linspace(70.0, 72.0, 21)
        lons = np.full_like(lats, 10.0)
        f = feather_mask(lats, lons)
        diffs = np.diff(f)
        assert (diffs <= 1e-6).all()


# ── Timing ────────────────────────────────────────────────────────────


class TestTiming:
    def test_floor_cycle_3h(self):
        # Friday 14:23 UTC → 12:00 UTC (3h cycle floor)
        ts = int(datetime(2026, 5, 1, 14, 23, tzinfo=timezone.utc).timestamp())
        floored = floor_cycle(ts)
        expected = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        assert floored == expected

    def test_floor_cycle_at_boundary(self):
        ts = int(datetime(2026, 5, 1, 15, 0, tzinfo=timezone.utc).timestamp())
        assert floor_cycle(ts) == ts

    def test_latest_published_run(self):
        now = int(datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc).timestamp())
        # 4h delay → should pick floor_cycle(10:00) = 09:00
        run = latest_published_run(now, 4 * 3600)
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
    def test_format_matches_dwd_pattern(self):
        run = datetime(2026, 5, 1, 12, tzinfo=timezone.utc)
        url = file_url(run, 5, "tot_prec")
        assert url.endswith(
            "/12/tot_prec/icon-eu_europe_regular-lat-lon_single-level_"
            "2026050112_005_TOT_PREC.grib2.bz2"
        )

    def test_step_zero_padded(self):
        run = datetime(2026, 5, 1, 0, tzinfo=timezone.utc)
        url = file_url(run, 0, "tot_prec")
        assert "_000_" in url


# ── Z-R conversion ────────────────────────────────────────────────────


class TestZR:
    def test_zero_rate_zero_encoded(self):
        encoded = precip_rate_to_dbz_encoded(np.array([0.0, 0.0]))
        assert (encoded == 0).all()

    def test_higher_rate_higher_dbz(self):
        encoded = precip_rate_to_dbz_encoded(np.array([0.5, 5.0, 50.0]))
        # Monotonically increasing
        assert encoded[0] < encoded[1] < encoded[2]
        # 50 mm/h hits ~50 dBZ; encoded = (50 + 32) * 2 = 164
        assert abs(int(encoded[2]) - 164) <= 2

    def test_handles_nan_and_negative(self):
        # NaN → 0, negative → 0
        encoded = precip_rate_to_dbz_encoded(
            np.array([np.nan, -1.0, 1.0])
        )
        assert encoded[0] == 0
        assert encoded[1] == 0
        assert encoded[2] > 0

    def test_dbz_offset_shifts_uniformly(self):
        # Encoding is pixel = (dBZ + 32) * 2, so +6 dBZ = +12 pixels.
        rates = np.array([1.0, 5.0, 25.0])
        base = precip_rate_to_dbz_encoded(rates, dbz_offset=0.0)
        shifted = precip_rate_to_dbz_encoded(rates, dbz_offset=6.0)
        for b, s in zip(base, shifted):
            if b > 0:
                assert int(s) - int(b) == 12

    def test_zero_rate_offset_still_zero(self):
        # Even with a positive offset, zero rate must encode to 0.
        encoded = precip_rate_to_dbz_encoded(
            np.array([0.0, 0.0]), dbz_offset=10.0,
        )
        assert (encoded == 0).all()


# ── Decode orientation ────────────────────────────────────────────────


class TestDecodeOrientation:
    def test_decode_flips_south_up_grib(self, monkeypatch):
        from contextlib import contextmanager
        from librewxr.sources.regional.europe.nwp.icon_eu import grid as iem

        # Synthetic cfgrib output: row 0 at the SOUTHERN edge.
        tp = np.zeros((ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH), dtype=np.float32)
        tp[0, 100] = 5.0   # marker at south
        tp[-1, 200] = 8.0  # marker at north

        lat = np.broadcast_to(
            np.linspace(ICON_EU_LAT_MIN, ICON_EU_LAT_MAX, ICON_EU_GRID_HEIGHT)[:, None],
            (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH),
        )
        lon = np.broadcast_to(
            np.linspace(-23.5, 62.5, ICON_EU_GRID_WIDTH)[None, :],
            (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH),
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
        monkeypatch.setattr(iem, "_suppress_eccodes_stderr", _noop)

        arr = iem.decode_tp_message(b"ignored")
        assert arr is not None
        # After flip: south marker (cfgrib row 0) → our row -1
        assert arr[-1, 100] == 5.0
        assert arr[0, 200] == 8.0


# ── Protocol + chain ──────────────────────────────────────────────────


def _inject_frame(g: ICONEUGrid, run_ts: int, lead_seconds: int, encoded_value: int):
    """Inject a uniform-value frame into the in-memory store."""
    arr = np.full(
        (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH),
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
        g = ICONEUGrid()
        assert isinstance(g, NWPSource)
        assert g.name == "icon_eu"

    def test_empty_grid_returns_zeros(self):
        g = ICONEUGrid()
        out = g.sample(np.array([52.5]), np.array([13.4]), timestamp=12345)
        assert out.shape == (1,)
        assert out[0] == 0

    def test_sample_at_exact_bracket(self, hourly_brackets):
        g = ICONEUGrid()
        run = int(datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 100)
        _inject_frame(g, run, 7200, 100)
        # Sample at run + 3600 (exact L0 frame)
        out = g.sample(np.array([52.5]), np.array([13.4]), timestamp=run + 3600)
        assert int(out[0]) == 100

    def test_sample_lerps_between_brackets(self, hourly_brackets):
        g = ICONEUGrid()
        run = int(datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 80)
        _inject_frame(g, run, 7200, 160)
        # Mid-point = (80 + 160) / 2 = 120
        out = g.sample(np.array([52.5]), np.array([13.4]), timestamp=run + 5400)
        assert abs(int(out[0]) - 120) <= 1

    def test_outside_domain_zero(self):
        g = ICONEUGrid()
        run = int(datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 200)
        _inject_frame(g, run, 7200, 200)
        # NYC is outside Europe
        out = g.sample(np.array([40.71]), np.array([-74.01]), timestamp=run + 3600)
        assert int(out[0]) == 0


class TestChainOrdering:
    def test_chain_prefers_icon_eu_inside_europe(self, hourly_brackets):
        from librewxr.sources.world.ifs.grid import ECMWFGrid
        from librewxr.sources.world.ifs.grid import (
            GRID_HEIGHT as IFS_H, GRID_WIDTH as IFS_W,
        )

        ifs = ECMWFGrid()
        ifs_dbz = np.full((IFS_H, IFS_W), 84, dtype=np.uint8)  # 10 dBZ
        ifs._timesteps[1000000] = (ifs_dbz, np.zeros_like(ifs_dbz, dtype=bool))
        ifs._sorted_timestamps = [1000000]

        icon_eu = ICONEUGrid()
        run = 1000000 - 1500
        _inject_frame(icon_eu, run, 0, 164)     # 50 dBZ
        _inject_frame(icon_eu, run, 3600, 164)

        chain = NWPChain([icon_eu, ifs])
        # Inside Europe: ICON-EU dominates
        out_eu = chain.sample(np.array([52.5]), np.array([13.4]), timestamp=1000000)
        assert abs(int(out_eu[0]) - 164) <= 1
        # Outside Europe (NYC): IFS fills
        out_us = chain.sample(np.array([40.71]), np.array([-74.01]), timestamp=1000000)
        assert int(out_us[0]) == 84


# ── Optical-flow interpolation ────────────────────────────────────────


def _make_blob(
    cy: int, cx: int, radius: int = 25, value: int = 150,
) -> np.ndarray:
    """Build a test precip grid with a circular blob at (cy, cx)."""
    grid = np.zeros((ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH), dtype=np.uint8)
    ys, xs = np.ogrid[0:ICON_EU_GRID_HEIGHT, 0:ICON_EU_GRID_WIDTH]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
    grid[mask] = value
    return grid


class TestInterpolateRunFrames:
    """``_interpolate_run_frames`` fills 10-min synthetics between hourly originals."""

    def test_fills_synthetic_leads_between_hourly_originals(self):
        grid = ICONEUGrid()
        run_ts = int(datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp())
        f0 = _make_blob(300, 500)
        f1 = _make_blob(300, 540)   # blob translated 40 px east
        grid._frames[(run_ts, 0)] = f0
        grid._frames[(run_ts, 3600)] = f1
        grid._latest_run_ts = run_ts

        added = grid._interpolate_run_frames(run_ts)
        assert added == 5  # leads 600, 1200, 1800, 2400, 3000
        for lead in (600, 1200, 1800, 2400, 3000):
            assert (run_ts, lead) in grid._frames
            arr = grid._frames[(run_ts, lead)]
            assert arr.shape == (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH)
            assert arr.dtype == np.uint8

    def test_idempotent_on_second_call(self):
        grid = ICONEUGrid()
        run_ts = int(datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp())
        grid._frames[(run_ts, 0)] = _make_blob(300, 500)
        grid._frames[(run_ts, 3600)] = _make_blob(300, 540)
        grid._latest_run_ts = run_ts

        first = grid._interpolate_run_frames(run_ts)
        second = grid._interpolate_run_frames(run_ts)
        assert first == 5
        assert second == 0

    def test_no_snow_masks_when_none_loaded(self):
        # When no snow masks are loaded for the run, the interpolator
        # produces precip frames only — no synthetic snow side appears.
        grid = ICONEUGrid()
        run_ts = int(datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp())
        grid._frames[(run_ts, 0)] = _make_blob(300, 500)
        grid._frames[(run_ts, 3600)] = _make_blob(300, 540)
        grid._latest_run_ts = run_ts

        grid._interpolate_run_frames(run_ts)
        # Snow dict exists (ICON-EU supports snow now) but is empty
        # because no T_2M data was decoded.
        assert grid.snow_mask_count == 0

    def test_returns_zero_when_run_has_one_or_fewer_frames(self):
        grid = ICONEUGrid()
        run_ts = int(datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp())
        # No frames yet
        assert grid._interpolate_run_frames(run_ts) == 0
        # Only one frame
        grid._frames[(run_ts, 0)] = _make_blob(300, 500)
        assert grid._interpolate_run_frames(run_ts) == 0

    def test_skips_other_runs(self):
        grid = ICONEUGrid()
        run_a = int(datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp())
        run_b = run_a + 3 * 3600   # ICON-EU cycle interval is 3 h
        grid._frames[(run_a, 0)] = _make_blob(300, 500)
        grid._frames[(run_a, 3600)] = _make_blob(300, 540)
        grid._frames[(run_b, 0)] = _make_blob(300, 500)
        grid._frames[(run_b, 3600)] = _make_blob(300, 540)

        added_a = grid._interpolate_run_frames(run_a)
        assert added_a == 5
        run_b_leads = [lead for (r, lead) in grid._frames if r == run_b]
        assert sorted(run_b_leads) == [0, 3600]


class TestPostInterpolationBracket:
    """Sample uses 10-min brackets when frames are interpolated."""

    @pytest.mark.asyncio
    async def test_sample_uses_10min_bracket_when_interpolation_enabled(self, tmp_path):
        grid = ICONEUGrid(cache_dir=tmp_path)
        run_ts = int(datetime(2026, 5, 8, 6, tzinfo=timezone.utc).timestamp())
        f0 = _make_blob(300, 500)
        f1 = _make_blob(300, 540)
        mm0 = grid._to_memmap(f"r{run_ts}_l0", f0)
        mm1 = grid._to_memmap(f"r{run_ts}_l3600", f1)
        grid._frames[(run_ts, 0)] = mm0
        grid._frames[(run_ts, 3600)] = mm1
        grid._latest_run_ts = run_ts

        grid._interpolate_run_frames(run_ts)

        # Bracket at 25 min in should be (1200, 1800), alpha=0.5.
        ts = run_ts + 25 * 60
        l0, l1, alpha = bracket_lead_seconds(ts - run_ts, 600)
        assert l0 == 1200
        assert l1 == 1800
        assert alpha == pytest.approx(0.5)
        assert (run_ts, 1200) in grid._frames
        assert (run_ts, 1800) in grid._frames
        assert grid._pick_run(ts) == run_ts
        await grid.close()


class TestRegionalInterpolationToggle:
    """The bracket interval follows ``LIBREWXR_REGIONAL_INTERPOLATION``."""

    def test_bracket_interval_is_hourly_when_disabled(self, hourly_brackets):
        grid = ICONEUGrid()
        assert grid._bracket_interval() == SOURCE_STEP_SECONDS

    def test_bracket_interval_is_10min_when_enabled(self):
        grid = ICONEUGrid()
        assert grid._bracket_interval() == STORED_INTERVAL_SECONDS


# ── T_2M decode orientation ──────────────────────────────────────────


class TestDecodeT2MOrientation:
    """``decode_t_2m_message`` flips south-up GRIBs and converts Kelvin → Celsius."""

    def test_decode_flips_south_up_and_converts_kelvin(self, monkeypatch):
        from contextlib import contextmanager
        from librewxr.sources.regional.europe.nwp.icon_eu import grid as icon_eu

        # Synthetic cfgrib output in Kelvin: row 0 at southern edge.
        # 283.15 K = +10 °C (warm south), 263.15 K = -10 °C (cold north)
        t2 = np.full(
            (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH), 273.15, dtype=np.float32,
        )
        t2[0, 100] = 283.15   # cfgrib row 0 = south
        t2[-1, 200] = 263.15  # cfgrib row -1 = north

        lat = np.linspace(
            ICON_EU_LAT_MIN, ICON_EU_LAT_MAX, ICON_EU_GRID_HEIGHT,
        )
        lon = np.linspace(-23.5, 62.5, ICON_EU_GRID_WIDTH)

        import xarray as xr
        fake_ds = xr.Dataset(
            {"t2m": (("latitude", "longitude"), t2)},
            coords={"latitude": ("latitude", lat),
                    "longitude": ("longitude", lon)},
        )

        @contextmanager
        def _noop():
            yield

        monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: fake_ds)
        monkeypatch.setattr(icon_eu, "_suppress_eccodes_stderr", _noop)

        arr = decode_t_2m_message(b"ignored")
        assert arr is not None
        # After flip: south marker → our row -1 (south)
        # After Kelvin → Celsius: 283.15 K - 273.15 = 10 °C
        assert arr[-1, 100] == pytest.approx(10.0)
        assert arr[0, 200] == pytest.approx(-10.0)


# ── T_2M URL ─────────────────────────────────────────────────────────


class TestT2MFileUrl:
    """The ``t_2m`` URL path mirrors the ``tot_prec`` shape."""

    def test_format_matches_dwd_pattern(self):
        run = datetime(2026, 5, 14, 12, tzinfo=timezone.utc)
        url = file_url(run, 3, "t_2m")
        assert url.endswith(
            "/12/t_2m/icon-eu_europe_regular-lat-lon_single-level_2026051412_003_T_2M.grib2.bz2"
        )


# ── Snow mask ────────────────────────────────────────────────────────


def _inject_frame_and_snow(
    grid: ICONEUGrid,
    run_ts: int,
    lead_seconds: int,
    *,
    snow_value: int | None = None,
) -> None:
    """Inject a uniform precip frame, optionally with a parallel snow mask."""
    fake = np.zeros(
        (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH), dtype=np.uint8,
    )
    grid._frames[(run_ts, lead_seconds)] = fake
    if grid._latest_run_ts is None or run_ts > grid._latest_run_ts:
        grid._latest_run_ts = run_ts
    if snow_value is not None:
        snow = np.full(
            (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH),
            snow_value & 0x01,
            dtype=np.uint8,
        )
        grid._snow_masks[(run_ts, lead_seconds)] = snow


class TestSnowMask:
    """ICONEUGrid.get_snow_mask end-to-end behaviour."""

    def test_supports_snow_is_true(self):
        grid = ICONEUGrid()
        assert grid.supports_snow is True

    def test_no_loaded_masks_returns_all_false(self):
        # Snow masks empty → falls through so the chain dispatcher
        # reaches the next snow-capable source (IFS globally).
        grid = ICONEUGrid()
        out = grid.get_snow_mask(np.array([50.0]), np.array([10.0]))
        assert out.dtype == np.bool_
        assert not out.any()

    def test_uniform_snow_returns_true_in_domain(self, hourly_brackets):
        grid = ICONEUGrid()
        run = int(datetime(2026, 5, 14, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame_and_snow(grid, run, 3 * 3600, snow_value=1)
        _inject_frame_and_snow(grid, run, 4 * 3600, snow_value=1)

        # Madrid — inside ICON-EU's domain (south of DINI)
        out = grid.get_snow_mask(
            np.array([40.42]), np.array([-3.70]),
            timestamp=run + 3 * 3600 + 1800,
        )
        assert out.tolist() == [True]

    def test_uniform_rain_returns_false_in_domain(self, hourly_brackets):
        grid = ICONEUGrid()
        run = int(datetime(2026, 5, 14, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame_and_snow(grid, run, 3 * 3600, snow_value=0)
        _inject_frame_and_snow(grid, run, 4 * 3600, snow_value=0)

        out = grid.get_snow_mask(
            np.array([41.90]), np.array([12.50]),   # Rome
            timestamp=run + 3 * 3600 + 1800,
        )
        assert out.tolist() == [False]

    def test_outside_domain_returns_false(self, hourly_brackets):
        grid = ICONEUGrid()
        run = int(datetime(2026, 5, 14, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame_and_snow(grid, run, 3 * 3600, snow_value=1)
        _inject_frame_and_snow(grid, run, 4 * 3600, snow_value=1)

        # New York — well outside ICON-EU
        out = grid.get_snow_mask(
            np.array([40.71]), np.array([-74.01]),
            timestamp=run + 3 * 3600 + 1800,
        )
        assert out.tolist() == [False]

    def test_lerp_bracket_majority_at_midpoint(self, hourly_brackets):
        grid = ICONEUGrid()
        run = int(datetime(2026, 5, 14, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame_and_snow(grid, run, 3 * 3600, snow_value=0)
        _inject_frame_and_snow(grid, run, 4 * 3600, snow_value=1)

        # alpha=0.25 → L0 wins (rain)
        out_low = grid.get_snow_mask(
            np.array([40.42]), np.array([-3.70]),
            timestamp=run + 3 * 3600 + 15 * 60,
        )
        assert out_low.tolist() == [False]

        # alpha=0.75 → L1 wins (snow)
        out_high = grid.get_snow_mask(
            np.array([40.42]), np.array([-3.70]),
            timestamp=run + 3 * 3600 + 45 * 60,
        )
        assert out_high.tolist() == [True]

    def test_partial_bracket_returns_false(self, hourly_brackets):
        # Precip at both leads, but only L0 has a snow mask.
        grid = ICONEUGrid()
        run = int(datetime(2026, 5, 14, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame_and_snow(grid, run, 3 * 3600, snow_value=1)
        _inject_frame_and_snow(grid, run, 4 * 3600, snow_value=None)

        out = grid.get_snow_mask(
            np.array([40.42]), np.array([-3.70]),
            timestamp=run + 3 * 3600 + 1800,
        )
        assert not out.any()


class TestSnowMaskPersistence:
    """Snow masks are atomic-write parallel files alongside precip frames."""

    @pytest.mark.asyncio
    async def test_snow_mask_round_trips_through_disk(self, tmp_path, hourly_brackets):
        run_ts = int(datetime(2026, 5, 14, 6, tzinfo=timezone.utc).timestamp())

        g1 = ICONEUGrid(cache_dir=tmp_path)
        fake_precip = np.zeros(
            (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH), dtype=np.uint8,
        )
        fake_snow = np.ones(
            (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH), dtype=np.uint8,
        )
        for lead in (3 * 3600, 4 * 3600):
            mm = g1._to_memmap(f"r{run_ts}_l{lead}", fake_precip)
            g1._frames[(run_ts, lead)] = mm
            mm_s = g1._to_memmap(f"r{run_ts}_l{lead}_snow", fake_snow)
            g1._snow_masks[(run_ts, lead)] = mm_s
        g1._latest_run_ts = run_ts
        await g1.close()

        cache_dir = tmp_path / "icon_eu"
        assert (cache_dir / f"r{run_ts}_l{3*3600}.dat").exists()
        assert (cache_dir / f"r{run_ts}_l{3*3600}_snow.dat").exists()

        g2 = ICONEUGrid(cache_dir=tmp_path)
        assert g2.frame_count == 2
        assert g2.snow_mask_count == 2
        assert (run_ts, 3 * 3600) in g2._snow_masks
        assert (run_ts, 4 * 3600) in g2._snow_masks

        sample_ts = run_ts + 3 * 3600 + 1800
        out = g2.get_snow_mask(
            np.array([40.42]), np.array([-3.70]),
            timestamp=sample_ts,
        )
        assert out.tolist() == [True]
        await g2.close()

    @pytest.mark.asyncio
    async def test_orphan_snow_mask_is_removed(self, tmp_path):
        cache_dir = tmp_path / "icon_eu"
        cache_dir.mkdir(parents=True)
        orphan = cache_dir / "r1234_l3600_snow.dat"
        size = ICON_EU_GRID_HEIGHT * ICON_EU_GRID_WIDTH
        orphan.write_bytes(b"\x00" * size)
        assert orphan.exists()

        g = ICONEUGrid(cache_dir=tmp_path)
        assert not orphan.exists()
        assert g.snow_mask_count == 0
        await g.close()

    @pytest.mark.asyncio
    async def test_eviction_removes_snow_files_too(self, tmp_path):
        run_ts = int(datetime(2026, 5, 14, 6, tzinfo=timezone.utc).timestamp())
        g = ICONEUGrid(cache_dir=tmp_path)
        fake_precip = np.zeros(
            (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH), dtype=np.uint8,
        )
        fake_snow = np.ones(
            (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH), dtype=np.uint8,
        )
        g._to_memmap(f"r{run_ts}_l3600", fake_precip)
        g._to_memmap(f"r{run_ts}_l3600_snow", fake_snow)
        mm = np.memmap(
            tmp_path / "icon_eu" / f"r{run_ts}_l3600.dat",
            dtype=np.uint8, mode="r",
            shape=(ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH),
        )
        g._frames[(run_ts, 3600)] = mm
        mm_s = np.memmap(
            tmp_path / "icon_eu" / f"r{run_ts}_l3600_snow.dat",
            dtype=np.uint8, mode="r",
            shape=(ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH),
        )
        g._snow_masks[(run_ts, 3600)] = mm_s

        far_future = run_ts + 7 * 24 * 3600
        g._evict_outside_window(far_future, far_future + 600)
        assert (run_ts, 3600) not in g._frames
        assert (run_ts, 3600) not in g._snow_masks
        assert not (tmp_path / "icon_eu" / f"r{run_ts}_l3600.dat").exists()
        assert not (tmp_path / "icon_eu" / f"r{run_ts}_l3600_snow.dat").exists()
        await g.close()


class TestChainSnowMaskWithIconEU:
    def test_chain_prefers_icon_eu_snow_inside_domain(self, hourly_brackets):
        from librewxr.sources.world.ifs.grid import ECMWFGrid
        from librewxr.sources.world.ifs.grid import GRID_HEIGHT as IFS_H, GRID_WIDTH as IFS_W

        # IFS says snow everywhere; ICON-EU says rain inside its domain.
        # Inside ICON-EU, ICON-EU wins → rain.  Outside (NYC), IFS wins → snow.
        ifs = ECMWFGrid()
        ifs_dbz = np.full((IFS_H, IFS_W), int((10 + 32) * 2), dtype=np.uint8)
        ifs_snow = np.ones((IFS_H, IFS_W), dtype=bool)
        ifs._timesteps[1000000] = (ifs_dbz, ifs_snow)
        ifs._sorted_timestamps = [1000000]

        icon = ICONEUGrid()
        run = 1000000 - 1800   # 30 min into the (0, 3600) bracket
        _inject_frame_and_snow(icon, run, 0, snow_value=0)
        _inject_frame_and_snow(icon, run, 3600, snow_value=0)

        chain = NWPChain([icon, ifs])

        # Madrid: ICON-EU says rain → False
        out = chain.get_snow_mask(
            np.array([40.42]), np.array([-3.70]), timestamp=1000000,
        )
        assert out.tolist() == [False]

        # New York: outside ICON-EU → IFS wins → True
        out = chain.get_snow_mask(
            np.array([40.71]), np.array([-74.01]), timestamp=1000000,
        )
        assert out.tolist() == [True]
