# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Unit tests for JMA MSM grid math, decode orientation, and chain integration."""
from __future__ import annotations

import inspect
from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.jma_msm

from librewxr.data.nwp_source import NWPChain, NWPSource
from librewxr.sources.regional.east_asia.japan.nwp.jma_msm.grid import (
    BRACKET_INTERVAL_SECONDS,
    CYCLE_INTERVAL_SECONDS,
    JMA_MSM_GRID_HEIGHT,
    JMA_MSM_GRID_WIDTH,
    JMA_MSM_LAT_MAX,
    JMA_MSM_LAT_MIN,
    JMA_MSM_LON_MAX,
    JMA_MSM_LON_MIN,
    SOURCE_STEP_SECONDS,
    STORED_INTERVAL_SECONDS,
    JMAMSMGrid,
    bracket_lead_seconds,
    domain_mask,
    feather_mask,
    file_key,
    floor_cycle,
    grid_indices,
    latest_published_run,
    precip_rate_to_dbz_encoded,
    run_dir,
)


# ── Lat/lon grid ──────────────────────────────────────────────────────


class TestLatLonGrid:
    @pytest.mark.parametrize(
        "name,lat,lon,inside",
        [
            ("Tokyo",       35.68, 139.69, True),
            ("Sapporo",     43.06, 141.35, True),
            ("Naha",        26.21, 127.68, True),
            ("Seoul",       37.57, 126.98, True),
            ("Taipei",      25.03, 121.57, True),
            ("Shanghai",    31.23, 121.47, True),
            # Just outside the bbox in each direction
            ("Vladivostok", 43.12, 131.89, True),  # inside (NE asia corner)
            ("Cairo",       30.04,  31.23, False),
            ("NYC",         40.71, -74.01, False),
            ("Sydney",     -33.86, 151.21, False),
            ("Manila",      14.60, 120.98, False),  # south of LAT_MIN=22.4
        ],
    )
    def test_domain_mask_known_points(self, name, lat, lon, inside):
        m = domain_mask(np.array([lat]), np.array([lon]))
        assert bool(m[0]) is inside, name

    def test_grid_origin_at_north_west(self):
        # Row 0 should be the NORTHERN edge (after our decode flip).
        row, col = grid_indices(
            np.array([JMA_MSM_LAT_MAX]), np.array([JMA_MSM_LON_MIN])
        )
        assert abs(row[0] - 0) < 1e-6
        assert abs(col[0] - 0) < 1e-6

    def test_grid_origin_at_south_east(self):
        row, col = grid_indices(
            np.array([JMA_MSM_LAT_MIN]), np.array([JMA_MSM_LON_MAX])
        )
        assert abs(row[0] - (JMA_MSM_GRID_HEIGHT - 1)) < 1e-6
        assert abs(col[0] - (JMA_MSM_GRID_WIDTH - 1)) < 1e-6

    def test_grid_dimensions_match_native_msm(self):
        # 0.0625° lon × 0.05° lat over 22.4-47.6°N × 120-150°E
        # → 481 × 505 cells (matches the .om container shape).
        assert JMA_MSM_GRID_HEIGHT == 505
        assert JMA_MSM_GRID_WIDTH == 481


# ── Feather ───────────────────────────────────────────────────────────


class TestFeatherMask:
    def test_inside_full_weight(self):
        # Mid-Japan, far from edges → 1.0
        f = feather_mask(np.array([36.0]), np.array([138.0]))
        assert f.dtype == np.float32
        assert f[0] == pytest.approx(1.0)

    def test_outside_zero(self):
        f = feather_mask(np.array([0.0]), np.array([0.0]))
        assert f[0] == 0.0

    def test_taper_monotonic_at_east_edge(self):
        lats = np.full(21, 35.0)
        lons = np.linspace(149.0, 151.0, 21)
        f = feather_mask(lats, lons)
        diffs = np.diff(f)
        assert (diffs <= 1e-6).all()


# ── Timing ────────────────────────────────────────────────────────────


class TestTiming:
    def test_floor_cycle_3h(self):
        # 14:23 UTC → 12:00 UTC (3-hour cycle floor)
        ts = int(datetime(2026, 5, 30, 14, 23, tzinfo=timezone.utc).timestamp())
        floored = floor_cycle(ts)
        expected = int(datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc).timestamp())
        assert floored == expected

    def test_floor_cycle_at_boundary(self):
        ts = int(datetime(2026, 5, 30, 15, 0, tzinfo=timezone.utc).timestamp())
        assert floor_cycle(ts) == ts

    def test_latest_published_run_with_5h_delay(self):
        # 14:00 UTC now, 5h publish delay → newest run is floor(09:00) = 09:00
        now = int(datetime(2026, 5, 30, 14, 0, tzinfo=timezone.utc).timestamp())
        run = latest_published_run(now, 5 * 3600)
        expected = int(datetime(2026, 5, 30, 9, 0, tzinfo=timezone.utc).timestamp())
        assert run == expected

    def test_latest_published_run_does_not_pick_future(self):
        # If 4h ago is still before the most recent cycle boundary, the
        # picker should NOT advance ahead — runs must be in the past.
        now = int(datetime(2026, 5, 30, 5, 30, tzinfo=timezone.utc).timestamp())
        # 5h delay → reach into "yesterday 00:30" which floors to yesterday
        # 00:00.  Just confirm the result is strictly less than ``now``.
        run = latest_published_run(now, 5 * 3600)
        assert run < now

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

    def test_cycle_interval_is_3_hours(self):
        assert CYCLE_INTERVAL_SECONDS == 3 * 3600
        assert BRACKET_INTERVAL_SECONDS == 3600


# ── S3 path construction ─────────────────────────────────────────────


class TestS3Path:
    def test_run_dir_format(self):
        run = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        d = run_dir(run)
        assert d == "data_spatial/jma_msm/2026/05/30/1800Z"

    def test_file_key_format(self):
        run = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        # Step 1 = valid_time 1900Z that same day
        key = file_key(run, 1)
        assert key == "data_spatial/jma_msm/2026/05/30/1800Z/2026-05-30T1900.om"

    def test_file_key_crosses_day(self):
        run = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
        # Step 12 → valid_time next day 0600Z
        key = file_key(run, 12)
        assert key == "data_spatial/jma_msm/2026/05/30/1800Z/2026-05-31T0600.om"


# ── Z-R conversion ────────────────────────────────────────────────────


class TestZR:
    def test_zero_rate_zero_encoded(self):
        encoded = precip_rate_to_dbz_encoded(np.array([0.0, 0.0]))
        assert (encoded == 0).all()

    def test_higher_rate_higher_dbz(self):
        encoded = precip_rate_to_dbz_encoded(np.array([0.5, 5.0, 50.0]))
        assert encoded[0] < encoded[1] < encoded[2]
        # 50 mm/h hits ~50 dBZ; encoded = (50 + 32) * 2 = 164
        assert abs(int(encoded[2]) - 164) <= 2

    def test_handles_nan_and_negative(self):
        encoded = precip_rate_to_dbz_encoded(
            np.array([np.nan, -1.0, 1.0])
        )
        assert encoded[0] == 0
        assert encoded[1] == 0
        assert encoded[2] > 0

    def test_dbz_offset_shifts_uniformly(self):
        rates = np.array([1.0, 5.0, 25.0])
        base = precip_rate_to_dbz_encoded(rates, dbz_offset=0.0)
        shifted = precip_rate_to_dbz_encoded(rates, dbz_offset=6.0)
        for b, s in zip(base, shifted):
            if b > 0:
                assert int(s) - int(b) == 12


# ── Decode orientation ────────────────────────────────────────────────


class TestDecodeOrientation:
    """``_decode_om_field`` flips south-up .om arrays to north-up."""

    def test_decode_flips_south_up_om_via_fake_reader(self, monkeypatch):
        from librewxr.sources.regional.east_asia.japan.nwp.jma_msm import grid as msm

        # Synthetic openmeteo .om payload: row 0 at the SOUTHERN edge.
        raw = np.zeros(
            (JMA_MSM_GRID_HEIGHT, JMA_MSM_GRID_WIDTH), dtype=np.float32,
        )
        raw[0, 100] = 5.0    # row 0 = south
        raw[-1, 200] = 8.0   # row -1 = north

        class _FakeChild:
            shape = raw.shape
            dtype = raw.dtype
            def __getitem__(self, _slc):
                return raw
            def close(self):
                pass

        class _FakeReader:
            def get_child_by_name(self, _name):
                return _FakeChild()
            def close(self):
                pass

        # Bypass retry_sync's exception swallowing — call our fake directly.
        monkeypatch.setattr(
            msm, "OmFileReader",
            type("F", (), {"from_fsspec": staticmethod(lambda *a, **kw: _FakeReader())}),
        )
        # retry_sync wraps OmFileReader.from_fsspec; with the fake above
        # in place, it'll just pass through.
        import librewxr.data.retry as retry
        monkeypatch.setattr(retry, "retry_sync", lambda fn, *a, **kw: fn(*a, **kw))

        arr = msm._decode_om_field(None, "ignored", "precipitation")
        assert arr is not None
        # After flip: south marker → our row -1; north marker → our row 0
        assert arr[-1, 100] == 5.0
        assert arr[0, 200] == 8.0


# ── Protocol + chain ──────────────────────────────────────────────────


def _inject_frame(g: JMAMSMGrid, run_ts: int, lead_seconds: int, encoded_value: int):
    """Inject a uniform-value frame into the in-memory store."""
    arr = np.full(
        (JMA_MSM_GRID_HEIGHT, JMA_MSM_GRID_WIDTH),
        encoded_value, dtype=np.uint8,
    )
    g._frames[(run_ts, lead_seconds)] = arr
    if g._latest_run_ts is None or run_ts > g._latest_run_ts:
        g._latest_run_ts = run_ts


@pytest.fixture
def hourly_brackets(monkeypatch):
    """Force legacy hourly bracket spacing for tests that don't interpolate."""
    from librewxr.config import settings as _settings
    monkeypatch.setattr(_settings, "regional_interpolation", False)


class TestProtocol:
    def test_satisfies_nwpsource(self):
        g = JMAMSMGrid()
        assert isinstance(g, NWPSource)
        assert g.name == "jma_msm"

    def test_empty_grid_returns_zeros(self):
        g = JMAMSMGrid()
        out = g.sample(np.array([35.68]), np.array([139.69]), timestamp=12345)
        assert out.shape == (1,)
        assert out[0] == 0

    def test_sample_at_exact_bracket(self, hourly_brackets):
        g = JMAMSMGrid()
        run = int(datetime(2026, 5, 30, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 100)
        _inject_frame(g, run, 7200, 100)
        out = g.sample(np.array([35.68]), np.array([139.69]), timestamp=run + 3600)
        assert int(out[0]) == 100

    def test_sample_lerps_between_brackets(self, hourly_brackets):
        g = JMAMSMGrid()
        run = int(datetime(2026, 5, 30, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 80)
        _inject_frame(g, run, 7200, 160)
        out = g.sample(np.array([35.68]), np.array([139.69]), timestamp=run + 5400)
        assert abs(int(out[0]) - 120) <= 1

    def test_outside_domain_zero(self, hourly_brackets):
        g = JMAMSMGrid()
        run = int(datetime(2026, 5, 30, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 200)
        _inject_frame(g, run, 7200, 200)
        # NYC is outside East Asia
        out = g.sample(np.array([40.71]), np.array([-74.01]), timestamp=run + 3600)
        assert int(out[0]) == 0


class TestFetchSignature:
    """Fetch signature must expose history_seconds + horizon_seconds.

    The MultiSourceFetcher's ``inspect.signature`` introspection in
    ``data/fetcher.py:_fetch_auxiliary_grids`` decides whether to pass
    these kwargs by checking the signature.  If they disappeared from
    the JMAMSMGrid signature, JMA MSM would silently fetch with a
    zero-second window every cycle.
    """

    def test_fetch_accepts_history_and_horizon(self):
        g = JMAMSMGrid()
        sig = inspect.signature(g.fetch)
        assert "history_seconds" in sig.parameters
        assert "horizon_seconds" in sig.parameters


class TestChainOrdering:
    def test_chain_prefers_jma_msm_inside_japan(self, hourly_brackets):
        from librewxr.sources.world.ifs.grid import (
            GRID_HEIGHT as IFS_H, GRID_WIDTH as IFS_W,
            ECMWFGrid,
        )

        ifs = ECMWFGrid()
        ifs_dbz = np.full((IFS_H, IFS_W), 84, dtype=np.uint8)  # ~10 dBZ
        ifs._timesteps[1000000] = (ifs_dbz, np.zeros_like(ifs_dbz, dtype=bool))
        ifs._sorted_timestamps = [1000000]

        jma = JMAMSMGrid()
        run = 1000000 - 1500
        _inject_frame(jma, run, 0, 164)      # ~50 dBZ
        _inject_frame(jma, run, 3600, 164)

        chain = NWPChain([jma, ifs])
        # Inside Japan: JMA MSM dominates
        out_jp = chain.sample(np.array([35.68]), np.array([139.69]), timestamp=1000000)
        assert abs(int(out_jp[0]) - 164) <= 1
        # Outside (NYC): IFS fills
        out_us = chain.sample(np.array([40.71]), np.array([-74.01]), timestamp=1000000)
        assert int(out_us[0]) == 84


# ── Provider gating ──────────────────────────────────────────────────


class TestProvider:
    def test_provider_returns_contribution_when_enabled(self, monkeypatch, tmp_path):
        from librewxr.config import settings as _settings
        from librewxr.sources.regional.east_asia.japan.nwp.jma_msm import (
            nwp_provider,
        )
        monkeypatch.setattr(_settings, "jma_msm_enabled", True)
        contrib = nwp_provider(_settings, tmp_path)
        assert contrib is not None
        assert contrib.priority == 20
        assert contrib.name == "JMA MSM"
        assert contrib.regional is True

    def test_provider_returns_none_when_disabled(self, monkeypatch, tmp_path):
        from librewxr.config import settings as _settings
        from librewxr.sources.regional.east_asia.japan.nwp.jma_msm import (
            nwp_provider,
        )
        monkeypatch.setattr(_settings, "jma_msm_enabled", False)
        assert nwp_provider(_settings, tmp_path) is None


# ── Snow mask ────────────────────────────────────────────────────────


def _inject_frame_and_snow(
    grid: JMAMSMGrid,
    run_ts: int,
    lead_seconds: int,
    *,
    snow_value: int | None = None,
) -> None:
    """Inject a uniform precip frame, optionally with a parallel snow mask."""
    fake = np.zeros(
        (JMA_MSM_GRID_HEIGHT, JMA_MSM_GRID_WIDTH), dtype=np.uint8,
    )
    grid._frames[(run_ts, lead_seconds)] = fake
    if grid._latest_run_ts is None or run_ts > grid._latest_run_ts:
        grid._latest_run_ts = run_ts
    if snow_value is not None:
        snow = np.full(
            (JMA_MSM_GRID_HEIGHT, JMA_MSM_GRID_WIDTH),
            snow_value & 0x01,
            dtype=np.uint8,
        )
        grid._snow_masks[(run_ts, lead_seconds)] = snow


class TestSnowMask:
    def test_supports_snow_is_true(self):
        grid = JMAMSMGrid()
        assert grid.supports_snow is True

    def test_no_loaded_masks_returns_all_false(self):
        grid = JMAMSMGrid()
        out = grid.get_snow_mask(np.array([43.0]), np.array([141.4]))
        assert out.dtype == np.bool_
        assert not out.any()

    def test_uniform_snow_returns_true_in_domain(self, hourly_brackets):
        grid = JMAMSMGrid()
        run = int(datetime(2026, 12, 14, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame_and_snow(grid, run, 3 * 3600, snow_value=1)
        _inject_frame_and_snow(grid, run, 4 * 3600, snow_value=1)
        # Sapporo — winter snow likely
        out = grid.get_snow_mask(
            np.array([43.06]), np.array([141.35]),
            timestamp=run + 3 * 3600 + 1800,
        )
        assert out.tolist() == [True]

    def test_uniform_rain_returns_false_in_domain(self, hourly_brackets):
        grid = JMAMSMGrid()
        run = int(datetime(2026, 6, 14, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame_and_snow(grid, run, 3 * 3600, snow_value=0)
        _inject_frame_and_snow(grid, run, 4 * 3600, snow_value=0)
        # Naha — subtropical, never snow
        out = grid.get_snow_mask(
            np.array([26.21]), np.array([127.68]),
            timestamp=run + 3 * 3600 + 1800,
        )
        assert out.tolist() == [False]


# ── Optical-flow interpolation ────────────────────────────────────────


def _make_blob(
    cy: int, cx: int, radius: int = 25, value: int = 150,
) -> np.ndarray:
    grid = np.zeros((JMA_MSM_GRID_HEIGHT, JMA_MSM_GRID_WIDTH), dtype=np.uint8)
    ys, xs = np.ogrid[0:JMA_MSM_GRID_HEIGHT, 0:JMA_MSM_GRID_WIDTH]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
    grid[mask] = value
    return grid


class TestInterpolateRunFrames:
    """``_interpolate_run_frames`` fills 10-min synthetics between hourly originals."""

    def test_fills_synthetic_leads_between_hourly_originals(self):
        grid = JMAMSMGrid()
        run_ts = int(datetime(2026, 5, 30, 6, tzinfo=timezone.utc).timestamp())
        grid._frames[(run_ts, 0)] = _make_blob(250, 200)
        grid._frames[(run_ts, 3600)] = _make_blob(250, 240)  # blob translated east
        grid._latest_run_ts = run_ts

        added = grid._interpolate_run_frames(run_ts)
        assert added == 5  # leads 600, 1200, 1800, 2400, 3000
        for lead in (600, 1200, 1800, 2400, 3000):
            assert (run_ts, lead) in grid._frames

    def test_idempotent_on_second_call(self):
        grid = JMAMSMGrid()
        run_ts = int(datetime(2026, 5, 30, 6, tzinfo=timezone.utc).timestamp())
        grid._frames[(run_ts, 0)] = _make_blob(250, 200)
        grid._frames[(run_ts, 3600)] = _make_blob(250, 240)
        grid._latest_run_ts = run_ts

        first = grid._interpolate_run_frames(run_ts)
        second = grid._interpolate_run_frames(run_ts)
        assert first == 5
        assert second == 0

    def test_returns_zero_when_run_has_one_or_fewer_frames(self):
        grid = JMAMSMGrid()
        run_ts = int(datetime(2026, 5, 30, 6, tzinfo=timezone.utc).timestamp())
        assert grid._interpolate_run_frames(run_ts) == 0
        grid._frames[(run_ts, 0)] = _make_blob(250, 200)
        assert grid._interpolate_run_frames(run_ts) == 0


class TestRegionalInterpolationToggle:
    def test_bracket_interval_is_hourly_when_disabled(self, hourly_brackets):
        grid = JMAMSMGrid()
        assert grid._bracket_interval() == SOURCE_STEP_SECONDS

    def test_bracket_interval_is_10min_when_enabled(self):
        grid = JMAMSMGrid()
        assert grid._bracket_interval() == STORED_INTERVAL_SECONDS
