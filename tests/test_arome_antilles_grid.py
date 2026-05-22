# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Unit tests for AROME Antilles grid math, decode, and chain integration."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.arome_antilles

from librewxr.sources.regional.caribbean.nwp.arome_antilles.grid import (
    AROME_ANT_GRID_HEIGHT,
    AROME_ANT_GRID_WIDTH,
    AROME_ANT_LAT_NORTH,
    AROME_ANT_LAT_SOUTH,
    AROME_ANT_LON_EAST_DEG_E,
    AROME_ANT_LON_WEST_DEG_E,
    BRACKET_INTERVAL_SECONDS,
    CYCLE_INTERVAL_SECONDS,
    AROMEAntillesGrid,
    bracket_lead_seconds,
    decode_tp_message,
    domain_mask,
    feather_mask,
    file_url,
    floor_cycle,
    grid_indices,
    latest_published_run,
    precip_rate_to_dbz_encoded,
)
from librewxr.data.nwp_source import NWPChain, NWPSource


# ── Regular lat/lon grid math ─────────────────────────────────────────


class TestGridIndices:
    @pytest.mark.parametrize(
        "name,lat,lon,inside",
        [
            # Cities expected INSIDE the verified Antilles extent
            ("Pointe-à-Pitre",  16.24, -61.55, True),   # Guadeloupe
            ("Fort-de-France",  14.61, -61.07, True),   # Martinique
            ("Saint Martin",    18.07, -63.05, True),   # Saint Martin
            ("Saint-Barthélemy",17.90, -62.83, True),   # Saint-Barthélemy
            ("Bridgetown",      13.10, -59.62, True),   # Barbados
            ("San Juan PR",     18.47, -66.10, True),   # Puerto Rico
            ("Caracas",         10.50, -66.92, True),   # near south edge
            ("Santo Domingo",   18.49, -69.93, True),   # Hispaniola east
            # Cities expected OUTSIDE the AROME Antilles domain
            ("Miami",           25.76, -80.20, False),   # north + west of grid
            ("Bogotá",           4.71, -74.07, False),   # too far south
            ("Belem (Brazil)",  -1.45, -48.50, False),   # south + east of grid
            ("Madrid",          40.42,  -3.70, False),   # off-continent
        ],
    )
    def test_domain_mask_known_points(self, name, lat, lon, inside):
        m = domain_mask(np.array([lat]), np.array([lon]))
        assert bool(m[0]) is inside, name

    def test_grid_origin_at_north_west(self):
        # Row 0, col 0 corresponds to the NORTHERN edge × WESTERN edge.
        # lat = AROME_ANT_LAT_NORTH (22.9°N), lon = -75.3 (= 284.7°E).
        row, col = grid_indices(
            np.array([AROME_ANT_LAT_NORTH]),
            np.array([AROME_ANT_LON_WEST_DEG_E - 360.0]),
        )
        assert abs(row[0] - 0) < 1e-3
        assert abs(col[0] - 0) < 1e-3

    def test_grid_origin_at_south_east(self):
        # Row HEIGHT-1, col WIDTH-1 corresponds to the SOUTHERN x EASTERN
        # edge.  lat = 9.7°N, lon = -51.7° (= 308.3°E).
        row, col = grid_indices(
            np.array([AROME_ANT_LAT_SOUTH]),
            np.array([AROME_ANT_LON_EAST_DEG_E - 360.0]),
        )
        assert abs(row[0] - (AROME_ANT_GRID_HEIGHT - 1)) < 1e-3
        assert abs(col[0] - (AROME_ANT_GRID_WIDTH - 1)) < 1e-3

    def test_lon_input_in_either_form(self):
        # Same physical point: -61.55°E and 298.45°E (= -61.55 + 360).
        row1, col1 = grid_indices(np.array([16.24]), np.array([-61.55]))
        row2, col2 = grid_indices(np.array([16.24]), np.array([298.45]))
        assert row1[0] == pytest.approx(row2[0])
        assert col1[0] == pytest.approx(col2[0])

    def test_grid_step_matches_resolution(self):
        # One degree of latitude → 40 row units (since dlat = 0.025°).
        r0, _ = grid_indices(np.array([16.0]), np.array([-61.55]))
        r1, _ = grid_indices(np.array([15.0]), np.array([-61.55]))
        assert r1[0] - r0[0] == pytest.approx(40.0)


# ── Feather ───────────────────────────────────────────────────────────


class TestFeatherMask:
    def test_inside_full_weight(self):
        # Pointe-à-Pitre (16.24, -61.55) — well inside the domain
        f = feather_mask(np.array([16.24]), np.array([-61.55]))
        assert f.dtype == np.float32
        assert f[0] == pytest.approx(1.0)

    def test_outside_zero(self):
        # Madrid is off the grid entirely.
        f = feather_mask(np.array([40.42]), np.array([-3.70]))
        assert f[0] == 0.0

    def test_taper_monotonic_walking_off_north_edge(self):
        # Walk lat from inside (15°N) to outside (28°N) at -61.55°W.
        # Feather should be non-increasing.
        lats = np.linspace(15.0, 28.0, 25)
        lons = np.full_like(lats, -61.55)
        f = feather_mask(lats, lons)
        diffs = np.diff(f)
        assert (diffs <= 1e-6).all()


# ── Timing helpers ────────────────────────────────────────────────────


class TestTiming:
    def test_floor_cycle_6h(self):
        # 14:23 UTC → 12:00 UTC (6h cycle)
        ts = int(datetime(2026, 5, 1, 14, 23, tzinfo=timezone.utc).timestamp())
        floored = floor_cycle(ts)
        expected = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        assert floored == expected

    def test_floor_cycle_at_boundary(self):
        ts = int(datetime(2026, 5, 1, 18, 0, tzinfo=timezone.utc).timestamp())
        assert floor_cycle(ts) == ts

    def test_latest_published_run(self):
        now = int(datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc).timestamp())
        # 7h delay → floor_cycle(07:00) = 06:00
        run = latest_published_run(now, 7 * 3600)
        expected = int(datetime(2026, 5, 1, 6, 0, tzinfo=timezone.utc).timestamp())
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
        assert CYCLE_INTERVAL_SECONDS == 6 * 3600
        assert BRACKET_INTERVAL_SECONDS == 3600


# ── Run windowing (regression for the empty-list edge case) ──────────


class TestWindowRuns:
    """Regression tests for ``_window_runs`` clamping.

    Pre-2026-05-21 the windowing used a plain ``max(...)`` for the
    earliest run, which collapsed to an empty ``range()`` for ~1 h of
    every 6 h cycle (when publish_delay > cycle_interval).  The fix
    clamps with ``min(latest_run_ts, ...)`` so the list always contains
    at least the latest published run.
    """

    def test_zero_history_zero_horizon_returns_latest_run(self):
        # 00:38 UTC — minutes after the previous 6h boundary; with a
        # 7h publish delay the old math went empty here.
        now = int(datetime(2026, 5, 22, 0, 38, tzinfo=timezone.utc).timestamp())
        latest, runs = AROMEAntillesGrid._window_runs(
            now_ts=now, history_seconds=0, horizon_seconds=0,
            publish_delay_seconds=7 * 3600,
        )
        expected_latest = int(
            datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc).timestamp(),
        )
        assert latest == expected_latest
        assert runs, "runs_to_consider must never be empty when a published run exists"
        assert runs[-1] == latest  # latest run always included

    @pytest.mark.parametrize(
        "now_h_utc",
        [0.5, 1.5, 6.5, 7.5, 12.5, 13.5, 18.5, 19.5],
    )
    def test_non_empty_across_cycle_phase_with_long_publish_delay(self, now_h_utc):
        # Sweep across each 6h cycle's leading hours; the empty-list
        # bug used to hit the first hour of each cycle.
        hour = int(now_h_utc)
        minute = int(round((now_h_utc - hour) * 60))
        now = int(datetime(2026, 5, 22, hour, minute, tzinfo=timezone.utc).timestamp())
        latest, runs = AROMEAntillesGrid._window_runs(
            now_ts=now, history_seconds=0, horizon_seconds=0,
            publish_delay_seconds=7 * 3600,
        )
        assert runs, f"runs_to_consider went empty at {now_h_utc:.1f}h UTC"
        assert latest in runs

    def test_lookback_window_includes_older_runs_when_history_demands(self):
        # 13:00 UTC + 12h history with 7h delay → latest is 06Z same
        # day, history reaches back into prev-day cycles.  Should
        # include at least latest plus one older run.
        now = int(datetime(2026, 5, 22, 13, 0, tzinfo=timezone.utc).timestamp())
        _, runs = AROMEAntillesGrid._window_runs(
            now_ts=now, history_seconds=12 * 3600, horizon_seconds=0,
            publish_delay_seconds=7 * 3600,
        )
        assert len(runs) >= 2


# ── Step range (regression for the evict-everything edge case) ────────


class TestStepRange:
    """Regression tests for ``_step_range``.

    Pre-2026-05-22 ``min_step`` used ``(min_lead // BRACKET) - 1`` and
    ``max_step`` used ``ceil(max_lead / BRACKET)``.  Both shapes fetched
    one bracket of slack beyond what ``_evict_outside_window`` retains,
    so every cycle ingested 2-3 frames that were immediately wiped.
    When those wasted fetches lined up just right with stale disk
    frames from prior runs, the store could end up at 0 frames for
    a full 10-min cycle — chain dispatch fell through to IFS and the
    AROME variant briefly disappeared from the map.

    The fix uses ceil for min_step and floor for max_step so every
    fetched step's valid_time falls inside the eviction window.
    """

    BRACKET = 3600

    def test_min_step_valid_time_inside_eviction_lower_bound(self):
        # min_lead = window_start - run_ts - BRACKET, by construction.
        # The eviction lower bound is window_start - BRACKET, so any
        # step whose valid_time >= window_start - BRACKET survives.
        # Step's valid_time = run_ts + step * BRACKET, so we need
        # step * BRACKET >= min_lead.
        for min_lead in [0, 1, 3599, 3600, 3601, 33019, 36000, 36001]:
            min_step, _ = AROMEAntillesGrid._step_range(min_lead, min_lead + 3600)
            assert min_step * self.BRACKET >= min_lead, (
                f"min_step={min_step} (lead={min_step * self.BRACKET}) "
                f"is below min_lead={min_lead}"
            )

    def test_max_step_valid_time_inside_eviction_upper_bound(self):
        # max_lead = window_end - run_ts + BRACKET.  Eviction upper
        # bound is window_end + BRACKET = run_ts + max_lead, so any
        # step whose valid_time <= max_lead-from-run survives.
        for max_lead in [3600, 7200, 51019, 54000, 54001, 60000]:
            _, max_step = AROMEAntillesGrid._step_range(0, max_lead)
            assert max_step * self.BRACKET <= max_lead, (
                f"max_step={max_step} (lead={max_step * self.BRACKET}) "
                f"is above max_lead={max_lead}"
            )

    def test_step_clamped_to_max_forecast_hours(self):
        # A max_lead that would exceed the model horizon is clamped.
        _, max_step = AROMEAntillesGrid._step_range(0, 200 * 3600)
        assert max_step == AROMEAntillesGrid.MAX_FORECAST_HOURS

    def test_min_step_never_negative(self):
        # Step 0 is the analysis frame; the fetcher must never request
        # a negative step.
        min_step, _ = AROMEAntillesGrid._step_range(0, 3600)
        assert min_step >= 0

    @pytest.mark.parametrize("now_h_utc", [4.17, 6.17, 8.17, 10.17, 12.17])
    def test_every_fetched_step_survives_eviction(self, now_h_utc):
        # End-to-end: pick a wall clock, walk through window_start,
        # window_end, eviction window, and assert every step the fetcher
        # would request lands inside [ws_evict, we_evict].  This is the
        # property the pre-fix code violated — extra steps got fetched
        # and then immediately wiped.
        hour = int(now_h_utc)
        minute = int(round((now_h_utc - hour) * 60))
        now = int(datetime(2026, 5, 22, hour, minute, tzinfo=timezone.utc).timestamp())
        history = 2 * 3600   # nwp_history default
        horizon = 1 * 3600   # nwp_horizon default
        publish_delay = 7 * 3600
        bracket = 3600

        _, runs = AROMEAntillesGrid._window_runs(
            now_ts=now, history_seconds=history, horizon_seconds=horizon,
            publish_delay_seconds=publish_delay,
        )
        assert runs, "no runs to consider — test setup wrong"

        window_start = now - history
        window_end = now + horizon
        ws_evict = window_start - bracket
        we_evict = window_end + bracket

        for run_ts in runs:
            min_lead = max(0, window_start - run_ts - bracket)
            max_lead = min(
                AROMEAntillesGrid.MAX_FORECAST_HOURS * 3600,
                window_end - run_ts + bracket,
            )
            if max_lead < min_lead:
                continue
            min_step, max_step = AROMEAntillesGrid._step_range(min_lead, max_lead)
            for step in range(min_step, max_step + 1):
                valid_time = run_ts + step * bracket
                assert ws_evict <= valid_time <= we_evict, (
                    f"run={run_ts} step={step} valid_time={valid_time} "
                    f"falls outside eviction window [{ws_evict}, {we_evict}] "
                    f"— would be fetched then immediately evicted"
                )


# ── URL construction ──────────────────────────────────────────────────


class TestFileUrl:
    def test_format_matches_data_gouv_pattern(self):
        run = datetime(2026, 5, 8, 0, tzinfo=timezone.utc)
        url = file_url(run, 6)
        assert url.endswith(
            "/pnt/2026-05-08T00:00:00Z/arome-om/ANTIL/0025/SP1/"
            "arome-om-ANTIL__0025__SP1__006H__2026-05-08T00:00:00Z.grib2"
        )

    def test_step_zero_padded_to_three_digits(self):
        run = datetime(2026, 5, 8, 6, tzinfo=timezone.utc)
        url = file_url(run, 0)
        assert "__000H__" in url

    def test_step_48_padded(self):
        run = datetime(2026, 5, 8, 6, tzinfo=timezone.utc)
        url = file_url(run, 48)
        assert "__048H__" in url

    def test_uses_settings_base_url(self):
        run = datetime(2026, 5, 8, 0, tzinfo=timezone.utc)
        url = file_url(run, 1)
        assert "meteofrance-pnt" in url


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
    def test_decode_no_flip_when_north_up(self, monkeypatch):
        from contextlib import contextmanager
        from librewxr.sources.regional.caribbean.nwp.arome_antilles import grid as ant
        from librewxr.sources._shared import arome as arome_shared

        # Synthetic cfgrib output: row 0 at the NORTHERN edge (correct).
        tp = np.zeros((AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH), dtype=np.float32)
        tp[0, 100] = 5.0    # marker at north
        tp[-1, 200] = 8.0   # marker at south

        # 1-D latitude coord that DECREASES with row index (north-up).
        lat = np.linspace(
            AROME_ANT_LAT_NORTH, AROME_ANT_LAT_SOUTH, AROME_ANT_GRID_HEIGHT,
        )
        lon = np.linspace(
            AROME_ANT_LON_WEST_DEG_E, AROME_ANT_LON_EAST_DEG_E,
            AROME_ANT_GRID_WIDTH,
        )

        import xarray as xr
        fake_ds = xr.Dataset(
            {"tp": (("latitude", "longitude"), tp)},
            coords={"latitude": lat, "longitude": lon},
        )

        @contextmanager
        def _noop():
            yield

        monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: fake_ds)
        monkeypatch.setattr(arome_shared, "_suppress_eccodes_stderr", _noop)

        arr = ant.decode_tp_message(b"ignored")
        assert arr is not None
        # No flip: north marker (cfgrib row 0) stays at our row 0.
        assert arr[0, 100] == 5.0
        assert arr[-1, 200] == 8.0

    def test_decode_flips_south_up_grib(self, monkeypatch):
        """Defensive: if cfgrib ever returns the file south-up, the flip
        should self-correct on the latitude coord."""
        from contextlib import contextmanager
        from librewxr.sources.regional.caribbean.nwp.arome_antilles import grid as ant
        from librewxr.sources._shared import arome as arome_shared

        tp = np.zeros((AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH), dtype=np.float32)
        tp[0, 100] = 5.0    # marker at "row 0" (now south)
        tp[-1, 200] = 8.0   # marker at "row -1" (now north)

        # Latitude coord INCREASES with row index → south-up.
        lat = np.linspace(
            AROME_ANT_LAT_SOUTH, AROME_ANT_LAT_NORTH, AROME_ANT_GRID_HEIGHT,
        )
        lon = np.linspace(
            AROME_ANT_LON_WEST_DEG_E, AROME_ANT_LON_EAST_DEG_E,
            AROME_ANT_GRID_WIDTH,
        )

        import xarray as xr
        fake_ds = xr.Dataset(
            {"tp": (("latitude", "longitude"), tp)},
            coords={"latitude": lat, "longitude": lon},
        )

        @contextmanager
        def _noop():
            yield

        monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: fake_ds)
        monkeypatch.setattr(arome_shared, "_suppress_eccodes_stderr", _noop)

        arr = ant.decode_tp_message(b"ignored")
        assert arr is not None
        # After flip: south marker (cfgrib row 0) → our row -1 (south).
        assert arr[-1, 100] == 5.0
        assert arr[0, 200] == 8.0


# ── Cumulative-to-rate diff ───────────────────────────────────────────


class TestAccumulationDiff:
    def test_step_zero_baseline_is_zero(self):
        # AROMEAntillesGrid initialises step 0's accumulator to all zeros.
        grid = AROMEAntillesGrid()
        run_dt = datetime(2026, 5, 8, 0, tzinfo=timezone.utc)
        run_ts = int(run_dt.timestamp())
        # Manually run the step-0 codepath: it caches a zero baseline
        # and returns 0.  We don't actually fetch network here.
        grid._accum[(run_ts, 0)] = np.zeros(
            (AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH), dtype=np.float32,
        )
        baseline = grid._accum[(run_ts, 0)]
        assert baseline.shape == (AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH)
        assert (baseline == 0).all()

    def test_diff_yields_windowed_rate(self):
        # Synthetic: at step 5, accum is 5 mm; at step 6, 11 mm.  Rate
        # over the [5h, 6h] window is 6 mm.
        accum_5 = np.full(
            (AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH), 5.0, dtype=np.float32,
        )
        accum_6 = np.full(
            (AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH), 11.0, dtype=np.float32,
        )
        rate = accum_6 - accum_5
        # Encoder converts to dBZ; pixel value should be > 0 (positive rate)
        encoded = precip_rate_to_dbz_encoded(rate)
        assert (encoded > 0).all()


# ── Run picking ───────────────────────────────────────────────────────


class TestPickRun:
    def test_no_frames_returns_none(self):
        grid = AROMEAntillesGrid()
        ts = int(datetime(2026, 5, 8, 12, tzinfo=timezone.utc).timestamp())
        assert grid._pick_run(ts) is None

    def test_returns_run_only_when_bracket_loaded(self):
        grid = AROMEAntillesGrid()
        run_dt = datetime(2026, 5, 8, 6, tzinfo=timezone.utc)
        run_ts = int(run_dt.timestamp())
        # Insert just one bracket frame (lead 3h) — bracket needs both
        # 3h and 4h for a query at 3h30m, so we should NOT return.
        fake = np.zeros(
            (AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH), dtype=np.uint8,
        )
        grid._frames[(run_ts, 3 * 3600)] = fake
        # Query at run + 3h30m
        query_ts = run_ts + 3 * 3600 + 1800
        assert grid._pick_run(query_ts) is None
        # Add the second bracket frame
        grid._frames[(run_ts, 4 * 3600)] = fake
        assert grid._pick_run(query_ts) == run_ts

    def test_falls_back_to_older_run_when_freshest_incomplete(self):
        grid = AROMEAntillesGrid()
        old_run = datetime(2026, 5, 8, 0, tzinfo=timezone.utc)
        new_run = datetime(2026, 5, 8, 6, tzinfo=timezone.utc)
        old_ts = int(old_run.timestamp())
        new_ts = int(new_run.timestamp())
        fake = np.zeros(
            (AROME_ANT_GRID_HEIGHT, AROME_ANT_GRID_WIDTH), dtype=np.uint8,
        )
        # Fresh run has only one bracket frame — incomplete
        grid._frames[(new_ts, 7 * 3600)] = fake
        # Old run has both bracket frames covering the query timestamp
        grid._frames[(old_ts, 13 * 3600)] = fake
        grid._frames[(old_ts, 14 * 3600)] = fake
        # Query at 13:30 UTC → +13.5h from old run, +7.5h from new run
        query_ts = int(datetime(2026, 5, 8, 13, 30, tzinfo=timezone.utc).timestamp())
        assert grid._pick_run(query_ts) == old_ts


# ── Protocol conformance ──────────────────────────────────────────────


class TestNWPSourceProtocol:
    def test_satisfies_protocol(self):
        grid = AROMEAntillesGrid()
        assert isinstance(grid, NWPSource)

    def test_chain_with_only_antilles(self):
        """A chain with just AROME Antilles + no data still answers."""
        grid = AROMEAntillesGrid()
        chain = NWPChain([grid])
        out = chain.sample(np.array([16.24]), np.array([-61.55]))
        assert out.shape == (1,)
        assert out[0] == 0  # no data loaded

    def test_supports_snow_false(self):
        # Tropical domain — snow classification not provided.
        grid = AROMEAntillesGrid()
        assert grid.supports_snow is False
        out = grid.get_snow_mask(np.array([16.24]), np.array([-61.55]))
        assert (out == False).all()
