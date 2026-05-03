# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Unit tests for ICON-EU grid math, decode orientation, and chain integration."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.icon_eu

from librewxr.data.icon_eu_grid import (
    BRACKET_INTERVAL_SECONDS,
    CYCLE_INTERVAL_SECONDS,
    ICON_EU_GRID_HEIGHT,
    ICON_EU_GRID_WIDTH,
    ICON_EU_LAT_MAX,
    ICON_EU_LAT_MIN,
    ICONEUGrid,
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


# ── Decode orientation ────────────────────────────────────────────────


class TestDecodeOrientation:
    def test_decode_flips_south_up_grib(self, monkeypatch):
        from contextlib import contextmanager
        from librewxr.data import icon_eu_grid as iem

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

    def test_sample_at_exact_bracket(self):
        g = ICONEUGrid()
        run = int(datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 100)
        _inject_frame(g, run, 7200, 100)
        # Sample at run + 3600 (exact L0 frame)
        out = g.sample(np.array([52.5]), np.array([13.4]), timestamp=run + 3600)
        assert int(out[0]) == 100

    def test_sample_lerps_between_brackets(self):
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
    def test_chain_prefers_icon_eu_inside_europe(self):
        from librewxr.data.ecmwf_grid import ECMWFGrid
        from librewxr.data.ecmwf_grid import (
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
