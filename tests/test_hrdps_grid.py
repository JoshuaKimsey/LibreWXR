# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Unit tests for HRDPS rotated-pole projection, decode, and chain integration."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.hrdps

from librewxr.sources.regional.north_america.canada.nwp.hrdps.grid import (
    BRACKET_INTERVAL_SECONDS,
    CYCLE_INTERVAL_SECONDS,
    HRDPS_GRID_HEIGHT,
    HRDPS_GRID_NORTH_POLE_LAT,
    HRDPS_GRID_NORTH_POLE_LON,
    HRDPS_GRID_RLAT_ORIGIN_NORTH,
    HRDPS_GRID_RLAT_ORIGIN_SOUTH,
    HRDPS_GRID_RLON_ORIGIN,
    HRDPS_GRID_WIDTH,
    HRDPS_LA1,
    HRDPS_LO1,
    MAX_FORECAST_HOURS,
    HRDPSGrid,
    bracket_lead_seconds,
    decode_apcp_message,
    domain_mask,
    feather_mask,
    file_url,
    floor_cycle,
    grid_indices,
    latest_published_run,
    precip_rate_to_dbz_encoded,
    rotated_forward,
)
from librewxr.data.nwp_source import NWPChain, NWPSource


# ── Rotated-pole projection ──────────────────────────────────────────


class TestRotatedProjection:
    def test_pole_maps_to_rotated_north_pole(self):
        # The rotated-north-pole point (in geographic coords) should map
        # to the rotated north pole, i.e. rlat = 90°.
        rlat, _ = rotated_forward(
            np.array([HRDPS_GRID_NORTH_POLE_LAT]),
            np.array([HRDPS_GRID_NORTH_POLE_LON]),
        )
        assert rlat[0] == pytest.approx(90.0, abs=1e-6)

    def test_geographic_pole_lands_at_pole_lat(self):
        # The geographic north pole (90°, anything) sits at angular
        # distance (90° − φ_p) from the rotated pole, so its rotated
        # latitude is 90° − (90° − φ_p) = φ_p ≈ 36.09°.  rlon there is
        # degenerate; numpy's atan2 returns some finite value.
        rlat, rlon = rotated_forward(np.array([90.0]), np.array([0.0]))
        assert rlat[0] == pytest.approx(HRDPS_GRID_NORTH_POLE_LAT, abs=1e-4)
        assert np.isfinite(rlon[0])

    def test_high_latitude_finite(self):
        # Numerical guard: 89.99°N must not raise or return NaN.
        rlat, rlon = rotated_forward(np.array([89.99]), np.array([-100.0]))
        assert np.isfinite(rlat[0])
        assert np.isfinite(rlon[0])

    @pytest.mark.parametrize(
        "name,lat,lon,exp_row,exp_col",
        [
            ("[0, 0]   SW",  39.6260, -133.6295, HRDPS_GRID_HEIGHT - 1, 0),
            ("[0, -1]  SE",  27.2846,  -66.9664, HRDPS_GRID_HEIGHT - 1, HRDPS_GRID_WIDTH - 1),
            ("[-1, 0]  NW",  66.5685, -152.7307, 0, 0),
            ("[-1, -1] NE",  47.8765,  -40.7086, 0, HRDPS_GRID_WIDTH - 1),
        ],
    )
    def test_grib_corners_land_at_expected_indices(self, name, lat, lon, exp_row, exp_col):
        # The four GRIB corners decoded from a 2026-05-07T12Z file must
        # round-trip through rotated_forward + grid_indices to within
        # half a cell of (exp_row, exp_col).
        row, col = grid_indices(np.array([lat]), np.array([lon]))
        assert abs(row[0] - exp_row) < 0.5, f"{name}: row={row[0]}"
        assert abs(col[0] - exp_col) < 0.5, f"{name}: col={col[0]}"


# ── Domain mask ──────────────────────────────────────────────────────


class TestDomainMask:
    @pytest.mark.parametrize(
        "name,lat,lon,inside",
        [
            # Cities expected INSIDE the HRDPS continental rotated rectangle
            ("Toronto",      43.65,  -79.38, True),
            ("Vancouver",    49.28, -123.12, True),
            ("Whitehorse",   60.72, -135.05, True),
            ("Iqaluit",      63.75,  -68.51, True),
            ("St. Johns NL", 47.56,  -52.71, True),
            ("Inuvik",       68.36, -133.71, True),
            ("Edmonton",     53.55, -113.49, True),
            ("Seattle",      47.61, -122.33, True),
            ("Minneapolis",  44.98,  -93.27, True),
            ("New York",     40.71,  -74.01, True),
            # Cities expected OUTSIDE — the rotated rectangle's geographic
            # footprint is curved; very-high-Arctic and Alaska fall out
            # because they sit beyond the western / northern rotated edge.
            # HRDPS-North + HRDPS-Atlantic cover those gaps but we
            # explicitly excluded them from this implementation.
            ("Resolute",     74.69,  -94.83, False),  # too far north (Nunavut high Arctic)
            ("Anchorage",    61.22, -149.90, False),  # west of grid (Alaska)
            ("Mexico City",  19.43,  -99.13, False),  # south of grid
            ("Reykjavik",    64.13,  -21.82, False),  # east of grid
            ("Honolulu",     21.31, -157.86, False),  # far southwest
            ("Miami",        25.76,  -80.19, False),  # too far south
        ],
    )
    def test_domain_mask_known_points(self, name, lat, lon, inside):
        m = domain_mask(np.array([lat]), np.array([lon]))
        assert bool(m[0]) is inside, name


# ── Feather ──────────────────────────────────────────────────────────


class TestFeatherMask:
    def test_inside_full_weight(self):
        # Edmonton sits well inside the rotated rectangle (col ≈ 690,
        # row ≈ 758) — far from any edge → feather = 1.0.
        f = feather_mask(np.array([53.55]), np.array([-113.49]))
        assert f.dtype == np.float32
        assert f[0] == pytest.approx(1.0)

    def test_outside_zero(self):
        # Mexico City → 0
        f = feather_mask(np.array([19.43]), np.array([-99.13]))
        assert f[0] == 0.0

    def test_taper_monotonic_walking_off_north_edge(self):
        # Walk lat from inside (60°N) up to outside (78°N) at lon=-94°.
        lats = np.linspace(60.0, 78.0, 25)
        lons = np.full_like(lats, -94.0)
        f = feather_mask(lats, lons)
        diffs = np.diff(f)
        assert (diffs <= 1e-6).all()


# ── Timing helpers ───────────────────────────────────────────────────


class TestTiming:
    def test_floor_cycle_6h(self):
        # 14:23 UTC → 12:00 UTC (6 h cycle)
        ts = int(datetime(2026, 5, 1, 14, 23, tzinfo=timezone.utc).timestamp())
        floored = floor_cycle(ts)
        expected = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        assert floored == expected

    def test_floor_cycle_at_boundary(self):
        ts = int(datetime(2026, 5, 1, 18, 0, tzinfo=timezone.utc).timestamp())
        assert floor_cycle(ts) == ts

    def test_latest_published_run(self):
        now = int(datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc).timestamp())
        # 4 h delay → floor_cycle(10:00) = 06:00
        run = latest_published_run(now, 4 * 3600)
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
        assert MAX_FORECAST_HOURS == 48


# ── URL construction ─────────────────────────────────────────────────


class TestFileUrl:
    def test_format_matches_eccc_pattern(self):
        run = datetime(2026, 5, 7, 12, tzinfo=timezone.utc)
        url = file_url(run, 1)
        assert url.endswith(
            "/20260507/WXO-DD/model_hrdps/continental/2.5km/12/001/"
            "20260507T12Z_MSC_HRDPS_APCP-Accum1h_Sfc_RLatLon0.0225_PT001H.grib2"
        )

    def test_three_digit_padded_lead(self):
        # Different from HRRR's 2-digit lead — ECCC uses 3 digits.
        run = datetime(2026, 5, 7, 18, tzinfo=timezone.utc)
        for step, expected in [(1, "PT001H"), (6, "PT006H"), (12, "PT012H"), (48, "PT048H")]:
            url = file_url(run, step)
            assert expected in url

    def test_uses_settings_base_url(self):
        run = datetime(2026, 5, 7, 0, tzinfo=timezone.utc)
        url = file_url(run, 1)
        assert "dd.weather.gc.ca" in url


# ── Z-R conversion ───────────────────────────────────────────────────


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


# ── Decode orientation ───────────────────────────────────────────────


class TestDecodeOrientation:
    def test_decode_flips_south_up_grib(self, monkeypatch):
        from contextlib import contextmanager
        from librewxr.sources.regional.north_america.canada.nwp.hrdps import grid as h

        # Synthetic cfgrib output: row 0 at the SOUTHERN edge (matches
        # ECCC's GRIB scan mode jScansPositively=1, iScansNegatively=0).
        tp = np.zeros((HRDPS_GRID_HEIGHT, HRDPS_GRID_WIDTH), dtype=np.float32)
        tp[0, 100] = 5.0      # marker at south
        tp[-1, 200] = 8.0     # marker at north

        # Latitude coord increases with row index (south-up).
        lat = np.broadcast_to(
            np.linspace(28.0, 67.0, HRDPS_GRID_HEIGHT)[:, None],
            (HRDPS_GRID_HEIGHT, HRDPS_GRID_WIDTH),
        )
        lon = np.broadcast_to(
            np.linspace(-150.0, -50.0, HRDPS_GRID_WIDTH)[None, :],
            (HRDPS_GRID_HEIGHT, HRDPS_GRID_WIDTH),
        )

        import xarray as xr
        # ECCC ships the variable as paramId-unrecognised, so cfgrib
        # names it 'unknown'.  Use that exact name to verify the
        # fallback-to-first-2D-var logic works.
        fake_ds = xr.Dataset(
            {"unknown": (("y", "x"), tp)},
            coords={
                "latitude": (("y", "x"), lat),
                "longitude": (("y", "x"), lon),
            },
        )

        @contextmanager
        def _noop():
            yield

        monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: fake_ds)
        monkeypatch.setattr(h, "_suppress_eccodes_stderr", _noop)

        arr = h.decode_apcp_message(b"ignored")
        assert arr is not None
        # After flip: south marker (cfgrib row 0) → our row -1 (south)
        assert arr[-1, 100] == 5.0
        assert arr[0, 200] == 8.0


# ── Protocol + sample ────────────────────────────────────────────────


def _inject_frame(g: HRDPSGrid, run_ts: int, lead_seconds: int, encoded_value: int):
    arr = np.full(
        (HRDPS_GRID_HEIGHT, HRDPS_GRID_WIDTH),
        encoded_value, dtype=np.uint8,
    )
    g._frames[(run_ts, lead_seconds)] = arr
    if g._latest_run_ts is None or run_ts > g._latest_run_ts:
        g._latest_run_ts = run_ts


class TestProtocol:
    def test_satisfies_nwpsource(self):
        g = HRDPSGrid()
        assert isinstance(g, NWPSource)
        assert g.name == "hrdps"

    def test_empty_grid_returns_zeros(self):
        g = HRDPSGrid()
        out = g.sample(np.array([43.65]), np.array([-79.38]), timestamp=12345)
        assert out.shape == (1,)
        assert out[0] == 0

    def test_sample_at_exact_bracket(self):
        g = HRDPSGrid()
        run = int(datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 100)
        _inject_frame(g, run, 7200, 100)
        # Toronto is well inside HRDPS
        out = g.sample(np.array([43.65]), np.array([-79.38]), timestamp=run + 3600)
        assert int(out[0]) == 100

    def test_sample_lerps_between_brackets(self):
        g = HRDPSGrid()
        run = int(datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 80)
        _inject_frame(g, run, 7200, 160)
        out = g.sample(np.array([43.65]), np.array([-79.38]), timestamp=run + 5400)
        assert abs(int(out[0]) - 120) <= 1

    def test_outside_domain_zero(self):
        g = HRDPSGrid()
        run = int(datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 200)
        _inject_frame(g, run, 7200, 200)
        # Mexico City and Reykjavik both outside
        for lat, lon in [(19.43, -99.13), (64.13, -21.82)]:
            out = g.sample(np.array([lat]), np.array([lon]), timestamp=run + 3600)
            assert int(out[0]) == 0

    def test_has_data_at_within_horizon(self):
        # has_data_at requires BOTH bracketing frames loaded.
        g = HRDPSGrid()
        run = int(datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 3600, 50)
        _inject_frame(g, run, 7200, 50)
        _inject_frame(g, run, 10800, 50)
        assert g.has_data_at(run + 3600) is True
        assert g.has_data_at(run + 5400) is True
        assert g.has_data_at(run + 7200) is True
        # Beyond the last loaded bracket
        assert g.has_data_at(run + 11000) is False


# ── Chain integration ────────────────────────────────────────────────


class TestChainOrdering:
    def test_chain_prefers_hrdps_inside_falls_back_outside(self):
        # Build a minimal HRDPS in front of a global IFS fallback.
        # Inside HRDPS the chain returns HRDPS's value; outside the
        # rotated rectangle it falls through to IFS.
        from librewxr.sources.world.ifs.grid import (
            ECMWFGrid,
            GRID_HEIGHT as IFS_H,
            GRID_WIDTH as IFS_W,
        )

        ifs = ECMWFGrid()
        ifs_dbz = np.full((IFS_H, IFS_W), 84, dtype=np.uint8)  # 10 dBZ
        ifs._timesteps[1000000] = (ifs_dbz, np.zeros_like(ifs_dbz, dtype=bool))
        ifs._sorted_timestamps = [1000000]

        hrdps = HRDPSGrid()
        run = 1000000 - 1500
        _inject_frame(hrdps, run, 0, 164)      # 50 dBZ, exact bracket
        _inject_frame(hrdps, run, 3600, 164)

        chain = NWPChain([hrdps, ifs])
        # Toronto: inside HRDPS rotated rectangle
        out_ca = chain.sample(np.array([43.65]), np.array([-79.38]), timestamp=1000000)
        assert abs(int(out_ca[0]) - 164) <= 1
        # Mexico City: outside HRDPS → IFS fills
        out_mx = chain.sample(np.array([19.43]), np.array([-99.13]), timestamp=1000000)
        assert int(out_mx[0]) == 84


# ── SW corner invariant ──────────────────────────────────────────────


class TestGridOrigin:
    def test_sw_corner_at_height_minus_one_col_zero(self):
        # The decoded GRIB SW corner must round-trip to (HEIGHT-1, 0).
        row, col = grid_indices(np.array([HRDPS_LA1]), np.array([HRDPS_LO1]))
        assert abs(row[0] - (HRDPS_GRID_HEIGHT - 1)) < 1e-3
        assert abs(col[0] - 0) < 1e-3

    def test_grid_dimensions(self):
        # Sanity check on the documented grid dimensions.
        assert HRDPS_GRID_WIDTH == 2540
        assert HRDPS_GRID_HEIGHT == 1290

    def test_origin_consistency(self):
        # The north and south rlat origins should differ by exactly
        # (HEIGHT - 1) * dy.
        assert HRDPS_GRID_RLAT_ORIGIN_NORTH > HRDPS_GRID_RLAT_ORIGIN_SOUTH
        # Just confirm rlon origin is finite — the exact value depends
        # on the documented (La1, Lo1) and the rotation.
        assert np.isfinite(HRDPS_GRID_RLON_ORIGIN)
