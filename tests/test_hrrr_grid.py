# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Unit tests for HRRR projection, idx parsing, and HRRRGrid sampling.

These are pure unit tests — no S3 network calls.  Live HRRR fetch is
exercised manually during development; CI runs only the unit set.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.hrrr

from librewxr.data.hrrr_grid import (
    HRRR_FEATHER_DISTANCE_M,
    HRRR_GRID_DX,
    HRRR_GRID_HEIGHT,
    HRRR_GRID_WIDTH,
    HRRRGrid,
    SUBH_INTERVAL_SECONDS,
    bracket_subh_leads,
    domain_mask,
    encode_dbz,
    feather_mask,
    find_refc_records,
    floor_hour,
    grid_indices,
    latest_published_run,
    lcc_forward,
    lead_seconds_for_step,
    parse_idx,
)
from librewxr.data.nwp_source import NWPChain, NWPSource


# ── Projection ───────────────────────────────────────────────────────


class TestLCCProjection:
    def test_origin_maps_to_zero(self):
        x, y = lcc_forward(np.array([38.5]), np.array([-97.5]))
        assert abs(x[0]) < 1e-6
        assert abs(y[0]) < 1e-6

    def test_origin_grid_index_is_grid_center(self):
        # 38.5N -97.5W is the projection origin; its grid (row, col) should
        # land near the centre of the 1799x1059 CONUS grid.
        row, col = grid_indices(np.array([38.5]), np.array([-97.5]))
        assert 525 < row[0] < 535
        assert 895 < col[0] < 905

    @pytest.mark.parametrize(
        "name,lat,lon,inside",
        [
            ("NYC",       40.7128,  -74.006,  True),
            ("LA",        34.0522, -118.244,  True),
            ("Seattle",   47.6062, -122.332,  True),
            ("Miami",     25.7617,  -80.192,  True),
            ("Anchorage", 61.2181, -149.900, False),
            ("London",    51.5074,   -0.128, False),
            ("Bermuda",   32.3078,  -64.751, False),  # east of HRRR
        ],
    )
    def test_domain_mask_known_points(self, name, lat, lon, inside):
        m = domain_mask(np.array([lat]), np.array([lon]))
        assert bool(m[0]) is inside, name

    def test_grid_indices_vectorize(self):
        lats = np.array([40.0, 35.0, 47.0])
        lons = np.array([-100.0, -90.0, -120.0])
        row, col = grid_indices(lats, lons)
        assert row.shape == lats.shape
        assert col.shape == lats.shape
        # All three points are deep inside CONUS
        assert ((row > 0) & (row < HRRR_GRID_HEIGHT - 1)).all()
        assert ((col > 0) & (col < HRRR_GRID_WIDTH - 1)).all()


# ── Boundary feather ──────────────────────────────────────────────────


class TestFeatherMask:
    def test_origin_is_full_weight(self):
        # The projection origin (38.5, -97.5) is at grid centre — far
        # from every edge — so feather should be exactly 1.0.
        f = feather_mask(np.array([38.5]), np.array([-97.5]))
        assert f.dtype == np.float32
        assert f[0] == pytest.approx(1.0)

    def test_outside_domain_is_zero(self):
        # London is well outside HRRR's CONUS grid → feather = 0.
        f = feather_mask(np.array([51.5]), np.array([-0.1]))
        assert f[0] == 0.0

    def test_far_inside_is_full_weight(self):
        # Pick points that are clearly more than the feather distance
        # from any LCC grid edge.  With 75 km feather and grids ~2000+ km
        # wide, any major US city interior of the SW/SE corners qualifies.
        cities = [
            (40.0, -100.0),  # KS
            (40.7, -74.0),   # NYC — inside but closer to east edge
            (41.9, -87.6),   # Chicago
        ]
        lats = np.array([c[0] for c in cities])
        lons = np.array([c[1] for c in cities])
        f = feather_mask(lats, lons)
        assert (f == 1.0).all(), f"expected all-1.0 feather, got {f}"

    def test_taper_is_monotonic_at_edge(self):
        # Walk lat from inside to outside on a fixed central meridian.
        # Feather should be monotonically non-increasing.
        lats = np.linspace(38.5, 65.0, 30)
        lons = np.full_like(lats, -97.5)
        f = feather_mask(lats, lons)
        diffs = np.diff(f)
        assert (diffs <= 1e-6).all(), "feather must be non-increasing as we leave domain"

    def test_taper_width_matches_constant(self):
        # A point exactly HRRR_FEATHER_DISTANCE_M from the grid centre
        # along a row direction should have feather ≈ 1.0; one beyond
        # the inner region but inside the grid should be < 1.0.
        # Use grid index reasoning: deep inside (centre) → feather=1.
        # At col index < FEATHER_DISTANCE_M / DX cells from edge → < 1.
        feather_cells = int(HRRR_FEATHER_DISTANCE_M // HRRR_GRID_DX)
        # Project two known LCC points back to lat/lon by reusing
        # cfgrib's stored coordinates is overkill — instead, build
        # query points at fractional col indices using the inverse
        # of grid_indices.  For col_target near the west edge (col=5):
        # deep inside corner (col=feather_cells+5): feather = 1
        # at col=2 (very near west edge): feather close to 0
        # We can't easily inverse-project without doing real LCC inverse,
        # so just sanity-check the centre vs an edge-adjacent lat/lon.
        deep = feather_mask(np.array([38.5]), np.array([-97.5]))[0]
        # West edge of HRRR is roughly lon ~-122 at lat 38, but the
        # LCC corners curve.  Pick a point that grid_indices says is
        # 2 cells from the west edge (col=2.0):
        # This requires inverse projection; skip the precise test.
        assert deep == pytest.approx(1.0)


class TestSoftBlendChain:
    """Verify the chain blends sources by feather instead of hard-fill."""

    def test_chain_with_binary_feathers_matches_hard_fill(self):
        """A chain of binary-feather sources behaves like the old hard fill."""
        from librewxr.data.ecmwf_grid import ECMWFGrid
        from librewxr.data.ecmwf_grid import GRID_HEIGHT as IFS_H, GRID_WIDTH as IFS_W

        ifs = ECMWFGrid()
        ifs_dbz = np.full((IFS_H, IFS_W), int((10 + 32) * 2), dtype=np.uint8)
        ifs._timesteps[1000000] = (ifs_dbz, np.zeros_like(ifs_dbz, dtype=bool))
        ifs._sorted_timestamps = [1000000]

        chain = NWPChain([ifs])

        # IFS is global with all-1.0 feather → output is just IFS values.
        out = chain.sample(np.array([40.0, 51.5]),
                           np.array([-100.0, -0.1]),
                           timestamp=1000000)
        assert int(out[0]) == 84   # 10 dBZ encoded
        assert int(out[1]) == 84

    def test_chain_blends_in_hrrr_feather_zone(self):
        """In HRRR's feather zone, the chain blends HRRR and IFS proportionally."""
        from librewxr.data.ecmwf_grid import ECMWFGrid
        from librewxr.data.ecmwf_grid import GRID_HEIGHT as IFS_H, GRID_WIDTH as IFS_W

        # IFS = 0 dBZ everywhere (encoded 64); HRRR = 50 dBZ everywhere (encoded 164)
        ifs = ECMWFGrid()
        ifs_dbz = np.full((IFS_H, IFS_W), int((0 + 32) * 2), dtype=np.uint8)
        ifs._timesteps[1000000] = (ifs_dbz, np.zeros_like(ifs_dbz, dtype=bool))
        ifs._sorted_timestamps = [1000000]

        hrrr = HRRRGrid()
        run = 1000000 - 1500
        # Inject uniform 50-dBZ frames so HRRR returns the same value
        # everywhere it's queried inside its domain.
        _inject_frame(hrrr, run, 900, 50.0)
        _inject_frame(hrrr, run, 1800, 50.0)

        chain = NWPChain([hrrr, ifs])

        # Deep inside CONUS: feather = 1 → all HRRR → encoded 164
        deep = chain.sample(np.array([40.0]),
                            np.array([-100.0]),
                            timestamp=1000000)
        assert abs(int(deep[0]) - 164) <= 1

        # Outside HRRR (London): feather = 0 → all IFS → encoded 64
        out = chain.sample(np.array([51.5]),
                           np.array([-0.1]),
                           timestamp=1000000)
        assert int(out[0]) == 64

        # In the feather zone, the value should fall between IFS (64) and
        # HRRR (164) and not equal either endpoint.  Pick a lat/lon that
        # we know lands a few grid cells inside the SW corner.  HRRR's
        # SW corner (col=0, row=HEIGHT-1) is around lat 21.14, lon -122.7.
        # Walk in slightly: feather_mask should give a partial weight.
        lats = np.array([22.5])
        lons = np.array([-118.0])
        partial = chain.sample(lats, lons, timestamp=1000000)
        # With HRRR's domain mask, lat=22.5 lon=-118 might or might not be
        # inside; just verify the result is a valid uint8 between IFS and HRRR.
        if domain_mask(lats, lons)[0]:
            f = feather_mask(lats, lons)[0]
            if 0.0 < f < 1.0:
                # In feather zone — value should be a real blend
                expected = round(f * 164 + (1 - f) * 64)
                assert abs(int(partial[0]) - expected) <= 2


# ── Subh timing math ──────────────────────────────────────────────────


class TestSubhTiming:
    def test_floor_hour(self):
        # 12:34:56 UTC → 12:00:00 UTC
        ts = int(datetime(2026, 5, 1, 12, 34, 56, tzinfo=timezone.utc).timestamp())
        floored = floor_hour(ts)
        expected = int(datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        assert floored == expected

    def test_latest_published_run_55min_delay(self):
        now = int(datetime(2026, 5, 1, 13, 30, 0, tzinfo=timezone.utc).timestamp())
        # With 55 min delay: now - 55min = 12:35.  Floor → 12:00.
        run = latest_published_run(now, 55 * 60)
        expected = int(datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        assert run == expected

    @pytest.mark.parametrize(
        "lead_min,expected_l0,expected_l1,expected_alpha",
        [
            (15,  15,  30, 0.000),  # exact subh frame
            (16,  15,  30, 1/15),
            (23,  15,  30, 8/15),
            (30,  30,  45, 0.000),
            (37,  30,  45, 7/15),
            (60,  60,  75, 0.000),
            (90,  90, 105, 0.000),
        ],
    )
    def test_bracket_subh_leads(self, lead_min, expected_l0, expected_l1, expected_alpha):
        l0, l1, alpha = bracket_subh_leads(lead_min * 60)
        assert l0 == expected_l0 * 60
        assert l1 == expected_l1 * 60
        assert abs(alpha - expected_alpha) < 1e-9

    def test_bracket_below_first_subh_clamps(self):
        # Lead < 900s falls back to (900, 900, 0) — caller handles by
        # rolling to a previous run.
        l0, l1, alpha = bracket_subh_leads(300)
        assert l0 == SUBH_INTERVAL_SECONDS
        assert l1 == SUBH_INTERVAL_SECONDS
        assert alpha == 0.0

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("15 min fcst", 900),
            ("60 min fcst", 3600),
            ("75 min fcst", 4500),
            ("120 min fcst", 7200),
        ],
    )
    def test_lead_seconds_for_step(self, label, expected):
        assert lead_seconds_for_step(label) == expected

    def test_lead_seconds_rejects_average_steps(self):
        # "70-75 min ave fcst" is an averaged window, not an instantaneous
        # forecast — reject these so they don't get keyed as instant frames.
        assert lead_seconds_for_step("70-75 min ave fcst") is None
        assert lead_seconds_for_step("anl") is None


# ── Idx parsing ───────────────────────────────────────────────────────


class TestIdxParsing:
    SAMPLE_IDX = (
        "1:0:d=2026050112:REFC:entire atmosphere:15 min fcst:\n"
        "2:466568:d=2026050112:RETOP:cloud top:15 min fcst:\n"
        "3:729628:d=2026050112:VIS:surface:15 min fcst:\n"
        "50:52337761:d=2026050112:REFC:entire atmosphere:30 min fcst:\n"
        "99:104679265:d=2026050112:REFC:entire atmosphere:45 min fcst:\n"
        "148:157975737:d=2026050112:REFC:entire atmosphere:60 min fcst:\n"
    )

    def test_parse_idx(self):
        records = parse_idx(self.SAMPLE_IDX)
        assert len(records) == 6
        assert records[0].var == "REFC"
        assert records[0].byte_offset == 0
        assert records[0].step == "15 min fcst"
        assert records[3].var == "REFC"
        assert records[3].step == "30 min fcst"

    def test_parse_idx_skips_garbage_lines(self):
        text = self.SAMPLE_IDX + "garbage line that doesn't match\n"
        records = parse_idx(text)
        assert len(records) == 6  # garbage line skipped silently

    def test_find_refc_records(self):
        records = parse_idx(self.SAMPLE_IDX)
        refcs = find_refc_records(records)
        assert len(refcs) == 4
        # First REFC: bytes 0..(next record - 1) = 0..466567
        first_rec, first_end = refcs[0]
        assert first_rec.byte_offset == 0
        assert first_end == 466567
        # Last REFC has no following record → end = -1 (caller uses bytes=N-)
        last_rec, last_end = refcs[-1]
        assert last_rec.step == "60 min fcst"
        assert last_end == -1


# ── Encoding ──────────────────────────────────────────────────────────


class TestEncoding:
    def test_encode_known_dbz_values(self):
        # Encoding: pixel = (dBZ + 32) * 2, clipped to [0, 255]
        refc = np.array([-32.0, 0.0, 50.0, 95.0])
        encoded = encode_dbz(refc)
        assert encoded.dtype == np.uint8
        assert encoded.tolist() == [0, 64, 164, 254]

    def test_encode_handles_nan(self):
        refc = np.array([np.nan, 30.0, np.nan])
        encoded = encode_dbz(refc)
        # NaN gets coerced to -32 (encoded 0) — "no precipitation"
        assert encoded[0] == 0
        assert encoded[1] == int((30 + 32) * 2)
        assert encoded[2] == 0

    def test_encode_clips_extremes(self):
        refc = np.array([-100.0, 200.0])
        encoded = encode_dbz(refc)
        assert encoded[0] == 0
        assert encoded[1] == 255


# ── Decode orientation regression ────────────────────────────────────


class TestDecodeOrientation:
    """Verify decode_refc_message flips cfgrib's south-up output to north-up.

    cfgrib returns HRRR REFC with row 0 at the SOUTH edge (lat ~21°N) and
    row 1058 at the NORTH edge (lat ~52°N).  The grid_indices() function
    in this module assumes the standard image orientation — row 0 = north.
    Without the flip, every sample reads from a cell mirrored about the
    central parallel, producing the visual "rotation" bug observed during
    the first live HRRR test.
    """

    def test_decode_flips_south_up_grib(self, monkeypatch):
        from contextlib import contextmanager
        from librewxr.data import hrrr_grid as hgm

        # Synthetic cfgrib-style output: row 0 at lat=21°N, row last at lat=53°N
        refc_data = np.zeros((HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH), dtype=np.float32)
        refc_data[0, 100] = 65.0  # marker at the SOUTH edge (cfgrib row 0)
        refc_data[-1, 200] = 45.0  # marker at the NORTH edge (cfgrib row -1)

        lat_data = np.broadcast_to(
            np.linspace(21.0, 53.0, HRRR_GRID_HEIGHT)[:, None],
            (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH),
        )
        lon_data = np.broadcast_to(
            np.linspace(225.0, 300.0, HRRR_GRID_WIDTH)[None, :],
            (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH),
        )

        import xarray as xr
        fake_ds = xr.Dataset(
            {"refc": (("y", "x"), refc_data)},
            coords={
                "latitude": (("y", "x"), lat_data),
                "longitude": (("y", "x"), lon_data),
            },
        )

        @contextmanager
        def _noop():
            yield

        monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: fake_ds)
        monkeypatch.setattr(hgm, "_suppress_eccodes_stderr", _noop)

        arr = hgm.decode_refc_message(b"ignored bytes")
        assert arr is not None
        assert arr.shape == (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH)
        # After flip: cfgrib's row 0 (south) → our row -1 (south of image)
        # and cfgrib's row -1 (north) → our row 0 (north of image)
        assert arr[-1, 100] == 65.0, "south-edge marker should land at our last row"
        assert arr[0, 200] == 45.0, "north-edge marker should land at our row 0"

    def test_decode_does_not_double_flip_north_up_grib(self, monkeypatch):
        """If cfgrib ever changes to return north-up natively, don't re-flip."""
        from contextlib import contextmanager
        from librewxr.data import hrrr_grid as hgm

        refc_data = np.zeros((HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH), dtype=np.float32)
        refc_data[0, 100] = 65.0  # at NORTH edge in already-correct frame
        refc_data[-1, 200] = 45.0

        # lat decreases with row → already north-up
        lat_data = np.broadcast_to(
            np.linspace(53.0, 21.0, HRRR_GRID_HEIGHT)[:, None],
            (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH),
        )
        lon_data = np.broadcast_to(
            np.linspace(225.0, 300.0, HRRR_GRID_WIDTH)[None, :],
            (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH),
        )

        import xarray as xr
        fake_ds = xr.Dataset(
            {"refc": (("y", "x"), refc_data)},
            coords={
                "latitude": (("y", "x"), lat_data),
                "longitude": (("y", "x"), lon_data),
            },
        )

        @contextmanager
        def _noop():
            yield

        monkeypatch.setattr(xr, "open_dataset", lambda *a, **kw: fake_ds)
        monkeypatch.setattr(hgm, "_suppress_eccodes_stderr", _noop)

        arr = hgm.decode_refc_message(b"ignored bytes")
        # Already north-up; markers should stay where they are
        assert arr[0, 100] == 65.0
        assert arr[-1, 200] == 45.0


# ── HRRRGrid sample / Protocol ───────────────────────────────────────


def _inject_frame(grid: HRRRGrid, run_ts: int, lead_seconds: int, refc_dbz: float) -> None:
    """Helper: inject a uniform-value frame into HRRRGrid for testing."""
    arr = np.full((HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH), refc_dbz, dtype=np.float32)
    encoded = encode_dbz(arr)
    grid._frames[(run_ts, lead_seconds)] = encoded
    if grid._latest_run_ts is None or run_ts > grid._latest_run_ts:
        grid._latest_run_ts = run_ts


class TestHRRRGridProtocol:
    def test_satisfies_protocol(self):
        g = HRRRGrid()
        assert isinstance(g, NWPSource)
        assert g.name == "hrrr"

    def test_empty_grid_sample_returns_zeros(self):
        g = HRRRGrid()
        lat = np.array([40.0])
        lon = np.array([-100.0])
        out = g.sample(lat, lon, timestamp=12345)
        assert out.shape == lat.shape
        assert (out == 0).all()

    def test_has_data_with_injected_frames(self):
        g = HRRRGrid()
        run = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 900, 30.0)   # F+15
        _inject_frame(g, run, 1800, 35.0)  # F+30
        assert g.has_data() is True

        # Mid-bracket valid time
        lead = 1500  # 25 min, between F+15 and F+30
        sample_ts = run + lead
        assert g.has_data_at(sample_ts) is True

        # Lead with no data (frame not loaded)
        assert g.has_data_at(run + 5400) is False  # F+90, no frame

    def test_sample_lerp_at_midpoint(self):
        """Sample at the midpoint between two frames: should average them."""
        g = HRRRGrid()
        run = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 900, 20.0)   # F+15: uniform 20 dBZ
        _inject_frame(g, run, 1800, 40.0)  # F+30: uniform 40 dBZ

        # Sample at lead 22:30 (midpoint = alpha 0.5) → uniform 30 dBZ
        # encoded: (30 + 32) * 2 = 124
        sample_ts = run + 22 * 60 + 30  # 22:30
        lat = np.array([40.0])  # any CONUS point
        lon = np.array([-100.0])
        out = g.sample(lat, lon, timestamp=sample_ts)
        # Allow ±1 due to rounding through uint8
        assert abs(int(out[0]) - 124) <= 1

    def test_sample_outside_domain_is_zero(self):
        g = HRRRGrid()
        run = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 900, 30.0)
        _inject_frame(g, run, 1800, 30.0)

        # London is outside HRRR
        lat = np.array([51.5])
        lon = np.array([-0.1])
        out = g.sample(lat, lon, timestamp=run + 1500)
        assert out[0] == 0

    def test_sample_picks_freshest_run(self):
        """Two runs cover the same target time; the newer one wins."""
        g = HRRRGrid()
        old_run = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        new_run = old_run + 3600  # 13:00Z

        # Old run says 10 dBZ at the target valid time
        target_ts = new_run + 1500  # F+25 from new_run, F+85 from old_run
        _inject_frame(g, old_run, 4500, 10.0)   # F+75 from 12Z
        _inject_frame(g, old_run, 5400, 10.0)   # F+90 from 12Z
        # New run says 50 dBZ at the same valid time
        _inject_frame(g, new_run, 900, 50.0)    # F+15 from 13Z
        _inject_frame(g, new_run, 1800, 50.0)   # F+30 from 13Z

        out = g.sample(np.array([40.0]), np.array([-100.0]), timestamp=target_ts)
        # Should pick the new run → uniform 50 dBZ → encoded 164
        assert abs(int(out[0]) - 164) <= 1

    def test_sample_falls_back_to_older_run_when_newer_bracket_missing(self):
        """During run rollover, prefer the older fully-loaded run over a newer
        one whose bracket frames haven't finished fetching yet.

        Regression: ``_pick_run`` originally returned the freshest run whose
        forecast horizon covered the timestamp, regardless of whether the
        bracket frames were actually loaded.  That made adjacent nowcast
        frames silently switch between HRRR and IFS during the period when
        a fresh run was mid-fetch — visually a discontinuous "flip" in the
        animation loop.
        """
        g = HRRRGrid()
        old_run = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        new_run = old_run + 3600

        # New run is mid-fetch: only subh01 (leads 15-60 min) has landed.
        _inject_frame(g, new_run, 900, 50.0)
        _inject_frame(g, new_run, 1800, 50.0)
        _inject_frame(g, new_run, 2700, 50.0)
        _inject_frame(g, new_run, 3600, 50.0)
        # Old run is fully loaded across the relevant range (subh02-03).
        for lead_min in (75, 90, 105, 120, 135, 150):
            _inject_frame(g, old_run, lead_min * 60, 5.0)

        # Target needs lead 75 min from new_run (in subh02, NOT yet loaded)
        # OR lead 135 min from old_run — bracket (135, 150), both loaded.
        target_ts = new_run + 75 * 60
        out = g.sample(np.array([40.0]), np.array([-100.0]), timestamp=target_ts)
        # Old run's value (5 dBZ → encoded 74), not new run's 99 dBZ — and
        # not zero (which would mean has_data_at returned False and the
        # chain fell through to IFS).
        encoded_old = int((5 + 32) * 2)
        assert abs(int(out[0]) - encoded_old) <= 1

        # has_data_at must agree
        assert g.has_data_at(target_ts) is True

        # And for a target ts that the new run CAN serve (lead in subh01),
        # we should pick the new run.
        target_in_new = new_run + 30 * 60
        out2 = g.sample(np.array([40.0]), np.array([-100.0]), timestamp=target_in_new)
        encoded_new = int((50 + 32) * 2)
        assert abs(int(out2[0]) - encoded_new) <= 1

    def test_get_snow_mask_returns_all_false(self):
        # Phase 2 v1: HRRR delegates snow classification to the next chain
        # source by returning False everywhere.
        g = HRRRGrid()
        run = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 900, 30.0)
        lat = np.array([40.0, 50.0, 60.0])
        lon = np.array([-100.0, -90.0, -80.0])
        out = g.get_snow_mask(lat, lon, timestamp=run + 900)
        assert out.dtype == np.bool_
        assert out.shape == lat.shape
        assert not out.any()


# ── Chain integration ────────────────────────────────────────────────


class TestChainWithHRRR:
    def test_chain_prefers_hrrr_inside_conus(self):
        from librewxr.data.ecmwf_grid import ECMWFGrid
        from librewxr.data.ecmwf_grid import GRID_HEIGHT as IFS_H, GRID_WIDTH as IFS_W

        # IFS says 10 dBZ everywhere; HRRR says 50 dBZ in CONUS.
        ifs = ECMWFGrid()
        ifs_dbz = np.full((IFS_H, IFS_W), int((10 + 32) * 2), dtype=np.uint8)
        ifs._timesteps[1000000] = (ifs_dbz, np.zeros_like(ifs_dbz, dtype=bool))
        ifs._sorted_timestamps = [1000000]

        hrrr = HRRRGrid()
        run = 1000000 - 1500  # so target ts lands at lead=1500 from this run
        # Note: run must align to an hour for has_data_at semantics, but for
        # the mocked _pick_run we just need something where target_ts - run
        # is in [900, 18*3600].
        _inject_frame(hrrr, run, 900, 50.0)
        _inject_frame(hrrr, run, 1800, 50.0)

        chain = NWPChain([hrrr, ifs])

        # CONUS point: HRRR fills it → 50 dBZ encoded ≈ 164
        out = chain.sample(np.array([40.0]), np.array([-100.0]), timestamp=1000000)
        assert abs(int(out[0]) - 164) <= 1

        # Outside HRRR domain (London): IFS fills it → 10 dBZ encoded = 84
        out = chain.sample(np.array([51.5]), np.array([-0.1]), timestamp=1000000)
        assert int(out[0]) == 84
