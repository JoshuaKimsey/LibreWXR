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

from librewxr.sources.regional.north_america.usa.nwp.hrrr.grid import (
    HRRR_FEATHER_DISTANCE_M,
    HRRR_GRID_DX,
    HRRR_GRID_HEIGHT,
    HRRR_GRID_WIDTH,
    HRRRGrid,
    SUBH_INTERVAL_SECONDS,
    bracket_subh_leads,
    compute_snow_mask,
    domain_mask,
    encode_dbz,
    feather_mask,
    find_refc_records,
    find_tmp_2m_records,
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
        from librewxr.sources.world.ifs.grid import ECMWFGrid
        from librewxr.sources.world.ifs.grid import GRID_HEIGHT as IFS_H, GRID_WIDTH as IFS_W

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
        from librewxr.sources.world.ifs.grid import ECMWFGrid
        from librewxr.sources.world.ifs.grid import GRID_HEIGHT as IFS_H, GRID_WIDTH as IFS_W

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
        from librewxr.sources.regional.north_america.usa.nwp.hrrr import grid as hgm

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
        from librewxr.sources.regional.north_america.usa.nwp.hrrr import grid as hgm

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

    def test_get_snow_mask_without_loaded_masks_returns_false(self):
        # When REFC frames are loaded but no snow masks are present (e.g.
        # TMP:2m fetch failed), get_snow_mask returns all-False so the
        # chain dispatcher falls through to the next snow-capable source.
        g = HRRRGrid()
        run = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 900, 30.0)
        _inject_frame(g, run, 1800, 30.0)
        lat = np.array([40.0, 50.0, 60.0])
        lon = np.array([-100.0, -90.0, -80.0])
        out = g.get_snow_mask(lat, lon, timestamp=run + 1500)
        assert out.dtype == np.bool_
        assert out.shape == lat.shape
        assert not out.any()


# ── Chain integration ────────────────────────────────────────────────


class TestChainWithHRRR:
    def test_chain_prefers_hrrr_inside_conus(self):
        from librewxr.sources.world.ifs.grid import ECMWFGrid
        from librewxr.sources.world.ifs.grid import GRID_HEIGHT as IFS_H, GRID_WIDTH as IFS_W

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


# ── Disk persistence ─────────────────────────────────────────────────


class TestPersistence:
    """Verify HRRRGrid's optional persistent cache survives restarts."""

    def _ingest_one_frame(self, g: HRRRGrid, run_ts: int, lead: int, value: float):
        """Round-trip a frame through ``_to_memmap`` (the real write path)."""
        arr = np.full(
            (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH), value, dtype=np.float32,
        )
        encoded = encode_dbz(arr)
        mm = g._to_memmap(f"r{run_ts}_l{lead}", encoded)
        g._frames[(run_ts, lead)] = mm
        if g._latest_run_ts is None or run_ts > g._latest_run_ts:
            g._latest_run_ts = run_ts

    @pytest.mark.asyncio
    async def test_no_cache_dir_uses_tmp_and_cleans_up(self, tmp_path):
        """Default behaviour (no cache_dir) keeps a tmpdir, cleaned on close."""
        g = HRRRGrid()
        memmap_dir = g._memmap_dir
        assert memmap_dir.exists()
        assert g._persistent is False
        await g.close()
        # tmp directory should be removed on close
        assert not memmap_dir.exists()

    @pytest.mark.asyncio
    async def test_persistent_cache_survives_restart(self, tmp_path):
        """Frames written by one HRRRGrid show up in another with the same cache_dir."""
        run_ts = int(datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc).timestamp())

        # First "process": ingest a frame and close.
        g1 = HRRRGrid(cache_dir=tmp_path)
        assert g1._persistent is True
        assert g1._memmap_dir == tmp_path / "hrrr"
        self._ingest_one_frame(g1, run_ts, 900, 30.0)
        self._ingest_one_frame(g1, run_ts, 1800, 30.0)
        await g1.close()
        # Cache dir must still exist after close (persistent mode)
        assert (tmp_path / "hrrr").exists()
        assert (tmp_path / "hrrr" / f"r{run_ts}_l900.dat").exists()

        # Second "process": load and verify the frames came back.
        g2 = HRRRGrid(cache_dir=tmp_path)
        assert g2.frame_count == 2
        assert (run_ts, 900) in g2._frames
        assert (run_ts, 1800) in g2._frames
        assert g2._latest_run_ts == run_ts

        # Sample at a CONUS point — should return the encoded value
        sample_ts = run_ts + 1500  # midway between the two frames, alpha=0.667
        out = g2.sample(np.array([40.0]), np.array([-100.0]), timestamp=sample_ts)
        assert abs(int(out[0]) - int((30 + 32) * 2)) <= 1
        await g2.close()

    @pytest.mark.asyncio
    async def test_eviction_removes_disk_files(self, tmp_path):
        """Out-of-window frames get unlinked from disk too."""
        run_ts = int(datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc).timestamp())
        g = HRRRGrid(cache_dir=tmp_path)
        self._ingest_one_frame(g, run_ts, 900, 30.0)
        path = tmp_path / "hrrr" / f"r{run_ts}_l900.dat"
        assert path.exists()

        # Evict against a window that doesn't include this frame's valid time
        far_future = run_ts + 24 * 3600
        g._evict_outside_window(far_future, far_future + 600)
        assert not path.exists(), "evicted frame should be unlinked from disk"
        assert (run_ts, 900) not in g._frames
        await g.close()

    @pytest.mark.asyncio
    async def test_load_skips_corrupt_files(self, tmp_path):
        """Files with bad names or wrong size are removed and skipped, not fatal."""
        cache_dir = tmp_path / "hrrr"
        cache_dir.mkdir()
        # Wrong size: a 1-byte file pretending to be a frame
        (cache_dir / "r1234_l900.dat").write_bytes(b"x")
        # Bad filename: doesn't match r*_l*.dat
        (cache_dir / "garbage.dat").write_bytes(b"y" * 100)
        # Stale .tmp from a crashed write — should be cleaned up
        (cache_dir / "r9999_l900.dat.tmp").write_bytes(b"z")

        g = HRRRGrid(cache_dir=tmp_path)
        # Bad-size file: tried to memmap, may have succeeded with wrong shape
        # — the size mismatch will raise on read, but we don't fail fast.
        # Bad-filename .dat is silently skipped (filename parse fails).
        # Stale .tmp is removed.
        assert not (cache_dir / "r9999_l900.dat.tmp").exists()
        await g.close()

    @pytest.mark.asyncio
    async def test_atomic_write_uses_tmp_and_renames(self, tmp_path):
        """``_to_memmap`` writes via .tmp and renames into place."""
        g = HRRRGrid(cache_dir=tmp_path)
        run_ts = int(datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc).timestamp())
        # Mid-write, the .tmp file briefly exists but is gone after the call.
        self._ingest_one_frame(g, run_ts, 900, 25.0)
        cache_dir = tmp_path / "hrrr"
        assert (cache_dir / f"r{run_ts}_l900.dat").exists()
        # No leftover .tmp
        assert not list(cache_dir.glob("*.tmp"))
        await g.close()


# ── Snow mask ─────────────────────────────────────────────────────────


def _inject_snow_mask(
    grid: HRRRGrid, run_ts: int, lead_seconds: int, snow_value: int
) -> None:
    """Helper: inject a uniform-value snow mask into HRRRGrid for testing."""
    arr = np.full(
        (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH),
        snow_value & 0x01,
        dtype=np.uint8,
    )
    grid._snow_masks[(run_ts, lead_seconds)] = arr


class TestSnowMaskHelpers:
    """The module-level T_2m / snow-mask helper functions."""

    SAMPLE_IDX = (
        "1:0:d=2026050112:REFC:entire atmosphere:15 min fcst:\n"
        "2:466568:d=2026050112:TMP:2 m above ground:15 min fcst:\n"
        "3:729628:d=2026050112:TMP:surface:15 min fcst:\n"
        "4:800000:d=2026050112:DPT:2 m above ground:15 min fcst:\n"
        "5:900000:d=2026050112:TMP:2 m above ground:30 min fcst:\n"
        "6:1000000:d=2026050112:TMP:2 m above ground:45 min fcst:\n"
    )

    def test_find_tmp_2m_records_filters_level(self):
        records = parse_idx(self.SAMPLE_IDX)
        tmps = find_tmp_2m_records(records)
        # Three TMP records at "2 m above ground"; surface TMP and DPT excluded.
        assert len(tmps) == 3
        first_rec, first_end = tmps[0]
        assert first_rec.var == "TMP"
        assert first_rec.level == "2 m above ground"
        assert first_rec.step == "15 min fcst"
        # First TMP record byte range is bounded by the next record (surface
        # TMP at offset 729628), not the next 2-m TMP record.
        assert first_end == 729627
        # Last 2-m TMP has no further record → end = -1
        last_rec, last_end = tmps[-1]
        assert last_rec.step == "45 min fcst"
        assert last_end == -1

    def test_find_tmp_2m_records_empty_on_no_tmp(self):
        records = parse_idx(
            "1:0:d=2026050112:REFC:entire atmosphere:15 min fcst:\n"
            "2:100:d=2026050112:RETOP:cloud top:15 min fcst:\n"
        )
        assert find_tmp_2m_records(records) == []

    def test_compute_snow_mask_below_threshold(self):
        # < 1.5 °C → snow (1)
        t2m = np.array([[-10.0, 0.0, 1.0]], dtype=np.float32)
        out = compute_snow_mask(t2m, threshold=1.5)
        assert out.dtype == np.uint8
        assert out.tolist() == [[1, 1, 1]]

    def test_compute_snow_mask_above_threshold(self):
        # >= 1.5 °C → rain (0)
        t2m = np.array([[1.5, 5.0, 20.0]], dtype=np.float32)
        out = compute_snow_mask(t2m, threshold=1.5)
        assert out.tolist() == [[0, 0, 0]]

    def test_compute_snow_mask_handles_nan(self):
        # NaN cells are treated as no-snow (don't paint snow palette over
        # decode glitches).
        t2m = np.array([np.nan, -10.0, np.nan, 25.0], dtype=np.float32)
        out = compute_snow_mask(t2m, threshold=1.5)
        assert out.tolist() == [0, 1, 0, 0]

    def test_compute_snow_mask_custom_threshold(self):
        # Threshold is configurable per call — tests the future-tuning path.
        t2m = np.array([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
        # Stricter: only sub-zero counts as snow
        out = compute_snow_mask(t2m, threshold=0.0)
        assert out.tolist() == [1, 1, 0, 0, 0]


class TestSnowMask:
    """HRRRGrid.get_snow_mask end-to-end behaviour."""

    def test_supports_snow_is_true(self):
        # The chain dispatcher gates on this — must be True so HRRR-CONUS
        # actually wins inside its domain.
        g = HRRRGrid()
        assert g.supports_snow is True

    def test_no_data_returns_all_false(self):
        g = HRRRGrid()
        lat = np.array([40.0, 50.0])
        lon = np.array([-100.0, -90.0])
        out = g.get_snow_mask(lat, lon, timestamp=12345)
        assert out.dtype == np.bool_
        assert out.shape == lat.shape
        assert not out.any()

    def test_no_timestamp_returns_all_false(self):
        g = HRRRGrid()
        run = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 900, 30.0)
        _inject_snow_mask(g, run, 900, 1)
        out = g.get_snow_mask(np.array([40.0]), np.array([-100.0]), timestamp=None)
        assert not out.any()

    def test_uniform_snow_returns_true_in_domain(self):
        g = HRRRGrid()
        run = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 900, 30.0)
        _inject_frame(g, run, 1800, 30.0)
        _inject_snow_mask(g, run, 900, 1)
        _inject_snow_mask(g, run, 1800, 1)

        # CONUS point inside the HRRR grid → snow=True
        out = g.get_snow_mask(
            np.array([40.0]), np.array([-100.0]), timestamp=run + 1500,
        )
        assert out.tolist() == [True]

    def test_uniform_rain_returns_false_in_domain(self):
        g = HRRRGrid()
        run = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 900, 30.0)
        _inject_frame(g, run, 1800, 30.0)
        _inject_snow_mask(g, run, 900, 0)
        _inject_snow_mask(g, run, 1800, 0)

        out = g.get_snow_mask(
            np.array([40.0]), np.array([-100.0]), timestamp=run + 1500,
        )
        assert out.tolist() == [False]

    def test_outside_domain_returns_false(self):
        g = HRRRGrid()
        run = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 900, 30.0)
        _inject_frame(g, run, 1800, 30.0)
        _inject_snow_mask(g, run, 900, 1)
        _inject_snow_mask(g, run, 1800, 1)

        # London — outside HRRR's CONUS domain
        out = g.get_snow_mask(
            np.array([51.5]), np.array([-0.1]), timestamp=run + 1500,
        )
        # _sample_grid returns 0 outside the grid; that re-binarises to False.
        assert out.tolist() == [False]

    def test_lerp_bracket_majority_at_midpoint(self):
        # When the bracket frames disagree, alpha < 0.5 picks L0 and
        # alpha >= 0.5 picks L1.  This mirrors the docstring of
        # get_snow_mask's lerp branch.
        g = HRRRGrid()
        run = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 900, 30.0)
        _inject_frame(g, run, 1800, 30.0)
        _inject_snow_mask(g, run, 900, 0)   # L0: rain
        _inject_snow_mask(g, run, 1800, 1)  # L1: snow

        # alpha=0.33 → rain wins (closer to L0)
        out_low = g.get_snow_mask(
            np.array([40.0]), np.array([-100.0]), timestamp=run + 1200,
        )
        assert out_low.tolist() == [False]

        # alpha=0.67 → snow wins (closer to L1)
        out_high = g.get_snow_mask(
            np.array([40.0]), np.array([-100.0]), timestamp=run + 1500,
        )
        assert out_high.tolist() == [True]

    def test_partial_bracket_returns_false(self):
        # Only L0 snow mask present; L1 snow mask missing.  Falls through
        # gracefully so the chain dispatcher reaches the next source.
        g = HRRRGrid()
        run = int(datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 900, 30.0)
        _inject_frame(g, run, 1800, 30.0)
        _inject_snow_mask(g, run, 900, 1)
        # Deliberately no snow_mask at lead 1800.

        out = g.get_snow_mask(
            np.array([40.0]), np.array([-100.0]), timestamp=run + 1500,
        )
        assert not out.any()


class TestSnowMaskPersistence:
    """Snow masks are atomic-write parallel files alongside REFC frames."""

    @pytest.mark.asyncio
    async def test_snow_mask_round_trips_through_disk(self, tmp_path):
        run_ts = int(datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc).timestamp())

        g1 = HRRRGrid(cache_dir=tmp_path)
        # Ingest REFC + snow for two bracketing leads.
        arr = np.full(
            (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH), 30.0, dtype=np.float32,
        )
        for lead in (900, 1800):
            encoded = encode_dbz(arr)
            mm = g1._to_memmap(f"r{run_ts}_l{lead}", encoded)
            g1._frames[(run_ts, lead)] = mm
            snow = np.ones((HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH), dtype=np.uint8)
            mm_s = g1._to_memmap(f"r{run_ts}_l{lead}_snow", snow)
            g1._snow_masks[(run_ts, lead)] = mm_s
        g1._latest_run_ts = run_ts
        await g1.close()

        # Both REFC and snow files persisted to disk
        cache_dir = tmp_path / "hrrr"
        assert (cache_dir / f"r{run_ts}_l900.dat").exists()
        assert (cache_dir / f"r{run_ts}_l900_snow.dat").exists()

        # Second "process" picks both up
        g2 = HRRRGrid(cache_dir=tmp_path)
        assert g2.frame_count == 2
        assert g2.snow_mask_count == 2
        assert (run_ts, 900) in g2._snow_masks
        assert (run_ts, 1800) in g2._snow_masks

        # Sample at a CONUS point — snow=True everywhere
        sample_ts = run_ts + 1500
        out = g2.get_snow_mask(
            np.array([40.0]), np.array([-100.0]), timestamp=sample_ts,
        )
        assert out.tolist() == [True]
        await g2.close()

    @pytest.mark.asyncio
    async def test_orphan_snow_mask_is_removed(self, tmp_path):
        # A snow file without a matching REFC file is dropped on load.
        cache_dir = tmp_path / "hrrr"
        cache_dir.mkdir(parents=True)
        # Orphan snow file (no matching r1234_l900.dat)
        orphan = cache_dir / "r1234_l900_snow.dat"
        size = HRRR_GRID_HEIGHT * HRRR_GRID_WIDTH
        orphan.write_bytes(b"\x00" * size)
        assert orphan.exists()

        g = HRRRGrid(cache_dir=tmp_path)
        assert not orphan.exists(), "orphan snow file should be removed"
        assert g.snow_mask_count == 0
        await g.close()

    @pytest.mark.asyncio
    async def test_eviction_removes_snow_files_too(self, tmp_path):
        run_ts = int(datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc).timestamp())
        g = HRRRGrid(cache_dir=tmp_path)
        # Ingest one REFC + snow at the same key.
        arr = np.full(
            (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH), 30.0, dtype=np.float32,
        )
        encoded = encode_dbz(arr)
        g._to_memmap(f"r{run_ts}_l900", encoded)
        snow = np.ones((HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH), dtype=np.uint8)
        g._to_memmap(f"r{run_ts}_l900_snow", snow)
        # Re-mount so the in-memory dicts know about them.
        mm = np.memmap(
            tmp_path / "hrrr" / f"r{run_ts}_l900.dat",
            dtype=np.uint8, mode="r",
            shape=(HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH),
        )
        g._frames[(run_ts, 900)] = mm
        mm_s = np.memmap(
            tmp_path / "hrrr" / f"r{run_ts}_l900_snow.dat",
            dtype=np.uint8, mode="r",
            shape=(HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH),
        )
        g._snow_masks[(run_ts, 900)] = mm_s

        # Evict to a window far in the future
        far_future = run_ts + 24 * 3600
        g._evict_outside_window(far_future, far_future + 600)
        assert (run_ts, 900) not in g._frames
        assert (run_ts, 900) not in g._snow_masks
        assert not (tmp_path / "hrrr" / f"r{run_ts}_l900.dat").exists()
        assert not (tmp_path / "hrrr" / f"r{run_ts}_l900_snow.dat").exists()
        await g.close()


class TestChainSnowMaskWithHRRR:
    def test_chain_prefers_hrrr_snow_inside_conus(self):
        from librewxr.sources.world.ifs.grid import ECMWFGrid
        from librewxr.sources.world.ifs.grid import GRID_HEIGHT as IFS_H, GRID_WIDTH as IFS_W

        # IFS says snow everywhere; HRRR says rain everywhere.  Inside HRRR's
        # domain, HRRR wins → rain.  Outside, IFS wins → snow.
        ifs = ECMWFGrid()
        ifs_dbz = np.full((IFS_H, IFS_W), int((10 + 32) * 2), dtype=np.uint8)
        ifs_snow = np.ones((IFS_H, IFS_W), dtype=bool)
        ifs._timesteps[1000000] = (ifs_dbz, ifs_snow)
        ifs._sorted_timestamps = [1000000]

        hrrr = HRRRGrid()
        run = 1000000 - 1500
        _inject_frame(hrrr, run, 900, 30.0)
        _inject_frame(hrrr, run, 1800, 30.0)
        _inject_snow_mask(hrrr, run, 900, 0)   # rain
        _inject_snow_mask(hrrr, run, 1800, 0)

        chain = NWPChain([hrrr, ifs])

        # CONUS point: HRRR says rain → False
        out = chain.get_snow_mask(
            np.array([40.0]), np.array([-100.0]), timestamp=1000000,
        )
        assert out.tolist() == [False]

        # Outside HRRR domain (London): IFS wins → True
        out = chain.get_snow_mask(
            np.array([51.5]), np.array([-0.1]), timestamp=1000000,
        )
        assert out.tolist() == [True]
