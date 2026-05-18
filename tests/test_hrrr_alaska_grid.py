# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Unit tests for HRRR-Alaska polar-stereo projection, idx step parsing,
hourly bracket lerp, decode orientation, and HRRRAlaskaGrid sampling.

Pure unit tests — no S3 network calls.  Live fetch against the
``noaa-hrrr-bdp-pds`` bucket is exercised manually during development.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytestmark = pytest.mark.hrrr_alaska

from librewxr.sources.regional.north_america.usa.nwp.hrrr_alaska.grid import (
    BRACKET_INTERVAL_SECONDS,
    CYCLE_INTERVAL_SECONDS,
    HRRR_AK_FEATHER_DISTANCE_M,
    HRRR_AK_GRID_DX,
    HRRR_AK_GRID_HEIGHT,
    HRRR_AK_GRID_WIDTH,
    MAX_FORECAST_HOURS,
    HRRRAlaskaGrid,
    bracket_lead_seconds,
    decode_refc_message,
    domain_mask,
    encode_dbz,
    feather_mask,
    floor_cycle,
    grid_indices,
    latest_published_run,
    lead_seconds_for_step,
    ps_forward,
    wrfsfcf_url,
)
from librewxr.sources.regional.north_america.usa.nwp.hrrr.grid import compute_snow_mask, find_tmp_2m_records, parse_idx
from librewxr.data.nwp_source import NWPChain, NWPSource


# ── Polar stereographic projection ──────────────────────────────────


class TestPolarStereoProjection:
    """Verify polar-stereographic forward maths against GRIB-reported corners.

    These corner values come from cfgrib decoding a real
    ``hrrr.t00z.wrfsfcf00.ak.grib2`` REFC message; the projected (x, y)
    must match the (col=0, row=0)..(col=NX-1, row=NY-1) layout the rest
    of the module assumes.
    """

    # First grid point in the GRIB stream is the SW corner (j=0, i=0).
    SW_CORNER_LAT = 41.612949
    SW_CORNER_LON = 185.117126
    SE_CORNER_LAT = 51.7256
    SE_CORNER_LON = 231.5269
    NW_CORNER_LAT = 55.6054
    NW_CORNER_LON = 156.4368
    NE_CORNER_LAT = 76.3377
    NE_CORNER_LON = 244.2243

    def test_sw_corner_projects_to_grid_origin(self):
        # SW corner = (col=0, row_unflipped=0).  After flipud, that lands
        # at row=NY-1 in the stored grid.
        x, y = ps_forward(
            np.array([self.SW_CORNER_LAT]),
            np.array([self.SW_CORNER_LON]),
        )
        # x at col=0 == GRID_X_ORIGIN; y at the south edge sits
        # (NY-1)*DY metres below GRID_Y_ORIGIN.
        from librewxr.sources.regional.north_america.usa.nwp.hrrr_alaska.grid import (
            HRRR_AK_GRID_X_ORIGIN, HRRR_AK_GRID_Y_ORIGIN, HRRR_AK_GRID_DY,
        )
        assert abs(x[0] - HRRR_AK_GRID_X_ORIGIN) < 1.0
        expected_y = HRRR_AK_GRID_Y_ORIGIN - (HRRR_AK_GRID_HEIGHT - 1) * HRRR_AK_GRID_DY
        assert abs(y[0] - expected_y) < 1.0

    def test_corners_land_at_expected_grid_indices(self):
        """All four GRIB-reported corners map to within 0.5 cells of corners."""
        # In the stored (flipped) grid:
        #   SW corner of grid → col=0, row=NY-1
        #   SE corner of grid → col=NX-1, row=NY-1
        #   NW corner of grid → col=0, row=0
        #   NE corner of grid → col=NX-1, row=0
        cases = [
            ("SW", self.SW_CORNER_LAT, self.SW_CORNER_LON, 0, HRRR_AK_GRID_HEIGHT - 1),
            ("SE", self.SE_CORNER_LAT, self.SE_CORNER_LON, HRRR_AK_GRID_WIDTH - 1, HRRR_AK_GRID_HEIGHT - 1),
            ("NW", self.NW_CORNER_LAT, self.NW_CORNER_LON, 0, 0),
            ("NE", self.NE_CORNER_LAT, self.NE_CORNER_LON, HRRR_AK_GRID_WIDTH - 1, 0),
        ]
        for name, lat, lon, ec, er in cases:
            row, col = grid_indices(np.array([lat]), np.array([lon]))
            assert abs(col[0] - ec) < 0.5, f"{name} col: got {col[0]}, expected {ec}"
            assert abs(row[0] - er) < 0.5, f"{name} row: got {row[0]}, expected {er}"

    @pytest.mark.parametrize(
        "name,lat,lon,inside",
        [
            ("Anchorage", 61.2181, -149.9003, True),
            ("Fairbanks", 64.8378, -147.7164, True),
            ("Juneau",    58.3019, -134.4197, True),
            ("Adak",      51.8800, -176.6580, True),   # Aleutians
            ("Barrow",    71.2906, -156.7886, True),   # northernmost
            ("Nome",      64.5011, -165.4064, True),
            # Out of domain:
            ("Seattle",   47.6062, -122.3321, False),  # too far south
            ("Tokyo",     35.6762,  139.6503, False),
            ("Honolulu",  21.3099, -157.8581, False),
        ],
    )
    def test_domain_mask_known_points(self, name, lat, lon, inside):
        m = domain_mask(np.array([lat]), np.array([lon]))
        assert bool(m[0]) is inside, f"{name}: expected inside={inside}, got {bool(m[0])}"

    def test_grid_indices_vectorize(self):
        lats = np.array([61.2, 64.8, 58.3])
        lons = np.array([-149.9, -147.7, -134.4])
        row, col = grid_indices(lats, lons)
        assert row.shape == lats.shape
        assert col.shape == lats.shape
        # All three Alaska cities are deep inside the polar-stereo grid
        assert ((row > 0) & (row < HRRR_AK_GRID_HEIGHT - 1)).all()
        assert ((col > 0) & (col < HRRR_AK_GRID_WIDTH - 1)).all()

    def test_lon_wrap_normalisation(self):
        """Both lon=185 and lon=-175 (same physical longitude) must agree."""
        # The grid spans from ~156°E across the dateline to ~244°E (=-116°).
        # Forward projection uses sin/cos so adding 360° must be a no-op.
        x1, y1 = ps_forward(np.array([60.0]), np.array([185.0]))
        x2, y2 = ps_forward(np.array([60.0]), np.array([185.0 - 360.0]))
        assert abs(x1[0] - x2[0]) < 1e-6
        assert abs(y1[0] - y2[0]) < 1e-6


# ── Boundary feather ──────────────────────────────────────────────────


class TestFeatherMask:
    def test_outside_domain_is_zero(self):
        f = feather_mask(np.array([47.0]), np.array([-122.0]))  # Seattle
        assert f[0] == 0.0

    def test_deep_inside_is_full_weight(self):
        # Pick interior Alaska points clearly more than 75 km from any edge.
        cities = [
            (61.2181, -149.9003),  # Anchorage
            (64.8378, -147.7164),  # Fairbanks
        ]
        lats = np.array([c[0] for c in cities])
        lons = np.array([c[1] for c in cities])
        f = feather_mask(lats, lons)
        assert f.dtype == np.float32
        assert (f == 1.0).all(), f"expected full weight, got {f}"

    def test_taper_is_monotonic_at_southern_edge(self):
        # Walk lat from inside (Anchorage at 61°N) toward the southern
        # domain edge.  feather_mask must be non-increasing.
        lats = np.linspace(56.0, 53.0, 30)
        lons = np.full_like(lats, -149.9)
        f = feather_mask(lats, lons)
        diffs = np.diff(f)
        assert (diffs <= 1e-6).all(), \
            "feather must be non-increasing as we leave domain"

    def test_taper_width_matches_constant(self):
        # Cell-distance check: the feather width is HRRR_AK_FEATHER_DISTANCE_M / DX
        # cells.  Picking a point exactly at the centre of the grid
        # guarantees feather=1.
        # Centre lat/lon roughly (61, 196.5°E) per the GRIB centre.
        f = feather_mask(np.array([60.8]), np.array([196.5]))
        assert f[0] == pytest.approx(1.0)
        feather_cells = int(HRRR_AK_FEATHER_DISTANCE_M // HRRR_AK_GRID_DX)
        assert feather_cells == 25  # 75km / 3km


# ── Hourly bracket timing ────────────────────────────────────────────


class TestHourlyBracket:
    def test_floor_cycle_3hourly(self):
        # 14:35 UTC → 12:00 UTC (most recent 3-hourly cycle)
        ts = int(datetime(2026, 5, 1, 14, 35, 0, tzinfo=timezone.utc).timestamp())
        cycle = floor_cycle(ts)
        expected = int(datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        assert cycle == expected

    def test_floor_cycle_at_exact_boundary(self):
        # 09:00 UTC → 09:00 UTC (already at a cycle boundary)
        ts = int(datetime(2026, 5, 1, 9, 0, 0, tzinfo=timezone.utc).timestamp())
        assert floor_cycle(ts) == ts

    def test_latest_published_run_80min_delay(self):
        # 14:00 UTC, 80 min delay: now - 80min = 12:40 → floor to 12:00
        now = int(datetime(2026, 5, 1, 14, 0, 0, tzinfo=timezone.utc).timestamp())
        run = latest_published_run(now, 80 * 60)
        expected = int(datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        assert run == expected

    def test_latest_published_run_just_after_cycle(self):
        # 12:30 UTC, 80 min delay: now - 80min = 11:10 → floor to 09:00
        # (not 12:00 since 12:00 isn't published yet)
        now = int(datetime(2026, 5, 1, 12, 30, 0, tzinfo=timezone.utc).timestamp())
        run = latest_published_run(now, 80 * 60)
        expected = int(datetime(2026, 5, 1, 9, 0, 0, tzinfo=timezone.utc).timestamp())
        assert run == expected

    @pytest.mark.parametrize(
        "lead_min,expected_l0_h,expected_l1_h,expected_alpha",
        [
            (0,    0, 1, 0.0),     # exact analysis frame
            (15,   0, 1, 0.25),
            (30,   0, 1, 0.5),
            (45,   0, 1, 0.75),
            (60,   1, 2, 0.0),     # exact +1h frame
            (90,   1, 2, 0.5),
            (120,  2, 3, 0.0),     # exact +2h frame
            (180,  3, 4, 0.0),
        ],
    )
    def test_bracket_lead_seconds(
        self, lead_min, expected_l0_h, expected_l1_h, expected_alpha
    ):
        l0, l1, alpha = bracket_lead_seconds(lead_min * 60)
        assert l0 == expected_l0_h * 3600
        assert l1 == expected_l1_h * 3600
        assert abs(alpha - expected_alpha) < 1e-9

    def test_bracket_negative_lead_clamps(self):
        # Lead < 0: caller is expected to roll back to a previous run.
        l0, l1, alpha = bracket_lead_seconds(-1)
        assert l0 == 0 and l1 == 0 and alpha == 0.0

    def test_intervals_are_3h_cycles_and_1h_steps(self):
        assert CYCLE_INTERVAL_SECONDS == 3 * 3600
        assert BRACKET_INTERVAL_SECONDS == 3600
        assert MAX_FORECAST_HOURS == 48


# ── Step-label parsing (hourly, not subh) ────────────────────────────


class TestStepLabelParsing:
    @pytest.mark.parametrize(
        "label,expected",
        [
            ("anl", 0),               # analysis step → lead 0
            ("1 hour fcst", 3600),
            ("6 hour fcst", 6 * 3600),
            ("12 hour fcst", 12 * 3600),
            ("48 hour fcst", 48 * 3600),
        ],
    )
    def test_hour_fcst_labels(self, label, expected):
        assert lead_seconds_for_step(label) == expected

    def test_unknown_step_label_returns_none(self):
        # Average / accumulation labels aren't REFC entries, but parser
        # must reject them rather than coerce to 0.
        assert lead_seconds_for_step("0-1 hour ave fcst") is None
        assert lead_seconds_for_step("15 min fcst") is None  # subh — wrong cadence
        assert lead_seconds_for_step("") is None
        assert lead_seconds_for_step("garbage") is None


# ── URL builder ──────────────────────────────────────────────────────


class TestUrlBuilder:
    def test_url_has_alaska_prefix_and_ak_infix(self):
        run = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
        url = wrfsfcf_url(run, 6, bucket="noaa-hrrr-bdp-pds")
        assert "/hrrr.20260501/alaska/" in url
        assert "hrrr.t12z.wrfsfcf06.ak.grib2" in url
        assert ".idx" not in url  # caller appends .idx

    def test_url_pads_lead_hour(self):
        run = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
        url0 = wrfsfcf_url(run, 0, bucket="b")
        url9 = wrfsfcf_url(run, 9, bucket="b")
        url48 = wrfsfcf_url(run, 48, bucket="b")
        assert "wrfsfcf00.ak.grib2" in url0
        assert "wrfsfcf09.ak.grib2" in url9
        assert "wrfsfcf48.ak.grib2" in url48


# ── Encoding ──────────────────────────────────────────────────────────


class TestEncoding:
    def test_encode_known_dbz_values(self):
        refc = np.array([-32.0, 0.0, 50.0, 95.0])
        encoded = encode_dbz(refc)
        assert encoded.dtype == np.uint8
        assert encoded.tolist() == [0, 64, 164, 254]

    def test_encode_handles_nan(self):
        refc = np.array([np.nan, 30.0, np.nan])
        encoded = encode_dbz(refc)
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

    cfgrib returns HRRR-Alaska REFC with row 0 at the SOUTH edge (the
    ``scanningMode=64`` first grid point at lat ~41.6°N) and row 918
    at the NORTH edge.  ``grid_indices()`` assumes the standard image
    orientation — row 0 = north — so without the flip every sample
    reads from a cell mirrored about the LoV meridian.
    """

    def test_decode_flips_south_up_grib(self, monkeypatch):
        from contextlib import contextmanager
        from librewxr.sources.regional.north_america.usa.nwp.hrrr_alaska import grid as hagm

        refc_data = np.zeros(
            (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH), dtype=np.float32
        )
        refc_data[0, 100] = 65.0   # marker at SOUTH edge (cfgrib row 0)
        refc_data[-1, 200] = 45.0  # marker at NORTH edge (cfgrib row -1)

        # Synthetic south-up coords: lat increases with row → row 0 = south
        lat_data = np.broadcast_to(
            np.linspace(41.6, 76.3, HRRR_AK_GRID_HEIGHT)[:, None],
            (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH),
        )
        lon_data = np.broadcast_to(
            np.linspace(185.0, 244.0, HRRR_AK_GRID_WIDTH)[None, :],
            (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH),
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
        # Patch the lazy-imported _suppress_eccodes_stderr inside the module
        import librewxr.data.sources as _sources_mod
        monkeypatch.setattr(_sources_mod, "_suppress_eccodes_stderr", _noop)

        arr = hagm.decode_refc_message(b"ignored bytes")
        assert arr is not None
        assert arr.shape == (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH)
        # After flip: cfgrib's row 0 (south) → our row -1, cfgrib's
        # row -1 (north) → our row 0.
        assert arr[-1, 100] == 65.0, "south-edge marker should land at our last row"
        assert arr[0, 200] == 45.0, "north-edge marker should land at our row 0"

    def test_decode_does_not_double_flip_north_up_grib(self, monkeypatch):
        """If cfgrib ever returns north-up natively, don't re-flip."""
        from contextlib import contextmanager
        from librewxr.sources.regional.north_america.usa.nwp.hrrr_alaska import grid as hagm

        refc_data = np.zeros(
            (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH), dtype=np.float32
        )
        refc_data[0, 100] = 65.0    # at NORTH edge in already-correct frame
        refc_data[-1, 200] = 45.0

        # lat decreases with row → already north-up
        lat_data = np.broadcast_to(
            np.linspace(76.3, 41.6, HRRR_AK_GRID_HEIGHT)[:, None],
            (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH),
        )
        lon_data = np.broadcast_to(
            np.linspace(185.0, 244.0, HRRR_AK_GRID_WIDTH)[None, :],
            (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH),
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
        import librewxr.data.sources as _sources_mod
        monkeypatch.setattr(_sources_mod, "_suppress_eccodes_stderr", _noop)

        arr = hagm.decode_refc_message(b"ignored bytes")
        # Already north-up; markers should stay where they are
        assert arr[0, 100] == 65.0
        assert arr[-1, 200] == 45.0


# ── HRRRAlaskaGrid sample / Protocol ─────────────────────────────────


def _inject_frame(
    grid: HRRRAlaskaGrid, run_ts: int, lead_seconds: int, refc_dbz: float
) -> None:
    """Inject a uniform-value frame into HRRRAlaskaGrid for testing."""
    arr = np.full(
        (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH), refc_dbz, dtype=np.float32
    )
    encoded = encode_dbz(arr)
    grid._frames[(run_ts, lead_seconds)] = encoded
    if grid._latest_run_ts is None or run_ts > grid._latest_run_ts:
        grid._latest_run_ts = run_ts


class TestHRRRAlaskaGridProtocol:
    def test_satisfies_protocol(self):
        g = HRRRAlaskaGrid()
        assert isinstance(g, NWPSource)
        assert g.name == "hrrr_alaska"

    def test_empty_grid_sample_returns_zeros(self):
        g = HRRRAlaskaGrid()
        lat = np.array([61.2])
        lon = np.array([-149.9])
        out = g.sample(lat, lon, timestamp=12345)
        assert out.shape == lat.shape
        assert (out == 0).all()

    def test_has_data_with_injected_frames(self):
        g = HRRRAlaskaGrid()
        run = int(
            datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        )
        # Inject the bracket for a query at run + 90 min (between L0=60min and L1=120min)
        _inject_frame(g, run, 3600, 30.0)
        _inject_frame(g, run, 7200, 30.0)
        target = run + 90 * 60
        assert g.has_data_at(target) is True
        # A query outside the loaded bracket horizon must report False
        assert g.has_data_at(run + 49 * 3600) is False

    def test_sample_at_anchorage_returns_injected_value(self):
        g = HRRRAlaskaGrid()
        run = int(
            datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        )
        _inject_frame(g, run, 0, 40.0)
        _inject_frame(g, run, 3600, 40.0)
        # 30 min into the run → bracket L0=0, L1=3600, alpha=0.5
        out = g.sample(
            np.array([61.2181]),
            np.array([-149.9003]),
            timestamp=run + 30 * 60,
        )
        # All injected frames are uniform 40 dBZ → encoded (40+32)*2 = 144
        assert int(out[0]) == 144

    def test_sample_outside_domain_returns_zero(self):
        g = HRRRAlaskaGrid()
        run = int(
            datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        )
        _inject_frame(g, run, 0, 40.0)
        _inject_frame(g, run, 3600, 40.0)
        # Tokyo: clearly outside HRRR-AK
        out = g.sample(
            np.array([35.6762]),
            np.array([139.6503]),
            timestamp=run + 30 * 60,
        )
        assert int(out[0]) == 0

    def test_pick_run_falls_back_when_bracket_incomplete(self):
        """During cycle rollover, an incompletely-loaded new run must
        defer to an older run whose bracket is complete."""
        g = HRRRAlaskaGrid()
        new_run = int(
            datetime(2026, 5, 1, 15, 0, 0, tzinfo=timezone.utc).timestamp()
        )
        old_run = new_run - 3 * 3600
        # Old run: complete bracket for query at new_run + 30 min
        # (= old_run + 3.5h, which sits between l0=3h and l1=4h)
        _inject_frame(g, old_run, 3 * 3600, 25.0)
        _inject_frame(g, old_run, 4 * 3600, 25.0)
        # New run: partial — only L0 loaded, no L1 yet for the query lead
        _inject_frame(g, new_run, 0, 25.0)  # missing the +1h frame

        target = new_run + 30 * 60
        # has_data_at must be True via the old run, despite the new run
        # being newer but incomplete.
        assert g.has_data_at(target) is True
        chosen = g._pick_run(target)
        assert chosen == old_run, "should fall back when newest bracket incomplete"


# ── Chain integration ────────────────────────────────────────────────


class TestChainIntegration:
    """Verify HRRR-Alaska soft-blends with IFS at its domain boundary."""

    def test_chain_uses_hrrr_alaska_inside_domain(self):
        from librewxr.sources.world.ifs.grid import (
            ECMWFGrid,
            GRID_HEIGHT as IFS_H,
            GRID_WIDTH as IFS_W,
        )

        # IFS = 0 dBZ (encoded 64); HRRR-AK = 50 dBZ (encoded 164)
        ifs = ECMWFGrid()
        ifs_dbz = np.full(
            (IFS_H, IFS_W), int((0 + 32) * 2), dtype=np.uint8
        )
        ifs._timesteps[1000000] = (
            ifs_dbz, np.zeros_like(ifs_dbz, dtype=bool),
        )
        ifs._sorted_timestamps = [1000000]

        ak = HRRRAlaskaGrid()
        run = 1000000 - 1800  # 30 min into the run
        _inject_frame(ak, run, 0, 50.0)
        _inject_frame(ak, run, 3600, 50.0)

        chain = NWPChain([ak, ifs])

        # Deep inside Alaska (Anchorage) → all HRRR-AK → encoded 164
        anc = chain.sample(
            np.array([61.2181]),
            np.array([-149.9003]),
            timestamp=1000000,
        )
        assert abs(int(anc[0]) - 164) <= 1

        # Outside HRRR-AK (Seattle) → all IFS → encoded 64
        sea = chain.sample(
            np.array([47.6062]),
            np.array([-122.3321]),
            timestamp=1000000,
        )
        assert int(sea[0]) == 64


# ── Snow mask ─────────────────────────────────────────────────────────


def _inject_snow_mask(
    grid: HRRRAlaskaGrid, run_ts: int, lead_seconds: int, snow_value: int
) -> None:
    """Inject a uniform-value snow mask into HRRRAlaskaGrid for testing."""
    arr = np.full(
        (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH),
        snow_value & 0x01,
        dtype=np.uint8,
    )
    grid._snow_masks[(run_ts, lead_seconds)] = arr


class TestSnowMaskIdxFiltering:
    """The shared ``find_tmp_2m_records`` works on HRRR-Alaska's wrfsfcf idx too."""

    SAMPLE_IDX = (
        "1:0:d=2026050112:REFC:entire atmosphere:0 min fcst:\n"
        "2:200000:d=2026050112:TMP:2 m above ground:0 min fcst:\n"
        "3:300000:d=2026050112:TMP:surface:0 min fcst:\n"
        "4:400000:d=2026050112:UGRD:10 m above ground:0 min fcst:\n"
        "5:500000:d=2026050112:TMP:2 m above ground:60 min fcst:\n"
    )

    def test_find_tmp_2m_records_filters_level(self):
        records = parse_idx(self.SAMPLE_IDX)
        tmps = find_tmp_2m_records(records)
        # Two TMP-at-2m records; surface TMP and UGRD excluded.
        assert len(tmps) == 2
        first_rec, first_end = tmps[0]
        assert first_rec.var == "TMP"
        assert first_rec.level == "2 m above ground"
        assert first_rec.step == "0 min fcst"
        # First TMP record byte range is bounded by the next record
        # (surface TMP at 300000), not the next 2-m TMP record.
        assert first_end == 299999
        # Last 2-m TMP has no further record → end = -1
        last_rec, last_end = tmps[-1]
        assert last_rec.step == "60 min fcst"
        assert last_end == -1


class TestSnowMask:
    """HRRRAlaskaGrid.get_snow_mask end-to-end behaviour."""

    def test_supports_snow_is_true(self):
        # The chain dispatcher gates on this — must be True so HRRR-Alaska
        # actually wins inside its domain.
        g = HRRRAlaskaGrid()
        assert g.supports_snow is True

    def test_no_data_returns_all_false(self):
        g = HRRRAlaskaGrid()
        lat = np.array([61.2, 64.8])
        lon = np.array([-149.9, -147.7])
        out = g.get_snow_mask(lat, lon, timestamp=12345)
        assert out.dtype == np.bool_
        assert out.shape == lat.shape
        assert not out.any()

    def test_no_timestamp_returns_all_false(self):
        g = HRRRAlaskaGrid()
        run = int(datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 0, 30.0)
        _inject_snow_mask(g, run, 0, 1)
        out = g.get_snow_mask(
            np.array([61.2]), np.array([-149.9]), timestamp=None,
        )
        assert not out.any()

    def test_uniform_snow_returns_true_in_domain(self):
        g = HRRRAlaskaGrid()
        run = int(datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 0, 30.0)
        _inject_frame(g, run, 3600, 30.0)
        _inject_snow_mask(g, run, 0, 1)
        _inject_snow_mask(g, run, 3600, 1)

        # Anchorage — inside HRRR-Alaska's domain → snow=True
        out = g.get_snow_mask(
            np.array([61.2181]), np.array([-149.9003]),
            timestamp=run + 30 * 60,
        )
        assert out.tolist() == [True]

    def test_uniform_rain_returns_false_in_domain(self):
        g = HRRRAlaskaGrid()
        run = int(datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 0, 30.0)
        _inject_frame(g, run, 3600, 30.0)
        _inject_snow_mask(g, run, 0, 0)
        _inject_snow_mask(g, run, 3600, 0)

        out = g.get_snow_mask(
            np.array([61.2181]), np.array([-149.9003]),
            timestamp=run + 30 * 60,
        )
        assert out.tolist() == [False]

    def test_outside_domain_returns_false(self):
        g = HRRRAlaskaGrid()
        run = int(datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 0, 30.0)
        _inject_frame(g, run, 3600, 30.0)
        _inject_snow_mask(g, run, 0, 1)
        _inject_snow_mask(g, run, 3600, 1)

        # Tokyo — outside HRRR-Alaska's domain
        out = g.get_snow_mask(
            np.array([35.6762]), np.array([139.6503]),
            timestamp=run + 30 * 60,
        )
        assert out.tolist() == [False]

    def test_lerp_bracket_majority_at_midpoint(self):
        # alpha < 0.5 picks L0; alpha >= 0.5 picks L1 (re-binarise at 0.5).
        g = HRRRAlaskaGrid()
        run = int(datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 0, 30.0)
        _inject_frame(g, run, 3600, 30.0)
        _inject_snow_mask(g, run, 0, 0)      # L0: rain
        _inject_snow_mask(g, run, 3600, 1)   # L1: snow

        # alpha=0.25 → L0 wins (closer)
        out_low = g.get_snow_mask(
            np.array([61.2181]), np.array([-149.9003]),
            timestamp=run + 15 * 60,
        )
        assert out_low.tolist() == [False]

        # alpha=0.75 → L1 wins (closer)
        out_high = g.get_snow_mask(
            np.array([61.2181]), np.array([-149.9003]),
            timestamp=run + 45 * 60,
        )
        assert out_high.tolist() == [True]

    def test_partial_bracket_returns_false(self):
        # Only L0 snow mask present; L1 snow mask missing — falls through
        # gracefully so the chain dispatcher reaches the next source.
        g = HRRRAlaskaGrid()
        run = int(datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        _inject_frame(g, run, 0, 30.0)
        _inject_frame(g, run, 3600, 30.0)
        _inject_snow_mask(g, run, 0, 1)
        # Deliberately no snow_mask at lead 3600.

        out = g.get_snow_mask(
            np.array([61.2181]), np.array([-149.9003]),
            timestamp=run + 30 * 60,
        )
        assert not out.any()


class TestSnowMaskPersistence:
    """Snow masks are atomic-write parallel files alongside REFC frames."""

    @pytest.mark.asyncio
    async def test_snow_mask_round_trips_through_disk(self, tmp_path):
        run_ts = int(datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc).timestamp())

        g1 = HRRRAlaskaGrid(cache_dir=tmp_path)
        arr = np.full(
            (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH), 30.0, dtype=np.float32,
        )
        for lead in (0, 3600):
            encoded = encode_dbz(arr)
            mm = g1._to_memmap(f"r{run_ts}_l{lead}", encoded)
            g1._frames[(run_ts, lead)] = mm
            snow = np.ones(
                (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH), dtype=np.uint8,
            )
            mm_s = g1._to_memmap(f"r{run_ts}_l{lead}_snow", snow)
            g1._snow_masks[(run_ts, lead)] = mm_s
        g1._latest_run_ts = run_ts
        await g1.close()

        cache_dir = tmp_path / "hrrr_alaska"
        assert (cache_dir / f"r{run_ts}_l0.dat").exists()
        assert (cache_dir / f"r{run_ts}_l0_snow.dat").exists()

        g2 = HRRRAlaskaGrid(cache_dir=tmp_path)
        assert g2.frame_count == 2
        assert g2.snow_mask_count == 2
        assert (run_ts, 0) in g2._snow_masks
        assert (run_ts, 3600) in g2._snow_masks

        sample_ts = run_ts + 30 * 60
        out = g2.get_snow_mask(
            np.array([61.2181]), np.array([-149.9003]),
            timestamp=sample_ts,
        )
        assert out.tolist() == [True]
        await g2.close()

    @pytest.mark.asyncio
    async def test_orphan_snow_mask_is_removed(self, tmp_path):
        # A snow file without a matching REFC file is dropped on load.
        cache_dir = tmp_path / "hrrr_alaska"
        cache_dir.mkdir(parents=True)
        orphan = cache_dir / "r1234_l0_snow.dat"
        size = HRRR_AK_GRID_HEIGHT * HRRR_AK_GRID_WIDTH
        orphan.write_bytes(b"\x00" * size)
        assert orphan.exists()

        g = HRRRAlaskaGrid(cache_dir=tmp_path)
        assert not orphan.exists(), "orphan snow file should be removed"
        assert g.snow_mask_count == 0
        await g.close()

    @pytest.mark.asyncio
    async def test_eviction_removes_snow_files_too(self, tmp_path):
        run_ts = int(datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc).timestamp())
        g = HRRRAlaskaGrid(cache_dir=tmp_path)
        arr = np.full(
            (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH), 30.0, dtype=np.float32,
        )
        encoded = encode_dbz(arr)
        g._to_memmap(f"r{run_ts}_l0", encoded)
        snow = np.ones(
            (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH), dtype=np.uint8,
        )
        g._to_memmap(f"r{run_ts}_l0_snow", snow)
        # Re-mount so the in-memory dicts know about them.
        mm = np.memmap(
            tmp_path / "hrrr_alaska" / f"r{run_ts}_l0.dat",
            dtype=np.uint8, mode="r",
            shape=(HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH),
        )
        g._frames[(run_ts, 0)] = mm
        mm_s = np.memmap(
            tmp_path / "hrrr_alaska" / f"r{run_ts}_l0_snow.dat",
            dtype=np.uint8, mode="r",
            shape=(HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH),
        )
        g._snow_masks[(run_ts, 0)] = mm_s

        # Evict to a window far in the future
        far_future = run_ts + 7 * 24 * 3600
        g._evict_outside_window(far_future, far_future + 600)
        assert (run_ts, 0) not in g._frames
        assert (run_ts, 0) not in g._snow_masks
        assert not (tmp_path / "hrrr_alaska" / f"r{run_ts}_l0.dat").exists()
        assert not (tmp_path / "hrrr_alaska" / f"r{run_ts}_l0_snow.dat").exists()
        await g.close()


class TestChainSnowMaskWithHRRRAlaska:
    def test_chain_prefers_hrrr_alaska_snow_inside_domain(self):
        from librewxr.sources.world.ifs.grid import ECMWFGrid
        from librewxr.sources.world.ifs.grid import GRID_HEIGHT as IFS_H, GRID_WIDTH as IFS_W

        # IFS says snow everywhere; HRRR-Alaska says rain everywhere.
        # Inside HRRR-AK's domain, HRRR-AK wins → rain.  Outside, IFS
        # wins → snow.
        ifs = ECMWFGrid()
        ifs_dbz = np.full((IFS_H, IFS_W), int((10 + 32) * 2), dtype=np.uint8)
        ifs_snow = np.ones((IFS_H, IFS_W), dtype=bool)
        ifs._timesteps[1000000] = (ifs_dbz, ifs_snow)
        ifs._sorted_timestamps = [1000000]

        ak = HRRRAlaskaGrid()
        run = 1000000 - 1800  # target lead = 30 min in (0, 3600) bracket
        _inject_frame(ak, run, 0, 30.0)
        _inject_frame(ak, run, 3600, 30.0)
        _inject_snow_mask(ak, run, 0, 0)      # rain
        _inject_snow_mask(ak, run, 3600, 0)

        chain = NWPChain([ak, ifs])

        # Anchorage: HRRR-AK says rain → False
        out = chain.get_snow_mask(
            np.array([61.2181]), np.array([-149.9003]), timestamp=1000000,
        )
        assert out.tolist() == [False]

        # Outside HRRR-AK domain (London): IFS wins → True
        out = chain.get_snow_mask(
            np.array([51.5]), np.array([-0.1]), timestamp=1000000,
        )
        assert out.tolist() == [True]
