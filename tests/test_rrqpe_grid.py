# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the NOAA Enterprise Rain Rate (RRQPE) GLB-5 blend radar source.

Synthetic data only — no network.  The fetch tests drive the real
``_fetch_sync`` path through ``httpx.MockTransport`` with an S3 listing
XML body and tiny in-memory NetCDF4 files.  The source-level fetch tests
stub the scan-store refresh so no real S3 traffic is ever attempted.
"""
from __future__ import annotations

import os
import tempfile
import time
import warnings
from datetime import datetime, timezone

import httpx
import numpy as np
import pytest

pytestmark = pytest.mark.rrqpe

from librewxr.config import settings
from librewxr.data.regions import REGIONS, RegionDef
from librewxr.sources.world.rrqpe.grid import (
    GLB5_KEY_RE,
    NATIVE_COLS,
    NATIVE_PIXEL,
    NATIVE_ROWS,
    RRQPEGrid,
    block_nanmean_downsample,
    downsampled_shape,
    effective_grid,
    hour_prefix,
    parse_s3_listing_keys,
    precip_rate_to_dbz_encoded,
    scan_ts_from_key,
)
from librewxr.sources.world.rrqpe.source import RRQPESource


def _inject_frame(grid: RRQPEGrid, ts: int, value: int):
    """Inject a uniform-value frame into the RRQPE scan store."""
    arr = np.full(grid.effective_shape, value, dtype=np.uint8)
    grid._timesteps[ts] = arr
    grid._sorted_timestamps = sorted(grid._timesteps)


# ── Z-R encoding ───────────────────────────────────────────────────────


class TestZREncoding:
    def test_zero_nan_negative_encode_zero(self):
        encoded = precip_rate_to_dbz_encoded(
            np.array([0.0, np.nan, -1.0, 5.0], dtype=np.float32),
            dbz_offset=6.0,
        )
        assert encoded[0] == 0
        assert encoded[1] == 0
        assert encoded[2] == 0
        assert encoded[3] > 0

    def test_encoded_monotonic_with_rate(self):
        encoded = precip_rate_to_dbz_encoded(
            np.array([0.1, 1.0, 10.0, 50.0], dtype=np.float32),
            dbz_offset=6.0,
        )
        assert encoded.dtype == np.uint8
        assert list(encoded) == sorted(encoded)
        assert encoded[0] < encoded[-1]

    def test_dbz_offset_shifts_uniformly(self):
        rates = np.array([1.0, 5.0, 25.0], dtype=np.float32)
        base = precip_rate_to_dbz_encoded(rates, dbz_offset=0.0)
        shifted = precip_rate_to_dbz_encoded(rates, dbz_offset=6.0)
        for b, s in zip(base, shifted):
            if b > 0:
                assert int(s) - int(b) == 12

    def test_trace_rate_zero_heavy_rain_high(self):
        encoded = precip_rate_to_dbz_encoded(
            np.array([0.005, 50.0], dtype=np.float32), dbz_offset=6.0,
        )
        assert encoded[0] == 0
        assert int(encoded[1]) >= 150  # 50 mm/h → ~56 dBZ → pixel ~176


# ── Grid geometry ──────────────────────────────────────────────────────


class TestGridGeometry:
    def test_effective_params_default_f2(self):
        pixel, north, west, rows, cols = effective_grid(2)
        assert pixel == pytest.approx(0.04)
        assert north == pytest.approx(69.99)
        assert west == pytest.approx(-179.99)
        assert rows == 3250
        assert cols == 9000

    def test_effective_params_f1(self):
        pixel, north, west, rows, cols = effective_grid(1)
        assert pixel == pytest.approx(0.02)
        assert north == pytest.approx(70.0)
        assert west == pytest.approx(-180.0)
        assert rows == NATIVE_ROWS
        assert cols == NATIVE_COLS

    def test_effective_params_f4(self):
        pixel, north, west, rows, cols = effective_grid(4)
        assert pixel == pytest.approx(0.08)
        assert north == pytest.approx(69.97)
        assert west == pytest.approx(-179.97)
        assert rows == 1625
        assert cols == 4500

    def test_downsampled_shape_crops_rows(self):
        assert downsampled_shape(1) == (NATIVE_ROWS, NATIVE_COLS)
        assert downsampled_shape(2) == (3250, 9000)
        assert downsampled_shape(4) == (1625, 4500)

    def test_descending_lat_row_mapping(self):
        """row 0 is the +70 band; index math is (north_eff - lat)/pixel."""
        pixel, north, _, rows, _ = effective_grid(2)
        row = ((north - np.array([69.99, 0.0, -59.99])) / pixel).astype(int)
        assert row[0] == 0
        assert row[1] == int(69.99 / 0.04)  # 1749
        assert row[2] == rows - 1


# ── Region registration ────────────────────────────────────────────────


class TestRegionRegistration:
    """RRQPE is a single coarse global region, always-on, no narrow group."""

    def test_region_fields_match_downsample(self):
        region = REGIONS["RRQPE"]
        F = max(1, int(settings.rrqpe_downsample))
        assert region.name == "RRQPE"
        assert (region.west, region.east, region.south, region.north) == (
            -180.0, 180.0, -60.0, 70.0,
        )
        assert region.proj == "latlon"
        assert region.pixel_size == pytest.approx(NATIVE_PIXEL * F)
        assert region.pixel_size_y == pytest.approx(NATIVE_PIXEL * F)
        assert region.grid_width == NATIVE_COLS // F
        assert region.grid_height == (NATIVE_ROWS // F) * F // F
        assert region.width == NATIVE_COLS // F
        assert region.height == NATIVE_ROWS // F
        assert region.is_global is True
        assert region.storm_cells is False

    def test_grid_shape_matches_region(self):
        region = REGIONS["RRQPE"]
        grid = RRQPEGrid()
        assert grid.effective_shape == (region.height, region.width)

    def test_in_regions_but_no_narrow_group(self):
        from librewxr.data.regions import REGION_GROUPS

        assert "RRQPE" in REGIONS
        for names in REGION_GROUPS.values():
            assert "RRQPE" not in names
        # The group label on the RegionDef itself is a plain label, not a
        # resolvable alias.
        assert REGIONS["RRQPE"].group == "GLOBAL"

    def test_block_centers_register_with_renderer_convention(self):
        """The renderer samples latlon grids with ``row = rint((north -
        lat)/ps_y)``, ``col = rint((lon - west)/ps_x)`` (edge-based, pixel
        centres at ``north - ps_y*(k+0.5)``).  Block ``k``'s centre must
        land on row/col ``k`` so the block-averaged grid registers exactly
        against the region bbox.
        """
        F = max(1, int(settings.rrqpe_downsample))
        region = REGIONS["RRQPE"]
        pixel = NATIVE_PIXEL * F
        _, north_eff, west_eff, rows, cols = effective_grid(F)

        for kr in (0, 1, rows // 2, rows - 1):
            lat = north_eff - pixel * kr  # block centre latitude
            row = int(np.rint((region.north - lat) / region._ps_y))
            assert row == kr
        for kc in (0, 1, cols // 2, cols - 1):
            lon = west_eff + pixel * kc  # block centre longitude
            col = int(np.rint((lon - region.west) / region.pixel_size))
            assert col == kc

        # Bbox edges stay in-bounds (row/col indices within the grid).
        lat_edge = region.south
        lon_edge = region.east
        row = int(np.rint((region.north - lat_edge) / region._ps_y))
        col = int(np.rint((lon_edge - region.west) / region.pixel_size))
        assert 0 <= row <= region.height
        assert 0 <= col <= region.width

    def test_coverage_polygon_is_full_inset_band(self):
        from librewxr.sources.world.rrqpe.regions import RRQPE_COVERAGE_POLYGON

        lats = [p[0] for p in RRQPE_COVERAGE_POLYGON]
        lons = [p[1] for p in RRQPE_COVERAGE_POLYGON]
        assert min(lats) == -58.0 and max(lats) == 68.0
        assert min(lons) == -180.0 and max(lons) == 180.0


# ── Lag-shifted relative matching ──────────────────────────────────────


def _floor_now() -> int:
    """Current wall clock floored to the 10-min scan cadence."""
    return (int(time.time()) // 600) * 600


class TestLagShift:
    """Lag-shifted 1:1 matching: frames are served the scan ``lag`` seconds
    their senior so animations step every frame instead of clamping the
    leading frames onto the newest scan.

    All timestamps are wall-clock-relative (10-min aligned slots) so the
    lag math serves every query.  ``_store_scans`` simulates a ~20-min
    publish lag: scans at the last ``len(values)`` aligned slots ending
    at ``floor_now - 1200``.
    """

    @staticmethod
    def _store_scans(values):
        """Inject scans at 10-min-aligned slots ending at ``floor_now -
        1200`` (a ~20-min publish lag).  ``values`` are oldest -> newest."""
        floor_now = _floor_now()
        newest = floor_now - 1200
        grid = RRQPEGrid(downsample=1)
        for i, value in enumerate(values):
            _inject_frame(grid, newest - (len(values) - 1 - i) * 600, value)
        return grid, floor_now, newest

    @staticmethod
    def _matched_value(grid, ts):
        """The uniform value of the scan matched at ``ts``, or None."""
        match = grid.match_timestamp(ts)
        if match is None:
            return None
        frame = grid.frame_at(match)
        return None if frame is None else int(np.asarray(frame).flat[0])

    def test_newest_frame_maps_to_newest_scan(self):
        grid, floor_now, _ = self._store_scans([101, 102, 103, 104])
        assert self._matched_value(grid, floor_now) == 104

    def test_consecutive_frames_map_to_consecutive_scans(self):
        """1:1 distinct: every past frame gets its own scan, stepping
        backward through the store (values oldest -> newest: 101..104)."""
        grid, floor_now, _ = self._store_scans([101, 102, 103, 104])
        assert self._matched_value(grid, floor_now) == 104
        assert self._matched_value(grid, floor_now - 600) == 103
        assert self._matched_value(grid, floor_now - 1800) == 101

    def test_missing_slot_falls_to_neighbor(self):
        """A single missed scan slot degrades to the adjacent scan — never a
        blink (zeros) and never a decline (None)."""
        grid, floor_now, newest = self._store_scans([101, 102, 103, 104])
        # Delete the scan that the floor_now - 600 frame maps to (value 103).
        del grid._timesteps[newest - 600]
        grid._sorted_timestamps = sorted(grid._timesteps)
        assert self._matched_value(grid, floor_now - 600) in (102, 104)

    def test_lag_at_cap_inclusive_serves(self):
        """Lag == tolerance (1800 s, floored) still serves."""
        now = int(time.time())
        grid = RRQPEGrid(downsample=1)
        _inject_frame(grid, now - 1900, 111)   # lag = 1800, at cap
        assert self._matched_value(grid, now) == 111

    def test_lag_beyond_cap_declines_whole_source(self):
        """Lag > tolerance (2400 s) declines the whole source: no past
        query matches."""
        now = int(time.time())
        grid = RRQPEGrid(downsample=1)
        _inject_frame(grid, now - 2500, 111)   # lag = 2400 > 1800
        assert grid.match_timestamp(now) is None
        assert grid.match_timestamp(now - 600) is None

    def test_empty_store_false(self):
        grid = RRQPEGrid()
        assert grid.match_timestamp(_floor_now()) is None
        assert grid.reference_time is None
        assert grid.timestep_count == 0
        assert grid.timestamps == []


# ── Wall-clock observed-only gate ──────────────────────────────────────


class TestWallClockGate:
    """The hard wall-clock gate rejects future-dated queries even when they
    fall inside the staleness tolerance (the nowcast-leak regression pin).

    Timestamps here are relative to the current wall clock so the pins
    stay valid no matter when the suite runs: stored scans are always in
    the past, and future queries exercise only the gate.
    """

    def test_newest_frame_matches_newest_past_query(self):
        """Frame 20 min old answers a 10-min-old query — the production case
        that the 1800 s tolerance exists to serve."""
        now = int(time.time())
        grid = RRQPEGrid(downsample=1)
        _inject_frame(grid, now - 1200, 100)
        assert grid.match_timestamp(now - 600) is not None

    def test_future_ts_within_tolerance_still_rejected(self):
        """now + 600 is 1800 s from the stored scan — inside the tolerance —
        but future-dated, so the wall-clock gate must reject it."""
        now = int(time.time())
        grid = RRQPEGrid(downsample=1)
        _inject_frame(grid, now - 1200, 100)
        assert grid.match_timestamp(now + 600) is None

    def test_no_match_for_future_ts(self):
        now = int(time.time())
        grid = RRQPEGrid(downsample=1)
        _inject_frame(grid, now - 1200, 100)
        assert grid.match_timestamp(now + 600) is None

    def test_stale_past_frame_rejected(self):
        """2100 s staleness is beyond the 1800 s tolerance for past frames."""
        now = int(time.time())
        grid = RRQPEGrid(downsample=1)
        _inject_frame(grid, now - 2400, 100)
        assert grid.match_timestamp(now - 300) is None


# ── Downsample ─────────────────────────────────────────────────────────


class TestDownsample:
    def test_block_nanmean_with_nan_and_all_nan_block(self):
        rate = np.array([
            [1.0, np.nan, 10.0, 20.0],
            [3.0, 4.0, 30.0, 40.0],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ds = block_nanmean_downsample(rate, 2)
        assert ds.shape == (2, 2)
        assert ds[0, 0] == pytest.approx(8.0 / 3.0)   # 3 finite members
        assert ds[0, 1] == pytest.approx(25.0)
        assert np.isnan(ds[1, 0])
        assert np.isnan(ds[1, 1])
        # All-NaN blocks encode to 0 through the Z-R path.
        encoded = precip_rate_to_dbz_encoded(ds, dbz_offset=6.0)
        assert encoded[0, 0] > 0
        assert encoded[0, 1] > 0
        assert encoded[1, 0] == 0
        assert encoded[1, 1] == 0

    def test_factor_one_is_identity(self):
        rate = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        ds = block_nanmean_downsample(rate, 1)
        assert np.array_equal(ds, rate)

    def test_crops_to_largest_multiple(self):
        rate = np.ones((5, 6), dtype=np.float32)
        ds = block_nanmean_downsample(rate, 2)
        assert ds.shape == (2, 3)


# ── Key parsing + XML ─────────────────────────────────────────────────


class TestKeyParsing:
    def test_glb5_regex_extracts_scan_ts(self):
        key = (
            "BLEND/RainRate-Blend-INST/2026/08/14/00/"
            "RRQPE-INST-GLB-5_v1r1_blend_s202608140000000"
            "_e202608140009599_c202608140023173.nc"
        )
        ts = scan_ts_from_key(key)
        assert ts is not None
        expected = int(datetime(2026, 8, 14, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        assert ts == expected

    def test_regex_rejects_other_blend_variants(self):
        key = (
            "BLEND/RainRate-Blend-INST/2026/08/14/00/"
            "RRQPE-INST-GLB-2_v1r1_blend_s202608140000000"
            "_e202608140009599_c202608140023173.nc"
        )
        assert GLB5_KEY_RE.search(key) is None
        assert scan_ts_from_key(key) is None

    def test_hour_prefix_format(self):
        ts = int(datetime(2026, 8, 14, 23, 10, 0, tzinfo=timezone.utc).timestamp())
        assert hour_prefix(ts) == "BLEND/RainRate-Blend-INST/2026/08/14/23/"

    def test_parse_s3_listing_keys(self):
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            "<Name>noaa-enterprise-rainrate-pds</Name>"
            "<KeyCount>2</KeyCount>"
            "<Contents><Key>BLEND/RainRate-Blend-INST/2026/08/14/00/"
            "RRQPE-INST-GLB-5_v1r1_blend_s202608140000000_"
            "e202608140009599_c202608140023173.nc</Key></Contents>"
            "<Contents><Key>BLEND/RainRate-Blend-INST/2026/08/14/00/"
            "RRQPE-INST-GLB-2_v1r1_blend_s202608140000000_"
            "e202608140009599_c202608140023173.nc</Key></Contents>"
            "</ListBucketResult>"
        )
        keys = parse_s3_listing_keys(xml.encode())
        assert len(keys) == 2
        assert keys[0].endswith(".nc")


# ── Fetch (mocked transport, synthetic NetCDF4) ───────────────────────


def _key_for(slot_ts: int) -> str:
    """Build the S3 key for a scan slot, mirroring the real filename."""
    dt = datetime.fromtimestamp(slot_ts, tz=timezone.utc)
    tok = dt.strftime("%Y%m%d%H%M%S")
    return (
        f"BLEND/RainRate-Blend-INST/{dt.year:04d}/{dt.month:02d}/{dt.day:02d}/"
        f"{dt.hour:02d}/RRQPE-INST-GLB-5_v1r1_blend_s{tok}000"
        f"_e{tok}959_c{tok}173.nc"
    )


def _listing_xml(keys: list[str]) -> str:
    contents = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
        f"<KeyCount>{len(keys)}</KeyCount>{contents}</ListBucketResult>"
    )


def _synthetic_nc_bytes(rows: int = 4, cols: int = 4, rate: float = 3.0) -> bytes:
    """Build a tiny NetCDF4 buffer mimicking the real RRQPE product.

    Dims/vars/attrs mirror the live GLB-5 files (scaled int16 RRQPE with
    DQF, where 3 = no-data) but at a small size so the fetch tests stay
    cheap.  Rate is uniform across the grid.
    """
    from netCDF4 import Dataset

    fd, path = tempfile.mkstemp(suffix=".nc")
    os.close(fd)
    try:
        ds = Dataset(path, "w", format="NETCDF4")
        ds.createDimension("Rows", rows)
        ds.createDimension("Columns", cols)
        ds.geospatial_lat_min = -60.0
        ds.geospatial_lat_max = 70.0
        ds.geospatial_lon_min = -180.0
        ds.geospatial_lon_max = 180.0
        ds.geospatial_lat_resolution = 0.02
        ds.geospatial_lon_resolution = 0.02
        rvar = ds.createVariable(
            "RRQPE", "i2", ("Rows", "Columns"), fill_value=np.int16(-9990),
        )
        rvar.scale_factor = np.float32(0.1)
        rvar.add_offset = np.float32(0.0)
        rvar.units = "mm/h"
        dvar = ds.createVariable("DQF", "i1", ("Rows", "Columns"))
        rvar[:] = np.full((rows, cols), int(rate / 0.1), dtype=np.int16)
        dvar[:] = np.zeros((rows, cols), dtype=np.int8)
        ds.close()
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


class _S3Transport:
    """MockTransport handler serving hour listings + .nc downloads."""

    def __init__(self, available_slots, nc_bytes):
        self._available_slots = list(available_slots)
        self._nc_bytes = nc_bytes

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "list-type=2" in url:
            prefix = request.url.params.get("prefix", "")
            hour_keys = [_key_for(s) for s in self._available_slots]
            return httpx.Response(200, text=_listing_xml(
                [k for k in hour_keys if k.startswith(prefix)],
            ))
        if url.endswith(".nc"):
            return httpx.Response(200, content=self._nc_bytes)
        return httpx.Response(404, text="not found")


class TestFetch:
    @staticmethod
    def _now():
        """Current wall clock floored to 10 min.

        Wall-clock-relative so the stored slots are always in the past
        (the observed-only gate rejects future-dated queries) regardless
        of when the suite runs.
        """
        return (int(time.time()) // 600) * 600

    async def test_fetch_stores_expected_slots(self, tmp_path):
        now_ts = self._now()
        slot_a = now_ts - 30 * 60
        slot_b = now_ts - 20 * 60
        grid = RRQPEGrid(cache_dir=tmp_path, downsample=1)
        nc = _synthetic_nc_bytes(rows=4, cols=4, rate=3.0)
        grid._client = httpx.Client(transport=httpx.MockTransport(
            _S3Transport([slot_a, slot_b], nc),
        ))
        try:
            await grid.fetch(now_ts=now_ts, history_seconds=3600)
            assert slot_a in grid._timesteps
            assert slot_b in grid._timesteps
            assert grid.timestep_count == 2
            assert grid.reference_time == slot_b
            # The newest past frame maps (lag-shifted) onto the newest scan.
            assert grid.match_timestamp(now_ts) == slot_b
            # Decode pipeline produced non-zero encoded data (rate 3 mm/h).
            assert int(np.asarray(grid._timesteps[slot_a]).max()) > 0
        finally:
            grid._client.close()

    async def test_fetch_evicts_out_of_window_frames(self, tmp_path):
        now_ts = self._now()
        old_slot = now_ts - 3 * 3600  # 09:00 — far outside the window
        slot_new = now_ts - 20 * 60   # 11:40
        grid = RRQPEGrid(cache_dir=tmp_path, downsample=4)
        old_mm = grid._to_memmap(
            str(old_slot), np.full(grid.effective_shape, 100, dtype=np.uint8),
        )
        grid._timesteps[old_slot] = old_mm
        grid._sorted_timestamps = sorted(grid._timesteps)

        nc = _synthetic_nc_bytes(rows=8, cols=8, rate=3.0)
        grid._client = httpx.Client(transport=httpx.MockTransport(
            _S3Transport([slot_new], nc),
        ))
        try:
            await grid.fetch(now_ts=now_ts, history_seconds=3600)
            assert old_slot not in grid._timesteps
            assert slot_new in grid._timesteps
            assert not (tmp_path / "rrqpe" / f"{old_slot}.dat").exists()
            assert (tmp_path / "rrqpe" / f"{slot_new}.dat").exists()
        finally:
            grid._client.close()

    async def test_fetch_skips_slots_missing_from_listing(self, tmp_path):
        now_ts = self._now()
        present_slot = now_ts - 20 * 60  # 11:40 — listed
        absent_slot = now_ts - 40 * 60   # 11:20 — needed but not listed
        grid = RRQPEGrid(cache_dir=tmp_path, downsample=1)
        nc = _synthetic_nc_bytes(rows=4, cols=4, rate=3.0)
        grid._client = httpx.Client(transport=httpx.MockTransport(
            _S3Transport([present_slot], nc),
        ))
        try:
            await grid.fetch(now_ts=now_ts, history_seconds=3600)
            assert present_slot in grid._timesteps
            assert absent_slot not in grid._timesteps
        finally:
            grid._client.close()

    async def test_fetch_total_failure_keeps_existing_frames(self, tmp_path):
        now_ts = self._now()
        grid = RRQPEGrid(cache_dir=tmp_path, downsample=4)
        keep_slot = now_ts - 30 * 60
        _inject_frame(grid, keep_slot, 100)
        # Transport raises on every request → per-file failure path.
        def boom(request):
            raise httpx.ConnectError("no network in tests")

        grid._client = httpx.Client(transport=httpx.MockTransport(boom))
        try:
            await grid.fetch(now_ts=now_ts, history_seconds=3600)
            assert grid.timestep_count == 1
            assert keep_slot in grid._timesteps
        finally:
            grid._client.close()


# ── Scan-store boot hygiene ────────────────────────────────────────────


class TestBootHygiene:
    async def test_stale_tmp_swept_at_boot(self, tmp_path):
        cache_dir = tmp_path / "rrqpe"
        cache_dir.mkdir(parents=True)
        stale = cache_dir / "123.dat.tmp"
        stale.write_bytes(b"\x00" * 8)
        grid = RRQPEGrid(cache_dir=tmp_path)
        assert not stale.exists()
        await grid.close()


# ── Radar-source fetch protocol (fetch_frame slotting) ─────────────────


def _rrqpe_region(factor: int = 4) -> RegionDef:
    """RegionDef matching an F=``factor`` grid so tests stay cheap."""
    pixel = NATIVE_PIXEL * factor
    return RegionDef(
        name="RRQPE",
        west=-180.0, east=180.0, south=-60.0, north=70.0,
        pixel_size=pixel, pixel_size_y=pixel, group="GLOBAL",
        grid_width=NATIVE_COLS // factor,
        grid_height=(NATIVE_ROWS // factor) * factor // factor,
        storm_cells=False,
    )


class TestFetchFrameSlotting:
    """Radar-source protocol: lag-shifted slotting with distinct scans per
    frame slot, None beyond the tolerance cap, and the scan-store refresh
    only on the newest-slot request."""

    @staticmethod
    def _store_scans(grid, values):
        """Inject scans at aligned slots ending at ``floor_now - 1200``."""
        floor_now = _floor_now()
        newest = floor_now - 1200
        for i, value in enumerate(values):
            arr = np.full(grid.effective_shape, value, dtype=np.uint8)
            grid._timesteps[newest - (len(values) - 1 - i) * 600] = arr
        grid._sorted_timestamps = sorted(grid._timesteps)
        return floor_now, newest

    async def test_fetch_frame_slots_map_to_distinct_scans(self, monkeypatch):
        grid = RRQPEGrid(downsample=4)
        self._store_scans(grid, [101, 102, 103, 104])

        async def _no_refresh(**kwargs):
            pass

        monkeypatch.setattr(grid, "fetch", _no_refresh)
        src = RRQPESource(grid)
        region = _rrqpe_region(4)
        # ``minutes_ago`` is in minutes (the fetcher passes i*interval_min).
        out_new = await src.fetch_frame(region, 0)
        out_mid = await src.fetch_frame(region, 10)
        out_old = await src.fetch_frame(region, 30)
        assert [int(np.asarray(o).flat[0]) for o in (out_new, out_mid, out_old)] == [
            104, 103, 101,
        ]
        assert len({id(o) for o in (out_new, out_mid, out_old)}) == 3
        for o in (out_new, out_mid, out_old):
            assert o.dtype == np.uint8
            assert o.shape == grid.effective_shape

    async def test_fetch_frame_none_beyond_tolerance_cap(self, monkeypatch):
        """A scan more than ``rrqpe_match_tolerance_seconds`` old declines
        the whole source — region absent from every frame."""
        now = int(time.time())
        grid = RRQPEGrid(downsample=4)
        arr = np.full(grid.effective_shape, 111, dtype=np.uint8)
        grid._timesteps[now - 2500] = arr
        grid._sorted_timestamps = [now - 2500]

        async def _no_refresh(**kwargs):
            pass

        monkeypatch.setattr(grid, "fetch", _no_refresh)
        src = RRQPESource(grid)
        region = _rrqpe_region(4)
        assert await src.fetch_frame(region, 0) is None
        assert await src.fetch_frame(region, 20) is None

    async def test_fetch_archive_frame_uses_lag_shift(self, monkeypatch):
        grid = RRQPEGrid(downsample=4)
        self._store_scans(grid, [101, 102, 103, 104])
        src = RRQPESource(grid)
        floor_now = _floor_now()
        region = _rrqpe_region(4)
        when = datetime.fromtimestamp(floor_now, tz=timezone.utc)
        out = await src.fetch_archive_frame(region, when)
        assert out is not None
        assert int(np.asarray(out).flat[0]) == 104
        # Naive datetimes are treated as UTC wall-clock (house convention).
        when_naive = datetime.fromtimestamp(
            floor_now - 600, tz=timezone.utc,
        ).replace(tzinfo=None)
        out2 = await src.fetch_archive_frame(region, when_naive)
        assert out2 is not None
        assert int(np.asarray(out2).flat[0]) == 103

    async def test_scan_refresh_triggered_only_on_newest_slot(self, monkeypatch):
        """The bulk scan-store refresh runs on the minutes_ago==0 request
        only — once per cycle — with a history window covering the whole
        frame span plus the lag tolerance."""
        grid = RRQPEGrid(downsample=4)
        calls = []

        async def _fake_fetch(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(grid, "fetch", _fake_fetch)
        src = RRQPESource(grid)
        region = _rrqpe_region(4)
        # No scans stored: every slot declines, but the refresh side effect
        # is what we pin here.  minutes_ago=10/110 are mid/old-frame slots.
        await src.fetch_frame(region, 0)
        await src.fetch_frame(region, 10)
        await src.fetch_frame(region, 110)
        assert len(calls) == 1
        expected_history = (
            settings.max_frames * settings.fetch_interval
            + settings.rrqpe_match_tolerance_seconds
        )
        assert calls[0]["history_seconds"] == expected_history


# ── Always-on enabled set ──────────────────────────────────────────────


class TestAlwaysEnabled:
    def test_contribution_defaults_to_not_always_enabled(self):
        from librewxr.sources._base import RadarSourceContribution

        contrib = RadarSourceContribution(
            regions=[], instance=None, group="X",  # type: ignore[arg-type]
        )
        assert contrib.always_enabled is False

    def test_rrqpe_contribution_flagged_always_enabled(self, monkeypatch):
        from librewxr.config import settings as real_settings
        from librewxr.sources import collect_radar_contributions

        monkeypatch.setattr(real_settings, "radar_enabled", True)
        real_settings.rrqpe_enabled = True
        contribs = collect_radar_contributions(real_settings)
        rrqpe = [
            c for c in contribs
            if any(r.name == "RRQPE" for r in c.regions)
        ]
        assert len(rrqpe) == 1
        assert rrqpe[0].always_enabled is True
        assert rrqpe[0].group == "GLOBAL"

    def test_rrqpe_kept_in_effective_enabled_set_under_narrow_spec(self, monkeypatch):
        """With ``enabled_regions=['USCOMP']`` the always-on RRQPE region
        remains in the effective enabled set."""
        from librewxr.config import settings as real_settings
        from librewxr.sources import enabled_regions_with_always_on

        monkeypatch.setattr(real_settings, "radar_enabled", True)
        real_settings.rrqpe_enabled = True
        monkeypatch.setattr(real_settings, "enabled_regions", "USCOMP")
        names = enabled_regions_with_always_on(real_settings)
        assert "USCOMP" in names
        assert "RRQPE" in names

    def test_rrqpe_dropped_from_effective_set_when_disabled(self, monkeypatch):
        from librewxr.config import settings as real_settings
        from librewxr.sources import enabled_regions_with_always_on

        monkeypatch.setattr(real_settings, "radar_enabled", True)
        real_settings.rrqpe_enabled = False
        monkeypatch.setattr(real_settings, "enabled_regions", "USCOMP")
        try:
            names = enabled_regions_with_always_on(real_settings)
            assert "USCOMP" in names
            assert "RRQPE" not in names
        finally:
            real_settings.rrqpe_enabled = True


# ── Compositor ordering pin ────────────────────────────────────────────


class TestCompositorOrdering:
    """RRQPE sorts last in the multi-region compositor (coarsest
    pixel_size) and fills only pixels no finer region claims — the fine
    region's authoritative zeros win inside its coverage."""

    # Tile z=4 x=8 y=5: lon 0..22.5, lat ~31.8..49 — inside both the fine
    # region's bbox and RRQPE's band.
    _Z, _X, _Y, _TILE = 4, 8, 5, 256

    def test_rrqpe_fills_only_unclaimed_pixels(self, monkeypatch):
        from librewxr.tiles import renderer as renderer_mod

        from librewxr.config import settings as real_settings

        monkeypatch.setattr(real_settings, "noise_floor_dbz", 10.0)

        fine = RegionDef(
            name="FINE", west=-10.0, east=10.0, south=30.0, north=50.0,
            pixel_size=0.01, group="TEST",
            grid_width=2000, grid_height=2000,
        )
        rrqpe = REGIONS["RRQPE"]
        monkeypatch.setattr(
            renderer_mod, "overlapping_regions",
            lambda z, x, y, enabled=None: [fine, rrqpe],
        )
        monkeypatch.setattr(
            renderer_mod, "sample_coverage",
            lambda name, lat, lon: (
                # FINE's bbox within the tile (lat 31.8..49 ⊂ 30..50, so
                # only the longitude edge bites); RRQPE covers everything.
                ((lon >= fine.west) & (lon <= fine.east))
                if name == "FINE"
                else np.ones(lat.shape, dtype=bool)
            ),
        )

        fine_frame = np.zeros((fine.height, fine.width), dtype=np.uint8)
        rrqpe_frame = np.full(
            (rrqpe.height, rrqpe.width), 200, dtype=np.uint8,
        )
        geom = renderer_mod.compute_tile_geometry(
            {"FINE": fine_frame, "RRQPE": rrqpe_frame},
            self._Z, self._X, self._Y, tile_size=self._TILE,
            nwp_chain=None,
        )
        assert geom.is_transparent is False
        values = geom.values
        assert values.shape == (self._TILE, self._TILE)
        # Inside FINE's coverage (lon ≤ 10 → first ~114 columns) the fine
        # region's zeros win — RRQPE's 200s must not bleed through.
        n_fine_cols = int(round((fine.east - 0.0) / (22.5 / self._TILE)))
        assert (values[:, :n_fine_cols] == 0).all()
        # Outside FINE's coverage RRQPE fills.
        assert (values[:, n_fine_cols:] == 200).all()
