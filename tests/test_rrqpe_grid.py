# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Tests for the NOAA Enterprise Rain Rate (RRQPE) GLB-5 blend grid.

Synthetic data only — no network.  The fetch tests drive the real
``_fetch_sync`` path through ``httpx.MockTransport`` with an S3 listing
XML body and tiny in-memory NetCDF4 files.
"""
from __future__ import annotations

import os
import pickle
import tempfile
import warnings
from datetime import datetime, timezone

import httpx
import numpy as np
import pytest

pytestmark = pytest.mark.rrqpe

from librewxr.data.nwp_source import NWPChain, NWPSource
from librewxr.sources.world.ifs.grid import (
    ECMWFGrid,
    GRID_HEIGHT as IFS_H,
    GRID_WIDTH as IFS_W,
)
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


def _inject_frame(grid: RRQPEGrid, ts: int, value: int):
    """Inject a uniform-value frame into the RRQPE store."""
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

    def test_domain_and_feather_weights(self):
        grid = RRQPEGrid()
        lats = np.array([0.0, 71.0, -61.0, 68.5, -59.0, -58.0])
        lons = np.zeros_like(lats)
        f = grid.feather_mask(lats, lons)
        assert f.dtype == np.float32
        assert f[0] == pytest.approx(1.0)    # mid-band full weight
        assert f[1] == 0.0                   # north of band
        assert f[2] == 0.0                   # south of band
        assert f[3] == pytest.approx(0.75)   # 68.5 → (70-68.5)/2
        assert f[4] == pytest.approx(0.5)    # -59 → (-59+60)/2
        assert f[5] == pytest.approx(1.0)    # -58 still full weight
        d = grid.domain_mask(lats, lons)
        assert d.dtype == np.bool_
        assert d.tolist() == [True, False, False, True, True, True]

    def test_feather_taper_monotonic(self):
        grid = RRQPEGrid()
        north = grid.feather_mask(np.linspace(67.0, 71.0, 25), np.zeros(25))
        south = grid.feather_mask(np.linspace(-57.0, -61.0, 25), np.zeros(25))
        assert (np.diff(north) <= 1e-6).all()
        assert (np.diff(south) <= 1e-6).all()


# ── Observed-only contract ────────────────────────────────────────────


class TestObservedOnlyContract:
    def _store(self, *slots):
        grid = RRQPEGrid(downsample=1)
        for ts, value in slots:
            _inject_frame(grid, ts, value)
        return grid

    def test_exact_slot_true(self):
        grid = self._store((1000000, 100))
        assert grid.has_data_at(1000000) is True

    def test_within_tolerance_true(self):
        grid = self._store((1000000, 100))
        assert grid.has_data_at(1000000 + 900) is True
        assert grid.has_data_at(1000000 - 899) is True

    def test_beyond_tolerance_false(self):
        grid = self._store((1000000, 100))
        assert grid.has_data_at(1000000 + 901) is False
        assert grid.has_data_at(1000000 - 901) is False

    def test_future_ts_false(self):
        grid = self._store((1000000, 100))
        assert grid.has_data_at(1000000 + 3600) is False
        assert grid.has_data_at(1000000 + 20000) is False

    def test_empty_store_false(self):
        grid = RRQPEGrid()
        assert grid.has_data_at(1000000) is False
        assert grid.has_data() is False
        assert grid.reference_time is None
        assert grid.timestep_count == 0

    def test_sample_zeros_when_no_match(self):
        grid = RRQPEGrid()
        out = grid.sample(np.array([0.0]), np.array([0.0]), timestamp=1000000)
        assert out.shape == (1,)
        assert out.dtype == np.uint8
        assert out[0] == 0

    def test_sample_returns_frame_within_tolerance(self):
        grid = self._store((1000000, 137))
        out = grid.sample(
            np.array([0.0]), np.array([0.0]), timestamp=1000000 + 600,
        )
        assert int(out[0]) == 137

    def test_bilinear_zero_guard(self):
        """Bilinear must not ghost precip into clear-sky neighbours."""
        grid = self._store((1000000, 137))
        lats = np.array([0.0, 0.0, 0.0])
        lons = np.array([-179.9, -179.8, -179.7])
        # Uniform-value frame: bilinear equals the frame value everywhere
        # (no zero neighbours to trigger the guard).
        out = grid.sample(lats, lons, timestamp=1000000, bilinear=True)
        assert (out == 137).all()


# ── Chain integration ──────────────────────────────────────────────────


class TestChainIntegration:
    def _chain(self, rrqpe_ts, rrqpe_value, ifs_value=84):
        grid = RRQPEGrid(downsample=1)
        _inject_frame(grid, rrqpe_ts, rrqpe_value)
        ifs = ECMWFGrid()
        ifs_dbz = np.full((IFS_H, IFS_W), ifs_value, dtype=np.uint8)
        ifs._timesteps[1000000] = (ifs_dbz, np.zeros_like(ifs_dbz, dtype=bool))
        ifs._sorted_timestamps = [1000000]
        return grid, ifs

    def test_past_ts_inside_band_prefers_rrqpe(self):
        rrqpe, ifs = self._chain(1000000, 200)
        chain = NWPChain([rrqpe, ifs])
        out = chain.sample(np.array([0.0]), np.array([0.0]), timestamp=1000000)
        assert int(out[0]) == 200

    def test_past_ts_outside_band_falls_to_ifs(self):
        rrqpe, ifs = self._chain(1000000, 200)
        chain = NWPChain([rrqpe, ifs])
        out = chain.sample(np.array([75.0]), np.array([0.0]), timestamp=1000000)
        assert int(out[0]) == 84

    def test_future_ts_inside_band_falls_to_ifs(self):
        """Critical regression pin: observations must never answer future ts."""
        rrqpe, ifs = self._chain(1000000, 200)
        chain = NWPChain([rrqpe, ifs])
        out = chain.sample(
            np.array([0.0]), np.array([0.0]), timestamp=1000000 + 3600,
        )
        assert int(out[0]) == 84

    def test_snow_mask_still_comes_from_ifs(self):
        rrqpe, ifs = self._chain(1000000, 200)
        ifs_snow = np.ones((IFS_H, IFS_W), dtype=bool)
        ifs._timesteps[1000000] = (
            np.full((IFS_H, IFS_W), 84, dtype=np.uint8), ifs_snow,
        )
        chain = NWPChain([rrqpe, ifs])
        out = chain.get_snow_mask(np.array([0.0]), np.array([0.0]), timestamp=1000000)
        assert out.tolist() == [True]

    def test_real_chain_places_rrqpe_before_hrrr(self, tmp_path, monkeypatch):
        """The real sorted contribution list puts RRQPE ahead of HRRR."""
        from librewxr.config import settings as real_settings
        from librewxr.sources import collect_nwp_contributions, nwp_grid_slug

        monkeypatch.setattr(real_settings, "regional_nwp_enabled", True)
        monkeypatch.setattr(real_settings, "na_nwp_source", "hrrr")
        monkeypatch.setattr(real_settings, "eu_nwp_profile", "ifs")
        contribs = collect_nwp_contributions(real_settings, cache_dir=tmp_path)
        slugs = [nwp_grid_slug(c) for c in contribs]
        assert slugs[0] == "rrqpe_grid"
        assert "hrrr_grid" in slugs
        assert slugs.index("rrqpe_grid") < slugs.index("hrrr_grid")


# ── Protocol conformance ───────────────────────────────────────────────


class TestProtocol:
    def test_satisfies_nwpsource(self):
        assert isinstance(RRQPEGrid(), NWPSource)
        assert RRQPEGrid().name == "rrqpe"

    def test_supports_snow_false(self):
        grid = RRQPEGrid()
        assert grid.supports_snow is False
        out = grid.get_snow_mask(np.array([0.0]), np.array([0.0]))
        assert out.dtype == np.bool_
        assert not out.any()


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
        return int(datetime(2026, 8, 14, 12, 0, 0, tzinfo=timezone.utc).timestamp())

    async def test_fetch_stores_expected_slots(self, tmp_path):
        now_ts = self._now()
        slot_a = now_ts - 30 * 60   # 11:30
        slot_b = now_ts - 20 * 60   # 11:40
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
            assert grid.has_data_at(slot_a) is True
            # Decode pipeline produced non-zero encoded data (rate 3 mm/h).
            assert int(np.asarray(grid._timesteps[slot_a]).max()) > 0
            # Sample maps inside the stored frame (top-left block centre).
            out = grid.sample(
                np.array([69.99]), np.array([-179.99]), timestamp=slot_a,
            )
            assert out.dtype == np.uint8
            assert int(out[0]) > 0
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


# ── Persistence ────────────────────────────────────────────────────────


class TestPersistence:
    async def test_getstate_setstate_round_trip(self, tmp_path):
        grid = RRQPEGrid(cache_dir=tmp_path, downsample=4)
        ts = 1000000
        value = 137
        mm = grid._to_memmap(
            str(ts), np.full(grid.effective_shape, value, dtype=np.uint8),
        )
        grid._timesteps[ts] = mm
        grid._sorted_timestamps = [ts]

        state = grid.__getstate__()
        assert state["timesteps"] == [ts]
        assert state["downsample"] == 4
        assert state["shape"] == list(grid.effective_shape)

        restored = pickle.loads(pickle.dumps(grid))
        assert restored.timestep_count == 1
        assert restored.reference_time == ts
        assert restored._rows == grid._rows
        out = restored.sample(np.array([0.0]), np.array([0.0]), timestamp=ts)
        assert int(out[0]) == value
        await grid.close()
        await restored.close()

    async def test_stale_tmp_swept_at_boot(self, tmp_path):
        cache_dir = tmp_path / "rrqpe"
        cache_dir.mkdir(parents=True)
        stale = cache_dir / "123.dat.tmp"
        stale.write_bytes(b"\x00" * 8)
        grid = RRQPEGrid(cache_dir=tmp_path)
        assert not stale.exists()
        await grid.close()
