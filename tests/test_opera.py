# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io
import logging
import struct
from datetime import datetime, timezone

import h5py
import httpx
import numpy as np
import pytest

pytestmark = pytest.mark.sources

from librewxr.config import Settings
from librewxr.data import retry
from librewxr.data.regions import REGIONS, resolve_regions
from librewxr.sources.regional.europe.radar.opera import (
    OperaSource,
    _parse_opera_hdf5,
    radar_provider,
)
from librewxr.tiles.coordinates import tile_overlaps_region

CLOUDFERRO = "https://s3.waw3-1.cloudferro.com/openradar-24h"
ECMWF_BACKUP = "https://object-store.os-api.cci1.ecmwf.int/openradar-24h-backup"
PRIMARY = "https://primary.test/openradar-24h"
BACKUP = "https://backup.test/openradar-24h-backup"


class TestOperaRegion:
    def test_opera_in_regions(self):
        assert "OPERA" in REGIONS

    def test_opera_is_laea(self):
        r = REGIONS["OPERA"]
        assert r.proj == "laea"
        assert r.group == "EUROPE"

    def test_opera_dimensions(self):
        r = REGIONS["OPERA"]
        assert r.width == 3800
        assert r.height == 4400

    def test_europe_group_resolution(self):
        # EUROPE bundles OPERA + ITCOMP (DPC Italian composite).  Group
        # membership is stored alphabetically by the discovery walker.
        assert resolve_regions("EUROPE") == ["ITCOMP", "OPERA"]

    def test_all_includes_opera(self):
        assert "OPERA" in resolve_regions("ALL")

    def test_mixed_regions(self):
        result = resolve_regions("EUROPE,CANADA")
        assert "OPERA" in result
        assert "CACOMP" in result


class TestOperaLAEA:
    def test_tile_over_europe_overlaps(self):
        region = REGIONS["OPERA"]
        # z=4, x=8, y=5 is central Europe
        assert tile_overlaps_region(region, z=4, x=8, y=5)

    def test_tile_over_america_does_not_overlap(self):
        region = REGIONS["OPERA"]
        # z=4, x=3, y=5 is eastern US
        assert not tile_overlaps_region(region, z=4, x=3, y=5)


class TestOperaHDF5Parser:
    def _make_opera_hdf5(
        self,
        data: np.ndarray,
        nodata: float = -9999000.0,
        undetect: float = -8888000.0,
        gain: float = 1.0,
        offset: float = 0.0,
    ) -> bytes:
        """Build a minimal OPERA-style ODIM HDF5 in memory."""
        buf = io.BytesIO()
        with h5py.File(buf, "w") as f:
            ds = f.create_group("dataset1")
            d1 = ds.create_group("data1")
            d1.create_dataset("data", data=data)
            what = d1.create_group("what")
            what.attrs["gain"] = gain
            what.attrs["offset"] = offset
            what.attrs["nodata"] = nodata
            what.attrs["undetect"] = undetect
            what.attrs["quantity"] = b"DBZH"
        return buf.getvalue()

    def test_basic_parsing(self):
        data = np.array([[20.0, 30.0], [-9999000.0, -8888000.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result is not None
        assert result.shape == (2, 2)

    def test_nodata_becomes_zero(self):
        data = np.array([[-9999000.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 0

    def test_undetect_becomes_zero(self):
        data = np.array([[-8888000.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 0

    def test_valid_dbz_encoding(self):
        # 20 dBZ -> (20+32)*2 = 104
        data = np.array([[20.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 104

    def test_high_dbz_encoding(self):
        # 60 dBZ -> (60+32)*2 = 184
        data = np.array([[60.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 184

    def test_negative_dbz_encoding(self):
        # -10 dBZ -> (-10+32)*2 = 44
        data = np.array([[-10.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 44

    def test_bad_data_returns_none(self):
        result = _parse_opera_hdf5(b"not hdf5")
        assert result is None

    def test_coverage_distinction(self):
        """Both nodata and undetect map to 0; only real precip is non-zero."""
        data = np.array([[-9999000.0, -8888000.0, 10.0]])
        raw = self._make_opera_hdf5(data)
        result = _parse_opera_hdf5(raw)
        assert result[0, 0] == 0    # nodata → 0
        assert result[0, 1] == 0    # undetect → 0 (gap-filler, let ECMWF fill)
        assert result[0, 2] >= 2    # real dBZ


class TestOperaSourceURL:
    def test_url_construction(self):
        src = OperaSource(["https://example.com/openradar-24h"])
        # 2026-04-10 09:05 UTC → ts=1775991900
        dt = datetime(2026, 4, 10, 9, 5, tzinfo=timezone.utc)
        ts = int(dt.timestamp())
        url = src._url_for_timestamp(ts, src._bucket_urls[0])
        assert "OPERA@20260410T0905@0@DBZH.h5" in url
        assert url.startswith("https://example.com/openradar-24h/2026/04/10/OPERA/COMP/")

    def test_url_rounds_to_5min(self):
        src = OperaSource(["https://example.com/openradar-24h"])
        # 09:07 should round to 09:05
        dt = datetime(2026, 4, 10, 9, 7, tzinfo=timezone.utc)
        url = src._url_for_timestamp(int(dt.timestamp()), src._bucket_urls[0])
        assert "T0905@" in url

    def test_url_for_each_bucket(self):
        # Trailing slashes are stripped; every bucket shares the key layout.
        src = OperaSource([CLOUDFERRO + "/", ECMWF_BACKUP])
        ts = int(datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc).timestamp())
        key = "2026/11/13/OPERA/COMP/OPERA@20261113T1200@0@DBZH.h5"
        assert [src._url_for_timestamp(ts, b) for b in src._bucket_urls] == [
            f"{CLOUDFERRO}/{key}",
            f"{ECMWF_BACKUP}/{key}",
        ]


class TestOperaBucketFallback:
    """Primary/backup bucket selection via ``httpx.MockTransport`` — no
    live network."""

    DT = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)

    @staticmethod
    def _composite() -> bytes:
        # One 20 dBZ pixel → uint8 104.
        return TestOperaHDF5Parser()._make_opera_hdf5(np.array([[20.0]]))

    def _make_source(self, handler, active: int = 0) -> OperaSource:
        src = OperaSource([PRIMARY, BACKUP])
        src._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        src._active = active
        return src

    def _serve(self, live_hosts: set[str], calls: list[tuple[str, str]]):
        """Handler: 200 + composite on ``live_hosts``, 404 elsewhere."""
        body = self._composite()

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.url.host, request.url.path.split("@")[1]))
            if request.url.host in live_hosts:
                return httpx.Response(200, content=body)
            return httpx.Response(404)
        return handler

    async def test_falls_back_when_primary_404s_every_slot(self):
        calls: list[tuple[str, str]] = []
        src = self._make_source(self._serve({"backup.test"}, calls))
        try:
            out = await src.fetch_archive_frame(REGIONS["OPERA"], self.DT)
        finally:
            await src.close()

        assert out is not None
        assert out[0, 0] == 104
        # The whole look-back runs on the primary before the backup is tried.
        assert calls == [
            ("primary.test", "20261113T1200"),
            ("primary.test", "20261113T1155"),
            ("primary.test", "20261113T1150"),
            ("primary.test", "20261113T1145"),
            ("backup.test", "20261113T1200"),
        ]
        assert src._active == 1

    async def test_prefers_the_bucket_that_last_worked(self):
        calls: list[tuple[str, str]] = []
        src = self._make_source(self._serve({"backup.test"}, calls))
        try:
            await src.fetch_archive_frame(REGIONS["OPERA"], self.DT)
            calls.clear()
            out = await src.fetch_frame(REGIONS["OPERA"], 0)
        finally:
            await src.close()

        assert out is not None
        # No wasted 404s on the quiet primary once the backup has served.
        assert [host for host, _ in calls] == ["backup.test"]

    async def test_returns_to_primary_when_backup_goes_quiet(self):
        calls: list[tuple[str, str]] = []
        src = self._make_source(self._serve({"primary.test"}, calls), active=1)
        try:
            out = await src.fetch_archive_frame(REGIONS["OPERA"], self.DT)
        finally:
            await src.close()

        assert out is not None
        assert [host for host, _ in calls] == ["backup.test"] * 4 + ["primary.test"]
        assert src._active == 0

    async def test_transport_error_tries_next_bucket(self, monkeypatch):
        monkeypatch.setattr(retry.settings, "download_retries", 0)
        body = self._composite()
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.host)
            if request.url.host == "primary.test":
                raise httpx.ConnectError("unreachable", request=request)
            return httpx.Response(200, content=body)

        src = self._make_source(handler)
        try:
            out = await src.fetch_archive_frame(REGIONS["OPERA"], self.DT)
        finally:
            await src.close()

        assert out is not None
        assert calls == ["primary.test", "backup.test"]
        assert src._active == 1

    async def test_403_on_every_slot_tries_next_bucket(self):
        # S3 hides missing keys behind 403 when anonymous listing is off.
        body = self._composite()
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.host)
            if request.url.host == "primary.test":
                return httpx.Response(403)
            return httpx.Response(200, content=body)

        src = self._make_source(handler)
        try:
            out = await src.fetch_archive_frame(REGIONS["OPERA"], self.DT)
        finally:
            await src.close()

        assert out is not None
        assert calls == ["primary.test"] * 4 + ["backup.test"]
        assert src._active == 1

    async def test_server_error_tries_next_bucket_at_once(self):
        body = self._composite()
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.host)
            if request.url.host == "primary.test":
                return httpx.Response(503)
            return httpx.Response(200, content=body)

        src = self._make_source(handler)
        try:
            out = await src.fetch_archive_frame(REGIONS["OPERA"], self.DT)
        finally:
            await src.close()

        assert out is not None
        # No look-back on a failing server: straight to the backup.
        assert calls == ["primary.test", "backup.test"]
        assert src._active == 1

    async def test_other_client_error_does_not_switch(self):
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.host)
            return httpx.Response(400)

        src = self._make_source(handler)
        try:
            out = await src.fetch_archive_frame(REGIONS["OPERA"], self.DT)
        finally:
            await src.close()

        assert out is None
        assert calls == ["primary.test"]
        assert src._active == 0

    async def test_all_buckets_empty_returns_none(self):
        calls: list[tuple[str, str]] = []
        src = self._make_source(self._serve(set(), calls))
        try:
            out = await src.fetch_archive_frame(REGIONS["OPERA"], self.DT)
        finally:
            await src.close()

        assert out is None
        assert [host for host, _ in calls] == (
            ["primary.test"] * 4 + ["backup.test"] * 4
        )
        assert src._active == 0

    async def test_logs_only_when_the_active_bucket_changes(self, caplog):
        calls: list[tuple[str, str]] = []
        src = self._make_source(self._serve({"backup.test"}, calls))
        caplog.set_level(logging.INFO, logger=OperaSource.__module__)
        try:
            for _ in range(3):
                await src.fetch_archive_frame(REGIONS["OPERA"], self.DT)
        finally:
            await src.close()

        switches = [r for r in caplog.records if "switched bucket" in r.getMessage()]
        assert len(switches) == 1
        assert switches[0].levelno == logging.WARNING
        assert BACKUP in switches[0].getMessage()


class TestOperaBucketConfig:
    @staticmethod
    def _settings(monkeypatch, base_url=None, bucket_urls=None) -> Settings:
        for name, value in (
            ("LIBREWXR_OPERA_BASE_URL", base_url),
            ("LIBREWXR_OPERA_BUCKET_URLS", bucket_urls),
        ):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        return Settings(_env_file=None)

    def test_default_bucket_order(self, monkeypatch):
        s = self._settings(monkeypatch)
        assert s.opera_bucket_urls == [CLOUDFERRO, ECMWF_BACKUP]

    def test_custom_base_url_replaces_primary_bucket(self, monkeypatch):
        s = self._settings(monkeypatch, base_url="https://mirror.example.org/")
        assert s.opera_bucket_urls == [
            "https://mirror.example.org/openradar-24h",
            ECMWF_BACKUP,
        ]

    def test_explicit_bucket_list_wins_over_base_url(self, monkeypatch):
        s = self._settings(
            monkeypatch,
            base_url="https://mirror.example.org",
            bucket_urls='["https://a.example/x/", "https://b.example/y"]',
        )
        assert s.opera_bucket_urls == ["https://a.example/x", "https://b.example/y"]

    def test_empty_bucket_list_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError):
            self._settings(monkeypatch, bucket_urls="[]")

    def test_provider_uses_bucket_list(self, monkeypatch):
        s = self._settings(monkeypatch, base_url="https://mirror.example.org")
        contribution = radar_provider(s)
        assert contribution.instance._bucket_urls == [
            "https://mirror.example.org/openradar-24h",
            ECMWF_BACKUP,
        ]
