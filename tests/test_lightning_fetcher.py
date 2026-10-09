# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for the GOES GLM lightning fetcher (Phase 1, pipeline side).

All HTTP is injected: the fetcher's ``_list_keys`` / ``_download`` /
``_get_listing_page`` seams are monkeypatched, so these tests never touch
real S3.  The synthetic netCDF writer mirrors the verified real GLM schema
(flat h5py, int16 ``_Unsigned`` storage with scale/offset attrs).
"""

import asyncio
import io
import json
import logging
from datetime import datetime, timezone

import h5py
import numpy as np
import pytest
from urllib.parse import quote

from librewxr.data.lightning_fetcher import (
    GOESGLMLightningFetcher,
    _key_window_epochs,
)
from librewxr.data.lightning_store import LightningStore, _POINT_DTYPE

pytestmark = pytest.mark.lightning

_LOGGER = "librewxr.data.lightning_fetcher"

# Real GLM scale/offset constants (verified live 2026-10-09).
_E_SCALE = 9.999959802209943e-16
_E_ADD = 2.8514998886357517e-16
_OFF_SCALE = 0.00038147560553625226
_OFF_ADD = -5.0
# Real GLM units string + the epoch it denotes (verified against a live
# 2026-10-09 file: product_time 844812660 + this epoch == the filename's
# 2026-10-09 10:11:00Z window start).
_UNITS = "seconds since 2000-01-01 12:00:00"
_EPOCH = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()  # 946728000.0

SAT = ("goes19", "noaa-goes19", 19)
NOW = 1_800_000_000


# ---------------------------------------------------------------------------
# Synthetic GLM netCDF writer (mirrors the real schema exactly)
# ---------------------------------------------------------------------------

def _pack16(values, scale, add) -> np.ndarray:
    raw = np.round((np.asarray(values, dtype=np.float64) - add) / scale)
    return np.clip(raw, 0, 65535).astype(np.uint16)


def _write_glm_nc(flashes, base_unix) -> bytes:
    """Build a flat GLM L2 LCFA netCDF.

    ``flashes`` is a list of ``(lat, lon, energy_J, first_offset_s, quality)``.
    ``base_unix`` is the window start (product_time).
    """
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        f.create_dataset(
            "flash_lat", data=np.array([fl[0] for fl in flashes], dtype=np.float32)
        )
        f.create_dataset(
            "flash_lon", data=np.array([fl[1] for fl in flashes], dtype=np.float32)
        )

        energy = _pack16([fl[2] for fl in flashes], _E_SCALE, _E_ADD)
        ds = f.create_dataset("flash_energy", data=energy.view(np.int16))
        ds.attrs["scale_factor"] = np.array([_E_SCALE])
        ds.attrs["add_offset"] = np.array([_E_ADD])
        ds.attrs["_Unsigned"] = "true"
        ds.attrs["_FillValue"] = -1
        ds.attrs["units"] = "J"

        offsets = _pack16([fl[3] for fl in flashes], _OFF_SCALE, _OFF_ADD)
        ds = f.create_dataset(
            "flash_time_offset_of_first_event", data=offsets.view(np.int16)
        )
        ds.attrs["scale_factor"] = np.array([_OFF_SCALE])
        ds.attrs["add_offset"] = np.array([_OFF_ADD])
        ds.attrs["_Unsigned"] = "true"
        ds.attrs["units"] = "s"

        quality = np.array([fl[4] for fl in flashes], dtype=np.int64) & 0xFFFF
        ds = f.create_dataset("flash_quality_flag", data=quality.astype(np.uint16).view(np.int16))
        ds.attrs["_Unsigned"] = "true"
        ds.attrs["_FillValue"] = -1
        ds.attrs["flag_values"] = np.array([0, 1, 3, 5], dtype=np.int16)

        pt = f.create_dataset(
            "product_time", data=np.float64(base_unix - _EPOCH)
        )
        pt.attrs["units"] = _UNITS
        f.create_dataset(
            "product_time_bounds",
            data=np.array([base_unix - _EPOCH, base_unix - _EPOCH + 20.0]),
        )
    return bio.getvalue()


def _stamp(ts: int) -> str:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y%j%H%M%S") + "0"


def _key(sat: int, start: int, end: int) -> str:
    return (
        f"OR_GLM-L2-LCFA_G{sat}_s{_stamp(start)}_e{_stamp(end)}"
        f"_c{_stamp(end)}.nc"
    )


def _make_fetcher(tmp_path, **kwargs) -> GOESGLMLightningFetcher:
    store = LightningStore(tmp_path)
    defaults = dict(cache_dir=tmp_path, interval_s=300, max_age_s=1800, satellites=(SAT,))
    defaults.update(kwargs)
    return GOESGLMLightningFetcher(store, **defaults)


# ---------------------------------------------------------------------------
# 1. Key window parsing
# ---------------------------------------------------------------------------

def test_key_window_epochs():
    key = (
        "GLM-L2-LCFA/2026/282/10/"
        "OR_GLM-L2-LCFA_G19_s20262821011000_e20262821011200_c20262821011223.nc"
    )
    assert _key_window_epochs(key) == (1791540660.0, 1791540680.0)
    assert _key_window_epochs("not-a-glm-key.nc") is None
    assert _key_window_epochs("OR_GLM-L2-LCFA_G19_c20262821011223.nc") is None


# ---------------------------------------------------------------------------
# 2. Decode + QC
# ---------------------------------------------------------------------------

def test_decode_qc(tmp_path):
    base = NOW
    flashes = [
        (30.0, -80.0, 5e-14, 3.0, 0),        # good
        (31.0, -81.0, 5e-14, 3.0, 3),        # flagged -> dropped
        (32.0, -82.0, 5e-14, 3.0, -1),       # fill -> dropped
        (float("nan"), -83.0, 5e-14, 3.0, 0),  # NaN lat -> dropped
    ]
    data = _write_glm_nc(flashes, base)
    fetcher = _make_fetcher(tmp_path)

    pts = fetcher._decode_file(data, 19)
    assert pts.dtype == _POINT_DTYPE
    assert len(pts) == 1

    p = pts[0]
    assert p["lat"] == np.float32(30.0)
    assert p["lon"] == np.float32(-80.0)
    assert p["satellite"] == 19

    raw_e = _pack16([5e-14], _E_SCALE, _E_ADD)[0]
    expected_energy = np.float32(raw_e * _E_SCALE + _E_ADD)
    assert p["energy"] == expected_energy

    raw_off = _pack16([3.0], _OFF_SCALE, _OFF_ADD)[0]
    expected_off = raw_off * _OFF_SCALE + _OFF_ADD
    assert p["time_s"] == int(base + expected_off)


# ---------------------------------------------------------------------------
# 3. Watermark resume + no re-download
# ---------------------------------------------------------------------------

async def test_watermark_resume_no_redownload(tmp_path, monkeypatch):
    fetcher = _make_fetcher(tmp_path)
    monkeypatch.setattr(fetcher, "_now_s", lambda: NOW)

    w1, w2, w3 = NOW - 600, NOW - 400, NOW - 200
    k1, k2, k3 = _key(19, w1, w1 + 20), _key(19, w2, w2 + 20), _key(19, w3, w3 + 20)
    data = {
        k1: _write_glm_nc([(10.0, -80.0, 5e-14, 1.0, 0)], w1),
        k2: _write_glm_nc([(11.0, -81.0, 5e-14, 1.0, 0)], w2),
        k3: _write_glm_nc([(12.0, -82.0, 5e-14, 1.0, 0)], w3),
    }
    listings = iter([[k1, k2], [k2, k3]])
    calls = {k1: 0, k2: 0, k3: 0}

    monkeypatch.setattr(fetcher, "_list_keys", lambda b, p, s: next(listings))

    def fake_download(bucket, key):
        calls[key] += 1
        return data[key]

    monkeypatch.setattr(fetcher, "_download", fake_download)

    await fetcher._fetch_once()
    await fetcher._fetch_once()

    assert calls == {k1: 1, k2: 1, k3: 1}
    assert fetcher._store.total_count == 3
    wm = json.loads((tmp_path / "lightning" / "watermark.json").read_text())
    assert wm["goes19"]["last_key"] == k3


# ---------------------------------------------------------------------------
# 4. Transient download miss stops the tail
# ---------------------------------------------------------------------------

async def test_transient_miss_stops_tail(tmp_path, monkeypatch):
    fetcher = _make_fetcher(tmp_path)
    monkeypatch.setattr(fetcher, "_now_s", lambda: NOW)

    w1, w2, w3 = NOW - 600, NOW - 400, NOW - 200
    k1, k2, k3 = _key(19, w1, w1 + 20), _key(19, w2, w2 + 20), _key(19, w3, w3 + 20)
    data = {
        k1: _write_glm_nc([(10.0, -80.0, 5e-14, 1.0, 0)], w1),
        k2: _write_glm_nc([(11.0, -81.0, 5e-14, 1.0, 0)], w2),
        k3: _write_glm_nc([(12.0, -82.0, 5e-14, 1.0, 0)], w3),
    }
    state = {"cycle": 0}
    attempts = []

    monkeypatch.setattr(fetcher, "_list_keys", lambda b, p, s: [k1, k2, k3])

    def fake_download(bucket, key):
        attempts.append((state["cycle"], key))
        if state["cycle"] == 0 and key == k2:
            return None
        return data[key]

    monkeypatch.setattr(fetcher, "_download", fake_download)

    await fetcher._fetch_once()
    assert (0, k3) not in attempts
    wm = json.loads((tmp_path / "lightning" / "watermark.json").read_text())
    assert wm["goes19"]["last_key"] == k1

    state["cycle"] = 1
    await fetcher._fetch_once()
    assert (1, k2) in attempts
    assert (1, k3) in attempts
    assert (1, k1) not in attempts  # already processed, skipped by watermark
    wm = json.loads((tmp_path / "lightning" / "watermark.json").read_text())
    assert wm["goes19"]["last_key"] == k3
    assert fetcher._store.total_count == 3


# ---------------------------------------------------------------------------
# 5. Cold-start cutoff
# ---------------------------------------------------------------------------

async def test_cold_start_cutoff(tmp_path, monkeypatch):
    fetcher = _make_fetcher(tmp_path)
    monkeypatch.setattr(fetcher, "_now_s", lambda: NOW)

    w_old = NOW - 5000   # window end NOW-4980 < cutoff (NOW-2400)
    w_mid = NOW - 1700   # window end NOW-1680 >= cutoff (and survives max-age)
    w_new = NOW - 100
    k_old, k_mid, k_new = (
        _key(19, w_old, w_old + 20),
        _key(19, w_mid, w_mid + 20),
        _key(19, w_new, w_new + 20),
    )
    data = {
        k_old: _write_glm_nc([(1.0, -80.0, 5e-14, 1.0, 0)], w_old),
        k_mid: _write_glm_nc([(2.0, -80.0, 5e-14, 1.0, 0)], w_mid),
        k_new: _write_glm_nc([(3.0, -80.0, 5e-14, 1.0, 0)], w_new),
    }
    downloaded = []

    monkeypatch.setattr(fetcher, "_list_keys", lambda b, p, s: [k_old, k_mid, k_new])

    def fake_download(bucket, key):
        downloaded.append(key)
        return data[key]

    monkeypatch.setattr(fetcher, "_download", fake_download)

    await fetcher._fetch_once()
    assert k_old not in downloaded
    assert set(downloaded) == {k_mid, k_new}
    assert fetcher._store.total_count == 2


# ---------------------------------------------------------------------------
# 6. Pagination
# ---------------------------------------------------------------------------

def test_list_keys_pagination(tmp_path, monkeypatch):
    fetcher = _make_fetcher(tmp_path)
    k1 = _key(19, NOW - 60, NOW - 40)
    k2 = _key(19, NOW - 40, NOW - 20)
    k3 = _key(19, NOW - 20, NOW)
    page1 = (
        "<ListBucketResult>"
        f"<Contents><Key>{k1}</Key></Contents>"
        f"<Contents><Key>{k2}</Key></Contents>"
        "<IsTruncated>true</IsTruncated>"
        "</ListBucketResult>"
    ).encode()
    page2 = (
        "<ListBucketResult>"
        f"<Contents><Key>{k3}</Key></Contents>"
        "<IsTruncated>false</IsTruncated>"
        "</ListBucketResult>"
    ).encode()

    urls = []
    pages = iter([page1, page2])

    def fake_page(url):
        urls.append(url)
        return next(pages)

    monkeypatch.setattr(fetcher, "_get_listing_page", fake_page)

    keys = fetcher._list_keys("noaa-goes19", "GLM-L2-LCFA/2026/282/10/", None)
    assert keys == [k1, k2, k3]
    assert len(urls) == 2
    assert "start-after=" + quote(k2) in urls[1]


# ---------------------------------------------------------------------------
# 7. Staleness warn-once + recovery
# ---------------------------------------------------------------------------

async def test_staleness_warn_once_and_recovery(tmp_path, monkeypatch, caplog):
    fetcher = _make_fetcher(tmp_path)
    monkeypatch.setattr(fetcher, "_now_s", lambda: NOW)

    stale_start = NOW - 4 * 300  # newest end NOW-1180 < NOW-900
    fresh_start = NOW - 100
    k_stale = _key(19, stale_start, stale_start + 20)
    k_fresh = _key(19, fresh_start, fresh_start + 20)
    data = {
        k_stale: _write_glm_nc([(1.0, -80.0, 5e-14, 1.0, 0)], stale_start),
        k_fresh: _write_glm_nc([(2.0, -80.0, 5e-14, 1.0, 0)], fresh_start),
    }
    state = {"keys": [k_stale]}
    monkeypatch.setattr(fetcher, "_list_keys", lambda b, p, s: list(state["keys"]))
    monkeypatch.setattr(fetcher, "_download", lambda b, k: data[k])

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await fetcher._fetch_once()
        assert fetcher._store.meta["satellites"]["goes19"]["stale"] is True
        await fetcher._fetch_once()  # latch: no second warning

    stale_warns = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "stale" in r.getMessage()
    ]
    assert len(stale_warns) == 1

    caplog.clear()
    state["keys"] = [k_fresh]
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await fetcher._fetch_once()
    assert fetcher._store.meta["satellites"]["goes19"]["stale"] is False
    assert not [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "stale" in r.getMessage()
    ]


# ---------------------------------------------------------------------------
# 8. Merge + age-out across quiet cycles
# ---------------------------------------------------------------------------

async def test_merge_age_out_quiet_cycle(tmp_path, monkeypatch):
    fetcher = _make_fetcher(tmp_path)
    clock = {"now": NOW}
    monkeypatch.setattr(fetcher, "_now_s", lambda: clock["now"])

    w_old = NOW - 1700  # ages out once "now" advances past the max age
    w_new = NOW
    k_old, k_new = _key(19, w_old, w_old + 20), _key(19, w_new, w_new + 20)
    data = {
        k_old: _write_glm_nc([(5.0, -80.0, 5e-14, 1.0, 0)], w_old),
        k_new: _write_glm_nc([(6.0, -81.0, 5e-14, 1.0, 0)], w_new),
    }
    listings = iter([[k_old, k_new], []])
    monkeypatch.setattr(fetcher, "_list_keys", lambda b, p, s: next(listings))
    monkeypatch.setattr(fetcher, "_download", lambda b, k: data[k])

    await fetcher._fetch_once()
    assert fetcher._store.total_count == 2
    version_after_first = fetcher._store.version

    clock["now"] = NOW + 200
    await fetcher._fetch_once()

    assert fetcher._store.total_count == 1
    assert fetcher._store.version > version_after_first
    assert fetcher._store.window_start_s >= (NOW + 200) - 1800
    assert fetcher._store.window_start_s == NOW

    reader = LightningStore(tmp_path)
    assert await reader.maybe_reload() is True
    assert reader.total_count == 1


# ---------------------------------------------------------------------------
# 9. Corrupt download skipped, tail continues
# ---------------------------------------------------------------------------

async def test_corrupt_download_skipped(tmp_path, monkeypatch, caplog):
    fetcher = _make_fetcher(tmp_path)
    monkeypatch.setattr(fetcher, "_now_s", lambda: NOW)

    w1, w2, w3 = NOW - 600, NOW - 400, NOW - 200
    k1, k2, k3 = _key(19, w1, w1 + 20), _key(19, w2, w2 + 20), _key(19, w3, w3 + 20)
    data = {
        k1: _write_glm_nc([(10.0, -80.0, 5e-14, 1.0, 0)], w1),
        k3: _write_glm_nc([(12.0, -82.0, 5e-14, 1.0, 0)], w3),
    }
    monkeypatch.setattr(fetcher, "_list_keys", lambda b, p, s: [k1, k2, k3])

    def fake_download(bucket, key):
        if key == k2:
            return b"not netcdf"
        return data[key]

    monkeypatch.setattr(fetcher, "_download", fake_download)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await fetcher._fetch_once()

    assert fetcher._store.total_count == 2
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    wm = json.loads((tmp_path / "lightning" / "watermark.json").read_text())
    assert wm["goes19"]["last_key"] == k3


# ---------------------------------------------------------------------------
# 10. start / close smoke
# ---------------------------------------------------------------------------

async def test_start_close_smoke(tmp_path, monkeypatch):
    fetcher = _make_fetcher(tmp_path)
    monkeypatch.setattr(fetcher, "_list_keys", lambda b, p, s: [])

    await fetcher.start()
    await asyncio.sleep(0)
    await fetcher.close()
    assert fetcher._task is None

    # close is idempotent
    await fetcher.close()
