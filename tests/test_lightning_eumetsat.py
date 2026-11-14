# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Tests for the EUMETSAT MTG LI lightning source (API-key family).

All HTTP is injected: each source's ``_search_products`` / ``_download``
seams and ``_TokenManager._fetch`` are monkeypatched, so these tests never
touch the EUMETSAT Data Store.  The synthetic netCDF writer mirrors the
verified LI L2 schema (flat h5py, SIGNED int16 latitude/longitude at
0.00275 deg, uint16 radiance with a file-defined ``scale_factor``, epoch
2000-01-01).
"""

import base64
import io
import logging
import time
import zipfile
from datetime import datetime, timezone

import h5py
import numpy as np
import pytest

from librewxr.data.lightning_eumetsat import (
    _EUMETSAT_TOKEN_URL,
    _LI_COLLECTION,
    _LI_EPOCH_FALLBACK,
    _SEARCH_PAGE_SIZE,
    _TOKEN_MARGIN_S,
    MTGLISource,
    _AuthError,
    _TokenManager,
    _parse_feature,
)
from librewxr.data.lightning_fetcher import Watermarks
from librewxr.data.lightning_store import _POINT_DTYPE

pytestmark = pytest.mark.lightning

_LOGGER = "librewxr.data.lightning_eumetsat"

# Real LI L2 constants (verified live 2026-11-13).
_LI_SCALE = 0.00275
_LI_UNITS = "seconds since 2000-01-01 00:00:00.0"
# Arbitrary file-defined radiance scale (the product ships its own attr).
_RAD_SCALE = 1.0e-6

NOW = 1_800_000_000

# Real product identity, copied from a live Data Store probe (2026-11-13).
# Product ids carry commas and ``+``; hrefs percent-encode them.
_LIVE_PRODUCT_ID = (
    "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+LI-2-LFL--FD--x-x--ARC-x"
    "_C_EUMT_20261113092023_L2PF_OPE_20261113091005_20261113092005"
    "_N__O_0056_0000"
)
_LIVE_HREF = (
    "https://api.eumetsat.int/data/download/1.0.0/collections/"
    "EO%3AEUM%3ADAT%3A0691/products/"
    "W_XX-EUMETSAT-Darmstadt%2CIMG%2BSAT%2CMTI1%2BLI-2-LFL--FD--x-x--ARC-x"
    "_C_EUMT_20261113092023_L2PF_OPE_20261113091005_20261113092005"
    "_N__O_0056_0000"
)
# The consolidated product ships a CHK-BODY and a CHK-TRAIL netCDF member
# over the same sensing window; only the BODY member carries the four flash
# variables, while the TRAIL member holds statistics alone.
_BODY_NC_NAME = (
    "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+LI-2-LFL--FD--CHK-BODY--ARC-NC4E"
    "_C_EUMT_20261113092023_L2PF_OPE_20261113091005_20261113092005"
    "_N__O_0056_0000.nc"
)
_BODY2_NC_NAME = (
    "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+LI-2-LFL--FD--CHK-BODY--ARC-NC4E"
    "_C_EUMT_20261113092023_L2PF_OPE_20261113091005_20261113092005"
    "_N__O_0056_0001.nc"
)
_TRAIL_NC_NAME = (
    "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+LI-2-LFL--FD--CHK-TRAIL--ARC-NC4E"
    "_C_EUMT_20261113092023_L2PF_OPE_20261113091005_20261113092005"
    "_N__O_0056_0000.nc"
)


# ---------------------------------------------------------------------------
# Synthetic LI netCDF + ZIP writers
# ---------------------------------------------------------------------------

def _write_li_nc(flashes, base_unix, *, qc=True) -> bytes:
    """Build a flat MTG LI L2 flashes netCDF.

    ``flashes`` is a list of ``(lat, lon, radiance, t_offset_s, warn)``;
    ``base_unix`` is the sensing-window start.
    """
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        ds = f.create_dataset(
            "flash_time",
            data=np.array(
                [base_unix + fl[3] - _LI_EPOCH_FALLBACK for fl in flashes],
                dtype=np.float64,
            ),
        )
        ds.attrs["units"] = _LI_UNITS

        ds = f.create_dataset(
            "latitude",
            data=np.array(
                [int(round(fl[0] / _LI_SCALE)) for fl in flashes],
                dtype=np.int16,
            ),
        )
        ds.attrs["scale_factor"] = np.array([_LI_SCALE])
        ds.attrs["add_offset"] = np.array([0.0])

        ds = f.create_dataset(
            "longitude",
            data=np.array(
                [int(round(fl[1] / _LI_SCALE)) for fl in flashes],
                dtype=np.int16,
            ),
        )
        ds.attrs["scale_factor"] = np.array([_LI_SCALE])
        ds.attrs["add_offset"] = np.array([0.0])

        ds = f.create_dataset(
            "radiance",
            data=np.array(
                [int(round(fl[2] / _RAD_SCALE)) for fl in flashes],
                dtype=np.uint16,
            ),
        )
        ds.attrs["scale_factor"] = np.array([_RAD_SCALE])
        ds.attrs["units"] = "mW.m-2.sr-1"

        if qc:
            f.create_dataset(
                "l1b_radiometric_warning",
                data=np.array([fl[4] for fl in flashes], dtype=np.uint8),
            )
    return bio.getvalue()


def _write_li_trailer_nc() -> bytes:
    """Build a trailer-shaped MTG LI netCDF: valid HDF5, no flash variables.

    Mirrors the real ``...CHK-TRAIL...`` member, which carries sensing-window
    statistics only.  ``h5py`` opens it happily, but it has none of the four
    flash variables, so the decoder's schema check rejects it.
    """
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        f.create_dataset(
            "number_of_flashes", data=np.array([3], dtype=np.uint32),
        )
        f.create_dataset(
            "number_of_events", data=np.array([5], dtype=np.uint32),
        )
        f.attrs["title"] = "MTG LI CHK-TRAIL statistics"
    return bio.getvalue()


def _zip_product(
    nc_bytes: bytes, *, nc_name="product.nc", extra_nc=None,
) -> bytes:
    """Wrap a netCDF buffer in the product ZIP (sidecars first on purpose).

    When ``extra_nc`` is given the ZIP mirrors the real consolidated product:
    two members named ``...CHK-BODY...`` and ``...CHK-TRAIL...`` over the same
    sensing window (real names carry commas and ``+``).
    """
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("manifest.xml", "<manifest/>")
        if extra_nc is not None:
            zf.writestr(_BODY_NC_NAME, nc_bytes)
            zf.writestr(_TRAIL_NC_NAME, extra_nc)
        else:
            zf.writestr(nc_name, nc_bytes)
        zf.writestr("quicklook.jpg", b"\xff\xd8\xff\xd9")
    return bio.getvalue()


def _zip_members(*members) -> bytes:
    """Wrap explicit ``(name, bytes)`` netCDF members in the product ZIP."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("manifest.xml", "<manifest/>")
        for name, payload in members:
            zf.writestr(name, payload)
        zf.writestr("quicklook.jpg", b"\xff\xd8\xff\xd9")
    return bio.getvalue()


def _zip_without_nc() -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("manifest.xml", "<manifest/>")
        zf.writestr("quicklook.jpg", b"\xff\xd8\xff\xd9")
    return bio.getvalue()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _parsed(pid, window_start, window_end) -> dict:
    """A parsed search feature (the ``_search_products`` return shape)."""
    return {
        "product_id": pid,
        "href": f"https://api.eumetsat.int/data/download/{pid}",
        "window_start": float(window_start),
        "window_end": float(window_end),
    }


def _feature(pid, start_iso, end_iso, *, href=None) -> dict:
    """A raw GeoJSON search feature (the Data Store response shape).

    Mirrors the live API (verified 2026-11-13): ``links`` nests under
    ``properties`` and ``links.data`` is a LIST of link dicts.
    """
    return {
        "type": "Feature",
        "id": pid,
        "geometry": {"type": "Point", "coordinates": [0, 0]},
        "properties": {
            "identifier": pid,
            "date": f"{start_iso}/{end_iso}",
            "links": {
                "type": "Links",
                "data": [
                    {
                        "type": "Link",
                        "href": href
                        or f"https://api.eumetsat.int/data/download/{pid}",
                        "mediaType": "application/zip",
                        "title": "Product download",
                    }
                ],
            },
        },
    }


def _make_source(**kwargs) -> MTGLISource:
    kwargs.setdefault("interval_s", 300)
    kwargs.setdefault("max_age_s", 1800)
    kwargs.setdefault("api_key", "key")
    kwargs.setdefault("api_secret", "secret")
    kwargs.setdefault("download_retries", 0)
    return MTGLISource(**kwargs)


def _watermarks(tmp_path) -> Watermarks:
    return Watermarks(tmp_path / "lightning" / "watermark.json")


def _wire(source, monkeypatch, features, download) -> None:
    monkeypatch.setattr(
        source, "_search_products", lambda start, end: list(features),
    )
    monkeypatch.setattr(source, "_download", download)


class _FakeResp:
    def __init__(self, status_code, payload, content=b""):
        self.status_code = status_code
        self._payload = payload
        self.content = content

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


# ---------------------------------------------------------------------------
# 1. Decode: signed coordinates, radiance scaling, epoch
# ---------------------------------------------------------------------------

def test_decode_signed_coords_scaling_and_epoch():
    src = _make_source()
    base = NOW
    # Exact multiples of the 0.00275 deg grid, with NEGATIVE lat AND lon.
    flashes = [
        (27.5, -41.25, 1.5e-4, 5, 0),
        (-27.5, -13.75, 2.25e-4, 8, 0),
    ]
    pts = src._decode_file(_zip_product(_write_li_nc(flashes, base)))

    assert pts.dtype == _POINT_DTYPE
    assert len(pts) == 2

    # SIGNED int16 storage survives decode (no _Unsigned byte flip).
    assert pts[0]["lat"] == np.float32(27.5)
    assert pts[0]["lon"] == np.float32(-41.25)
    assert pts[1]["lat"] == np.float32(-27.5)
    assert pts[1]["lon"] == np.float32(-13.75)

    # Radiance rides the energy slot, scaled by the file's scale_factor.
    assert pts[0]["energy"] == pytest.approx(1.5e-4, rel=1e-6)
    assert pts[1]["energy"] == pytest.approx(2.25e-4, rel=1e-6)

    # Per-flash nominal time via the 2000-01-01 epoch.
    assert int(pts[0]["time_s"]) == base + 5
    assert int(pts[1]["time_s"]) == base + 8

    assert all(int(p["satellite"]) == 12 for p in pts)


def test_decode_qc_filtering():
    src = _make_source()
    flashes = [
        (11.0, 22.0, 1.0e-4, 1, 0),    # good
        (22.0, 33.0, 1.0e-4, 2, 1),    # warning -> dropped
        (33.0, -22.0, 1.0e-4, 3, 255),  # warning -> dropped
    ]
    pts = src._decode_file(_zip_product(_write_li_nc(flashes, NOW)))
    assert len(pts) == 1
    assert pts[0]["lat"] == np.float32(11.0)
    assert int(pts[0]["time_s"]) == NOW + 1


def test_decode_missing_qc_variable_is_tolerated():
    src = _make_source()
    flashes = [
        (11.0, 22.0, 1.0e-4, 1, 1),
        (33.0, -22.0, 1.0e-4, 2, 1),
    ]
    pts = src._decode_file(
        _zip_product(_write_li_nc(flashes, NOW, qc=False))
    )
    assert len(pts) == 2


def test_decode_invalid_coordinates_filtered():
    src = _make_source()
    flashes = [
        (90.0735, 0.0, 1.0e-4, 1, 0),    # > 90 -> dropped
        (-90.0735, 0.0, 1.0e-4, 2, 0),   # < -90 -> dropped
        (11.0, 22.0, 1.0e-4, 3, 0),      # kept
    ]
    pts = src._decode_file(_zip_product(_write_li_nc(flashes, NOW)))
    assert len(pts) == 1
    assert int(pts[0]["time_s"]) == NOW + 3


def test_zip_member_selection_ignores_sidecars():
    src = _make_source()
    nc = _write_li_nc([(11.0, 22.0, 1.0e-4, 4, 0)], NOW)
    pts = src._decode_file(_zip_product(nc, nc_name="LI/flashes.nc"))
    assert len(pts) == 1


def test_decode_no_netcdf_member_raises():
    src = _make_source()
    with pytest.raises(ValueError, match="no netCDF member"):
        src._decode_file(_zip_without_nc())


def test_decode_concatenates_multiple_body_members():
    src = _make_source()
    body_a = _write_li_nc(
        [
            (11.0, 22.0, 1.0e-4, 1, 0),      # flash A
            (13.75, 24.75, 1.1e-4, 2, 0),    # flash B
        ],
        NOW,
    )
    body_b = _write_li_nc([(16.5, 27.5, 1.2e-4, 3, 0)], NOW)  # flash C

    pts = src._decode_file(
        _zip_members((_BODY_NC_NAME, body_a), (_BODY2_NC_NAME, body_b))
    )

    assert len(pts) == 3
    assert [int(p["time_s"]) for p in pts] == [NOW + 1, NOW + 2, NOW + 3]
    assert [float(p["lat"]) for p in pts] == [11.0, 13.75, 16.5]
    assert all(int(p["satellite"]) == 12 for p in pts)


def test_decode_skips_trailer_member(caplog):
    """The statistics-only CHK-TRAIL member is never decoded."""
    src = _make_source()
    body = _write_li_nc([(11.0, 22.0, 1.0e-4, 1, 0)], NOW)
    trailer = _write_li_trailer_nc()

    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        pts = src._decode_file(
            _zip_members(
                (_BODY_NC_NAME, body), (_TRAIL_NC_NAME, trailer),
            )
        )

    assert len(pts) == 1
    assert pts[0]["lat"] == np.float32(11.0)
    assert int(pts[0]["time_s"]) == NOW + 1

    # No decode-failure warning: the trailer is filtered before the attempt.
    assert not [
        r for r in caplog.records
        if r.levelno >= logging.WARNING
        and "decode failed for member" in r.getMessage()
    ]
    # The skip itself is reported (DEBUG only).
    skips = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG
        and "skipping non-body member" in r.getMessage()
        and _TRAIL_NC_NAME in r.getMessage()
    ]
    assert len(skips) == 1


def test_decode_trailer_only_raises(caplog):
    """With no CHK-BODY member the fallback attempts the trailer and fails."""
    src = _make_source()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        with pytest.raises(ValueError, match="no decodable netCDF member"):
            src._decode_file(
                _zip_members((_TRAIL_NC_NAME, _write_li_trailer_nc()))
            )

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert _TRAIL_NC_NAME in warnings[0].getMessage()
    assert "decode failed for member" in warnings[0].getMessage()


def test_decode_corrupt_body_member_keeps_valid_body(caplog):
    src = _make_source()
    good = _write_li_nc([(11.0, 22.0, 1.0e-4, 1, 0)], NOW)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        pts = src._decode_file(
            _zip_members((_BODY_NC_NAME, b"garbage"), (_BODY2_NC_NAME, good))
        )

    assert len(pts) == 1
    assert pts[0]["lat"] == np.float32(11.0)
    assert int(pts[0]["time_s"]) == NOW + 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert _BODY_NC_NAME in warnings[0].getMessage()
    assert "decode failed for member" in warnings[0].getMessage()


def test_decode_all_members_corrupt_raises():
    src = _make_source()
    with pytest.raises(ValueError, match="no decodable netCDF member"):
        src._decode_file(_zip_product(b"garbage", extra_nc=b"garbage"))


# ---------------------------------------------------------------------------
# 2. fetch_cycle: watermark, cutoff, tail, auth, staleness
# ---------------------------------------------------------------------------

def test_undecodable_zip_advances_watermark(tmp_path, monkeypatch):
    src = _make_source()
    wm = _watermarks(tmp_path)
    # A real product id carries commas and ``+``; prove the watermark JSON
    # round-trips it and the status carries it verbatim.
    feat = _parsed(_LIVE_PRODUCT_ID, NOW - 600, NOW - 300)
    _wire(src, monkeypatch, [feat], lambda href: _zip_without_nc())

    pts, status = src.fetch_cycle(NOW, wm)

    assert len(pts) == 0
    assert status["last_key"] == _LIVE_PRODUCT_ID
    assert status["last_e"] == NOW - 300
    assert status["last_fetch_ok"] is True
    entry = wm.get("meteosat12")
    assert entry["last_key"] == _LIVE_PRODUCT_ID
    assert entry["source"] == _LI_COLLECTION
    # Round-trip through a fresh reader (i.e. the on-disk JSON).
    assert (
        _watermarks(tmp_path).get("meteosat12")["last_key"] == _LIVE_PRODUCT_ID
    )


def test_watermark_resume_skips_processed(tmp_path, monkeypatch):
    src = _make_source()
    wm = _watermarks(tmp_path)
    wm.update(
        "meteosat12", source=_LI_COLLECTION, last_key="p1", last_e=NOW - 900,
    )
    f1 = _parsed("p1", NOW - 1200, NOW - 900)
    f2 = _parsed("p2", NOW - 600, NOW - 300)
    payload = _zip_product(
        _write_li_nc([(11.0, 22.0, 1.0e-4, 1, 0)], NOW - 300)
    )
    calls = []

    def download(href):
        calls.append(href)
        return payload

    _wire(src, monkeypatch, [f1, f2], download)

    pts, status = src.fetch_cycle(NOW, wm)

    assert calls == [f2["href"]]
    assert len(pts) == 1
    assert status["last_key"] == "p2"


def test_cold_start_skips_pre_cutoff(tmp_path, monkeypatch):
    src = _make_source()
    wm = _watermarks(tmp_path)
    old = _parsed("old", NOW - 5000, NOW - 4980)
    new = _parsed("new", NOW - 600, NOW - 300)
    payload = _zip_product(
        _write_li_nc([(11.0, 22.0, 1.0e-4, 1, 0)], NOW - 300)
    )
    calls = []

    def download(href):
        calls.append(href)
        return payload

    _wire(src, monkeypatch, [old, new], download)

    pts, _status = src.fetch_cycle(NOW, wm)

    assert calls == [new["href"]]
    assert len(pts) == 1


def test_transient_miss_stops_tail_and_retries(tmp_path, monkeypatch):
    src = _make_source()
    wm = _watermarks(tmp_path)
    f1 = _parsed("p1", NOW - 1200, NOW - 900)
    f2 = _parsed("p2", NOW - 600, NOW - 300)
    f3 = _parsed("p3", NOW - 300, NOW)
    payloads = {
        f["href"]: _zip_product(
            _write_li_nc([(11.0, 22.0, 1.0e-4, 1, 0)], f["window_end"])
        )
        for f in (f1, f2, f3)
    }
    state = {"cycle": 0}
    attempts = []

    def download(href):
        attempts.append((state["cycle"], href))
        if state["cycle"] == 0 and href == f2["href"]:
            return None
        return payloads[href]

    _wire(src, monkeypatch, [f1, f2, f3], download)

    pts, status = src.fetch_cycle(NOW, wm)
    assert len(pts) == 1
    assert status["last_key"] == "p1"
    assert (0, f3["href"]) not in attempts
    assert wm.get("meteosat12")["last_key"] == "p1"

    state["cycle"] = 1
    pts, status = src.fetch_cycle(NOW, wm)
    assert len(pts) == 2
    assert (1, f1["href"]) not in attempts
    assert (1, f2["href"]) in attempts
    assert (1, f3["href"]) in attempts
    assert status["last_key"] == "p3"


def test_auth_error_retries_once_then_succeeds(tmp_path, monkeypatch):
    src = _make_source()
    wm = _watermarks(tmp_path)
    feat = _parsed("p1", NOW - 600, NOW - 300)
    payload = _zip_product(
        _write_li_nc([(11.0, 22.0, 1.0e-4, 1, 0)], NOW - 300)
    )
    state = {"downloads": 0, "invalidated": 0}

    def download(href):
        state["downloads"] += 1
        if state["downloads"] == 1:
            raise _AuthError("401")
        return payload

    _wire(src, monkeypatch, [feat], download)
    monkeypatch.setattr(
        src._tokens,
        "invalidate",
        lambda: state.__setitem__("invalidated", state["invalidated"] + 1),
    )

    pts, status = src.fetch_cycle(NOW, wm)

    assert len(pts) == 1
    assert status["last_fetch_ok"] is True
    assert state["downloads"] == 2
    assert state["invalidated"] == 1


def test_persistent_auth_failure_keeps_earlier_points(
    tmp_path, monkeypatch, caplog,
):
    src = _make_source()
    wm = _watermarks(tmp_path)
    f1 = _parsed("p1", NOW - 1200, NOW - 900)
    f2 = _parsed("p2", NOW - 600, NOW - 300)
    payload = _zip_product(
        _write_li_nc([(11.0, 22.0, 1.0e-4, 1, 0)], NOW - 900)
    )

    def download(href):
        if href == f1["href"]:
            return payload
        raise _AuthError("403")

    _wire(src, monkeypatch, [f1, f2], download)
    monkeypatch.setattr(src._tokens, "invalidate", lambda: None)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        pts, status = src.fetch_cycle(NOW, wm)

    assert len(pts) == 1
    assert status["last_fetch_ok"] is False
    assert status["last_key"] == "p1"
    assert wm.get("meteosat12")["last_key"] == "p1"
    assert any(
        "LIBREWXR_LIGHTNING_EUMETSAT_API_KEY" in r.getMessage()
        for r in caplog.records
    )


def test_stale_empty_search_with_watermark(tmp_path, monkeypatch, caplog):
    src = _make_source()
    wm = _watermarks(tmp_path)
    wm.update(
        "meteosat12", source=_LI_COLLECTION, last_key="old", last_e=NOW - 4000,
    )
    _wire(src, monkeypatch, [], lambda href: None)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        pts, status = src.fetch_cycle(NOW, wm)

    assert status["stale"] is True
    assert len(pts) == 0
    assert any(
        "feed appears stale" in r.getMessage() for r in caplog.records
    )


def test_stale_ancient_window_end(tmp_path, monkeypatch):
    src = _make_source()
    wm = _watermarks(tmp_path)
    feat = _parsed("p-old", NOW - 4000, NOW - 1200)  # end < NOW - 900
    _wire(src, monkeypatch, [feat], lambda href: None)

    _pts, status = src.fetch_cycle(NOW, wm)

    assert status["stale"] is True


def test_empty_search_no_watermark_not_stale(tmp_path, monkeypatch):
    src = _make_source()
    wm = _watermarks(tmp_path)
    _wire(src, monkeypatch, [], lambda href: None)

    pts, status = src.fetch_cycle(NOW, wm)

    assert status["stale"] is False
    assert status["last_fetch_ok"] is True
    assert len(pts) == 0


def test_duplicate_product_ids_processed_once(tmp_path, monkeypatch):
    src = _make_source()
    wm = _watermarks(tmp_path)
    dup_a = _parsed("dup", NOW - 900, NOW - 600)
    dup_b = _parsed("dup", NOW - 900, NOW - 600)
    payload = _zip_product(
        _write_li_nc([(11.0, 22.0, 1.0e-4, 1, 0)], NOW - 600)
    )
    calls = []

    def download(href):
        calls.append(href)
        return payload

    _wire(src, monkeypatch, [dup_a, dup_b], download)

    pts, status = src.fetch_cycle(NOW, wm)

    assert len(calls) == 1
    assert len(pts) == 1
    assert status["last_key"] == "dup"


def test_status_dict_shape(tmp_path, monkeypatch):
    src = _make_source()
    _wire(src, monkeypatch, [], lambda href: None)

    pts, status = src.fetch_cycle(NOW, _watermarks(tmp_path))

    assert pts.dtype == _POINT_DTYPE
    assert set(status) == {
        "source", "units", "last_key", "last_e", "stale", "last_fetch_ok",
    }
    assert status["source"] == _LI_COLLECTION
    assert status["units"] == "mW.m-2.sr-1"
    assert status["last_key"] is None
    assert status["last_e"] is None


def test_fetch_cycle_swallows_exceptions(tmp_path, monkeypatch, caplog):
    src = _make_source()

    def boom(start, end):
        raise RuntimeError("boom")

    monkeypatch.setattr(src, "_search_products", boom)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        pts, status = src.fetch_cycle(NOW, _watermarks(tmp_path))

    assert len(pts) == 0
    assert status["last_fetch_ok"] is False
    assert any(
        "fetch cycle failed" in r.getMessage() for r in caplog.records
    )


# ---------------------------------------------------------------------------
# 3. Search parsing / pagination
# ---------------------------------------------------------------------------

class _FakeClient:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def get(self, url, params=None, headers=None):
        self.calls.append(dict(params or {}))
        page = (int(params["si"]) - 1) // _SEARCH_PAGE_SIZE
        return _FakeResp(200, self.pages[page])


def test_search_products_paginates_and_parses(monkeypatch):
    src = _make_source()
    page0 = {
        "totalResults": 3,
        "features": [
            _feature(
                "p1", "2026-11-13T10:00:00Z", "2026-11-13T10:10:00Z",
                href="https://x/p1",
            ),
            _feature(
                "p2", "2026-11-13T10:10:00Z", "2026-11-13T10:20:00Z",
                href="https://x/p2",
            ),
        ],
    }
    page1 = {
        "totalResults": 3,
        "features": [
            _feature(
                "p3", "2026-11-13T10:20:00Z", "2026-11-13T10:30:00Z",
                href="https://x/p3",
            ),
        ],
    }
    client = _FakeClient([page0, page1])
    monkeypatch.setattr(src, "_get_client", lambda: client)

    features = src._search_products(
        "2026-11-13T09:00:00Z", "2026-11-13T11:00:00Z",
    )

    assert [f["product_id"] for f in features] == ["p1", "p2", "p3"]
    assert features[0]["href"] == "https://x/p1"
    assert len(client.calls) == 2
    assert client.calls[0]["pi"] == _LI_COLLECTION
    assert client.calls[0]["si"] == 1
    assert client.calls[1]["si"] == _SEARCH_PAGE_SIZE + 1


def test_search_parses_live_feature_shape():
    """A raw feature copied near-verbatim from the live Data Store probe."""
    feature = {
        "type": "Feature",
        "id": _LIVE_PRODUCT_ID,
        "geometry": {"type": "Point", "coordinates": [0, 0]},
        "properties": {
            "identifier": _LIVE_PRODUCT_ID,
            "date": "2026-11-13T09:10:05Z/2026-11-13T09:20:05Z",
            "links": {
                "type": "Links",
                "data": [
                    {
                        "type": "Link",
                        "href": _LIVE_HREF,
                        "mediaType": "application/zip",
                        "title": "Product download",
                    }
                ],
            },
        },
    }

    item = _parse_feature(feature)

    assert item is not None
    assert item["product_id"] == _LIVE_PRODUCT_ID
    assert item["href"] == _LIVE_HREF
    assert item["window_start"] == datetime(
        2026, 11, 13, 9, 10, 5, tzinfo=timezone.utc
    ).timestamp()
    assert item["window_end"] == datetime(
        2026, 11, 13, 9, 20, 5, tzinfo=timezone.utc
    ).timestamp()


def test_feature_top_level_links_fallback(monkeypatch):
    """The defensive top-level ``links`` shape is still accepted."""
    src = _make_source()
    page = {
        "totalResults": 2,
        "features": [
            {
                "properties": {
                    "date": "2026-11-13T10:00:00Z/2026-11-13T10:10:00Z",
                },
                "id": "feat-id",
                "links": {"data": {"href": "https://x/a"}},
            },
            {
                "properties": {
                    "date": "2026-11-13T10:10:00Z/2026-11-13T10:20:00Z",
                },
                "links": {"data": {"href": "https://x/weird%20name.nc"}},
            },
        ],
    }
    monkeypatch.setattr(src, "_get_client", lambda: _FakeClient([page]))

    features = src._search_products("x", "y")

    assert [f["product_id"] for f in features] == [
        "feat-id", "weird name.nc",
    ]


def test_search_failure_returns_empty(monkeypatch):
    src = _make_source()
    client = _FakeClient([{"totalResults": 0, "features": []}])

    def boom(url, params=None, headers=None):
        raise RuntimeError("network down")

    client.get = boom
    monkeypatch.setattr(src, "_get_client", lambda: client)

    assert src._search_products("x", "y") == []


# ---------------------------------------------------------------------------
# 4. Download status handling
# ---------------------------------------------------------------------------

def test_download_status_handling(monkeypatch):
    src = _make_source()
    monkeypatch.setattr(src._tokens, "get", lambda: "tok")
    seen = {}

    class _Client:
        def __init__(self, code, content=b""):
            self.code = code
            self.content = content

        def get(self, url, headers=None):
            seen["headers"] = headers
            return _FakeResp(self.code, None, self.content)

    monkeypatch.setattr(src, "_get_client", lambda: _Client(200, b"zip"))
    assert src._download("https://x/a") == b"zip"
    assert seen["headers"]["Authorization"] == "Bearer tok"

    monkeypatch.setattr(src, "_get_client", lambda: _Client(404))
    assert src._download("https://x/a") is None

    monkeypatch.setattr(src, "_get_client", lambda: _Client(401))
    with pytest.raises(_AuthError):
        src._download("https://x/a")

    monkeypatch.setattr(src, "_get_client", lambda: _Client(403))
    with pytest.raises(_AuthError):
        src._download("https://x/a")

    monkeypatch.setattr(src, "_get_client", lambda: _Client(500))
    assert src._download("https://x/a") is None


# ---------------------------------------------------------------------------
# 5. Token manager
# ---------------------------------------------------------------------------

def test_token_manager_caches_and_invalidates(monkeypatch):
    tm = _TokenManager("key", "secret")
    calls = []

    def fake_fetch():
        calls.append(1)
        return f"tok{len(calls)}", time.time() + 3600

    monkeypatch.setattr(tm, "_fetch", fake_fetch)

    assert tm.get() == "tok1"
    assert tm.get() == "tok1"
    assert len(calls) == 1

    tm.invalidate()
    assert tm.get() == "tok2"
    assert len(calls) == 2


def test_token_manager_refreshes_within_margin(monkeypatch):
    tm = _TokenManager("key", "secret")
    calls = []

    def fake_fetch():
        calls.append(1)
        return f"tok{len(calls)}", time.time() + _TOKEN_MARGIN_S - 10

    monkeypatch.setattr(tm, "_fetch", fake_fetch)

    assert tm.get() == "tok1"
    assert tm.get() == "tok2"  # inside the refresh margin
    assert len(calls) == 2


def test_token_fetch_basic_auth_and_form(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _FakeResp(200, {"access_token": "abc", "expires_in": 28800})

    monkeypatch.setattr(
        "librewxr.data.lightning_eumetsat.httpx.post", fake_post,
    )

    token, expires_at = _TokenManager("key", "secret")._fetch()

    assert token == "abc"
    assert captured["url"] == _EUMETSAT_TOKEN_URL
    assert captured["data"] == {"grant_type": "client_credentials"}
    expected = base64.b64encode(b"key:secret").decode("ascii")
    assert captured["headers"]["Authorization"] == f"Basic {expected}"
    assert expires_at > time.time()


def test_token_fetch_failure_raises(monkeypatch):
    monkeypatch.setattr(
        "librewxr.data.lightning_eumetsat.httpx.post",
        lambda url, **kwargs: _FakeResp(401, None),
    )
    with pytest.raises(_AuthError):
        _TokenManager("key", "secret")._fetch()
