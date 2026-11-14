# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""OPERA pan-European CIRRUS composite source.

Downloads the CIRRUS MAX reflectivity composite (DBZH) as ODIM HDF5
directly from the MeteoGate S3 buckets.  Rolling 24-hour archive,
5-minute cadence.

Object key (identical in every bucket):
    YYYY/MM/DD/OPERA/COMP/OPERA@YYYYMMDDTHHMM@0@DBZH.h5
Buckets (``LIBREWXR_OPERA_BUCKET_URLS``, tried in order):
    https://s3.waw3-1.cloudferro.com/openradar-24h                    (primary)
    https://object-store.os-api.cci1.ecmwf.int/openradar-24h-backup   (backup)

EUMETNET moves Open Radar Data to the ECMWF backup bucket while the
Cloudferro one is unavailable.  The source keeps using the bucket that
last served data and only falls through to the next one when every
look-back slot is missing (404, or 403 from a bucket that hides
missing keys), the server errors (5xx) or the request fails at the
transport level.

License: data is published under the EUMETNET OPERA data policy
(open, gratis, anonymous).  See README and ``docs/coverage.md`` for the
attribution block.
"""
import io
import logging
import time
from collections.abc import Sequence
from datetime import datetime, timezone

import h5py
import httpx
import numpy as np

from librewxr.data.regions import RegionDef
from librewxr.data.retry import retry_get
from librewxr.sources._helpers import HDF5_LOCK, _dbz_float_to_uint8

logger = logging.getLogger(__name__)

# Returned by ``_fetch_from_bucket`` when the bucket had nothing for the
# requested slots or could not serve them, so the caller should try the
# next bucket.
_NEXT_BUCKET = object()


class OperaSource:
    """OPERA pan-European radar composite from MeteoGate S3.

    Downloads the CIRRUS MAX reflectivity composite (DBZH) as ODIM HDF5
    from an ordered list of S3 buckets (primary + backup).  Rolling
    24-hour archive, 5-minute cadence.
    """

    # OPERA files are published with a ~5-10 minute delay; try up to
    # 3 older 5-minute slots if the target timestamp 404s.
    _MAX_FALLBACK_STEPS = 3

    def __init__(self, bucket_urls: Sequence[str]):
        self._bucket_urls = [u.rstrip("/") for u in bucket_urls]
        # Index of the bucket that served the last successful fetch.  It is
        # tried first, so a healthy cycle costs no extra 404s on a bucket
        # that has gone quiet.
        self._active = 0
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(90.0, connect=15.0),
                follow_redirects=True,
            )
        return self._client

    def _url_for_timestamp(self, ts: int, bucket_url: str) -> str:
        """Build S3 URL for a unix timestamp (rounded to 5-min cadence)."""
        rounded = (ts // 300) * 300
        dt = datetime.fromtimestamp(rounded, tz=timezone.utc)
        fname = dt.strftime("OPERA@%Y%m%dT%H%M@0@DBZH.h5")
        path = dt.strftime(f"%Y/%m/%d/OPERA/COMP/{fname}")
        return f"{bucket_url}/{path}"

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int
    ) -> np.ndarray | None:
        # OPERA composites are published every 5 minutes; round to nearest
        # 5-min slot so the fallback chain finds the right file.
        now_rounded = int(time.time() // 300) * 300
        target_ts = now_rounded - minutes_ago * 60
        # Snap to the nearest 5-min slot for the target as well
        target_ts = int(target_ts // 300) * 300
        return await self._fetch_hdf5(target_ts)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime
    ) -> np.ndarray | None:
        return await self._fetch_hdf5(int(dt.timestamp()))

    async def _fetch_hdf5(self, ts: int) -> np.ndarray | None:
        """Download and parse, trying each bucket in turn.

        The bucket that last served data goes first, then the others in
        configured order.  The next bucket is tried only when the current
        one has none of the look-back slots, returns a server error or
        fails at the transport level.
        """
        client = await self._get_client()
        first = self._active
        order = [first] + [
            i for i in range(len(self._bucket_urls)) if i != first
        ]
        for idx in order:
            result = await self._fetch_from_bucket(
                client, self._bucket_urls[idx], ts,
            )
            if result is _NEXT_BUCKET:
                continue
            if result is not None:
                self._set_active(idx)
            return result
        logger.warning(
            "OPERA fetch failed: no bucket served %s or the %d slots before it",
            self._url_for_timestamp(ts, "").split("/")[-1],
            self._MAX_FALLBACK_STEPS,
        )
        return None

    async def _fetch_from_bucket(
        self, client: httpx.AsyncClient, bucket_url: str, ts: int,
    ):
        """Fetch one bucket, falling back to older slots on a missing key.

        Returns the parsed frame, ``None`` on any other HTTP error or a
        parse failure, or ``_NEXT_BUCKET`` when every slot is missing, the
        server errors (5xx) or the request fails at the transport level.
        """
        for step in range(self._MAX_FALLBACK_STEPS + 1):
            url = self._url_for_timestamp(ts - step * 300, bucket_url)
            resp = await retry_get(client, url, log_name="OPERA")
            if resp is None:
                return _NEXT_BUCKET
            try:
                if resp.status_code == 200:
                    return _parse_opera_hdf5(resp.content)
                # S3 answers 403 instead of 404 for a missing key when the
                # bucket does not allow anonymous listing.
                if resp.status_code in (403, 404):
                    continue  # try older slot
                if resp.status_code >= 500:
                    logger.debug(
                        "OPERA: HTTP %d from %s, trying next bucket",
                        resp.status_code, bucket_url,
                    )
                    return _NEXT_BUCKET
                logger.warning(
                    "OPERA fetch failed: HTTP %d (%s)",
                    resp.status_code, url,
                )
                return None
            except Exception:
                logger.exception("Error parsing OPERA composite")
                return None
        return _NEXT_BUCKET

    def _set_active(self, idx: int) -> None:
        """Remember the bucket that served data; log only when it changes."""
        if idx == self._active:
            return
        # Leaving the primary is worth a warning; returning to it is not.
        log = logger.info if idx == 0 else logger.warning
        log(
            "OPERA: switched bucket from %s to %s",
            self._bucket_urls[self._active], self._bucket_urls[idx],
        )
        self._active = idx

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


def _parse_opera_hdf5(data: bytes) -> np.ndarray | None:
    """Parse an OPERA CIRRUS ODIM HDF5 file into a uint8 dBZ array.

    OPERA files use float64 with gain=1.0, offset=0.0 — the raw values
    ARE dBZ directly.  Sentinel values:
      nodata  = -9999000.0  (no radar coverage)
      undetect = -8888000.0 (coverage but below detection threshold)

    Both ``nodata`` and ``undetect`` are encoded as 0 — OPERA acts as a
    gap-filler that only contributes pixels with actual precipitation.
    Clear-sky areas fall through to ECMWF, avoiding the problem that
    OPERA marks inconsistent swaths of ocean as "undetect."
    """
    try:
        with HDF5_LOCK, h5py.File(io.BytesIO(data), "r") as f:
            raw = f["dataset1/data1/data"][:]
            what = f["dataset1/data1/what"]
            nodata_val = float(what.attrs["nodata"])
            undetect_val = float(what.attrs["undetect"])
            gain = float(what.attrs["gain"])
            offset = float(what.attrs["offset"])

        # Apply gain/offset (usually 1.0/0.0 for OPERA CIRRUS)
        dbz = raw.astype(np.float32) * gain + offset

        # Mark nodata and undetect as below threshold → 0 in uint8
        invalid = np.isclose(raw, nodata_val, atol=1.0) | np.isclose(
            raw, undetect_val, atol=1.0
        )
        dbz[invalid] = -33.0

        return _dbz_float_to_uint8(dbz)
    except Exception:
        logger.exception("Failed to parse OPERA HDF5")
        return None
