# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""OPERA pan-European CIRRUS composite source.

Downloads the CIRRUS MAX reflectivity composite (DBZH) as ODIM HDF5
directly from Cloudferro S3.  Rolling 24-hour archive, 5-minute cadence.

URL pattern:
    s3://openradar-24h/YYYY/MM/DD/OPERA/COMP/OPERA@YYYYMMDDTHHMM@0@DBZH.h5
HTTP:
    https://s3.waw3-1.cloudferro.com/openradar-24h/...

License: data is published under the EUMETNET OPERA data policy
(open, gratis, anonymous).  See README and ``docs/coverage.md`` for the
attribution block.
"""
import io
import logging
import time
from datetime import datetime, timezone

import h5py
import httpx
import numpy as np

from librewxr.data.regions import RegionDef
from librewxr.data.retry import retry_get
from librewxr.sources._helpers import HDF5_LOCK, _dbz_float_to_uint8

logger = logging.getLogger(__name__)


class OperaSource:
    """OPERA pan-European radar composite from MeteoGate S3.

    Downloads the CIRRUS MAX reflectivity composite (DBZH) as ODIM HDF5
    directly from Cloudferro S3.  Rolling 24-hour archive, 5-minute cadence.
    """

    _S3_PATH = "/openradar-24h"
    # OPERA files are published with a ~5-10 minute delay; try up to
    # 3 older 5-minute slots if the target timestamp 404s.
    _MAX_FALLBACK_STEPS = 3

    def __init__(self, base_url: str = "https://s3.waw3-1.cloudferro.com"):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(90.0, connect=15.0),
                follow_redirects=True,
            )
        return self._client

    def _url_for_timestamp(self, ts: int) -> str:
        """Build S3 URL for a unix timestamp (rounded to 5-min cadence)."""
        rounded = (ts // 300) * 300
        dt = datetime.fromtimestamp(rounded, tz=timezone.utc)
        fname = dt.strftime("OPERA@%Y%m%dT%H%M@0@DBZH.h5")
        path = dt.strftime(f"%Y/%m/%d/OPERA/COMP/{fname}")
        return f"{self._base_url}{self._S3_PATH}/{path}"

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
        """Download and parse, falling back to older slots on 404."""
        client = await self._get_client()
        for step in range(self._MAX_FALLBACK_STEPS + 1):
            url = self._url_for_timestamp(ts - step * 300)
            resp = await retry_get(client, url, log_name="OPERA")
            if resp is None:
                return None
            try:
                if resp.status_code == 200:
                    return _parse_opera_hdf5(resp.content)
                if resp.status_code == 404 and step < self._MAX_FALLBACK_STEPS:
                    continue  # try older slot
                logger.warning(
                    "OPERA fetch failed: HTTP %d (%s)",
                    resp.status_code, url.split("/")[-1],
                )
                return None
            except Exception:
                logger.exception("Error parsing OPERA composite")
                return None
        return None

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
