# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import bisect
import gzip
import io
import logging
import os
import re
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import h5py
import httpx
import numpy as np
import xarray as xr
from PIL import Image

from librewxr.data.regions import RegionDef
from librewxr.data.retry import retry_get

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_eccodes_stderr():
    """Redirect OS-level stderr to /dev/null during the block.

    The eccodes C library (used by cfgrib) writes non-actionable
    ``dataTime`` truncation messages directly to stderr.  This silences
    them without affecting Python logging or other error reporting.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    original = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(original, 2)
        os.close(devnull)
        os.close(original)


class IEMSource:
    """Iowa Environmental Mesonet NEXRAD composite source.

    Fetches radar composites for any region (USCOMP, AKCOMP, etc.)
    from IEM's live and archive image endpoints.
    """

    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,
            )
        return self._client

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int
    ) -> np.ndarray | None:
        """Fetch live N0Q frame for a region."""
        frame_idx = minutes_ago // 5
        if frame_idx < 0 or frame_idx > 11:
            return None

        url = (
            f"{self._base_url}/data/gis/images/4326"
            f"/{region.live_dir}/n0q_{frame_idx}.png"
        )
        return await self._download_and_parse(url, region)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime
    ) -> np.ndarray | None:
        """Fetch archived N0Q frame for a specific UTC datetime."""
        minute = (dt.minute // 5) * 5
        dt = dt.replace(minute=minute, second=0, microsecond=0)
        path = dt.strftime(
            f"%Y/%m/%d/GIS/{region.archive_dir}/n0q_%Y%m%d%H%M.png"
        )
        url = f"{self._base_url}/archive/data/{path}"
        return await self._download_and_parse(url, region)

    async def _download_and_parse(
        self, url: str, region: RegionDef
    ) -> np.ndarray | None:
        try:
            client = await self._get_client()
            resp = await retry_get(client, url, log_name="IEM")
            if resp is None:
                return None
            if resp.status_code != 200:
                logger.warning("Failed to fetch %s: HTTP %d", url, resp.status_code)
                return None

            return _parse_n0q_png(resp.content, region)
        except Exception:
            logger.exception("Error fetching %s", url)
            return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


def _parse_n0q_png(data: bytes, region: RegionDef) -> np.ndarray | None:
    """Parse an IEM N0Q PNG into a raw uint8 numpy array.

    The PNGs are palette-indexed. We extract the raw index values,
    not the RGB colors.
    """
    try:
        img = Image.open(io.BytesIO(data))
        if img.mode == "P":
            arr = np.array(img, dtype=np.uint8)
        else:
            arr = np.array(img.convert("L"), dtype=np.uint8)

        expected = (region.height, region.width)
        if arr.shape != expected:
            logger.warning(
                "Unexpected %s dimensions: %s (expected %s)",
                region.name, arr.shape, expected,
            )
        return arr
    except Exception:
        logger.exception("Failed to parse N0Q PNG for %s", region.name)
        return None



def _dbz_float_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert float32 dBZ values to uint8 using IEM's encoding.

    Formula: pixel = clamp((dBZ + 32) * 2, 0, 255)
    NODATA (anything <= -32) maps to 0 (transparent in all color schemes).
    """
    nodata_mask = arr <= -32.0
    result = np.clip((arr + 32.0) * 2.0, 0, 255).astype(np.uint8)
    result[nodata_mask] = 0
    return result


# ── MSC Canada (GeoMet WMS) source ───────────────────────────────────

# RADAR_1KM_RRAI discrete palette (Radar-Rain_Dis-14colors style).
#
# MSC's WMS serves Canadian radar as pre-colored PNG only (no WCS/TIFF
# access to raw data).  The "Dis" style uses 14 discrete buckets mapping
# RGB → precipitation rate in mm/h.  Each entry is (R, G, B, rate) where
# rate is the geometric mean of the bucket's [lower, upper) edges — the
# typical value within the bucket, given that precipitation rates are
# log-distributed.  Using lower-edge labels directly systematically
# under-reports by ~30% within each bucket and (crucially) pushes the
# lowest bucket below the 10 dBZ noise floor, making broad light-rain
# regions invisible compared to Rain Viewer and other MSC consumers.
#
# The top bucket (≥200 mm/h, unbounded) is represented as 250 mm/h —
# a reasonable "typical extreme" that preserves headroom below the
# clamp ceiling without requiring an arbitrary upper edge.
#
# Bucket edges from the legend: 0.1, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0,
# 24.0, 32.0, 50.0, 64.0, 100.0, 125.0, 200.0, (∞).
#
# Colors were extracted from the legend graphic; composite pixels are
# within ±1 of these values due to server-side rendering rounding, so we
# use nearest-anchor lookup with a small Euclidean distance threshold.
_MSC_CANADA_PALETTE: tuple[tuple[int, int, int, float], ...] = (
    (152, 203, 254, 0.3162),   # √(0.1 × 1.0)
    (0, 152, 254, 1.4142),     # √(1.0 × 2.0)
    (0, 254, 102, 2.8284),     # √(2.0 × 4.0)
    (0, 203, 0, 5.6569),       # √(4.0 × 8.0)
    (0, 152, 0, 9.7980),       # √(8.0 × 12.0)
    (0, 102, 0, 13.8564),      # √(12.0 × 16.0)
    (254, 254, 0, 19.5959),    # √(16.0 × 24.0)
    (254, 203, 0, 27.7128),    # √(24.0 × 32.0)
    (254, 152, 0, 40.0),       # √(32.0 × 50.0)
    (254, 102, 0, 56.5685),    # √(50.0 × 64.0)
    (254, 0, 0, 80.0),         # √(64.0 × 100.0)
    (254, 2, 152, 111.8034),   # √(100.0 × 125.0)
    (152, 51, 203, 158.1139),  # √(125.0 × 200.0)
    (102, 0, 152, 250.0),      # top bucket (unbounded): typical extreme
)

# Max Euclidean RGB distance for nearest-anchor matching.  Legend vs
# composite colors differ by ≤±1 per channel (≈1.7 total); any pixel
# farther than this from every anchor is probably an artifact or
# unexpected color and is treated as no-data.
_MSC_CANADA_MAX_RGB_DIST = 8.0


def _mmhr_to_dbz(rate: np.ndarray) -> np.ndarray:
    """Convert precipitation rate (mm/h) to reflectivity (dBZ).

    Uses the Marshall-Palmer Z-R relationship: Z = 200 * R^1.6.
    NaN inputs propagate as NaN (no-data).  Rates ≤ 0 map to NaN.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        z = 200.0 * np.power(rate, 1.6)
        dbz = 10.0 * np.log10(z)
    dbz[~np.isfinite(dbz)] = np.nan
    return dbz


def _decode_msc_canada_png(data: bytes) -> np.ndarray | None:
    """Decode an MSC GeoMet WMS PNG into a uint8 dBZ array.

    Steps:
    1. Open as RGBA (transparent pixels → no-data).
    2. For each opaque pixel, find the nearest palette anchor in RGB
       space.  Pixels beyond the distance threshold become no-data.
    3. Convert anchor mm/h values to dBZ via Marshall-Palmer.
    4. Encode to uint8 using the shared scheme: (dBZ + 32) * 2, clamped.
    """
    try:
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)
    except Exception:
        logger.exception("Failed to decode MSC Canada PNG")
        return None

    h, w = arr.shape[:2]
    # int32 — per-channel squared diffs can reach ~65k and we sum three of
    # them, which overflows int16.
    rgb = arr[..., :3].astype(np.int32)
    alpha = arr[..., 3]

    # Build palette arrays: shape (N, 3) for colors, (N,) for rates
    anchors_rgb = np.array(
        [(r, g, b) for r, g, b, _ in _MSC_CANADA_PALETTE], dtype=np.int32
    )
    anchors_rate = np.array(
        [rate for *_, rate in _MSC_CANADA_PALETTE], dtype=np.float32
    )

    # Flatten pixels for vectorized nearest-anchor lookup
    flat = rgb.reshape(-1, 3)  # (H*W, 3)

    # Squared distance from each pixel to each anchor: (H*W, N)
    # Using broadcasting: (H*W, 1, 3) - (1, N, 3) → (H*W, N, 3)
    diffs = flat[:, None, :] - anchors_rgb[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)  # (H*W, N)

    nearest_idx = np.argmin(dist2, axis=1)  # (H*W,)
    nearest_dist2 = dist2[np.arange(len(flat)), nearest_idx]

    # Map nearest index → mm/h rate
    rate_flat = anchors_rate[nearest_idx]  # (H*W,)

    # Mask out: transparent pixels, or pixels too far from any anchor
    valid = (alpha.reshape(-1) > 0) & (
        nearest_dist2 <= _MSC_CANADA_MAX_RGB_DIST ** 2
    )
    rate_flat = np.where(valid, rate_flat, np.nan)

    # Convert mm/h → dBZ
    dbz_flat = _mmhr_to_dbz(rate_flat)
    dbz = dbz_flat.reshape(h, w)

    # NaN → -33 sentinel so the shared uint8 encoder maps it to 0
    dbz = np.where(np.isnan(dbz), -33.0, dbz)
    return _dbz_float_to_uint8(dbz)


class MSCCanadaSource:
    """Environment and Climate Change Canada radar composite source.

    Fetches the RADAR_1KM_RRAI composite from MSC GeoMet WMS as a
    pre-colored PNG (MSC does not publish raw radar data in any open
    format).  Uses the "Radar-Rain_Dis-14colors" discrete style and
    reverse-engineers the color palette back to precipitation rate,
    then converts to dBZ via Marshall-Palmer.

    The WMS time dimension gives a rolling ~3-hour history at 6-minute
    cadence — sufficient for live + archive playback.
    """

    _WMS_PATH = "/geomet"
    _STYLE = "Radar-Rain_Dis-14colors"
    _LAYER = "RADAR_1KM_RRAI"

    def __init__(self, base_url: str = "https://geo.weather.gc.ca"):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=15.0),
                follow_redirects=True,
            )
        return self._client

    def _build_url(self, region: RegionDef, time_iso: str | None) -> str:
        params = [
            "SERVICE=WMS",
            "VERSION=1.3.0",
            "REQUEST=GetMap",
            f"LAYERS={self._LAYER}",
            f"STYLES={self._STYLE}",
            "CRS=EPSG:4326",
            # WMS 1.3.0 EPSG:4326 axis order is (lat, lon)
            f"BBOX={region.south},{region.west},{region.north},{region.east}",
            f"WIDTH={region.width}",
            f"HEIGHT={region.height}",
            "FORMAT=image/png",
            "TRANSPARENT=TRUE",
        ]
        if time_iso:
            params.append(f"TIME={time_iso}")
        return f"{self._base_url}{self._WMS_PATH}?" + "&".join(params)

    # MSC publishes radar composites on a ~6-minute cadence.  When a
    # requested timestamp is not yet available (common for the most recent
    # slots or when our aligned timestamps drift slightly ahead of real
    # time), we fall back to progressively older 6-minute slots.
    _CADENCE_SEC = 360  # 6 minutes
    _MAX_TIME_RETRIES = 2

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int
    ) -> np.ndarray | None:
        """Fetch a live frame.  minutes_ago=0 → server default (latest)."""
        if minutes_ago <= 0:
            return await self._fetch(region, None)
        target_ts = int(time.time()) - minutes_ago * 60
        target = datetime.fromtimestamp(target_ts, tz=timezone.utc)
        return await self._fetch_with_fallback(region, target)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime
    ) -> np.ndarray | None:
        """Fetch a specific historical frame via WMS TIME parameter."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return await self._fetch_with_fallback(region, dt)

    async def _fetch(
        self, region: RegionDef, time_iso: str | None
    ) -> np.ndarray | None:
        url = self._build_url(region, time_iso)
        try:
            client = await self._get_client()
            resp = await retry_get(client, url, log_name="MSC Canada")
            if resp is None:
                return None
            if resp.status_code != 200:
                logger.warning(
                    "MSC Canada WMS fetch failed: HTTP %d (time=%s)",
                    resp.status_code, time_iso,
                )
                return None
            # MSC returns a ServiceExceptionReport (XML) with 200 when the
            # requested TIME is not yet available — this is normal for the
            # most recent slots, not an error worth warning about.
            if resp.headers.get("content-type", "").startswith("text/xml"):
                logger.debug(
                    "MSC Canada WMS returned XML exception (time=%s)",
                    time_iso,
                )
                return None
            return _decode_msc_canada_png(resp.content)
        except Exception:
            logger.exception("Error fetching MSC Canada WMS")
            return None

    async def _fetch_with_fallback(
        self, region: RegionDef, target_dt: datetime
    ) -> np.ndarray | None:
        """Fetch with progressive fallback to older MSC timesteps.

        MSC cadence is ~6 minutes and the most recent 1-2 slots are often
        not yet published.  If the requested TIME returns an XML exception,
        step backwards in 6-minute increments up to ``_MAX_TIME_RETRIES``.
        """
        from datetime import timedelta

        for attempt in range(self._MAX_TIME_RETRIES + 1):
            ts_iso = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            result = await self._fetch(region, ts_iso)
            if result is not None:
                if attempt > 0:
                    logger.debug(
                        "MSC Canada fallback succeeded: %s (attempt %d)",
                        ts_iso, attempt,
                    )
                return result
            target_dt -= timedelta(seconds=self._CADENCE_SEC)

        logger.debug(
            "MSC Canada WMS no data after %d fallback attempts",
            self._MAX_TIME_RETRIES,
        )
        return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ── OPERA (pan-European CIRRUS) source ───────────────────────────────


class OperaSource:
    """OPERA pan-European radar composite from MeteoGate S3.

    Downloads the CIRRUS MAX reflectivity composite (DBZH) as ODIM HDF5
    directly from Cloudferro S3.  Rolling 24-hour archive, 5-minute cadence.

    URL pattern:
        s3://openradar-24h/YYYY/MM/DD/OPERA/COMP/OPERA@YYYYMMDDTHHMM@0@DBZH.h5
    HTTP:
        https://s3.waw3-1.cloudferro.com/openradar-24h/...
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
        f = h5py.File(io.BytesIO(data), "r")
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


# ── MRMS (NOAA Multi-Radar/Multi-Sensor) source ─────────────────────


# MRMS MergedReflectivityQCComposite per-region product paths and extents.
# Each US territory has its own MRMS regional product on the NCEP server.
MRMS_PRODUCTS: dict[str, str] = {
    "USCOMP": "MergedReflectivityQCComposite",
    "CACOMP": "MergedReflectivityQCComposite",
    "AKCOMP": "ALASKA/MergedReflectivityQCComposite",
    "HICOMP": "HAWAII/MergedReflectivityQCComposite",
    "PRCOMP": "CARIB/MergedReflectivityQCComposite",
    "GUCOMP": "GUAM/MergedReflectivityQCComposite",
}

# MRMS grid extents per region (south, north, west, east) in degrees.
# These match the actual GRIB2 grid bounds from NCEP.
MRMS_EXTENTS: dict[str, tuple[float, float, float, float]] = {
    "USCOMP": (20.005, 54.995, -129.995, -60.005),
    "CACOMP": (20.005, 54.995, -129.995, -60.005),
    "AKCOMP": (50.005, 71.995, -175.995, -126.005),
    "HICOMP": (15.002, 25.997, -163.998, -151.002),
    "PRCOMP": (10.005, 24.995, -89.995, -60.005),
    "GUCOMP": (9.002, 17.997, 140.002, 149.998),
}

class MRMSSource:
    """NOAA MRMS MergedReflectivityQCComposite source.

    Fetches the quality-controlled composite reflectivity product from the
    NCEP real-time GRIB2 endpoint.  Supports both the CONUS product
    (USCOMP/CACOMP) and regional products for Alaska, Hawaii, Caribbean
    (Puerto Rico), and Guam.

    The live endpoint publishes a ``.latest.grib2.gz`` file updated every
    ~2 minutes.  Archive files follow the pattern
    ``MRMS_MergedReflectivityQCComposite_00.50_YYYYMMDD-HHMMSS.grib2.gz``.

    No-data is encoded as -999.0; valid values are dBZ.
    """

    _TIMESTAMP_RE = re.compile(
        r"MRMS_MergedReflectivityQCComposite_00\.50_(\d{8}-\d{6})\.grib2\.gz"
    )

    def __init__(
        self,
        base_url: str = "https://mrms.ncep.noaa.gov/2D",
        region_name: str = "USCOMP",
    ):
        self._base_url = base_url.rstrip("/")
        self._region_name = region_name
        self._product = MRMS_PRODUCTS[region_name]
        self._client: httpx.AsyncClient | None = None
        # Directory listing cache: sorted list of (datetime, filename) tuples.
        # Refreshed once per fetch cycle.
        self._dir_cache: list[tuple[datetime, str]] | None = None
        self._dir_cache_time: float = 0.0
        # Serialises refreshes so parallel backfill coroutines don't each
        # issue their own HTTP fetch when the cache is cold or stale.
        self._dir_cache_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=15.0),
                follow_redirects=True,
            )
        return self._client

    def _latest_url(self) -> str:
        product_name = self._product.split("/")[-1]
        return (
            f"{self._base_url}/{self._product}"
            f"/MRMS_{product_name}.latest.grib2.gz"
        )

    def _archive_url(self, dt: datetime) -> str:
        ts = dt.strftime("%Y%m%d-%H%M%S")
        product_name = self._product.split("/")[-1]
        return (
            f"{self._base_url}/{self._product}"
            f"/MRMS_{product_name}_00.50_{ts}.grib2.gz"
        )

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int
    ) -> np.ndarray | None:
        """Fetch live MRMS frame.

        For minutes_ago == 0, uses the ``.latest`` endpoint.
        For minutes_ago > 0, scans the directory listing to find the
        file closest to the target time.
        """
        if minutes_ago <= 0:
            return await self._fetch_and_parse(self._latest_url(), region)

        # Calculate target timestamp and find nearest file
        target_ts = int(time.time()) - minutes_ago * 60
        target_dt = datetime.fromtimestamp(target_ts, tz=timezone.utc)
        url = await self._find_nearest_url(target_dt)
        if url is not None:
            return await self._fetch_and_parse(url, region)

        # Fallback to .latest if directory scan failed
        logger.warning("MRMS directory scan failed, falling back to .latest")
        return await self._fetch_and_parse(self._latest_url(), region)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime
    ) -> np.ndarray | None:
        """Fetch archived MRMS frame for a specific UTC datetime.

        Scans the NCEP directory listing to find the file whose timestamp
        is closest to the requested time.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        url = await self._find_nearest_url(dt)
        if url is not None:
            return await self._fetch_and_parse(url, region)
        return None

    async def _find_nearest_url(self, target: datetime) -> str | None:
        """Find the MRMS file whose timestamp is closest to *target*.

        Fetches the NCEP directory listing (cached for 5 minutes), parses
        the filenames to extract timestamps, and returns the URL of the
        file closest to the target time.  Returns None if the directory
        listing cannot be fetched or parsed.
        """
        await self._refresh_dir_cache()
        if not self._dir_cache:
            return None

        target_ts = target.timestamp()
        timestamps = [e[0].timestamp() for e in self._dir_cache]
        idx = bisect.bisect_left(timestamps, target_ts)

        if idx == 0:
            best_idx = 0
        elif idx == len(timestamps):
            best_idx = len(timestamps) - 1
        else:
            before = timestamps[idx - 1]
            after = timestamps[idx]
            best_idx = idx - 1 if (target_ts - before) <= (after - target_ts) else idx

        dt, filename = self._dir_cache[best_idx]
        logger.debug(
            "MRMS nearest to %s: %s (delta=%ds)",
            target.strftime("%Y%m%d-%H%M%S"),
            filename,
            int(abs((target - dt).total_seconds())),
        )
        return f"{self._base_url}/{self._product}/{filename}"

    async def _refresh_dir_cache(self) -> None:
        """Fetch and parse the MRMS directory listing if stale.

        Caches for 5 minutes to avoid hammering the server. Uses
        double-checked locking so parallel backfill coroutines coalesce
        into a single HTTP fetch instead of each refreshing on their own.
        """
        if self._dir_cache is not None and (time.time() - self._dir_cache_time) < 300:
            return

        async with self._dir_cache_lock:
            # Re-check under the lock: another coroutine may have already
            # refreshed while we were waiting.
            if self._dir_cache is not None and (time.time() - self._dir_cache_time) < 300:
                return

            url = f"{self._base_url}/{self._product}/"
            client = await self._get_client()
            resp = await retry_get(client, url, log_name="MRMS directory")
            if resp is None:
                return
            if resp.status_code != 200:
                logger.warning("MRMS directory listing failed: HTTP %d", resp.status_code)
                return

            entries: list[tuple[datetime, str]] = []
            for match in self._TIMESTAMP_RE.finditer(resp.text):
                ts_str = match.group(1)
                try:
                    dt = datetime.strptime(ts_str, "%Y%m%d-%H%M%S").replace(
                        tzinfo=timezone.utc
                    )
                    entries.append((dt, match.group(0)))
                except ValueError:
                    continue

            if not entries:
                logger.warning("MRMS directory listing: no timestamps found")
                return

            entries.sort(key=lambda e: e[0])
            self._dir_cache = entries
            self._dir_cache_time = time.time()
            logger.info(
                "MRMS directory cache refreshed: %d files, %s to %s",
                len(entries),
                entries[0][0].strftime("%Y%m%d-%H%M%S"),
                entries[-1][0].strftime("%Y%m%d-%H%M%S"),
            )

    async def _fetch_and_parse(
        self, url: str, region: RegionDef
    ) -> np.ndarray | None:
        """Download a GRIB2.gz file, parse, crop and resample to region."""
        from librewxr.config import settings as _settings

        client = await self._get_client()
        for attempt in range(_settings.download_retries + 1):
            resp = await retry_get(client, url, log_name="MRMS")
            if resp is None:
                return None
            if resp.status_code != 200:
                logger.warning(
                    "MRMS fetch failed: HTTP %d (%s)", resp.status_code, url
                )
                return None

            try:
                ds = _parse_mrms_grib2(resp.content)
            except EOFError:
                # Truncated download (server dropped connection mid-stream).
                # Retry the full download cycle once before giving up.
                if attempt < _settings.download_retries:
                    logger.info(
                        "MRMS gzip truncated, retrying download: %s", url
                    )
                    await asyncio.sleep(1)
                    continue
                logger.warning(
                    "MRMS gzip truncated after %d retries: %s",
                    _settings.download_retries, url,
                )
                return None
            except Exception:
                logger.exception("Failed to parse MRMS GRIB2 from %s", url)
                return None

            if ds is None:
                return None

            return _resample_mrms_to_region(ds, region)

        return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._dir_cache = None


def _parse_mrms_grib2(data: bytes) -> xr.Dataset | None:
    """Decompress and parse an MRMS GRIB2 file into an xarray Dataset.

    Returns a Dataset with latitude, longitude, and a single reflectivity
    variable.  Returns None on any parse failure.

    Raises:
        EOFError: if the gzip stream is truncated (incomplete download).
    """
    try:
        raw = gzip.decompress(data)
    except EOFError:
        # Truncated download — let the caller retry.  Don't log here so
        # the retry logic in _fetch_and_parse can decide the message.
        raise
    except Exception:
        logger.exception("Failed to decompress MRMS GRIB2")
        return None

    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
        tmp.write(raw)
        tmp.close()
        # Suppress eccodes "truncating time" noise written directly to stderr.
        with _suppress_eccodes_stderr():
            ds = xr.open_dataset(tmp.name, engine="cfgrib")
        # Force load into memory so the temp file can be deleted
        ds = ds.compute()
        return ds
    except Exception:
        logger.exception("Failed to parse MRMS GRIB2 with cfgrib")
        return None
    finally:
        if tmp is not None:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except OSError:
                pass


def _resample_mrms_to_region(
    ds: xr.Dataset, region: RegionDef
) -> np.ndarray:
    """Crop and resample an MRMS Dataset to a region's lat/lon grid.

    Steps:
    1. Extract the reflectivity variable (first data var).
    2. Slice the MRMS grid to the region's bounding box (with 1-cell
       padding to avoid edge effects in nearest-neighbor sampling).
    3. Replace -999.0 (MRMS no-data) with NaN.
    4. Build target lat/lon axes from region bounds and pixel_size.
    5. Resample via nearest-neighbor (upscale for USCOMP, downsample
       for CACOMP) using numpy index mapping.
    6. Convert float dBZ to uint8 using the shared ``_dbz_float_to_uint8``
       encoder.
    """
    var_name = list(ds.data_vars)[0]
    data = ds[var_name].values.astype(np.float32)

    lats = ds.latitude.values  # north-to-south (54.99 → 20.01)
    lons = ds.longitude.values  # west-to-east, may be 0-360 or -180-180

    # Normalize longitude to -180..180 if needed
    if lons.max() > 180:
        lons = np.where(lons > 180, lons - 360, lons).astype(lons.dtype)

    # Slice to region bbox with 1-cell padding
    pad = 2  # extra cells beyond bbox for safety
    south_idx = np.searchsorted(-lats, -region.south)  # lats are descending
    north_idx = np.searchsorted(-lats, -region.north)
    west_idx = np.searchsorted(lons, region.west)
    east_idx = np.searchsorted(lons, region.east)

    south_idx = max(0, south_idx - pad)
    north_idx = min(len(lats), north_idx + pad)
    west_idx = max(0, west_idx - pad)
    east_idx = min(len(lons), east_idx + pad)

    data = data[north_idx:south_idx, west_idx:east_idx]
    lats = lats[north_idx:south_idx]
    lons = lons[west_idx:east_idx]

    # Build target grid axes.
    # Target lats go north-to-south (descending) so that row 0 of the
    # output array corresponds to the northernmost pixel, matching the
    # coordinate convention used by the renderer:
    #   row = (region.north - lat) / pixel_size_y
    # Pixel centers are offset by half a pixel from the grid edge.
    target_ps = region.pixel_size
    target_ps_y = region._ps_y
    north_center = region.north - target_ps_y / 2
    south_center = region.south + target_ps_y / 2
    target_lats = np.linspace(north_center, south_center, region.height)
    target_lons = np.arange(region.west, region.east, target_ps)

    if len(target_lats) == 0 or len(target_lons) == 0:
        logger.warning("MRMS resample: empty target grid for %s", region.name)
        return np.zeros((region.height, region.width), dtype=np.uint8)

    # Nearest-neighbor resampling: map each target pixel to the closest
    # source pixel.  Both source and target lats are north-to-south
    # (descending), so negating gives ascending arrays suitable for
    # searchsorted.
    target_lat_rows = np.searchsorted(-lats, -target_lats)
    target_lat_rows = np.clip(target_lat_rows, 0, len(lats) - 1)

    target_lon_cols = np.searchsorted(lons, target_lons)
    target_lon_cols = np.clip(target_lon_cols, 0, len(lons) - 1)

    # Index into the cropped data array
    resampled = data[target_lat_rows[:, None], target_lon_cols[None, :]]

    # Replace MRMS no-data (-999.0) with NaN before encoding
    resampled = np.where(resampled < -900, np.nan, resampled)

    # Convert to uint8 using shared encoder (NaN → -33 → 0)
    resampled = np.where(np.isnan(resampled), -33.0, resampled)
    return _dbz_float_to_uint8(resampled)


# ── MARN (El Salvador SNET) source ──────────────────────────────────

# SNET's 120 km radar product (``esar82``) encodes reflectivity as a
# continuous HSV-style gradient running from green (low dBZ) through
# cyan and blue to magenta (high dBZ).  Every opaque pixel sits exactly
# on the fully-saturated hue ring with one channel at 0, another at 255,
# and the third varying — so decode is a direct arc-detect + linear
# hue→dBZ map (no nearest-anchor search needed).
#
#   Arc 1 (G=255, R=0, B 0→255):   hue 120° (green)   → 180° (cyan)
#   Arc 2 (B=255, R=0, G 255→0):   hue 180° (cyan)    → 240° (blue)
#   Arc 3 (B=255, G=0, R 0→255):   hue 240° (blue)    → 300° (magenta)
#
# dBZ range is provisional: green=10 dBZ, magenta=70 dBZ (linear in
# hue).  SNET publishes no per-bin dBZ calibration, so refine after
# sampling real precipitation cells against a reference (e.g. MRMS over
# adjacent overlapping coverage, or NEXRAD next time a tropical system
# tracks over the Caribbean side).  The discrete legend at
# ``snet.gob.sv/UserFiles/SNET/Image/meteorologia/escalaPropuesta2013SNEThW_.png``
# is for a *different* product (the 60 km multi-radar composite) — do
# not use it for this decoder.

_MARN_HUE_MIN = 120.0   # green at the low-dBZ end of the gradient
_MARN_HUE_MAX = 300.0   # magenta at the high-dBZ end
_MARN_DBZ_MIN = 10.0
_MARN_DBZ_MAX = 70.0


class MARNSource:
    """El Salvador MARN/SNET radar composite source.

    Single S-band radar at San Andrés volcano, 120 km range product
    (``esar82/Images/``).  Files are PNGs published to an anonymous
    Google Cloud Storage bucket (``radar-images-sv``) at 5-minute
    cadence.  Filenames embed the radar timestamp in El Salvador local
    time (UTC-6, no DST) as ``YYYY-MM-DD HH-MM-SS.png``.

    License: MARN explicitly permits full or partial reproduction with
    citation — see README and docs/coverage.md for the attribution.
    """

    _PRODUCT_PREFIX = "esar82/Images/"
    _BUCKET = "radar-images-sv"
    _LOCAL_TZ_OFFSET = -6     # El Salvador is UTC-6 year-round (no DST)
    _CADENCE_SEC = 300        # 5 minutes
    _MAX_FALLBACK_STEPS = 3   # 4 attempts × 5 min = 20 min lookback
    _DIR_CACHE_TTL_SEC = 120  # Refresh bucket listing every 2 minutes

    def __init__(self, base_url: str = "https://storage.googleapis.com"):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        # Cached bucket listing: list of (UTC datetime, full object name).
        self._dir_cache: list[tuple[datetime, str]] | None = None
        self._dir_cache_time: float = 0.0
        self._dir_cache_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=15.0),
                follow_redirects=True,
            )
        return self._client

    @classmethod
    def _filename_to_utc(cls, name: str) -> datetime | None:
        """Parse ``esar82/Images/YYYY-MM-DD HH-MM-SS.png`` → UTC datetime.

        Filenames embed El Salvador local time (UTC-6); we convert to
        UTC here so all downstream code can stay in UTC.
        """
        stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        try:
            local = datetime.strptime(stem, "%Y-%m-%d %H-%M-%S")
        except ValueError:
            return None
        return local.replace(tzinfo=timezone.utc) - timedelta(
            hours=cls._LOCAL_TZ_OFFSET
        )

    def _object_url(self, name: str) -> str:
        """Build the public download URL for a bucket object."""
        # Use the GCS-native media path — quoted to handle spaces in
        # the filename.  Using ``httpx.URL`` here is overkill; manual
        # quoting is cheaper and stable.
        from urllib.parse import quote
        return f"{self._base_url}/{self._BUCKET}/{quote(name)}"

    async def _refresh_dir_cache(self, target_utc: datetime) -> None:
        """Refresh the cached bucket listing if stale.

        Lists today's and yesterday's local-date prefixes so that a
        target straddling local midnight still finds the right file.
        """
        if self._dir_cache is not None and (
            time.time() - self._dir_cache_time
        ) < self._DIR_CACHE_TTL_SEC:
            return

        async with self._dir_cache_lock:
            if self._dir_cache is not None and (
                time.time() - self._dir_cache_time
            ) < self._DIR_CACHE_TTL_SEC:
                return

            # El Salvador local time for the target.
            local = target_utc + timedelta(hours=self._LOCAL_TZ_OFFSET)
            dates = {
                local.strftime("%Y-%m-%d"),
                (local - timedelta(days=1)).strftime("%Y-%m-%d"),
            }

            client = await self._get_client()
            entries: list[tuple[datetime, str]] = []
            for date_str in sorted(dates):
                prefix = f"{self._PRODUCT_PREFIX}{date_str}"
                url = (
                    f"{self._base_url}/storage/v1/b/{self._BUCKET}/o"
                    f"?prefix={prefix}&maxResults=500"
                )
                resp = await retry_get(client, url, log_name="MARN dir")
                if resp is None or resp.status_code != 200:
                    if resp is not None:
                        logger.warning(
                            "MARN bucket listing failed: HTTP %d (%s)",
                            resp.status_code, date_str,
                        )
                    continue
                try:
                    payload = resp.json()
                except Exception:
                    logger.exception("MARN bucket listing: bad JSON")
                    continue
                for item in payload.get("items", []):
                    name = item.get("name", "")
                    dt = self._filename_to_utc(name)
                    # Skip the 1899-12-30 placeholder upload that GCS
                    # bucket-Owner-only sentinel objects use.
                    if dt is None or dt.year < 2000:
                        continue
                    entries.append((dt, name))

            if not entries:
                logger.warning("MARN bucket listing: no usable entries")
                return

            entries.sort(key=lambda e: e[0])
            self._dir_cache = entries
            self._dir_cache_time = time.time()
            logger.info(
                "MARN bucket cache refreshed: %d files, %s..%s",
                len(entries),
                entries[0][0].strftime("%Y-%m-%dT%H:%MZ"),
                entries[-1][0].strftime("%Y-%m-%dT%H:%MZ"),
            )

    async def _find_nearest(
        self, target: datetime
    ) -> list[tuple[datetime, str]]:
        """Return candidate (datetime, name) entries ordered for fallback.

        Best match (nearest at-or-before *target*) first; older slots
        next.  Caller iterates this list until one fetch succeeds.
        """
        await self._refresh_dir_cache(target)
        if not self._dir_cache:
            return []

        target_ts = target.timestamp()
        timestamps = [e[0].timestamp() for e in self._dir_cache]
        idx = bisect.bisect_right(timestamps, target_ts)
        if idx == 0:
            return [self._dir_cache[0]]
        # Start with the file at-or-before target, then walk backwards.
        start = idx - 1
        return list(reversed(self._dir_cache[: start + 1]))

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int
    ) -> np.ndarray | None:
        """Fetch a live frame.  ``minutes_ago=0`` → latest available."""
        now_rounded = int(time.time() // self._CADENCE_SEC) * self._CADENCE_SEC
        target_ts = now_rounded - minutes_ago * 60
        target_ts = (target_ts // self._CADENCE_SEC) * self._CADENCE_SEC
        target = datetime.fromtimestamp(target_ts, tz=timezone.utc)
        return await self._fetch_nearest(region, target)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime
    ) -> np.ndarray | None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return await self._fetch_nearest(region, dt)

    async def _fetch_nearest(
        self, region: RegionDef, target: datetime
    ) -> np.ndarray | None:
        """Fetch the nearest available frame at or before *target*."""
        candidates = await self._find_nearest(target)
        client = await self._get_client()
        for step, (dt, name) in enumerate(candidates):
            if step > self._MAX_FALLBACK_STEPS:
                break
            url = self._object_url(name)
            resp = await retry_get(client, url, log_name="MARN")
            if resp is None:
                return None
            if resp.status_code == 404 and step < self._MAX_FALLBACK_STEPS:
                continue
            if resp.status_code != 200:
                logger.warning(
                    "MARN fetch failed: HTTP %d (%s)", resp.status_code, name,
                )
                return None
            arr = _decode_marn_png(resp.content, region)
            if arr is not None and step > 0:
                logger.debug(
                    "MARN fallback succeeded at step %d: %s", step, name,
                )
            return arr
        return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._dir_cache = None


def _decode_marn_png(data: bytes, region: RegionDef) -> np.ndarray | None:
    """Decode a SNET radar PNG into a uint8 dBZ array.

    SNET PNGs are RGB with a tRNS chunk marking RGB=(0, 0, 0) as
    transparent (no-data).  Every opaque pixel sits on the saturated
    HSV hue ring described at the top of this section, so each one
    falls into exactly one of three arcs based on which two channels
    are pinned to {0, 255}.
    """
    try:
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)
    except Exception:
        logger.exception("Failed to decode MARN PNG")
        return None

    if arr.shape[:2] != (region.height, region.width):
        logger.warning(
            "Unexpected %s dimensions: %s (expected %s)",
            region.name, arr.shape, (region.height, region.width),
        )

    r = arr[..., 0].astype(np.int32)
    g = arr[..., 1].astype(np.int32)
    b = arr[..., 2].astype(np.int32)
    alpha = arr[..., 3]

    hue = np.full(arr.shape[:2], np.nan, dtype=np.float32)

    # Arc 1: green → cyan, hue 120..180
    arc1 = (g == 255) & (r == 0)
    hue[arc1] = 120.0 + b[arc1].astype(np.float32) * (60.0 / 255.0)

    # Arc 2: cyan → blue, hue 180..240.  Excludes pure cyan (already
    # handled by arc1 when B=255).
    arc2 = (b == 255) & (r == 0) & (g != 255)
    hue[arc2] = 240.0 - g[arc2].astype(np.float32) * (60.0 / 255.0)

    # Arc 3: blue → magenta, hue 240..300.  Excludes pure blue
    # (already handled by arc2 when G=0).
    arc3 = (b == 255) & (g == 0) & (r != 0)
    hue[arc3] = 240.0 + r[arc3].astype(np.float32) * (60.0 / 255.0)

    dbz_span = _MARN_DBZ_MAX - _MARN_DBZ_MIN
    hue_span = _MARN_HUE_MAX - _MARN_HUE_MIN
    dbz = _MARN_DBZ_MIN + (hue - _MARN_HUE_MIN) * (dbz_span / hue_span)

    # Transparent and off-palette → no-data (NaN → -33 sentinel → 0).
    nodata = (alpha == 0) | np.isnan(hue)
    dbz = np.where(nodata, -33.0, dbz)
    return _dbz_float_to_uint8(dbz)


# ── CWA Taiwan (QPESUMS composite reflectivity) source ──────────────

# CWA O-A0059-001 publishes the QPESUMS 7-radar composite as a single
# UTF-8 XML per 10-min frame on anonymous AWS S3.  Each frame is ~9 MB
# of comma-separated scientific-notation floats inside one <content>
# element, with sentinels -99 (invalid) and -999 (outside radar range).
#
# Archive key uses NO separator dot between the timestamp and the
# product name (``{ts}compref_mosaic.xml``).  The interleaved QPESUMS
# gauge keys ``{ts}.QPESUMS_GAUGE.10M.xml`` *do* use a dot — easy to
# mix up.
#
# Timestamps in keys are Taipei local (UTC+8, no DST).

_CWA_NAMESPACE = "urn:cwa:gov:tw:cwacommon:0.1"


class CWASource:
    """Taiwan CWA QPESUMS composite reflectivity (O-A0059-001) source.

    Anonymous S3 (``cwaopendata`` in ``ap-northeast-1``).  10-min
    cadence, clock-aligned in Taipei local time (UTC+8).  Decoder
    parses scientific-notation floats from the XML ``<content>``
    element and flips vertically (south-to-north → north-up).

    License: data.gov.tw Open Government Data License v1.0,
    attribution required.  See README and docs/coverage.md for the
    citation.
    """

    _ARCHIVE_PREFIX = "/history/Observation"
    _LOCAL_TZ_OFFSET = 8           # Taipei is UTC+8 year-round (no DST)
    _CADENCE_SEC = 600             # 10 minutes
    # Files publish ~6 min after their frame time, so the most-recent
    # 1-2 slots are often 404.  Walk back up to 3 older slots.
    _MAX_FALLBACK_STEPS = 3

    def __init__(
        self,
        base_url: str = "https://cwaopendata.s3.ap-northeast-1.amazonaws.com",
    ):
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
        """Build the archive URL for a unix timestamp.

        Rounds to a 10-min slot, converts UTC → UTC+8, and formats
        as ``{YYYYMMDDHHMM}compref_mosaic.xml`` (note: no separator
        dot between timestamp and product name).
        """
        rounded = (ts // self._CADENCE_SEC) * self._CADENCE_SEC
        local = datetime.fromtimestamp(
            rounded, tz=timezone.utc
        ) + timedelta(hours=self._LOCAL_TZ_OFFSET)
        fname = local.strftime("%Y%m%d%H%M") + "compref_mosaic.xml"
        return f"{self._base_url}{self._ARCHIVE_PREFIX}/{fname}"

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int
    ) -> np.ndarray | None:
        now_rounded = (
            int(time.time() // self._CADENCE_SEC) * self._CADENCE_SEC
        )
        target_ts = now_rounded - minutes_ago * 60
        return await self._fetch_xml(target_ts, region)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime
    ) -> np.ndarray | None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return await self._fetch_xml(int(dt.timestamp()), region)

    async def _fetch_xml(
        self, ts: int, region: RegionDef
    ) -> np.ndarray | None:
        """Download and parse, falling back to older slots on 404."""
        client = await self._get_client()
        for step in range(self._MAX_FALLBACK_STEPS + 1):
            url = self._url_for_timestamp(ts - step * self._CADENCE_SEC)
            resp = await retry_get(client, url, log_name="CWA")
            if resp is None:
                return None
            if resp.status_code == 200:
                return _parse_cwa_xml(resp.content, region)
            if resp.status_code == 404 and step < self._MAX_FALLBACK_STEPS:
                continue
            logger.warning(
                "CWA fetch failed: HTTP %d (%s)",
                resp.status_code, url.rsplit("/", 1)[-1],
            )
            return None
        return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


def _parse_cwa_xml(data: bytes, region: RegionDef) -> np.ndarray | None:
    """Parse a CWA O-A0059-001 XML into a uint8 dBZ array.

    Steps:
    1. Locate the namespaced ``<content>`` element.
    2. Parse comma-separated scientific-notation floats into a 1D array.
    3. Reshape to ``(height, width)`` (the XML order is row-major,
       south-to-north — first value is the SW corner) and flip
       vertically so row 0 ends up north.
    4. Map sentinels (``-99``, ``-999``) to the shared no-data sentinel
       (-33), then encode to uint8 via the shared encoder.
    """
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        logger.exception("Failed to parse CWA XML")
        return None

    content_el = root.find(f".//{{{_CWA_NAMESPACE}}}content")
    if content_el is None or content_el.text is None:
        logger.warning("CWA XML missing <content> element")
        return None

    try:
        flat = np.fromstring(content_el.text, sep=",", dtype=np.float32)
    except Exception:
        logger.exception("Failed to parse CWA <content> floats")
        return None

    expected = region.width * region.height
    if flat.size != expected:
        logger.warning(
            "CWA grid size mismatch for %s: %d (expected %d)",
            region.name, flat.size, expected,
        )
        return None

    # Row-major south-to-north → flip vertically to north-up convention.
    grid = flat.reshape(region.height, region.width)[::-1]

    # Sentinels: -99 (invalid) and -999 (outside radar range / QC-removed)
    invalid = (grid <= -99.0)
    grid = np.where(invalid, -33.0, grid)
    return _dbz_float_to_uint8(grid)


# ── Per-source module re-exports ─────────────────────────────────
#
# Following the convention established by per-NWP-grid files
# (hrrr_grid.py, dmi_dini_grid.py, etc.), radar sources also live in
# their own modules as they get touched.  Re-export the classes here
# so the fetcher and tests can keep importing from the canonical
# ``librewxr.data.sources`` namespace alongside the older inline
# sources.  Inline sources (IEM, MRMS, MSC, OPERA, CWA, MARN) stay in
# this file until they're next touched.
from librewxr.data.mmd_source import MMDSource  # noqa: E402,F401
