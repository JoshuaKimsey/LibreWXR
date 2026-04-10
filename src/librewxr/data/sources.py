# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io
import logging
import time
from datetime import datetime, timezone

import h5py
import httpx
import numpy as np
from PIL import Image

from librewxr.data.regions import RegionDef

logger = logging.getLogger(__name__)

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
            resp = await client.get(url)
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

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int
    ) -> np.ndarray | None:
        """Fetch a live frame.  minutes_ago=0 → server default (latest)."""
        if minutes_ago <= 0:
            return await self._fetch(region, None)
        # MSC cadence is 6 minutes — let the server snap to nearest timestep
        target_ts = int(time.time()) - minutes_ago * 60
        target = datetime.fromtimestamp(target_ts, tz=timezone.utc)
        return await self._fetch(region, target.strftime("%Y-%m-%dT%H:%M:%SZ"))

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime
    ) -> np.ndarray | None:
        """Fetch a specific historical frame via WMS TIME parameter."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return await self._fetch(
            region, dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        )

    async def _fetch(
        self, region: RegionDef, time_iso: str | None
    ) -> np.ndarray | None:
        url = self._build_url(region, time_iso)
        try:
            client = await self._get_client()
            resp = await client.get(url)
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
        now_rounded = int(time.time() // 300) * 300
        target_ts = now_rounded - minutes_ago * 60
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
            try:
                resp = await client.get(url)
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
                logger.exception("Error fetching OPERA composite")
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


