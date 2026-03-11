# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import io
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx
import numpy as np
from PIL import Image

from librewrx.data.regions import RegionDef

logger = logging.getLogger(__name__)

# Filename pattern for MET Norway Nordic reflectivity files
_NORDIC_FILENAME_PREFIX = (
    "yrwms-nordic.mos.pcappi-0-dbz."
    "noclass-clfilter-novpr-clcorr-block.nordiclcc-1000."
)
_NORDIC_TIMESTAMP_RE = re.compile(r"(\d{8}T\d{6}Z)\.nc$")


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


class METNordicSource:
    """MET Norway Nordic radar composite source.

    Fetches reflectivity composites covering Norway, Sweden, Finland, and
    Denmark from MET Norway's THREDDS WCS endpoint.  Data is delivered as
    float32 GeoTIFF (raw dBZ) and converted to the same uint8 encoding
    used by IEM: pixel = clamp((dBZ + 32) * 2, 0, 255).
    """

    _CATALOG_PATH = (
        "/thredds/catalog/remotesensing/reflectivity-nordic"
        "/latest/catalog.xml"
    )
    _WCS_PATH = (
        "/thredds/wcs/remotesensing/reflectivity-nordic/latest"
    )
    _CATALOG_MAX_AGE = 300  # re-fetch catalog at most every 5 minutes

    def __init__(self, base_url: str = "https://thredds.met.no"):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        # Cached catalog: list of (unix_timestamp, filename) sorted newest-first
        self._catalog: list[tuple[int, str]] = []
        self._catalog_fetched_at: float = 0

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=15.0),
                follow_redirects=True,
            )
        return self._client

    # -- Catalog discovery ---------------------------------------------------

    async def _refresh_catalog(self) -> None:
        """Fetch and parse the THREDDS catalog XML for available files."""
        if time.time() - self._catalog_fetched_at < self._CATALOG_MAX_AGE:
            return

        url = f"{self._base_url}{self._CATALOG_PATH}"
        try:
            client = await self._get_client()
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning("Nordic catalog fetch failed: HTTP %d", resp.status_code)
                return

            self._catalog = _parse_nordic_catalog(resp.content)
            self._catalog_fetched_at = time.time()
            logger.debug("Nordic catalog: %d files available", len(self._catalog))
        except Exception:
            logger.exception("Error fetching Nordic catalog")

    def _find_closest_file(self, target_ts: int) -> str | None:
        """Find the catalog file closest to the target unix timestamp."""
        if not self._catalog:
            return None

        best_file = None
        best_diff = float("inf")
        for ts, filename in self._catalog:
            diff = abs(ts - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_file = filename

        # Only accept matches within 10 minutes of the target
        if best_diff > 600:
            return None
        return best_file

    # -- Frame fetching ------------------------------------------------------

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int
    ) -> np.ndarray | None:
        """Fetch a Nordic radar frame for the given minutes-ago offset."""
        await self._refresh_catalog()
        now_rounded = int(time.time() // 300) * 300
        target_ts = now_rounded - minutes_ago * 60
        filename = self._find_closest_file(target_ts)
        if filename is None:
            return None
        return await self._fetch_wcs(filename, region)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime
    ) -> np.ndarray | None:
        """Fetch a Nordic radar frame for a specific UTC datetime."""
        await self._refresh_catalog()
        target_ts = int(dt.timestamp())
        filename = self._find_closest_file(target_ts)
        if filename is None:
            return None
        return await self._fetch_wcs(filename, region)

    async def _fetch_wcs(
        self, filename: str, region: RegionDef
    ) -> np.ndarray | None:
        """Download a single frame via WCS GetCoverage as float32 GeoTIFF."""
        # Request the full native LCC grid — the server returns it in
        # its native projection regardless of CRS request, so we use the
        # LCC projection parameters in coordinates.py to map pixels.
        url = (
            f"{self._base_url}{self._WCS_PATH}/{filename}"
            f"?service=WCS&version=1.0.0&request=GetCoverage"
            f"&coverage=equivalent_reflectivity_factor"
            f"&CRS=OGC:CRS84"
            f"&BBOX={region.west},{region.south},{region.east},{region.north}"
            f"&format=GeoTIFF_Float"
        )
        try:
            client = await self._get_client()
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning("Nordic WCS fetch failed: HTTP %d (%s)", resp.status_code, filename)
                return None

            return _parse_nordic_geotiff(resp.content, region)
        except Exception:
            logger.exception("Error fetching Nordic WCS frame %s", filename)
            return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


def _parse_nordic_catalog(data: bytes) -> list[tuple[int, str]]:
    """Parse THREDDS catalog XML into a list of (unix_timestamp, filename)."""
    entries: list[tuple[int, str]] = []
    try:
        root = ET.fromstring(data)
        # THREDDS catalog uses a namespace
        ns = {"thredds": "http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0"}
        for dataset in root.iter("{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}dataset"):
            name = dataset.get("name", "")
            if not name.endswith(".nc"):
                continue
            match = _NORDIC_TIMESTAMP_RE.search(name)
            if not match:
                continue
            ts_str = match.group(1)  # e.g. "20260310T233000Z"
            dt = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            entries.append((int(dt.timestamp()), name))
    except Exception:
        logger.exception("Failed to parse Nordic catalog XML")
        return []

    # Sort newest-first
    entries.sort(key=lambda x: x[0], reverse=True)
    return entries


def _parse_nordic_geotiff(data: bytes, region: RegionDef) -> np.ndarray | None:
    """Parse a float32 GeoTIFF from MET Norway WCS into a uint8 dBZ array.

    Converts float32 dBZ values to the same uint8 encoding used by IEM:
    pixel = clamp((dBZ + 32) * 2, 0, 255).  NODATA (-32.5) maps to 0.
    """
    try:
        img = Image.open(io.BytesIO(data))
        arr = np.array(img, dtype=np.float32)

        return _dbz_float_to_uint8(arr)
    except Exception:
        logger.exception("Failed to parse Nordic GeoTIFF for %s", region.name)
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
