# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NOAA MRMS MergedReflectivityQCComposite source.

Fetches the quality-controlled composite reflectivity product from the
NCEP real-time GRIB2 endpoint.  Supports both the CONUS product
(USCOMP/CACOMP) and regional products for Alaska, Hawaii, Caribbean
(Puerto Rico), and Guam.

The live endpoint publishes a ``.latest.grib2.gz`` file updated every
~2 minutes.  Archive files follow the pattern
``MRMS_MergedReflectivityQCComposite_00.50_YYYYMMDD-HHMMSS.grib2.gz``.

No-data is encoded as -999.0; valid values are dBZ.

MRMS routes by product (each US territory has its own regional GRIB
path), so one MRMSSource instance covers a single product.  The
``MRMSCompositeSource`` wrapper below presents the registry-friendly
single-instance facade required by the discovery walker, while
internally maintaining one MRMSSource per unique product path.
"""
import asyncio
import bisect
import gzip
import logging
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import xarray as xr

from librewxr.data.regions import RegionDef
from librewxr.data.retry import retry_get
# Shared with the NWP grid modules — kept in ``data/sources.py`` until
# Phase 3/4 of the refactor relocates it.
from librewxr.sources._helpers import _dbz_float_to_uint8

from .products import MRMS_PRODUCTS

logger = logging.getLogger(__name__)


class MRMSSource:
    """NOAA MRMS MergedReflectivityQCComposite source (one product).

    Each instance binds to a single MRMS product path (e.g.
    ``MergedReflectivityQCComposite`` for CONUS, ``ALASKA/...`` for
    AKCOMP).  The :class:`MRMSCompositeSource` wrapper below manages a
    pool of these for multi-region serving.
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
                # gzip decompress + cfgrib decode run in a worker thread
                # (full-CONUS GRIB2 parse can take seconds).
                ds = await asyncio.to_thread(_parse_mrms_grib2, resp.content)
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

            # Bilinear resample of the full CONUS grid — also heavy, so
            # run it in a worker thread.
            return await asyncio.to_thread(_resample_mrms_to_region, ds, region)

        return None

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._dir_cache = None


class MRMSCompositeSource:
    """Multi-product MRMS facade — one outer source for all US-group regions.

    The discovery walker contributes a single ``RadarSource`` instance per
    contribution.  MRMS however routes by product path (a separate
    GRIB2 series per US territory), so we pool one inner
    :class:`MRMSSource` per unique product behind this facade.  Regions
    that share a product (e.g. USCOMP and CACOMP both use the bare
    CONUS path) share one inner instance — one HTTP client, one
    directory cache, one GRIB2 download per fetch cycle.

    Calls to ``fetch_frame`` / ``fetch_archive_frame`` route by
    ``region.name`` to the right inner ``MRMSSource``.
    """

    def __init__(self, base_url: str):
        self._base_url = base_url
        # region_name -> MRMSSource.  Regions sharing a product share an
        # instance via _by_product.
        self._by_region: dict[str, MRMSSource] = {}
        self._by_product: dict[str, MRMSSource] = {}

    def _resolve(self, region: RegionDef) -> MRMSSource:
        cached = self._by_region.get(region.name)
        if cached is not None:
            return cached
        product = MRMS_PRODUCTS[region.name]
        inst = self._by_product.get(product)
        if inst is None:
            inst = MRMSSource(self._base_url, region_name=region.name)
            self._by_product[product] = inst
        self._by_region[region.name] = inst
        return inst

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int
    ) -> np.ndarray | None:
        return await self._resolve(region).fetch_frame(region, minutes_ago)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime
    ) -> np.ndarray | None:
        return await self._resolve(region).fetch_archive_frame(region, dt)

    async def close(self) -> None:
        closed: set[int] = set()
        for inst in self._by_product.values():
            if id(inst) in closed:
                continue
            await inst.close()
            closed.add(id(inst))


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
    2. Slice the MRMS grid to the region's bounding box (with extra-cell
       padding so bilinear interpolation has neighbours at the edges).
    3. Mask -999.0 (MRMS no-data) → 0.0 dBZ so it interpolates as clear
       sky instead of poisoning bilinear neighbours.
    4. Build target lat/lon axes from region bounds and pixel_size.
    5. Resample via bilinear interpolation (upscale for USCOMP/HICOMP,
       1:1 for AKCOMP/PRCOMP, mild downsample for GUCOMP).  Bilinear
       removes the doubled-pixel pattern that nearest-neighbor would
       bake into the upsampled USCOMP grid.
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

    # Slice to region bbox with extra-cell padding (bilinear needs a
    # neighbour on either side of the target span)
    pad = 2
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

    # Track MRMS no-data (-999) explicitly: substitute 0 dBZ for the
    # bilinear math (so adjacent valid pixels don't get pulled toward
    # -999 at coverage edges), and carry a parallel validity mask that
    # also gets bilinearly interpolated.  After interpolation, any
    # target pixel whose source neighbours were mostly no-data is
    # restored to the no-data sentinel (-33 → 0 in the encoder).
    nodata_mask = (data < -900).astype(np.float32)
    data = np.where(data < -900, 0.0, data).astype(np.float32)

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

    # Source grid is uniform (after crop): derive float row/col indices
    # for each target pixel by linear inversion of the grid axes.
    src_lat0 = float(lats[0])
    src_lon0 = float(lons[0])
    src_lat_step = float(lats[1] - lats[0])  # negative (descending lats)
    src_lon_step = float(lons[1] - lons[0])  # positive

    rows_f = (target_lats - src_lat0) / src_lat_step
    cols_f = (target_lons - src_lon0) / src_lon_step

    r0 = np.clip(np.floor(rows_f).astype(np.int32), 0, data.shape[0] - 1)
    c0 = np.clip(np.floor(cols_f).astype(np.int32), 0, data.shape[1] - 1)
    r1 = np.minimum(r0 + 1, data.shape[0] - 1)
    c1 = np.minimum(c0 + 1, data.shape[1] - 1)

    dr = (rows_f - r0).astype(np.float32)[:, None]
    dc = (cols_f - c0).astype(np.float32)[None, :]

    rr0 = r0[:, None]
    rr1 = r1[:, None]
    cc0 = c0[None, :]
    cc1 = c1[None, :]
    w00 = (1 - dr) * (1 - dc)
    w01 = (1 - dr) * dc
    w10 = dr * (1 - dc)
    w11 = dr * dc

    resampled = (
        data[rr0, cc0] * w00
        + data[rr0, cc1] * w01
        + data[rr1, cc0] * w10
        + data[rr1, cc1] * w11
    )

    # Bilinearly interpolate the no-data mask too.  Target pixels whose
    # source neighbourhood was majority no-data get restored to the
    # no-data sentinel; the encoder maps that to 0 (transparent).
    nodata_interp = (
        nodata_mask[rr0, cc0] * w00
        + nodata_mask[rr0, cc1] * w01
        + nodata_mask[rr1, cc0] * w10
        + nodata_mask[rr1, cc1] * w11
    )
    resampled = np.where(nodata_interp > 0.5, -33.0, resampled)

    return _dbz_float_to_uint8(resampled)
