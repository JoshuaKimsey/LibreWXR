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
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import cv2
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

# ---------------------------------------------------------------------------
# Decoded-dataset memo (Work item 2)
#
# USCOMP and CACOMP share one MRMS product (the bare CONUS path), so a single
# ``MRMSSource`` instance serves both.  Without a memo every cycle downloaded
# and decoded the same ~7000x3500 CONUS GRIB once per region.  The memo below
# keys decoded Datasets by (product, timestamp) and is bounded to a few
# entries (a decoded CONUS Dataset is ~100-200 MB), evicting the least
# recently used (oldest timestamp) first.  Four slots comfortably hold the
# ``.latest`` alias + timestamp pair plus a backfill neighbour in both the
# pipeline and single-mode budgets, reducing decode churn during backfill.
_DECODED_CACHE_MAX = 4
# The ``.latest`` URL carries no timestamp, so entries fetched through it are
# aliased under this key and time-boxed (see ``_latest_memo_ttl``): the live
# file is republished every ~2 minutes and the default fetch cycle runs every
# 10, so an entry older than the bound must never be served to a later cycle.
_LATEST_MEMO_KEY = "latest"
# Cap on the per-key fetch locks (one per in-flight (product, timestamp)).
_DECODED_LOCK_MAX = 16

# ---------------------------------------------------------------------------
# cv2.remap index-grid cache (Work item 1)
#
# The resample used to materialize ~12 full target-shaped float32 arrays per
# frame (dr/dc, four weight products, four corner gathers, resampled,
# nodata_mask, nodata_interp) — for USCOMP (12200x5400) that is a ~2.5 GB
# transient.  cv2.remap does the same bilinear gather in C with small
# per-row buffers; its map_x/map_y index grids are deterministic per
# (cropped source grid shape, region name), so they are built once and
# cached instead of per frame (~2 x 263 MB for USCOMP).  The pad depths that
# reproduce the legacy clipped-index corner handling are stored alongside.
# ---------------------------------------------------------------------------
_MAP_CACHE_MAX = 8


class _ResampleMaps(NamedTuple):
    """Cached cv2.remap index grids for one (source grid, region) pair.

    map_x/map_y are float32 (height, width) grids holding the target-pixel
    coordinates in the *continuation-padded* source array (see
    ``_pad_continuation``); the pad depths are stored alongside so the
    per-frame padding is built identically every frame.
    """

    map_x: np.ndarray
    map_y: np.ndarray
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


# keyed by ((cropped_src_rows, cropped_src_cols), region.name).  Shared across
# all MRMSSource instances (all products) and guarded by a lock because the
# resample can run concurrently in worker threads (asyncio.to_thread).
_MAP_CACHE: OrderedDict[tuple[tuple[int, int], str], _ResampleMaps] = OrderedDict()
_MAP_CACHE_LOCK = threading.Lock()


def _latest_memo_ttl() -> float:
    """Freshness bound (seconds) for ``.latest`` memo entries.

    Must stay above the spread of the concurrent same-cycle fetch batch
    (USCOMP + CACOMP fire together) yet strictly below the fetch cycle
    length so an entry is never served to a later cycle.  MRMS republishes
    the live file every ~2 minutes; with the default 600 s cycle a 180 s
    bound satisfies both.  For short custom fetch intervals the bound
    shrinks below the 30 s same-cycle floor (and below 180 s) so the
    "never served to a later cycle" guarantee survives even sub-30 s
    cycles.
    """
    try:
        from librewxr.config import settings as _settings

        cycle = float(_settings.fetch_interval)
        # 180 s cap (live file republished ~2 min), 30 s same-cycle batch
        # floor, and strictly less than one fetch cycle.
        return min(180.0, max(30.0, cycle - 60.0), max(1.0, cycle - 1.0))
    except Exception:
        return 180.0


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
        # Decoded-dataset memo: (product, timestamp) -> (Dataset, fetched_at).
        # Regions that share a product (USCOMP + CACOMP both use the CONUS
        # GRIB) reuse one download + full-grid decode per fetch cycle.
        # LRU-bounded (see _DECODED_CACHE_MAX); ``.latest`` entries are
        # additionally time-boxed (see _latest_memo_ttl).
        self._decoded_cache: OrderedDict[
            tuple[str, str], tuple[xr.Dataset, float]
        ] = OrderedDict()
        # Per-key asyncio locks so concurrent fetches for the same
        # (product, timestamp) coalesce into one HTTP GET + GRIB decode; the
        # second caller waits on the lock, then hits the warm cache.
        self._decoded_locks: dict[tuple[str, str], asyncio.Lock] = {}

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
            logger.debug(
                "MRMS directory cache refreshed: %d files, %s to %s",
                len(entries),
                entries[0][0].strftime("%Y%m%d-%H%M%S"),
                entries[-1][0].strftime("%Y%m%d-%H%M%S"),
            )

    async def _fetch_and_parse(
        self, url: str, region: RegionDef
    ) -> np.ndarray | None:
        """Download a GRIB2.gz file, parse, crop and resample to region.

        Decoded datasets are memoized per (product, timestamp) so regions
        that share one MRMS product (USCOMP and CACOMP both use the bare
        CONUS path) reuse a single download + full-grid decode per fetch
        cycle instead of each doing their own.  The get-or-compute is
        guarded by a per-key asyncio lock: concurrent fetches for the same
        key coalesce (the second caller waits, then hits the warm cache).
        A failed fetch never stores an entry, so the retry / fallback
        (IEM / MSC) semantics are unchanged.
        """
        from librewxr.config import settings as _settings

        url_key = self._memo_key_for_url(url)
        lock_key = url_key if url_key is not None else (
            self._product, _LATEST_MEMO_KEY
        )

        async with self._get_decoded_lock(lock_key):
            cached = self._lookup_decoded(
                url_key if url_key is not None else lock_key
            )
            if cached is not None:
                return await asyncio.to_thread(
                    _resample_mrms_to_region, cached, region
                )

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

                # Only memoize on success — a failed download must not
                # poison the memo (the fallback paths rely on a None result).
                self._store_decoded(url_key, lock_key, ds)

                # Bilinear resample of the full CONUS grid — also heavy, so
                # run it in a worker thread.
                return await asyncio.to_thread(
                    _resample_mrms_to_region, ds, region
                )

        return None

    def _memo_key_for_url(self, url: str) -> tuple[str, str] | None:
        """(product, timestamp) memo key for an archive URL, else None.

        The ``.latest`` URL carries no timestamp, so it returns None and the
        fetch path derives the key from the decoded dataset instead.
        """
        m = self._TIMESTAMP_RE.search(url)
        if m is not None:
            return (self._product, m.group(1))
        return None

    def _get_decoded_lock(self, key: tuple[str, str]) -> asyncio.Lock:
        """Return (creating if needed) the per-key fetch lock."""
        lock = self._decoded_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._decoded_locks[key] = lock
            self._trim_decoded_locks()
        return lock

    def _trim_decoded_locks(self) -> None:
        """Drop unlocked per-key locks once the map outgrows its cap."""
        if len(self._decoded_locks) <= _DECODED_LOCK_MAX:
            return
        for key in list(self._decoded_locks):
            if len(self._decoded_locks) <= _DECODED_LOCK_MAX:
                break
            # Never evict the ``.latest`` lock: coalescing callers (e.g.
            # USCOMP + CACOMP in one cycle) take it in turn, and trimming
            # it between their awaits would make the second caller build a
            # fresh lock and fetch independently instead of hitting the
            # alias memo.
            if key[1] == _LATEST_MEMO_KEY:
                continue
            if not self._decoded_locks[key].locked():
                del self._decoded_locks[key]

    def _lookup_decoded(self, key: tuple[str, str]) -> xr.Dataset | None:
        """Return the cached Dataset for *key*, or None.

        ``.latest`` alias entries are time-boxed: an entry fetched more than
        ``_latest_memo_ttl()`` seconds ago is dropped so a later fetch cycle
        always re-fetches the live product (MRMS republishes it every
        ~2 minutes while a default cycle runs every 10).
        """
        entry = self._decoded_cache.get(key)
        if entry is None:
            return None
        ds, fetched_at = entry
        if key[1] == _LATEST_MEMO_KEY and (
            time.time() - fetched_at
        ) > _latest_memo_ttl():
            del self._decoded_cache[key]
            return None
        self._decoded_cache.move_to_end(key)
        return ds

    def _store_decoded(
        self, url_key: tuple[str, str] | None, lock_key: tuple[str, str], ds: xr.Dataset
    ) -> None:
        """Insert *ds* into the memo, LRU-evicting down to the cap.

        Archive URLs key by the timestamp embedded in the filename.  The
        ``.latest`` URL has no timestamp, so its entry is keyed by the
        dataset's own validity time when one can be derived, plus an alias
        under the lock key so a concurrent caller that waited on the lock
        (and so doesn't know the derived timestamp) can still hit the warm
        cache.  The cap is small, so inserting a newer timestamp evicts the
        oldest one — the fetch window slides every cycle and stale datasets
        are never served.
        """
        now = time.time()
        canonical = url_key
        if canonical is None:
            ts = _dataset_timestamp(ds)
            canonical = (self._product, ts) if ts is not None else lock_key
        self._decoded_cache[canonical] = (ds, now)
        self._decoded_cache.move_to_end(canonical)
        if lock_key != canonical:
            self._decoded_cache[lock_key] = (ds, now)
            self._decoded_cache.move_to_end(lock_key)
        while len(self._decoded_cache) > _DECODED_CACHE_MAX:
            self._decoded_cache.popitem(last=False)

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._dir_cache = None
        self._decoded_cache.clear()
        self._decoded_locks.clear()


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


def _dataset_timestamp(ds: xr.Dataset) -> str | None:
    """Best-effort 'YYYYMMDD-HHMMSS' of the dataset's validity time.

    cfgrib decodes a ``time`` (and for forecast data a ``valid_time``)
    coordinate for real GRIB2 files.  Used to key ``.latest`` memo entries by
    the file's actual data timestamp; returns None when no time coordinate is
    present (e.g. synthetic test datasets) or it can't be coerced.
    """
    for coord in ("valid_time", "time"):
        if coord not in ds.coords:
            continue
        try:
            val = np.asarray(ds[coord].values).reshape(-1)
            if val.size == 0:
                continue
            ts_ns = int(np.asarray(val[0], dtype="datetime64[ns]").astype(np.int64))
            return datetime.fromtimestamp(
                ts_ns / 1e9, tz=timezone.utc
            ).strftime("%Y%m%d-%H%M%S")
        except (TypeError, ValueError, OverflowError):
            continue
    return None


def _build_resample_maps(
    rows_f: np.ndarray,
    cols_f: np.ndarray,
    src_rows: int,
    src_cols: int,
) -> _ResampleMaps:
    """Build cached cv2.remap index grids for one (source grid, region) pair.

    rows_f / cols_f are the float row/col indices of every target pixel in
    the cropped source grid (deterministic per pair — the source MRMS grid is
    fixed).  The legacy broadcast implementation clamped the corner indices
    (r0 = clip(floor(rows_f), 0, H-1), r1 = min(r0+1, H-1)) while keeping the
    *unclamped* fractional offsets, which reproduces:

    - north/west overshoot (rows_f < 0 / cols_f < 0): a linear continuation
      of the edge gradient between source rows/cols 0-1 (r0 = 0, r1 = 1);
    - south/east overshoot (rows_f >= H-1): a constant edge row/col
      (r0 = r1 = H-1).

    cv2.remap cannot express either, so the source arrays are padded with the
    continuation above/left and a replication below/right (see
    ``_pad_continuation``) and the maps carry the padded coordinates, keeping
    every sample in-bounds.  The pad depths are derived from the extrema of
    rows_f / cols_f and stored with the maps.
    """
    pad_top = max(0, -int(np.floor(np.min(rows_f))))
    pad_bottom = max(1, int(np.floor(np.max(rows_f))) - (src_rows - 1) + 1)
    pad_left = max(0, -int(np.floor(np.min(cols_f))))
    pad_right = max(1, int(np.floor(np.max(cols_f))) - (src_cols - 1) + 1)

    shape = (len(rows_f), len(cols_f))
    map_x = np.broadcast_to(
        (cols_f + pad_left).astype(np.float32)[None, :], shape
    ).astype(np.float32)
    map_y = np.broadcast_to(
        (rows_f + pad_top).astype(np.float32)[:, None], shape
    ).astype(np.float32)

    return _ResampleMaps(
        map_x, map_y, pad_top, pad_bottom, pad_left, pad_right
    )


def _get_resample_maps(
    rows_f: np.ndarray,
    cols_f: np.ndarray,
    src_rows: int,
    src_cols: int,
    region: RegionDef,
) -> _ResampleMaps:
    """Return (building if needed) the cached index grids for this pair."""
    key = ((src_rows, src_cols), region.name)
    with _MAP_CACHE_LOCK:
        entry = _MAP_CACHE.get(key)
        if entry is not None:
            _MAP_CACHE.move_to_end(key)
            return entry
        entry = _build_resample_maps(rows_f, cols_f, src_rows, src_cols)
        _MAP_CACHE[key] = entry
        while len(_MAP_CACHE) > _MAP_CACHE_MAX:
            _MAP_CACHE.popitem(last=False)
        return entry


def _pad_continuation(
    arr: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
) -> np.ndarray:
    """Pad *arr* so cv2.remap reproduces the legacy clipped-index bilinear.

    Top/left overshoot interpolates between source rows/cols 0-1 with
    unclamped fractional weights — a linear continuation of the edge
    gradient, so those pad rows/cols are ``(1+k)*edge0 - k*edge1``.
    Bottom/right overshoot collapses to a constant edge (both clipped
    corners equal the last row/col), so those pads replicate the edge.
    Rows are padded first so corner cells (overshoot in both axes)
    interpolate the same four real corner cells the legacy code gathered.
    """
    if pad_top > 0:
        # k runs pad_top..1 (topmost row is the deepest continuation).  With
        # map_y = rows_f + pad_top in (0, 1] the remap then samples the
        # k=pad_top..pad_top-1 rows — exactly the unclamped weight pair the
        # legacy (1 - dr) * edge0 + dr * edge1 produces for dr = -rows_f.
        k = (pad_top - np.arange(pad_top, dtype=np.float32))[:, None]
        arr = np.concatenate([(1 + k) * arr[0:1, :] - k * arr[1:2, :], arr], axis=0)
    if pad_bottom > 0:
        arr = np.concatenate([arr, np.repeat(arr[-1:, :], pad_bottom, axis=0)], axis=0)
    if pad_left > 0:
        k = (pad_left - np.arange(pad_left, dtype=np.float32))[None, :]
        arr = np.concatenate([(1 + k) * arr[:, 0:1] - k * arr[:, 1:2], arr], axis=1)
    if pad_right > 0:
        arr = np.concatenate([arr, np.repeat(arr[:, -1:], pad_right, axis=1)], axis=1)
    return arr


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
    5. Resample via cv2.remap bilinear interpolation (upscale for
       USCOMP/HICOMP, 1:1 for AKCOMP/PRCOMP, mild downsample for GUCOMP).
       The map_x/map_y index grids are cached per (source grid, region), and
       the source is continuation-padded so the legacy clipped-index corner
       behaviour (linear continuation north/west, constant edge south/east)
       is reproduced exactly.  The no-data mask is resampled the same way so
       target pixels whose source neighbourhood was majority no-data are
       restored to the no-data sentinel.
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

    maps = _get_resample_maps(
        rows_f, cols_f, data.shape[0], data.shape[1], region
    )

    padded = _pad_continuation(
        data, maps.pad_top, maps.pad_bottom, maps.pad_left, maps.pad_right,
    )
    padded_mask = _pad_continuation(
        nodata_mask,
        maps.pad_top, maps.pad_bottom, maps.pad_left, maps.pad_right,
    )

    # cv2.remap bilinear interpolation on the float32 source.  The maps
    # carry coordinates in the padded array, so every sample lands
    # in-bounds and the border mode is never involved.
    resampled = cv2.remap(padded, maps.map_x, maps.map_y, cv2.INTER_LINEAR)
    nodata_interp = cv2.remap(
        padded_mask, maps.map_x, maps.map_y, cv2.INTER_LINEAR
    )
    resampled = np.where(nodata_interp > 0.5, -33.0, resampled)

    return _dbz_float_to_uint8(resampled)
