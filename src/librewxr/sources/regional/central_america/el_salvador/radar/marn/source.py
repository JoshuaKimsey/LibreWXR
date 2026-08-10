# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""El Salvador MARN/SNET radar composite source.

SNET's 120 km radar product (``esar82``) encodes reflectivity as a
continuous HSV-style gradient running from green (low dBZ) through
cyan and blue to magenta (high dBZ).  Every opaque pixel sits exactly
on the fully-saturated hue ring with one channel at 0, another at 255,
and the third varying — so decode is a direct arc-detect + linear
hue→dBZ map (no nearest-anchor search needed).

  Arc 1 (G=255, R=0, B 0→255):   hue 120° (green)   → 180° (cyan)
  Arc 2 (B=255, R=0, G 255→0):   hue 180° (cyan)    → 240° (blue)
  Arc 3 (B=255, G=0, R 0→255):   hue 240° (blue)    → 300° (magenta)

dBZ range is provisional: green=10 dBZ, magenta=70 dBZ (linear in
hue).  SNET publishes no per-bin dBZ calibration, so refine after
sampling real precipitation cells against a reference (e.g. MRMS over
adjacent overlapping coverage, or NEXRAD next time a tropical system
tracks over the Caribbean side).  The discrete legend at
``snet.gob.sv/UserFiles/SNET/Image/meteorologia/escalaPropuesta2013SNEThW_.png``
is for a *different* product (the 60 km multi-radar composite) — do
not use it for this decoder.

License: MARN explicitly permits full or partial reproduction with
citation — see README and ``docs/coverage.md`` for the attribution.
"""
import asyncio
import bisect
import io
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
from PIL import Image

from librewxr.data.regions import RegionDef
from librewxr.data.retry import retry_get
from librewxr.sources._helpers import _dbz_float_to_uint8

logger = logging.getLogger(__name__)


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
            logger.debug(
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
