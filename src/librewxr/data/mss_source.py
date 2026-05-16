# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Singapore MSS (Meteorological Service Singapore) radar source.

Fetches the 50 km / 5-min rectangular rain-area product served
anonymously from ``weather.gov.sg/files/rainarea/50km/v2/`` as RGBA
PNGs.  Despite the marketing name "50 km", the upstream file path uses
``dpsri_70km_...`` (70 km radial range, cropped to a horizontal display
rectangle of ~108×60 km centred on MSS Changi).

This source replaced the older 480 km super-regional product when MET
Malaysia coverage landed.  MET Malaysia covers Peninsular Malaysia +
Borneo + Brunei + N. Sumatra at higher resolution, so the 480 km
overlap was redundant; the 30-min native cadence on the 480 km
product was also a poor fit for our 10-min store cadence (required
optical-flow interpolation between every native frame).

The 50 km product's 5-min native cadence aligns trivially with our
10-min store grid — every requested slot lands on an exact native
frame.  No interpolation, no flow cache, no walk-back fallback: just
fetch the PNG, decode the palette, return the grid.

License: Singapore Open Data Licence v1.0 (data.gov.sg / MSS / NEA).
Attribution recorded in README and docs/coverage.md.
"""
import asyncio
import io
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np
from PIL import Image

from librewxr.data.regions import RegionDef
from librewxr.data.retry import retry_get

logger = logging.getLogger(__name__)


# ── Palette ─────────────────────────────────────────────────────────
#
# 31 discrete RGB stops used by every MSS rain-area product (50 km, 240
# km, and 480 km all share the same palette).  Each stop maps to a dBZ
# value spanning ~5 dBZ (lightest cyan) to 75 dBZ (extreme magenta) in
# even ~2.3 dBZ steps.  Treats dBR (rain-rate output of the DPSRI
# product) as dBZ for visualisation — strict M-P conversion would shift
# absolute values slightly but doesn't change relative animation.
#
# Match is by squared-RGB distance with a ``_MSS_MAX_RGB_DIST2``
# threshold below which the nearest anchor wins; pixels outside the
# threshold are treated as no-data.  The PNG palette is lossless so
# anchors arrive exact in practice — tolerance exists to absorb future
# anti-aliasing tweaks without breaking the decoder.
_MSS_PALETTE: tuple[tuple[int, int, int, float], ...] = (
    # Cyan family (lightest precipitation)
    (0, 239, 239, 5.0),
    (0, 255, 255, 7.3),
    (0, 209, 213, 9.7),
    (0, 186, 191, 12.0),
    (0, 151, 154, 14.3),
    (0, 131, 125, 16.7),
    # Green family (moderate precipitation)
    (0, 128, 69, 19.0),
    (0, 137, 56, 21.3),
    (0, 162, 53, 23.7),
    (0, 183, 41, 26.0),
    (0, 202, 17, 28.3),
    (0, 218, 13, 30.7),
    (0, 245, 7, 33.0),
    (0, 255, 0, 35.3),
    (67, 255, 65, 37.7),
    (72, 255, 70, 40.0),
    # Yellow → orange → red (heavy precipitation)
    (255, 255, 59, 42.3),
    (255, 255, 0, 44.7),
    (255, 240, 0, 47.0),
    (255, 220, 0, 49.3),
    (255, 198, 0, 51.7),
    (255, 178, 0, 54.0),
    (255, 165, 0, 56.3),
    (255, 138, 0, 58.7),
    (255, 114, 0, 61.0),
    (255, 73, 0, 63.3),
    (255, 31, 0, 65.7),
    # Red family (severe)
    (229, 0, 0, 68.0),
    (193, 0, 0, 70.3),
    # Magenta (extreme)
    (182, 0, 106, 72.7),
    (210, 0, 165, 75.0),
)
_MSS_MAX_RGB_DIST2 = 64

# Native cadence (5 min — twice our store cadence; every 10-min slot
# lands on a native frame).
_CADENCE_SEC = 300
# How long to cache a confirmed 404 before retrying the same URL.
# Short enough that a slot which publishes a few minutes after we
# first asked still gets picked up on the next fetch cycle.
_NONE_CACHE_TTL_SEC = 120
# Cap on the per-ts cache size (≈ 4 hours of 5-min natives).
_CACHE_MAX = 48
# Singapore local time = UTC+8 year-round (no DST).  MSS filenames
# encode the data timestamp in SGT, not UTC — same convention as MET
# Malaysia and CWA Taiwan.
_LOCAL_TZ_OFFSET_HOURS = 8


def _decode_mss_png(
    png_bytes: bytes, region: RegionDef,
) -> np.ndarray | None:
    """Decode an MSS rain-area RGBA PNG into a uint8 dBZ array.

    Snaps each opaque pixel to the nearest ``_MSS_PALETTE`` anchor by
    squared-RGB distance; pixels farther than ``_MSS_MAX_RGB_DIST2``
    from every anchor are treated as no-data.  Returns ``None`` if the
    PNG fails to parse.
    """
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception:
        logger.exception("Failed to decode MSS PNG")
        return None

    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    if (h, w) != (region.height, region.width):
        logger.warning(
            "MSS PNG shape mismatch: got %dx%d, expected %dx%d for %s",
            h, w, region.height, region.width, region.name,
        )

    rgb = arr[..., :3].astype(np.int32)
    alpha = arr[..., 3]

    anchors_rgb = np.array(
        [(r, g, b) for r, g, b, _ in _MSS_PALETTE], dtype=np.int32,
    )
    anchors_dbz = np.array(
        [dbz for *_, dbz in _MSS_PALETTE], dtype=np.float32,
    )

    flat = rgb.reshape(-1, 3)
    diffs = flat[:, None, :] - anchors_rgb[None, :, :]
    dist2 = np.sum(diffs * diffs, axis=2)

    nearest_idx = np.argmin(dist2, axis=1)
    nearest_dist2 = dist2[np.arange(len(flat)), nearest_idx]

    dbz_flat = anchors_dbz[nearest_idx]
    valid = (alpha.reshape(-1) > 0) & (nearest_dist2 <= _MSS_MAX_RGB_DIST2)
    dbz_flat = np.where(valid, dbz_flat, -33.0)

    # Shared uint8 encoding: clamp((dBZ + 32) * 2, 0, 255); NODATA → 0.
    dbz_grid = dbz_flat.reshape(h, w)
    nodata = dbz_grid <= -32.0
    out = np.clip((dbz_grid + 32.0) * 2.0, 0, 255).astype(np.uint8)
    out[nodata] = 0
    return out


class MSSSource:
    """Singapore MSS 50 km rain-area radar source.

    Fetches one PNG per requested 10-min store slot from
    ``weather.gov.sg/files/rainarea/50km/v2/``.  The native 5-min
    cadence aligns cleanly with our 10-min slots (every 10-min
    boundary is also a 5-min boundary), so requested slots always map
    1:1 to native frames — no interpolation, no flow cache.

    Cache + 404 handling mirror MMD's pattern: per-ts dict with a
    short TTL on cached ``None`` so a slot that publishes a few
    minutes after we first asked still gets picked up.

    License: Singapore Open Data Licence v1.0.  Attribution required.
    """

    def __init__(
        self, base_url: str = "https://www.weather.gov.sg/files/rainarea/50km/v2",
    ):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        # ts -> uint8 grid (or None for cached 404)
        self._cache: dict[int, np.ndarray | None] = {}
        self._cache_time: dict[int, float] = {}
        self._cache_order: list[int] = []
        self._locks: dict[int, asyncio.Lock] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,
            )
        return self._client

    @classmethod
    def _aligned_ts(cls, ts: int) -> int:
        """Round *ts* down to the native 5-min cadence boundary."""
        return (ts // _CADENCE_SEC) * _CADENCE_SEC

    def _url_for_ts(self, ts: int) -> str:
        """Build the PNG URL for a 5-min-aligned UTC unix timestamp.

        The filename encodes Singapore local time (UTC+8), not UTC —
        same convention as CWA Taiwan and MET Malaysia.  The product
        name "dpsri_70km" reflects the 70 km radial range despite
        weather.gov.sg marketing it as the "50 km" view.
        """
        sgt = datetime.fromtimestamp(
            ts, tz=timezone.utc,
        ) + timedelta(hours=_LOCAL_TZ_OFFSET_HOURS)
        fname = (
            f"dpsri_70km_{sgt.strftime('%Y%m%d%H%M')}0000dBR.dpsri.png"
        )
        return f"{self._base_url}/{fname}"

    # Backwards-compat alias for tests that exercise the URL builder
    # via the older entry point name.
    def _url_for_timestamp(self, ts: int) -> str:
        return self._url_for_ts(self._aligned_ts(ts))

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int,
    ) -> np.ndarray | None:
        # LibreWXR's 10-min store cadence is aligned to clock boundaries;
        # every such boundary is also a 5-min boundary.
        now_aligned = (int(time.time()) // 600) * 600
        target_ts = now_aligned - minutes_ago * 60
        return await self._fetch_for_ts(target_ts, region)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime,
    ) -> np.ndarray | None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return await self._fetch_for_ts(int(dt.timestamp()), region)

    async def _fetch_for_ts(
        self, ts: int, region: RegionDef,
    ) -> np.ndarray | None:
        ts = self._aligned_ts(ts)
        if self._cache_hit(ts):
            return self._cache[ts]

        lock = self._locks.setdefault(ts, asyncio.Lock())
        async with lock:
            # Re-check after acquiring the lock — another coroutine may
            # have populated the cache while we were waiting.
            if self._cache_hit(ts):
                return self._cache[ts]

            client = await self._get_client()
            url = self._url_for_ts(ts)
            resp = await retry_get(client, url, log_name="MSS")

            frame: np.ndarray | None = None
            if resp is None:
                frame = None
            elif resp.status_code == 200:
                frame = _decode_mss_png(resp.content, region)
            elif resp.status_code == 404:
                frame = None
            else:
                logger.warning(
                    "MSS fetch failed: HTTP %d (%s)",
                    resp.status_code, url.rsplit("/", 1)[-1],
                )
                frame = None

            self._cache_put(ts, frame)
            self._locks.pop(ts, None)
            return frame

    def _cache_hit(self, ts: int) -> bool:
        if ts not in self._cache:
            return False
        if self._cache[ts] is not None:
            return True
        # None entry — only trust within TTL window.
        if time.time() - self._cache_time.get(ts, 0.0) < _NONE_CACHE_TTL_SEC:
            return True
        # Stale miss: drop bookkeeping so the fetch path proceeds.
        self._cache.pop(ts, None)
        self._cache_time.pop(ts, None)
        if ts in self._cache_order:
            self._cache_order.remove(ts)
        return False

    def _cache_put(self, ts: int, frame: np.ndarray | None) -> None:
        self._cache[ts] = frame
        self._cache_time[ts] = time.time()
        if ts not in self._cache_order:
            self._cache_order.append(ts)
        while len(self._cache_order) > _CACHE_MAX:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
            self._cache_time.pop(evict, None)

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
