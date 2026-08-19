# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""PAGASA PANAHON radar composite source.

Fetches the Philippine national radar mosaic served anonymously from
``cdn.panahon.gov.ph/api/v1`` — a JSON timeline endpoint that returns
six frames at 15-min cadence with explicit UTC ``observed_at_unix``
timestamps, paired with 2048×2048 RGBA PNGs in EPSG:4326
equirectangular projection.  The mosaic is a union of ~240 km coverage
circles around the national station network (nine stations in the
app's current list: Baguio, Baler, Echague, Daet, Guiuan, Iloilo,
Davao, Kabacan, Panabo) — a 2026-08-19 probe found a uniform ~240 km
cap on every station that could be tested.

License: public domain per Philippine IP code RA 8293 §176 (government-
works exception); attribution to PAGASA / DOST recorded in README.

Two architectural quirks unique to PAGASA among LibreWXR sources:

  * **15-min native cadence vs 10-min store cadence.** The fetcher
    requests frames at the 10-min grid; the source rounds each request
    to the nearest native frame, ≤7.5 min off, invisible in a
    RainViewer-style animation.  No interpolation step.

  * **Discrete alpha-encoded palette.** The PNG uses exact-RGB palette
    stops (zero anti-aliasing), with alpha discretised per stop: the
    three "weak-echo" gray stops have α=60/120/200, all precip stops
    have α=255.  Treating α=0 as no-data is sufficient; we also drop
    the three grays so output dBZ never falls below LibreWXR's typical
    noise floor (default 10 dBZ).
"""
import asyncio
import io
import json
import logging
import time
from datetime import datetime, timezone

import httpx
import numpy as np
from PIL import Image

from librewxr.data.regions import RegionDef
from librewxr.data.retry import retry_get
from librewxr.sources._helpers import _dbz_float_to_uint8

logger = logging.getLogger(__name__)


# ── Palette ─────────────────────────────────────────────────────────
#
# 13 exact RGB stops from the PANAHON JS bundle's
# ``generateGradientColormap([...], 13, [0, 75])`` call — a linear
# 0–75 dBZ ramp at 6.25 dBZ per stop.  The three grays (alpha 60/120/
# 200 in the source PNG) are below LibreWXR's default noise floor and
# get treated as no-data; the 10 precip stops (alpha 255) decode to
# real dBZ values.  The PNG renders without anti-aliasing in the RGB
# channel — every visible pixel is an exact palette match — so
# nearest-RGB lookup with a small tolerance is sufficient.
_PAGASA_PALETTE: tuple[tuple[int, int, int, float], ...] = (
    # (R, G, B, dBZ).  First three are weak-echo grays, treated as
    # clutter / no-data downstream.
    (0x9c, 0x9c, 0x9c,  0.00),
    (0xb4, 0xb4, 0xb4,  6.25),
    (0xc8, 0xc8, 0xc8, 12.50),
    (0x00, 0xff, 0x6e, 18.75),
    (0x00, 0xe6, 0x00, 25.00),
    (0x00, 0xc8, 0x00, 31.25),
    (0xff, 0xff, 0x00, 37.50),
    (0xff, 0xd2, 0x00, 43.75),
    (0xff, 0x8c, 0x00, 50.00),
    (0xff, 0x50, 0x50, 56.25),
    (0xff, 0x00, 0x00, 62.50),
    (0xc8, 0x00, 0x00, 68.75),
    (0xff, 0x00, 0xff, 75.00),
)

# Number of leading palette stops to drop as weak-echo / clutter.
# These render in the source PNG with α<255 — the PNG itself flags
# them as below-confident-precip — and at ≤12.5 dBZ they're below
# the default ``noise_floor_dbz=10`` anyway.  Dropping them here keeps
# the source's output clean and matches the alpha-zero treatment.
_PAGASA_CLUTTER_STOPS = 3

# Squared RGB distance below which a pixel is considered a palette
# match.  4 = ~1 per channel — tighter than MMD's 64 because PAGASA's
# PNG is unaliased so palette hits land exactly on a stop.  The
# allowance only forgives 1-bit rounding in case the upstream renderer
# changes its bit-depth.
_PAGASA_MAX_RGB_DIST2 = 4

# Native PNG dimensions (square 2048×2048) — declared up front so a
# malformed response can't sneak past the decoder with a different
# shape and produce a region grid mismatch later.
_PAGASA_EXPECTED_WIDTH = 2048
_PAGASA_EXPECTED_HEIGHT = 2048

# Number of frames in the timeline endpoint.  PAGASA serves a rolling
# 6-frame buffer (≈75 min of backfill at 15-min cadence).
_PAGASA_EXPECTED_FRAMES = 6

# Native cadence (15 min).  Store cadence is 10 min; the source maps
# each requested store slot to the nearest native frame.
_PAGASA_NATIVE_CADENCE_SEC = 900

# Maximum acceptable distance (seconds) between a requested store slot
# and the nearest native PAGASA frame.  Half the native cadence
# (7.5 min) is the worst-case round-to-nearest offset; anything farther
# means the requested slot predates the rolling buffer.
_MAX_NEAREST_OFFSET_SEC = _PAGASA_NATIVE_CADENCE_SEC // 2


def _build_palette_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rgb_int32 [N,3], dbz_float32 [N], is_clutter_bool [N])."""
    rgb = np.array(
        [(r, g, b) for r, g, b, _ in _PAGASA_PALETTE], dtype=np.int32,
    )
    dbz = np.array(
        [d for *_, d in _PAGASA_PALETTE], dtype=np.float32,
    )
    is_clutter = np.array(
        [i < _PAGASA_CLUTTER_STOPS for i in range(len(_PAGASA_PALETTE))],
        dtype=bool,
    )
    return rgb, dbz, is_clutter


def _decode_pagasa_png(rgba: np.ndarray) -> np.ndarray:
    """Decode a (H,W,4) RGBA array into a uint8 dBZ array (H,W).

    Pixels with α=0 are no-data (transparent background — almost the
    entire ocean).  For visible pixels, find the nearest palette stop
    by squared RGB distance; pixels whose nearest match is one of the
    leading clutter stops are also treated as no-data (the PNG flags
    these with α<255).  Final encoding is LibreWXR's shared uint8 dBZ
    scheme: clamp((dBZ+32)*2, 0, 255), no-data → 0.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"expected (H,W,4) RGBA array, got {rgba.shape}")

    h, w, _ = rgba.shape
    alpha = rgba[..., 3]
    rgb = rgba[..., :3]

    palette_rgb, palette_dbz, palette_clutter = _build_palette_arrays()

    # Only run the (expensive) nearest-stop search on visible pixels.
    # The PNG is sparse (<1% visible coverage in typical frames), so
    # this skips ~99% of the array.
    visible_mask = alpha > 0
    visible_pixels = rgb[visible_mask].astype(np.int32)

    out = np.zeros((h, w), dtype=np.uint8)
    if visible_pixels.size == 0:
        return out

    # Squared distance from each visible pixel to each palette entry.
    dists = np.sum(
        (visible_pixels[:, None, :] - palette_rgb[None, :, :]) ** 2,
        axis=2,
    )
    nearest_idx = np.argmin(dists, axis=1)
    nearest_dist2 = dists[np.arange(visible_pixels.shape[0]), nearest_idx]
    palette_hit = nearest_dist2 <= _PAGASA_MAX_RGB_DIST2
    not_clutter = ~palette_clutter[nearest_idx]
    valid = palette_hit & not_clutter

    dbz_vals = palette_dbz[nearest_idx[valid]]
    # Shared uint8 encoding via librewxr.sources._helpers:
    # clamp(((dBZ + 32) * 2), 0, 255); NODATA → 0.
    encoded = _dbz_float_to_uint8(dbz_vals)

    # Scatter the valid hits back into the full-frame output.
    visible_indices = np.flatnonzero(visible_mask)
    out.flat[visible_indices[valid]] = encoded
    return out


class PAGASASource:
    """PANAHON-mosaic radar composite source for the Philippines.

    A single timeline fetch returns six 15-min-spaced PNG URLs with
    explicit UTC timestamps.  We fetch them in parallel, decode each,
    and cache by frame timestamp.  Per-region requests at the 10-min
    store grid round to the nearest cached native timestamp (≤7.5 min
    off, invisible in animation).

    The fetcher's burst of concurrent per-slot calls inside one cycle
    is coalesced via ``_REFRESH_TTL_SEC`` — after a timeline refresh,
    all subsequent calls hit the in-memory cache until the next cycle.
    """

    _TIMELINE_PATH = "/api/v1/radar/timeline"
    _IMAGE_PATH = "/api/v1/radar-image"
    _SUBLAYER = "hybrid-reflectivity"

    # Don't re-fetch the timeline more than once per ~2 min — covers
    # the fetcher's burst of concurrent per-slot calls in one cycle
    # without unnecessary network round trips.
    _REFRESH_TTL_SEC = 120

    # Keep at most this many decoded frames in cache (~2 full timeline
    # buffers + headroom for store walk-back during cycle overlap).
    _CACHE_MAX = 24

    def __init__(self, base_url: str = "https://cdn.panahon.gov.ph"):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._frame_cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []
        self._last_fetch_unix: float = 0.0
        self._refresh_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,
            )
        return self._client

    @property
    def _timeline_url(self) -> str:
        return f"{self._base_url}{self._TIMELINE_PATH}"

    def _image_url(self, index: int) -> str:
        return (
            f"{self._base_url}{self._IMAGE_PATH}"
            f"?sublayer={self._SUBLAYER}&index={index}"
        )

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int,
    ) -> np.ndarray | None:
        """Return the frame for ``minutes_ago`` slots back, or ``None``.

        ``minutes_ago`` is on the 10-min store grid; PAGASA's native
        grid is 15 min, so this rounds to the nearest native frame.
        """
        now_rounded = (int(time.time()) // 600) * 600
        target_ts = now_rounded - minutes_ago * 60
        return await self._fetch_for_ts(target_ts, region)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime,
    ) -> np.ndarray | None:
        """Best-effort archive lookup — the timeline endpoint only
        carries ~75 min of backfill, so anything older returns None.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return await self._fetch_for_ts(int(dt.timestamp()), region)

    async def _fetch_for_ts(
        self, ts: int, region: RegionDef,
    ) -> np.ndarray | None:
        if region.name != "PHCOMP":
            raise ValueError(
                f"PAGASASource cannot decode region {region.name!r}; "
                "expected PHCOMP"
            )

        # Try the in-memory cache first.
        nearest = self._nearest_cached(ts)
        if nearest is not None:
            return self._frame_cache[nearest]

        # Cache miss — refresh once per TTL window so a flurry of
        # concurrent per-slot calls in one cycle shares one fetch.
        if time.time() - self._last_fetch_unix >= self._REFRESH_TTL_SEC:
            await self._refresh()
            nearest = self._nearest_cached(ts)
            if nearest is not None:
                return self._frame_cache[nearest]

        return None

    def _nearest_cached(self, ts: int) -> int | None:
        """Return the cached frame timestamp closest to ``ts``, within
        ``_MAX_NEAREST_OFFSET_SEC`` — or None if no cached frame is in
        range.  Keeps the 15-min → 10-min mapping invisible to callers.
        """
        if not self._cache_order:
            return None
        nearest = min(self._cache_order, key=lambda t: abs(t - ts))
        if abs(nearest - ts) > _MAX_NEAREST_OFFSET_SEC:
            return None
        return nearest

    async def _refresh(self) -> None:
        """Fetch the timeline + all 6 image frames; populate the cache."""
        async with self._refresh_lock:
            # Re-check inside the lock — another coroutine may already
            # have refreshed while we were queued.
            if time.time() - self._last_fetch_unix < self._REFRESH_TTL_SEC:
                return
            self._last_fetch_unix = time.time()

            client = await self._get_client()
            resp = await retry_get(
                client, self._timeline_url, log_name="PAGASA-timeline",
            )
            if resp is None or resp.status_code != 200:
                logger.warning(
                    "PAGASA timeline fetch failed: status=%s",
                    "None" if resp is None else resp.status_code,
                )
                return

            try:
                payload = resp.json()
            except json.JSONDecodeError:
                logger.warning("PAGASA timeline returned non-JSON body")
                return

            entries = (payload.get("data") or {}).get("timeline") or []
            if not entries:
                logger.warning("PAGASA timeline empty: %s", payload)
                return

            # Fetch all image frames in parallel by their stable index.
            # The timeline gives index → ts mapping; the image endpoint
            # is keyed by that same index.  We keep using the CDN
            # endpoint rather than the ``image_url`` field in the JSON
            # (it points at the origin host; a 2026-08-19 probe found it
            # now works without a CSRF token via a 301 to HTTPS, but the
            # CDN host remains the preferred path).
            tasks = [
                self._fetch_image(client, i)
                for i in range(min(len(entries), _PAGASA_EXPECTED_FRAMES))
            ]
            pngs = await asyncio.gather(*tasks, return_exceptions=True)

            for i, (entry, png) in enumerate(zip(entries, pngs)):
                if isinstance(png, BaseException):
                    logger.warning(
                        "PAGASA image %d fetch raised: %s", i, png,
                    )
                    continue
                if png is None:
                    continue
                try:
                    ts = int(entry["observed_at_unix"])
                except (KeyError, TypeError, ValueError):
                    logger.warning(
                        "PAGASA timeline entry %d missing observed_at_unix",
                        i,
                    )
                    continue
                try:
                    grid = self._decode_png(png)
                except Exception:
                    logger.exception("PAGASA frame %d decode failed", i)
                    continue
                self._frame_cache[ts] = grid
                if ts not in self._cache_order:
                    self._cache_order.append(ts)

            self._cache_order.sort()
            while len(self._cache_order) > self._CACHE_MAX:
                evict = self._cache_order.pop(0)
                self._frame_cache.pop(evict, None)

    async def _fetch_image(
        self, client: httpx.AsyncClient, index: int,
    ) -> bytes | None:
        url = self._image_url(index)
        resp = await retry_get(client, url, log_name=f"PAGASA-img-{index}")
        if resp is None or resp.status_code != 200:
            return None
        return resp.content

    def _decode_png(self, png_bytes: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(png_bytes))
        if img.size != (_PAGASA_EXPECTED_WIDTH, _PAGASA_EXPECTED_HEIGHT):
            raise ValueError(
                f"PAGASA PNG unexpected size {img.size}, "
                f"expected ({_PAGASA_EXPECTED_WIDTH},"
                f" {_PAGASA_EXPECTED_HEIGHT})"
            )
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return _decode_pagasa_png(np.array(img))

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()