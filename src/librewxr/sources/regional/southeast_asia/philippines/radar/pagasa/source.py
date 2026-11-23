# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""PAGASA PANAHON radar composite source.

Fetches the Philippine national radar mosaic served anonymously from
``panahon.gov.ph`` -- a same-origin JSON+PNG API that replaced the retired
``cdn.panahon.gov.ph`` host in November 2026.  The timeline endpoint returns
nine frames at 10-min cadence with explicit UTC ``observed_at_unix``
timestamps; the companion ``radar-data-image`` endpoint serves the matching
2048x2048 RGBA PNGs in EPSG:4326 equirectangular projection, keyed by that
timestamp.  The mosaic is a union of ~240 km coverage circles around the
national station network (nine stations in the app's current list: Baguio,
Baler, Echague, Daet, Guiuan, Iloilo, Davao, Kabacan, Panabo) -- a
2026-08-19 probe found a uniform ~240 km cap on every station that could be
tested.

License: public domain per Philippine IP code RA 8293 section 176
(government-works exception); attribution to PAGASA / DOST recorded in
README.

Architectural quirks unique to PAGASA among LibreWXR sources:

  * **10-min native cadence.**  The timeline publishes at the same 10-min
    cadence as the store grid, so each requested store slot maps 1:1 onto
    a native frame.  The nearest-frame lookup stays for clock skew and
    archive requests.

  * **Timeline-provided palette.**  The palette is no longer hard-coded:
    every timeline response carries its ``colormap`` (14 stops today) and
    ``scale.max`` (80 dBZ), and those are authoritative.  The module keeps
    a verbatim fallback table for a response that omits or mangles the
    field -- upstream has changed its palette silently before.

  * **Synthesized image URLs.**  The per-entry ``image_url`` is always
    empty now, so the image URL is built from ``observed_at_unix`` plus the
    response's ``tile_version``.

  * **Placeholder sentinel.**  When the upstream radar renderer is down,
    the image endpoint answers HTTP 200 with a universal 1x1 RGBA
    placeholder instead of an error.  The 2048x2048 decode-size guard
    rejects it, so such frames are skipped with a warning rather than
    silently replacing real data.

  * **Browser User-Agent.**  The host soft-blocks non-browser user agents
    with a 1280x720 decoy PNG at HTTP 200 (verified 2026-11-23), so the
    client sends a browser UA to reach the real JSON and PNG endpoints.
"""
import asyncio
import io
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
# Fallback palette: the 14 stops the new timeline API reported on
# 2026-11-23, verbatim.  Each stop is ((dBZ_min, dBZ_max),
# (R, G, B, A)).  The live timeline response's ``colormap`` is the
# authoritative source (see ``_build_palette_arrays``); this table is
# only used when that field is missing or malformed, because upstream
# has rewritten its palette without notice before.
_PAGASA_FALLBACK_COLORMAP: tuple[
    tuple[tuple[float, float], tuple[int, int, int, int]], ...
] = (
    ((0.0, 5.0), (85, 85, 85, 45)),
    ((5.0, 10.0), (127, 127, 127, 80)),
    ((10.0, 15.0), (168, 168, 168, 125)),
    ((15.0, 20.0), (83, 228, 83, 255)),
    ((20.0, 25.0), (0, 201, 0, 255)),
    ((25.0, 30.0), (19, 163, 0, 255)),
    ((30.0, 35.0), (229, 239, 0, 255)),
    ((35.0, 40.0), (255, 196, 0, 255)),
    ((40.0, 45.0), (255, 112, 0, 255)),
    ((45.0, 50.0), (249, 10, 0, 255)),
    ((50.0, 55.0), (198, 0, 0, 255)),
    ((55.0, 60.0), (166, 0, 34, 255)),
    ((60.0, 65.0), (234, 0, 237, 255)),
    ((65.0, 999.0), (157, 0, 255, 255)),
)

# dBZ ceiling reported alongside the fallback palette.  Range midpoints
# with an upper bound above the ceiling are clamped to it, so the
# open-ended top stop ``[65, 999]`` decodes to 72.5 dBZ at the default
# 80 dBZ scale.
_PAGASA_FALLBACK_SCALE_MAX = 80.0

# Squared RGB distance below which a pixel is considered a palette
# match.  4 = ~1 per channel -- tighter than MMD's 64 because PAGASA's
# PNG is unaliased so palette hits land exactly on a stop.  The
# allowance only forgives 1-bit rounding in case the upstream renderer
# changes its bit-depth.
_PAGASA_MAX_RGB_DIST2 = 4

# Native PNG dimensions (square 2048x2048) -- declared up front so a
# malformed response can't sneak past the decoder with a different shape
# and produce a region grid mismatch later.  Also the tripwire for the
# upstream 1x1 placeholder PNG served while its radar renderer is down.
_PAGASA_EXPECTED_WIDTH = 2048
_PAGASA_EXPECTED_HEIGHT = 2048

# Number of frames in the timeline endpoint.  PAGASA serves a rolling
# 9-frame buffer (~90 min of backfill at 10-min cadence).
_PAGASA_EXPECTED_FRAMES = 9

# Native cadence (10 min), matching the store grid 1:1.
_PAGASA_NATIVE_CADENCE_SEC = 600

# Maximum acceptable distance (seconds) between a requested store slot
# and the nearest native PAGASA frame.  Half the native cadence (5 min)
# is the worst-case round-to-nearest offset; anything farther means the
# requested slot predates the rolling buffer.
_MAX_NEAREST_OFFSET_SEC = _PAGASA_NATIVE_CADENCE_SEC // 2

# Image-URL cache-busting version used until the timeline reports its
# own.  Upstream currently ignores ``v`` but the real client sends it,
# so we do too.
_DEFAULT_TILE_VERSION = 5

# The host soft-blocks non-browser user agents: a default httpx or curl
# UA gets HTTP 200 with a 1280x720 decoy PNG from both endpoints
# (verified 2026-11-23), while a browser UA gets the real JSON timeline
# and the radar PNGs.  Send the UA the PANAHON web client itself sends.
_PAGASA_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _build_palette_arrays(
    colormap, scale_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build (rgb_int32 [N,3], dbz_float32 [N], is_clutter_bool [N]).

    ``colormap`` is the timeline response's list of
    ``[[dBZ_min, dBZ_max], [R, G, B, A]]`` stops.  Each stop decodes to
    the midpoint of its dBZ range, with the upper bound clamped to
    ``scale_max`` -- so the open-ended ``[65, 999]`` stop decodes to 72.5
    at the default 80 dBZ scale.

    Leading stops whose R, G and B channels are equal are the upstream
    weak-echo / clutter grays (alpha < 255 in the PNG); their count is derived
    from the table itself rather than assumed, so a colormap swap cannot
    desync it.  The fallback table above carries the three grays the
    retired API used, which is also what a response without a usable
    colormap produces here.

    Returns ``None`` when the table is missing or malformed so the caller
    can fall back to ``_PAGASA_FALLBACK_COLORMAP``.
    """
    if not colormap:
        return None
    try:
        rgb_rows: list[tuple[int, int, int]] = []
        dbz_rows: list[float] = []
        for stop in colormap:
            dbz_range, colour = stop
            if len(dbz_range) != 2 or len(colour) != 4:
                return None
            dbz_min = float(dbz_range[0])
            dbz_max = float(dbz_range[1])
            rgb_rows.append(
                (int(colour[0]), int(colour[1]), int(colour[2])),
            )
            dbz_rows.append(
                (dbz_min + min(dbz_max, float(scale_max))) / 2.0,
            )
    except (TypeError, ValueError, IndexError):
        return None
    if not rgb_rows:
        return None

    clutter_stops = 0
    for r, g, b in rgb_rows:
        if r == g == b:
            clutter_stops += 1
        else:
            break

    return (
        np.array(rgb_rows, dtype=np.int32),
        np.array(dbz_rows, dtype=np.float32),
        np.array(
            [i < clutter_stops for i in range(len(rgb_rows))], dtype=bool,
        ),
    )


_PAGASA_FALLBACK_PALETTE = _build_palette_arrays(
    _PAGASA_FALLBACK_COLORMAP, _PAGASA_FALLBACK_SCALE_MAX,
)


def _decode_pagasa_png(
    rgba: np.ndarray,
    palette: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Decode a (H,W,4) RGBA array into a uint8 dBZ array (H,W).

    Pixels with alpha=0 are no-data (transparent background -- almost the
    entire ocean).  For visible pixels, find the nearest palette stop by
    squared RGB distance; pixels whose nearest match is one of the
    leading clutter stops are also treated as no-data (the PNG flags
    these with alpha < 255).  Final encoding is LibreWXR's shared uint8 dBZ
    scheme: clamp((dBZ+32)*2, 0, 255), no-data -> 0.

    ``palette`` is the LUT built from the current timeline response; it
    defaults to the fallback table.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"expected (H,W,4) RGBA array, got {rgba.shape}")

    h, w, _ = rgba.shape
    alpha = rgba[..., 3]
    rgb = rgba[..., :3]

    palette_rgb, palette_dbz, palette_clutter = (
        palette if palette is not None else _PAGASA_FALLBACK_PALETTE
    )

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
    # clamp(((dBZ + 32) * 2), 0, 255); NODATA -> 0.
    encoded = _dbz_float_to_uint8(dbz_vals)

    # Scatter the valid hits back into the full-frame output.
    visible_indices = np.flatnonzero(visible_mask)
    out.flat[visible_indices[valid]] = encoded
    return out


class PAGASASource:
    """PANAHON-mosaic radar composite source for the Philippines.

    A single timeline fetch returns nine 10-min-spaced frame timestamps
    plus the palette to decode them with.  We fetch the matching PNGs in
    parallel (URLs synthesized from ``observed_at_unix``), decode each
    with the response's own colormap, and cache by frame timestamp.
    Requests at the 10-min store grid land on native frames 1:1; the
    nearest-frame lookup only covers clock skew and archive requests.

    The fetcher's burst of concurrent per-slot calls inside one cycle is
    coalesced via ``_REFRESH_TTL_SEC`` -- after a timeline refresh, all
    subsequent calls hit the in-memory cache until the next cycle.
    """

    _TIMELINE_PATH = "/api/v1/radar/timeline"
    _IMAGE_PATH = "/api/v1/radar-data-image"
    _SUBLAYER = "mosaic-reflectivity"

    # Don't re-fetch the timeline more than once per ~2 min -- covers the
    # fetcher's burst of concurrent per-slot calls in one cycle without
    # unnecessary network round trips.
    _REFRESH_TTL_SEC = 120

    # Keep at most this many decoded frames in cache (~2 full timeline
    # buffers + headroom for store walk-back during cycle overlap).
    _CACHE_MAX = 24

    def __init__(self, base_url: str = "https://panahon.gov.ph"):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._frame_cache: dict[int, np.ndarray] = {}
        self._cache_order: list[int] = []
        self._last_fetch_unix: float = 0.0
        self._refresh_lock = asyncio.Lock()
        self._tile_version: int = _DEFAULT_TILE_VERSION
        self._palette: tuple[np.ndarray, np.ndarray, np.ndarray] = (
            _PAGASA_FALLBACK_PALETTE
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,
                headers={"User-Agent": _PAGASA_USER_AGENT},
            )
        return self._client

    @property
    def _timeline_url(self) -> str:
        return (
            f"{self._base_url}{self._TIMELINE_PATH}"
            f"?sublayer={self._SUBLAYER}"
        )

    def _image_url(self, ts: int) -> str:
        return (
            f"{self._base_url}{self._IMAGE_PATH}"
            f"?sublayer={self._SUBLAYER}&t={ts}&mode=dbz"
            f"&size={_PAGASA_EXPECTED_WIDTH}&v={self._tile_version}"
        )

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int,
    ) -> np.ndarray | None:
        """Return the frame for ``minutes_ago`` slots back, or ``None``.

        ``minutes_ago`` is on the 10-min store grid, which now matches
        PAGASA's native cadence, so each slot maps 1:1 onto a frame.
        """
        now_rounded = (int(time.time()) // 600) * 600
        target_ts = now_rounded - minutes_ago * 60
        return await self._fetch_for_ts(target_ts, region)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime,
    ) -> np.ndarray | None:
        """Best-effort archive lookup -- the timeline endpoint only
        carries ~90 min of backfill, so anything older returns None.
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

        # Cache miss -- refresh once per TTL window so a flurry of
        # concurrent per-slot calls in one cycle shares one fetch.
        if time.time() - self._last_fetch_unix >= self._REFRESH_TTL_SEC:
            await self._refresh()
            nearest = self._nearest_cached(ts)
            if nearest is not None:
                return self._frame_cache[nearest]

        return None

    def _nearest_cached(self, ts: int) -> int | None:
        """Return the cached frame timestamp closest to ``ts``, within
        ``_MAX_NEAREST_OFFSET_SEC`` -- or None if no cached frame is in
        range.  Native and store cadences are both 10 min, so this only
        absorbs clock skew and archive requests.
        """
        if not self._cache_order:
            return None
        nearest = min(self._cache_order, key=lambda t: abs(t - ts))
        if abs(nearest - ts) > _MAX_NEAREST_OFFSET_SEC:
            return None
        return nearest

    def _resolve_palette(
        self, data: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the decode LUT from the timeline's colormap + scale.

        The response is authoritative (upstream has silently changed the
        palette before); the module fallback table only covers a missing
        or malformed field, with a warning so the drift is visible.
        """
        scale = data.get("scale") or {}
        try:
            scale_max = float(scale.get("max", _PAGASA_FALLBACK_SCALE_MAX))
        except (TypeError, ValueError):
            scale_max = _PAGASA_FALLBACK_SCALE_MAX

        palette = _build_palette_arrays(data.get("colormap"), scale_max)
        if palette is None:
            logger.warning(
                "PAGASA timeline colormap missing or malformed; "
                "using fallback palette",
            )
            return _PAGASA_FALLBACK_PALETTE
        return palette

    async def _refresh(self) -> None:
        """Fetch the timeline + all 9 image frames; populate the cache."""
        async with self._refresh_lock:
            # Re-check inside the lock -- another coroutine may already
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
            except ValueError:
                # JSONDecodeError and UnicodeDecodeError both land here --
                # e.g. the host's decoy PNG when it decides to block us.
                logger.warning("PAGASA timeline returned non-JSON body")
                return

            data = payload.get("data") or {}
            entries = data.get("timeline") or []
            if not entries:
                logger.warning("PAGASA timeline empty: %s", payload)
                return

            try:
                self._tile_version = int(
                    data.get("tile_version", _DEFAULT_TILE_VERSION),
                )
            except (TypeError, ValueError):
                self._tile_version = _DEFAULT_TILE_VERSION

            palette = self._resolve_palette(data)
            self._palette = palette

            # The per-entry ``image_url`` is always empty now, so the
            # image URL is synthesized from each entry's
            # ``observed_at_unix`` plus the response's ``tile_version``.
            targets: list[tuple[int, int]] = []
            for i, entry in enumerate(entries[:_PAGASA_EXPECTED_FRAMES]):
                try:
                    ts = int(entry["observed_at_unix"])
                except (KeyError, TypeError, ValueError):
                    logger.warning(
                        "PAGASA timeline entry %d missing observed_at_unix",
                        i,
                    )
                    continue
                targets.append((i, ts))

            tasks = [self._fetch_image(client, ts) for _, ts in targets]
            pngs = await asyncio.gather(*tasks, return_exceptions=True)

            guard_rejected: list[tuple[int, str]] = []
            for (i, ts), png in zip(targets, pngs):
                if isinstance(png, BaseException):
                    logger.warning(
                        "PAGASA image %d fetch raised: %s", i, png,
                    )
                    continue
                if png is None:
                    continue
                try:
                    grid = self._decode_png(png, palette)
                except ValueError as exc:
                    # The decode guards (unexpected image size / channel
                    # count) fire on the upstream's placeholder and decoy
                    # PNGs.  A traceback per frame adds noise, not
                    # information -- collect and summarize after the loop.
                    guard_rejected.append((ts, str(exc)))
                    continue
                except Exception:
                    logger.exception("PAGASA frame %d decode failed", i)
                    continue
                self._frame_cache[ts] = grid
                if ts not in self._cache_order:
                    self._cache_order.append(ts)

            if guard_rejected:
                if len(guard_rejected) == len(targets):
                    logger.warning(
                        "PAGASA: all %d frames rejected by the decode "
                        "guard (%s) -- the upstream is serving placeholder "
                        "or decoy PNGs; its radar renderer is likely "
                        "unavailable",
                        len(guard_rejected), guard_rejected[0][1],
                    )
                else:
                    logger.warning(
                        "PAGASA: %d of %d frames rejected by the decode "
                        "guard -- first: ts=%d (%s)",
                        len(guard_rejected), len(targets),
                        guard_rejected[0][0], guard_rejected[0][1],
                    )

            self._cache_order.sort()
            while len(self._cache_order) > self._CACHE_MAX:
                evict = self._cache_order.pop(0)
                self._frame_cache.pop(evict, None)

    async def _fetch_image(
        self, client: httpx.AsyncClient, ts: int,
    ) -> bytes | None:
        url = self._image_url(ts)
        resp = await retry_get(client, url, log_name=f"PAGASA-img-{ts}")
        if resp is None or resp.status_code != 200:
            return None
        return resp.content

    def _decode_png(
        self,
        png_bytes: bytes,
        palette: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        img = Image.open(io.BytesIO(png_bytes))
        if img.size != (_PAGASA_EXPECTED_WIDTH, _PAGASA_EXPECTED_HEIGHT):
            raise ValueError(
                f"PAGASA PNG unexpected size {img.size}, "
                f"expected ({_PAGASA_EXPECTED_WIDTH},"
                f" {_PAGASA_EXPECTED_HEIGHT})"
            )
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return _decode_pagasa_png(
            np.array(img), palette if palette is not None else self._palette,
        )

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
