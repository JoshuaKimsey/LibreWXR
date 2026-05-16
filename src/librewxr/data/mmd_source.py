# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""MET Malaysia (Jabatan Meteorologi Malaysia) radar composite source.

Fetches the combined Peninsular + East Malaysia animated GIF served
anonymously from ``api.met.gov.my/static/images/radar-latest.gif`` under
CC-BY-4.0.  Each fetch returns a 1352×570 GIF89a holding 6 frames at
10-min cadence (60 min of backfill), with metadata (timestamp, legend,
range, sensor list) burned into the right-side chrome panel.

The two coverage zones — Peninsular Malaysia + N. Sumatra and East
Malaysia + Brunei — are sub-rectangles of the radar map area (x∈[0,1100),
y∈[0,570)) within the combined GIF.  The vertical band between them
(x≈[424,460]) is the South China Sea — pure water in the rendering, and
discarded.  Each region is decoded into its own ``RegionDef``
(``MYPENINSULAR``, ``MYEAST``) via an 18-stop palette → dBZ table; both
regions feed the ``SOUTHEAST_ASIA`` group alongside MSS Singapore.

The GIF carries no per-frame timestamps in any structured form (only
burned-in chrome text), so frame times are anchored by the HTTP
``Last-Modified`` header minus a configurable publication-lag estimate
(default 10 min, see :attr:`Settings.mmd_publish_lag_sec`).  The 6
frames are then mapped to the 6 most recent 10-min slots, oldest first.

License: CC-BY-4.0 (METMalaysia / api.met.gov.my).  Attribution recorded
in README and docs/coverage.md.
"""
import asyncio
import io
import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
import numpy as np
from PIL import Image, ImageSequence

from librewxr.data.regions import RegionDef
from librewxr.data.retry import retry_get

logger = logging.getLogger(__name__)


# ── Palette ─────────────────────────────────────────────────────────
#
# 18 discrete RGB stops observed in the GIF colorbar, paired with dBZ
# values derived from labeled mm/h ticks via Marshall-Palmer
# (Z = 200·R^1.6 → dBZ = 23.01 + 16·log10(R)).  The 5 unlabeled
# intermediate stops (200/20/2/0.2 mm/h and the >400 cap) interpolate
# cleanly between the labeled ticks on the same M-P curve.  Order is
# arbitrary — decoder uses nearest-RGB match.
_MMD_PALETTE: tuple[tuple[int, int, int, float], ...] = (
    (210, 10, 210,  65.0),   # >400 mm/h (cap)
    (255, 55, 255,  64.6),   # 400 mm/h
    (255, 115, 255, 62.7),   # 300 mm/h
    (180, 0,   0,   59.8),   # 200 mm/h (intermediate)
    (220, 0,   0,   55.0),   # 100 mm/h
    (255, 38,  0,   53.5),   # 80 mm/h
    (247, 119, 0,   50.2),   # 50 mm/h
    (247, 165, 0,   43.8),   # 20 mm/h (intermediate)
    (255, 209, 0,   39.0),   # 10 mm/h
    (255, 255, 0,   37.5),   # 8 mm/h
    (0,   240, 0,   34.2),   # 5 mm/h
    (0,   200, 0,   27.8),   # 2 mm/h (intermediate)
    (0,   172, 0,   23.0),   # 1 mm/h
    (0,   135, 0,   21.5),   # 0.8 mm/h
    (52,  206, 236, 18.2),   # 0.5 mm/h
    (5,   155, 255, 11.8),   # 0.2 mm/h (intermediate)
    (0,   113, 226, 7.0),    # 0.1 mm/h
    (255, 255, 255, 2.2),    # 0.05 mm/h
)

# Squared RGB distance below which a pixel is considered a palette hit.
# 64 = ~8 per channel — same threshold MSS uses; covers the GIF's
# anti-aliased palette quantisation without false-positive matches on
# the basemap (land brown / sea blue / range circles).
_MMD_MAX_RGB_DIST2 = 64

# ── Sub-rectangle crops within the combined GIF ─────────────────────
#
# Combined GIF: 1352×570.  Radar map area: x∈[0,1100), y∈[0,570).
# Right side x∈[1100,1352] is the chrome panel (legend + metadata).
# Within the radar area, peninsular and east coverage are equirectangular
# sub-rectangles over the union bounding box (96.92°E to 121.19°E,
# 9.18°N to -1.48°S).  Pixel scales: 45.323 px/° lon, 53.471 px/° lat.
#
#   MYPENINSULAR bounds: 96.92°E..106.28°E, -1.33°S..8.97°N
#                        → x=[0, 424), y=[11, 562)  → 424×551
#   MYEAST       bounds: 107.08°E..121.19°E, -1.48°S..9.18°N
#                        → x=[460, 1100), y=[0, 570) → 640×570
#
# The 36-px gap x∈[424,460] is the South China Sea strip between the
# two coverage zones and is discarded.
_MMD_SUBRECTS: dict[str, tuple[int, int, int, int]] = {
    # (y_top, y_bot, x_left, x_right)
    "MYPENINSULAR": (11, 562, 0, 424),
    "MYEAST":       (0, 570, 460, 1100),
}

# Hard caps so a malformed upstream GIF can't blow past expected bounds.
_MMD_EXPECTED_WIDTH = 1352
_MMD_EXPECTED_HEIGHT = 570
_MMD_EXPECTED_FRAMES = 6

# Native cadence (10 min — matches LibreWXR store cadence exactly).
_CADENCE_SEC = 600


def _build_palette_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return (rgb_int32 [N,3], dbz_float32 [N]) for nearest-match lookup."""
    rgb = np.array(
        [(r, g, b) for r, g, b, _ in _MMD_PALETTE], dtype=np.int32,
    )
    dbz = np.array(
        [d for *_, d in _MMD_PALETTE], dtype=np.float32,
    )
    return rgb, dbz


def _decode_mmd_palette(rgb: np.ndarray) -> np.ndarray:
    """Decode an RGB array (H,W,3) into a uint8 dBZ array (H,W).

    For each pixel, find the nearest palette stop by squared RGB
    distance.  Pixels farther than :data:`_MMD_MAX_RGB_DIST2` from every
    stop are treated as "no data" (uint8 0) — covers basemap pixels
    (land brown, sea blue, coastlines, range circles, etc.).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H,W,3) RGB array, got {rgb.shape}")

    palette_rgb, palette_dbz = _build_palette_arrays()
    flat = rgb.reshape(-1, 3).astype(np.int32)
    # Squared distance from each pixel to each palette entry: (N_pix, N_pal)
    dists = np.sum(
        (flat[:, None, :] - palette_rgb[None, :, :]) ** 2, axis=2,
    )
    nearest_idx = np.argmin(dists, axis=1)
    nearest_dist2 = dists[np.arange(flat.shape[0]), nearest_idx]
    valid = nearest_dist2 <= _MMD_MAX_RGB_DIST2

    dbz = np.full(flat.shape[0], -33.0, dtype=np.float32)
    dbz[valid] = palette_dbz[nearest_idx[valid]]
    dbz = dbz.reshape(rgb.shape[:2])

    # Shared uint8 encoding: clamp(((dBZ + 32) * 2), 0, 255); NODATA → 0.
    nodata = dbz <= -32.0
    out = np.clip((dbz + 32.0) * 2.0, 0, 255).astype(np.uint8)
    out[nodata] = 0
    return out


def _extract_region(full_rgb: np.ndarray, region_name: str) -> np.ndarray:
    """Crop ``full_rgb`` (570×1352×3) to the sub-rectangle for region_name."""
    y0, y1, x0, x1 = _MMD_SUBRECTS[region_name]
    return full_rgb[y0:y1, x0:x1]


def _decode_mmd_frame(
    frame_rgb: np.ndarray, region: RegionDef,
) -> np.ndarray:
    """Crop one combined-GIF frame to ``region`` and decode to uint8 dBZ."""
    if region.name not in _MMD_SUBRECTS:
        raise ValueError(
            f"MMDSource cannot decode region {region.name!r}; "
            f"expected one of {sorted(_MMD_SUBRECTS)}"
        )
    sub = _extract_region(frame_rgb, region.name)
    decoded = _decode_mmd_palette(sub)
    if decoded.shape != (region.height, region.width):
        raise ValueError(
            f"MMD region {region.name} decoded to {decoded.shape}, "
            f"expected ({region.height}, {region.width})"
        )
    return decoded


def _parse_last_modified(value: str | None) -> int | None:
    """Parse an HTTP Last-Modified header to a UTC unix timestamp."""
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _frame_timestamps(
    last_modified_unix: int, publish_lag_sec: int,
) -> list[int]:
    """Compute the 6 frame timestamps (oldest first) from Last-Modified.

    The GIF lands at the endpoint a fixed-ish lag after the newest
    frame's data time.  Snap (LM − lag) down to the 10-min grid to get
    the newest frame's timestamp, then walk back 5 cadences for the
    rest.  Order matches PIL frame iteration order (oldest at index 0).
    """
    latest = ((last_modified_unix - publish_lag_sec) // _CADENCE_SEC) * _CADENCE_SEC
    return [latest - (_MMD_EXPECTED_FRAMES - 1 - i) * _CADENCE_SEC
            for i in range(_MMD_EXPECTED_FRAMES)]


class MMDSource:
    """MET Malaysia animated-GIF radar composite source.

    A single fetch of ``api.met.gov.my/static/images/radar-latest.gif``
    returns 6 frames at 10-min cadence (60 min of backfill).  One
    HTTP fetch per cycle is shared across both regions
    (``MYPENINSULAR``, ``MYEAST``) and all the 10-min store slots via a
    per-cycle cache keyed by frame timestamp.

    The native cadence matches LibreWXR's 10-min store cadence exactly,
    so there's no interpolation step (unlike MSS Singapore's 30-min →
    10-min path).  Aligned slots get the exact native frame; misaligned
    slots (which shouldn't happen at this cadence, but might during
    cold-start before the GIF is decoded) return ``None``.

    Caching strategy:
      * ``_frame_cache`` keys decoded per-region frames by their UTC
        unix timestamp so the fetcher's 6-slot × 2-region call pattern
        (12 lookups per cycle) shares one HTTP fetch + one decode pass.
      * ``_last_fetch_unix`` throttles re-fetches inside one cycle —
        repeated ``fetch_frame`` calls within ``_REFRESH_TTL_SEC`` of
        each other reuse the existing decoded cache.

    License: CC-BY-4.0 (METMalaysia).  Attribution recorded in README.
    """

    _GIF_PATH = "/static/images/radar-latest.gif"
    # Don't re-fetch the GIF more than once per ~2 min — covers the
    # fetcher's burst of concurrent per-region per-slot calls in one
    # cycle without unnecessary network round trips.
    _REFRESH_TTL_SEC = 120
    # Keep at most this many frame timestamps in cache (= 2 full cycles
    # of 6 frames + some headroom for store walk-back).
    _CACHE_MAX = 24

    def __init__(
        self,
        base_url: str = "https://api.met.gov.my",
        publish_lag_sec: int = 600,
    ):
        self._base_url = base_url.rstrip("/")
        self._publish_lag_sec = publish_lag_sec
        self._client: httpx.AsyncClient | None = None
        # ts -> region_name -> uint8 dBZ grid
        self._frame_cache: dict[int, dict[str, np.ndarray]] = {}
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
    def _gif_url(self) -> str:
        return f"{self._base_url}{self._GIF_PATH}"

    async def fetch_frame(
        self, region: RegionDef, minutes_ago: int,
    ) -> np.ndarray | None:
        """Return the frame for ``minutes_ago`` slots back, or ``None``."""
        now_rounded = (int(time.time()) // _CADENCE_SEC) * _CADENCE_SEC
        target_ts = now_rounded - minutes_ago * 60
        return await self._fetch_for_ts(target_ts, region)

    async def fetch_archive_frame(
        self, region: RegionDef, dt: datetime,
    ) -> np.ndarray | None:
        """Best-effort archive lookup — the API endpoint only carries the
        latest 60 min, so archive requests outside that window return None.
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return await self._fetch_for_ts(int(dt.timestamp()), region)

    async def _fetch_for_ts(
        self, ts: int, region: RegionDef,
    ) -> np.ndarray | None:
        # Snap to the 10-min grid so callers that pass slightly-off
        # timestamps (clock skew, archive lookups) still land on a real
        # frame timestamp.
        ts_aligned = (ts // _CADENCE_SEC) * _CADENCE_SEC

        # Cheap hit on the existing cache first.
        cached = self._cache_lookup(ts_aligned, region.name)
        if cached is not None:
            return cached

        # Miss → consider a refresh.  Throttle by TTL so a flurry of
        # concurrent fetches inside one cycle doesn't pile up HTTP work.
        if time.time() - self._last_fetch_unix >= self._REFRESH_TTL_SEC:
            await self._refresh_gif()
            cached = self._cache_lookup(ts_aligned, region.name)
            if cached is not None:
                return cached

        # No data for this slot — either before the GIF's earliest frame
        # or the publish lag pushed the newest frame one cadence past us.
        return None

    def _cache_lookup(
        self, ts: int, region_name: str,
    ) -> np.ndarray | None:
        per_region = self._frame_cache.get(ts)
        if per_region is None:
            return None
        return per_region.get(region_name)

    async def _refresh_gif(self) -> None:
        """Fetch the latest GIF and rebuild the frame cache atomically."""
        async with self._refresh_lock:
            # Re-check after acquiring the lock — another coroutine may
            # have already refreshed while we were waiting.
            if time.time() - self._last_fetch_unix < self._REFRESH_TTL_SEC:
                return

            client = await self._get_client()
            url = self._gif_url
            resp = await retry_get(client, url, log_name="MMD")
            # Always advance the timer so a failed fetch doesn't busy-loop
            # the retry path; the next 10-min cycle will try again.
            self._last_fetch_unix = time.time()

            if resp is None:
                logger.warning("MMD fetch returned None (retries exhausted)")
                return
            if resp.status_code != 200:
                logger.warning(
                    "MMD fetch failed: HTTP %d (%s)",
                    resp.status_code, url,
                )
                return

            lm = _parse_last_modified(resp.headers.get("Last-Modified"))
            if lm is None:
                # No Last-Modified → fall back to "now" as the reference.
                # Snapping (now − lag) lands us on the most recent fully-
                # cooked 10-min slot, which is also the newest frame in
                # the GIF as long as the publication is on schedule.
                lm = int(time.time())

            try:
                new_cache = self._decode_gif(resp.content, lm)
            except Exception:
                logger.exception("MMD decode failed")
                return

            # Merge into the existing cache rather than replacing — keeps
            # older frames available for store walk-back during cold
            # starts and cycle overlap.
            for ts, per_region in new_cache.items():
                self._frame_cache[ts] = per_region
                if ts not in self._cache_order:
                    self._cache_order.append(ts)
            self._cache_order.sort()
            while len(self._cache_order) > self._CACHE_MAX:
                evict = self._cache_order.pop(0)
                self._frame_cache.pop(evict, None)

    def _decode_gif(
        self, gif_bytes: bytes, last_modified_unix: int,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Decode an MMD GIF into a {ts: {region_name: grid}} mapping."""
        img = Image.open(io.BytesIO(gif_bytes))
        if img.size != (_MMD_EXPECTED_WIDTH, _MMD_EXPECTED_HEIGHT):
            raise ValueError(
                f"MMD GIF unexpected size {img.size}, "
                f"expected ({_MMD_EXPECTED_WIDTH}, {_MMD_EXPECTED_HEIGHT})"
            )
        if getattr(img, "n_frames", 1) != _MMD_EXPECTED_FRAMES:
            logger.warning(
                "MMD GIF has %d frames, expected %d",
                getattr(img, "n_frames", 1), _MMD_EXPECTED_FRAMES,
            )

        timestamps = _frame_timestamps(
            last_modified_unix, self._publish_lag_sec,
        )

        out: dict[int, dict[str, np.ndarray]] = {}
        # Re-resolve regions on every decode in case bounds ever change.
        # (Cheap — just two dict reads from REGIONS.)
        from librewxr.data.regions import REGIONS
        regions = [REGIONS[name] for name in _MMD_SUBRECTS.keys()]

        for i, frame in enumerate(ImageSequence.Iterator(img)):
            if i >= len(timestamps):
                break
            rgb = np.array(frame.convert("RGB"))
            ts = timestamps[i]
            per_region: dict[str, np.ndarray] = {}
            for region in regions:
                per_region[region.name] = _decode_mmd_frame(rgb, region)
            out[ts] = per_region

        return out

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
