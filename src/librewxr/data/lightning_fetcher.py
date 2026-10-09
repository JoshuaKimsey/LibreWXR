# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Pipeline-owned GOES GLM lightning fetch loop.

Fetches NOAA NODD GOES-West/East GLM L2 LCFA flash files over anonymous S3,
decodes the flash point arrays with h5py, applies QC, and writes the merged
points through :class:`~librewxr.data.lightning_store.LightningStore`.  The
store's on-disk artifact is the cross-process handoff -- lightning has no
``state.json`` section (docs/lightning-implementation-plan.md ruling 2).

The loop mirrors the WMO alerts fetcher's clock-aligned background-task shape
(``data/alerts_fetcher.py``) but owns its own interval knob, so disabling
alerts never disables lightning and vice versa.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence
from urllib.parse import quote

import h5py
import httpx
import numpy as np

from librewxr.data.lightning_store import _POINT_DTYPE, LightningStore
from librewxr.data.retry import retry_sync

logger = logging.getLogger(__name__)

# (name, bucket, bird number).  Satellite identity is INTERNAL -- never a
# user-facing config surface.
_NOAA_SATS: tuple[tuple[str, str, int], ...] = (
    ("goes18", "noaa-goes18", 18),
    ("goes19", "noaa-goes19", 19),
)

# GLM L2 LCFA object-key prefix under each bucket.
_GLM_PREFIX_TMPL = "GLM-L2-LCFA"

# Per-satellite resume file, relative to the shared cache dir.
_WATERMARK_RELPATH = Path("lightning") / "watermark.json"

# A satellite whose newest published key is older than this many fetch
# intervals is reported stale (a degraded hemisphere, never an error).
_STALE_INTERVALS = 3

# Safety cap on ``start-after`` pagination re-requests per listing.
_MAX_LIST_PAGES = 20

# GLM netCDF constants.  ``product_time`` is a scalar in units
# "seconds since 2000-01-01 12:00:00" (the true 12:00 epoch is 946728000,
# derived from the units string); the fallback below is only used when the
# units attribute is absent/unparseable.
_GLM_EPOCH_FALLBACK = 946728000.0
_KEY_RE = re.compile(r"GLM-L2-LCFA.*?_s(\d{14})_e(\d{14})")
_KEY_TAG_RE = re.compile(r"<Key>(.*?)</Key>", re.DOTALL)
_TRUNCATED_RE = re.compile(r"<IsTruncated>\s*(true|false)\s*</IsTruncated>")


# ---------------------------------------------------------------------------
# Key / time helpers
# ---------------------------------------------------------------------------

def _parse_glm_stamp(tok: str) -> float | None:
    """Parse a 14-digit GLM filename stamp to a unix epoch (float).

    Format: year(4) doy(3) hour(2) minute(2) second(2) tenth(1).  Returns
    ``None`` when the token is malformed.
    """
    try:
        year = int(tok[0:4])
        doy = int(tok[4:7])
        hour = int(tok[7:9])
        minute = int(tok[9:11])
        second = int(tok[11:13])
        tenth = int(tok[13:14])
    except (ValueError, IndexError):
        return None
    try:
        base = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
            days=doy - 1, hours=hour, minutes=minute, seconds=second,
        )
    except ValueError:
        return None
    return base.timestamp() + tenth / 10.0


def _key_window_epochs(key: str) -> tuple[float, float] | None:
    """Return ``(window_start_unix, window_end_unix)`` for a GLM key.

    ``None`` when the key does not match the GLM L2 LCFA pattern.
    """
    m = _KEY_RE.search(key)
    if m is None:
        return None
    start = _parse_glm_stamp(m.group(1))
    end = _parse_glm_stamp(m.group(2))
    if start is None or end is None:
        return None
    return (start, end)


def _epoch_from_units(units: str | bytes | None) -> float:
    """Parse ``seconds since <epoch>`` netCDF units to a unix epoch offset."""
    if isinstance(units, (bytes, bytearray)):
        units = units.decode("utf-8", errors="replace")
    m = re.search(r"seconds since\s+(.+)", units or "")
    if m is None:
        return _GLM_EPOCH_FALLBACK
    token = m.group(1).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(token, fmt)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(token)
    except ValueError:
        return _GLM_EPOCH_FALLBACK
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _scalar_attr(var, name: str, default: float) -> float:
    """Read a possibly 1-element-array numeric netCDF attr as a float."""
    if name not in var.attrs:
        return default
    arr = np.asarray(var.attrs[name]).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[0])


def _decode_int16_var(var) -> np.ndarray:
    """Decode a ``_Unsigned="true"`` int16 netCDF variable to float64.

    True storage is unsigned 16-bit; the attrs carry the scale/offset.
    Identity when the attrs are absent.
    """
    raw = np.asarray(var[()])
    if raw.dtype != np.int16:
        raw = raw.astype(np.int16)
    unsigned = raw.view(np.uint16).astype(np.float64)
    scale = _scalar_attr(var, "scale_factor", 1.0)
    offset = _scalar_attr(var, "add_offset", 0.0)
    return unsigned * scale + offset


# ---------------------------------------------------------------------------
# GOESGLMLightningFetcher
# ---------------------------------------------------------------------------

class GOESGLMLightningFetcher:
    """Clock-aligned GOES GLM flash fetcher (pipeline-owned writer)."""

    def __init__(
        self,
        store: LightningStore,
        *,
        cache_dir: Path | str | None,
        interval_s: int = 300,
        max_age_s: int = 1800,
        satellites: Sequence[tuple[str, str, int]] = _NOAA_SATS,
        download_retries: int | None = None,
    ) -> None:
        self._store = store
        self._interval_s = int(interval_s)
        self._max_age_s = int(max_age_s)
        self._satellites = tuple(satellites)
        self._download_retries = download_retries

        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._watermark_path = (
            self._cache_dir / _WATERMARK_RELPATH
            if self._cache_dir is not None
            else None
        )
        self._watermark_lock = threading.Lock()
        self._watermarks = self._load_watermarks()
        # Per-satellite stale-warn latch: warn once per stale episode,
        # INFO once on recovery.
        self._stale_warned: dict[str, bool] = {}

        self._client: httpx.Client | None = None
        self._task: asyncio.Task | None = None

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Kick off the background fetch task."""
        self._task = asyncio.create_task(self._fetch_loop())

    async def close(self) -> None:
        """Cancel the background task and close the sync HTTP client.

        Idempotent: safe to call before ``start`` or twice.
        """
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        client = self._client
        self._client = None
        if client is not None:
            client.close()

    # -- background loop -----------------------------------------------------

    async def _fetch_loop(self) -> None:
        """Fetch immediately, then on each clock-aligned interval boundary."""
        try:
            await self._fetch_once()
        except Exception:
            logger.exception("Initial GLM lightning fetch failed")

        while True:
            now = time.time()
            interval = self._interval_s
            next_boundary = (int(now // interval) + 1) * interval
            sleep_secs = max(next_boundary - now, 1.0)
            logger.debug("Next GLM lightning fetch in %.1fs", sleep_secs)
            await asyncio.sleep(sleep_secs)

            try:
                await self._fetch_once()
            except Exception:
                logger.exception("GLM lightning fetch failed")

    # -- per-cycle body ------------------------------------------------------

    async def _fetch_once(self) -> None:
        """Fetch every satellite in parallel, merge, trim, and persist."""
        now = self._now_s()

        if self._satellites:
            results = await asyncio.gather(
                *(asyncio.to_thread(self._fetch_sat, sat) for sat in self._satellites)
            )
        else:
            results = []

        fresh = [points for points, _status in results if len(points)]
        # Concatenating the store's current points keeps age-out working on a
        # quiet feed; the watermark prevents double-adding.
        merged = np.concatenate([self._store.points, *fresh])

        satellites = {
            sat[0]: status for sat, (_points, status) in zip(self._satellites, results)
        }
        await self._store.replace_points(
            merged,
            last_seen_s=now,
            max_age_s=self._max_age_s,
            meta={"units": "J", "satellites": satellites},
        )
        await self._store.save_snapshot()

    # -- seams ---------------------------------------------------------------

    def _now_s(self) -> int:
        """Current unix time as an int (monkeypatched in tests)."""
        return int(time.time())

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True,
            )
        return self._client

    # -- per-satellite fetch -------------------------------------------------

    def _fetch_sat(self, sat: tuple[str, str, int]) -> tuple[np.ndarray, dict]:
        """Fetch one satellite; NEVER raises, returns (points, status)."""
        name, bucket, sat_id = sat
        status: dict = {
            "bucket": bucket,
            "last_key": None,
            "last_e": None,
            "stale": False,
            "last_fetch_ok": False,
        }
        try:
            now = self._now_s()
            cutoff = now - self._max_age_s - 2 * self._interval_s

            now_utc = datetime.fromtimestamp(now, tz=timezone.utc)
            prefix = f"{_GLM_PREFIX_TMPL}/{now_utc:%Y}/{now_utc:%j}/"

            wm = self._watermarks.get(name) or {}
            last_key = wm.get("last_key")
            last_e = wm.get("last_e")
            if last_key:
                start_after = last_key
            else:
                # Cold start: begin one hour before the cutoff hour.  That
                # hour's dir string sorts before all of its keys, so the
                # listing picks up any boundary file the cutoff filter keeps.
                cold = datetime.fromtimestamp(cutoff - 3600, tz=timezone.utc)
                start_after = (
                    f"{_GLM_PREFIX_TMPL}/{cold:%Y}/{cold:%j}/{cold:%H}/"
                )

            keys = self._list_keys(bucket, prefix, start_after)

            # -- staleness (newest published key window-end) -----------------
            newest_e: float | None = None
            for key in reversed(keys):
                window = _key_window_epochs(key)
                if window is not None:
                    newest_e = window[1]
                    break
            stale = False
            if keys:
                if (
                    newest_e is not None
                    and newest_e < now - _STALE_INTERVALS * self._interval_s
                ):
                    stale = True
            elif last_key:
                # Watermark present but no data published this cycle.
                stale = True
            status["stale"] = stale
            if stale:
                if not self._stale_warned.get(name):
                    logger.warning(
                        "GLM %s: feed appears stale (newest window end %s, "
                        "now %s); holding last-known points",
                        name, newest_e, now,
                    )
                    self._stale_warned[name] = True
            elif self._stale_warned.get(name):
                logger.info("GLM %s: feed recovered", name)
                self._stale_warned[name] = False

            # -- tail the listing in ascending order -------------------------
            points: list[np.ndarray] = []
            for key in keys:
                window = _key_window_epochs(key)
                if window is None:
                    continue
                window_end = window[1]
                if window_end < cutoff:
                    continue
                if last_e is not None and window_end <= last_e:
                    continue

                data = self._download(bucket, key)
                if data is None:
                    # Transient miss / not yet published: stop the tail so
                    # the next cycle re-tails from the last processed key
                    # and never skips past a gap.
                    break

                try:
                    pts = self._decode_file(data, sat_id)
                except Exception as exc:  # noqa: BLE001 - corrupt file path
                    logger.warning(
                        "GLM %s: decode failed for %s (%s); skipping",
                        name, key, exc,
                    )
                    # Advance past a permanently corrupt file so it cannot
                    # wedge the loop.
                    last_key = key
                    last_e = window_end
                    continue

                points.append(pts)
                last_key = key
                last_e = window_end

            status["last_key"] = last_key
            status["last_e"] = last_e
            status["last_fetch_ok"] = True

            if last_key is not None:
                self._update_watermark(name, bucket, last_key, last_e)

            merged = np.concatenate(points) if points else np.empty(0, dtype=_POINT_DTYPE)
            return merged, status
        except Exception:
            logger.warning("GLM %s: fetch cycle failed", name, exc_info=True)
            status["last_fetch_ok"] = False
            return np.empty(0, dtype=_POINT_DTYPE), status

    # -- HTTP surface --------------------------------------------------------

    def _get_listing_page(self, url: str) -> bytes | None:
        """GET one S3 listing page; ``None`` on a degraded request."""
        client = self._get_client()
        resp = retry_sync(
            client.get,
            url,
            retries=self._download_retries,
            log_name="glm listing",
        )
        if resp is None:
            logger.warning("GLM: listing request failed after retries (%s)", url)
            return None
        try:
            resp.raise_for_status()
        except Exception:
            logger.warning("GLM: listing HTTP error (%s)", url, exc_info=True)
            return None
        return resp.content

    def _list_keys(
        self, bucket: str, prefix: str, start_after: str | None,
    ) -> list[str]:
        """List every key under ``prefix`` after ``start_after`` (ascending).

        Paginates with ``start-after=<last-key-of-page>`` until S3 reports
        ``IsTruncated=false``.  Returns ``[]`` on any listing failure -- a
        failed listing is a degraded cycle, never an exception.
        """
        base = f"https://{bucket}.s3.amazonaws.com"
        keys: list[str] = []
        start = start_after
        for _page in range(_MAX_LIST_PAGES):
            url = f"{base}/?list-type=2&prefix={quote(prefix)}"
            if start:
                url += f"&start-after={quote(start)}"
            xml = self._get_listing_page(url)
            if xml is None:
                return []
            try:
                page_keys, truncated = self._parse_listing_page(xml)
            except Exception:
                logger.warning("GLM: failed to parse listing (%s)", url, exc_info=True)
                return []
            keys.extend(page_keys)
            if not truncated or not page_keys:
                break
            start = page_keys[-1]
        else:
            logger.warning(
                "GLM: listing page cap (%d) hit for prefix %s",
                _MAX_LIST_PAGES, prefix,
            )
        return keys

    @staticmethod
    def _parse_listing_page(xml: bytes) -> tuple[list[str], bool]:
        """Parse a ListObjectsV2 XML page into (keys, is_truncated)."""
        text = (
            xml.decode("utf-8", errors="replace")
            if isinstance(xml, (bytes, bytearray))
            else str(xml)
        )
        keys = _KEY_TAG_RE.findall(text)
        m = _TRUNCATED_RE.search(text)
        truncated = m is not None and m.group(1) == "true"
        return keys, truncated

    def _download(self, bucket: str, key: str) -> bytes | None:
        """Download one GLM object; ``None`` when not (yet) available."""
        client = self._get_client()
        url = f"https://{bucket}.s3.amazonaws.com/{quote(key)}"
        resp = retry_sync(
            client.get,
            url,
            retries=self._download_retries,
            log_name=f"glm {key}",
        )
        if resp is None:
            return None
        if resp.status_code != 200:
            # A not-yet-published key is a normal miss, not an error.
            logger.info("GLM: %s not available yet (HTTP %d)", key, resp.status_code)
            return None
        return resp.content

    # -- decode --------------------------------------------------------------

    def _decode_file(self, data: bytes, sat_id: int) -> np.ndarray:
        """Decode a flat GLM L2 LCFA netCDF buffer into flash points.

        Raises ``ValueError`` on an unexpected schema (missing variable) so
        the caller treats it as a decode failure and skips the file.
        """
        try:
            with h5py.File(io.BytesIO(data), "r") as f:
                product_time_var = f["product_time"]
                product_time = float(product_time_var[()])
                epoch = _epoch_from_units(product_time_var.attrs.get("units"))

                lat = np.asarray(f["flash_lat"][()], dtype=np.float64)
                lon = np.asarray(f["flash_lon"][()], dtype=np.float64)
                energy = _decode_int16_var(f["flash_energy"])
                first_offset = _decode_int16_var(
                    f["flash_time_offset_of_first_event"]
                )
                quality = _decode_int16_var(f["flash_quality_flag"])
        except KeyError as exc:
            raise ValueError("unexpected GLM schema") from exc

        mask = (
            (quality == 0)
            & np.isfinite(lat)
            & np.isfinite(lon)
            & (lat >= -90.0)
            & (lat <= 90.0)
            & (lon >= -180.0)
            & (lon <= 180.0)
        )

        product_time_unix = product_time + epoch
        out = np.empty(int(mask.sum()), dtype=_POINT_DTYPE)
        out["time_s"] = np.asarray(
            product_time_unix + first_offset[mask], dtype=np.int64
        )
        out["lat"] = lat[mask].astype(np.float32)
        out["lon"] = lon[mask].astype(np.float32)
        out["energy"] = energy[mask].astype(np.float32)
        out["satellite"] = np.int8(sat_id)
        return out

    # -- watermarks ----------------------------------------------------------

    def _load_watermarks(self) -> dict:
        """Load the per-satellite resume file; ``{}`` when absent/corrupt."""
        path = self._watermark_path
        if path is None:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            logger.warning(
                "GLM: corrupt watermark file %s (%s); starting fresh", path, exc,
            )
            return {}
        if not isinstance(data, dict):
            logger.warning("GLM: watermark file %s is not a JSON object", path)
            return {}
        return data

    def _update_watermark(
        self, name: str, bucket: str, last_key: str, last_e: float | None,
    ) -> None:
        """Merge one satellite's entry and persist atomically.

        Read-modify-write under a lock: the satellites fetch concurrently and
        must not clobber each other's entries.
        """
        entry = {"bucket": bucket, "last_key": last_key, "last_e": last_e}
        with self._watermark_lock:
            if self._watermark_path is not None:
                current = self._load_watermarks()
            else:
                current = dict(self._watermarks)
            current[name] = entry
            self._watermarks = current
            if self._watermark_path is not None:
                self._write_watermarks(current)

    def _write_watermarks(self, data: dict) -> None:
        """Best-effort atomic watermark write (pid+uuid tmp + os.replace)."""
        path = self._watermark_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(
                f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("GLM: failed to persist watermark: %s", exc)
