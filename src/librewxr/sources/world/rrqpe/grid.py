# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NOAA Enterprise Rain Rate (RRQPE) GLB-5 blend — observed precip scan store.

The RRQPE GLB-5 blend is a satellite-derived 10-minute rain-rate product
built by NOAA's Enterprise Rain Rate algorithm from the geostationary
constellation.  It is *observed* precipitation, ingested as a radar
source: the package's ``source.py`` serves its frames into the FrameStore
via the standard RadarSource fetch protocol (lag-shifted relative
matching — see below), and the region participates in the radar
compositor, nowcast extrapolation, carry-forward, and state sync like
any other region.

Data layout (verified against the live anonymous S3 bucket
``noaa-enterprise-rainrate-pds``):

    BLEND/RainRate-Blend-INST/{YYYY}/{MM}/{DD}/{HH}/
        RRQPE-INST-GLB-5_v1r1_blend_s{YYYYMMDDHHMMSSmmm}_e{...}_c{...}.nc

NetCDF4, dims ``Rows=6501``, ``Columns=18000``, uniform 0.02° grid:
lat(row) = 70.0 - 0.02*row (row 0 = +70, row 6500 = -60),
lon(col) = -180.0 + 0.02*col.  No coordinate variables; the global
``geospatial_lat_min/max`` / ``geospatial_lon_min/max`` /
``geospatial_lat_resolution`` / ``geospatial_lon_resolution`` attrs
carry the grid description.

Variables:
    RRQPE   int16, scale_factor=0.1, add_offset=0.0, _FillValue=-9990,
            units mm/h
    DQF     int8; DQF==3 is no-data (masked to NaN before the block
            nanmean downsample), DQF==2 is degraded-but-valid (kept)

Valid time = the filename / global-attr ``s`` timestamp (scan start,
10-min aligned).  Median publish lag ~17 min; the configurable publish
delay keeps the fetch window clear of not-yet-published slots.

Frame → scan matching is *lag-shifted relative matching*: because the
newest scan is always ~15-25 min behind the newest radar frame, every
past frame is served the scan ``lag`` seconds its senior (where ``lag``
is the newest scan's current age, floored to whole 10-min slots).  The
resulting ~15-25 min uniform temporal fib is deliberate — absolute
nearest-match would clamp the leading 2-3 frames onto the same newest
scan and the fill would freeze, then lurch.  Lag-shifted matching steps
the fill every frame: less accurate in absolute time, but smooth — and
still observational.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx
import numpy as np

from librewxr.config import settings
from librewxr.data.retry import retry_sync
from librewxr.sources._helpers import HDF5_LOCK

logger = logging.getLogger(__name__)

# Native grid — verified against live GLB-5 files.  Uniform 0.02°
# plate-carrée with no coordinate variables.
NATIVE_ROWS = 6501
NATIVE_COLS = 18000
NATIVE_PIXEL = 0.02
NORTH_EDGE = 70.0
SOUTH_EDGE = -60.0
WEST_EDGE = -180.0

# Z-R relationship (Marshall-Palmer, stratiform rain).
ZR_A_RAIN = 200.0
ZR_B_RAIN = 1.6

# S3 layout: one hour directory per hour, GLB-5 files only (the bucket
# carries other blend variants in the same prefix — GLB-2/GLB-4 — so the
# regex must pin the GLB-5 token).
BUCKET_PREFIX_TEMPLATE = (
    "BLEND/RainRate-Blend-INST/{year:04d}/{month:02d}/{day:02d}/{hour:02d}/"
)
GLB5_KEY_RE = re.compile(
    r"RRQPE-INST-GLB-5_v1r1_blend_s(\d{14})\d*_e\d*_c\d*\.nc"
)

# Scan cadence — 10 minutes.  Used to enumerate the needed scan-start
# slots in the fetch window.
SCAN_INTERVAL_SECONDS = 600

# Throttle between scan-store refresh passes.  The fetch cycle is 600 s,
# but the source's lazy refresh may be invoked twice in quick succession
# at startup (initial backfill + the first full cycle); collapsing the
# second call into the first avoids a redundant S3 listing pass.
_REFRESH_THROTTLE_SECONDS = 120


# ── Pure helpers (grid math, keys, XML) ────────────────────────────────


def downsampled_shape(factor: int) -> tuple[int, int]:
    """Effective grid shape after cropping + block-averaging by ``factor``.

    Rows are cropped to the largest multiple of ``factor`` first
    (6501 → 6500 for factor 2); columns are already a multiple.
    """
    rows = NATIVE_ROWS // factor
    cols = NATIVE_COLS // factor
    return (rows, cols)


def effective_grid(factor: int) -> tuple[float, float, float, int, int]:
    """Effective grid parameters for a downsample factor.

    Returns ``(pixel_eff, north_eff, west_eff, rows, cols)``.  Block ``k``
    spans native rows ``[k*f, (k+1)*f)``; its centre latitude is
    ``70.0 - 0.02*f*k - 0.01*(f-1)``, so the top block centre (k=0) is
    ``north_eff = 70.0 - 0.01*(f-1)`` and the effective pixel size is
    ``0.02*f``.  Columns mirror the west edge.
    """
    pixel_eff = NATIVE_PIXEL * factor
    north_eff = NORTH_EDGE - 0.01 * (factor - 1)
    west_eff = WEST_EDGE + 0.01 * (factor - 1)
    rows, cols = downsampled_shape(factor)
    return pixel_eff, north_eff, west_eff, rows, cols


def block_nanmean_downsample(rate: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2D float32 rate grid by integer ``factor`` (block nanmean).

    Each ``factor × factor`` block is averaged over its finite members
    (NaN = missing, i.e. DQF==3).  A block with no finite members becomes
    NaN.  Rows/cols are cropped to the largest multiple of ``factor``
    first.

    Implemented as nansum/count instead of bare ``np.nanmean`` so an
    all-NaN block can never emit a RuntimeWarning — the division only
    runs where ``count > 0``.
    """
    crop_rows = (rate.shape[0] // factor) * factor
    crop_cols = (rate.shape[1] // factor) * factor
    rate = rate[:crop_rows, :crop_cols]
    out_rows = crop_rows // factor
    out_cols = crop_cols // factor
    reshaped = rate.reshape(out_rows, factor, out_cols, factor)
    finite = np.isfinite(reshaped)
    nansum = np.where(finite, reshaped, 0.0).sum(axis=(1, 3))
    count = finite.sum(axis=(1, 3))
    mean = np.divide(
        nansum, count,
        out=np.full(nansum.shape, np.nan, dtype=np.float32),
        where=(count > 0),
    )
    return mean.astype(np.float32)


def precip_rate_to_dbz_encoded(
    rate: np.ndarray, *, dbz_offset: float,
) -> np.ndarray:
    """Convert a mm/h rain-rate grid to uint8 dBZ-encoded values.

    ``Z = 200 * R^1.6``, ``dBZ = 10*log10(Z) + dbz_offset``, then the
    project's standard ``pixel = (dBZ + 32) * 2`` encoding.  Zero / NaN /
    rates at or below 0.01 mm/h (trace or clear sky) encode to 0.  Invalid
    rates are zeroed before the Z-R math so no NaN flows through to the
    uint8 cast.
    """
    valid = np.isfinite(rate) & (rate > 0.01)
    safe = np.where(valid, rate, 0.0)
    z = ZR_A_RAIN * np.power(safe, ZR_B_RAIN)
    dbz = 10.0 * np.log10(np.maximum(z, 1e-10)) + dbz_offset
    encoded = np.clip((dbz + 32.0) * 2.0, 0, 255).astype(np.uint8)
    encoded[~valid] = 0
    return encoded


def scan_ts_from_key(key: str) -> int | None:
    """Extract the scan-start Unix timestamp from a GLB-5 S3 key.

    The ``s`` token is ``YYYYMMDDHHMMSS`` plus 3-digit milliseconds;
    the milliseconds are dropped (the scan start is 10-min aligned).
    """
    m = GLB5_KEY_RE.search(key)
    if m is None:
        return None
    tok = m.group(1)
    try:
        dt = datetime.strptime(tok, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(dt.timestamp())


def hour_prefix(ts: int) -> str:
    """Return the S3 hour-directory prefix for a scan timestamp."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return BUCKET_PREFIX_TEMPLATE.format(
        year=dt.year, month=dt.month, day=dt.day, hour=dt.hour,
    )


def parse_s3_listing_keys(xml_bytes: bytes) -> list[str]:
    """Parse S3 ListObjectsV2 XML into a list of object keys.

    Tolerates the default ``http://s3.amazonaws.com/doc/2006-03-01/``
    namespace by matching on the local tag name.
    """
    root = ET.fromstring(xml_bytes)
    keys = []
    for elem in root.iter():
        if elem.tag.endswith("Key"):
            keys.append(elem.text or "")
    return keys


def _fmt_slot(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%MZ")


# ── RRQPEGrid: the scan store behind the radar source ──────────────────


class RRQPEGrid:
    """NOAA Enterprise Rain Rate GLB-5 blend scan store.

    Fetch-side only: downloads any missing 10-min scan slots in the
    recent window, decodes them to uint8 dBZ-encoded memmaps keyed by
    scan timestamp (``dict[int, np.ndarray]``), and answers lag-shifted
    relative matches (see the module docstring).  The radar source
    (``source.RRQPESource``) drives the fetch cycle and serves matched
    scans as radar frames; the frames themselves flow through the
    FrameStore's state sync, so this class carries no cross-process
    pickling surface.
    """

    name = "rrqpe"

    def __init__(self, cache_dir: Path | None = None, downsample: int | None = None):
        self._downsample = (
            downsample if downsample is not None else settings.rrqpe_downsample
        )
        (
            self._pixel_eff, self._north_eff, self._west_eff,
            self._rows, self._cols,
        ) = effective_grid(self._downsample)

        self._timesteps: dict[int, np.ndarray] = {}
        self._sorted_timestamps: list[int] = []
        self._client: httpx.Client | None = None
        self._fetch_lock = asyncio.Lock()
        # Monotonic timestamp of the last completed refresh pass, for the
        # ``_REFRESH_THROTTLE_SECONDS`` gate in ``fetch``.
        self._last_refresh_monotonic = 0.0

        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "rrqpe"
            self._persistent = True
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_rrqpe_"))
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        # Drop stale .tmp files from a crash mid-write.
        for path in self._memmap_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        logger.debug(
            "RRQPE memmap directory: %s (persistent=%s, downsample=%d)",
            self._memmap_dir, self._persistent, self._downsample,
        )

    # ── Public state ──────────────────────────────────────────────────

    @property
    def reference_time(self) -> int | None:
        """Latest stored scan timestamp (Unix seconds) or None."""
        if not self._sorted_timestamps:
            return None
        return self._sorted_timestamps[-1]

    @property
    def timestep_count(self) -> int:
        return len(self._timesteps)

    @property
    def effective_shape(self) -> tuple[int, int]:
        return (self._rows, self._cols)

    @property
    def timestamps(self) -> list[int]:
        """Sorted stored scan timestamps."""
        return list(self._sorted_timestamps)

    # ── Cache management ──────────────────────────────────────────────

    def _to_memmap(self, name: str, data: np.ndarray) -> np.ndarray:
        """Write array to disk atomically and return a read-only memmap.

        Atomic write (``.tmp`` → ``os.replace``) ensures readers in other
        processes never see a half-written file.  The scan store is
        fetch-side only (radar frames flow through the FrameStore), but
        the atomic write pattern is kept for consistency with the rest of
        the project's memmap usage.
        """
        final = self._memmap_dir / f"{name}.dat"
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(tmp, dtype=data.dtype, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        del mm
        os.replace(tmp, final)
        return np.memmap(final, dtype=data.dtype, mode="r", shape=data.shape)

    # ── Frame → scan matching (lag-shifted relative) ──────────────────

    def match_timestamp(self, timestamp: int | None) -> int | None:
        """Lag-shifted relative match: serve the frame the scan ``lag``
        seconds its senior so client animations stay smooth.

        The product's newest scan is always ~15-25 min behind the newest
        radar frame (publish latency).  Absolute nearest-match would clamp
        the leading 2-3 frames onto the same newest scan — the fill
        freezes for 2-3 animation frames, then lurches.  Instead every
        past frame gets its own scan, shifted by the current lag: a
        uniform temporal fib (~15-25 min) that steps the fill every
        frame.  Less accurate in absolute time, but smooth — and still
        observational.
        """
        if timestamp is None:
            if not self._sorted_timestamps:
                return None
            return self._sorted_timestamps[-1]
        # Hard observed-only gate: nowcast frames are always future-dated
        # and past frames never are, so anything beyond a small clock-skew
        # slack is not an observed time and can't be answered.  This is
        # what stops RRQPE leaking into nowcast tiles.
        if timestamp > int(time.time()) + 120:
            return None
        ts_list = self._sorted_timestamps
        if not ts_list:
            return None
        # Current publish lag, floored to whole 10-min slots.  The newest
        # stored scan is (now - newest) seconds old; every query is shifted
        # back by that lag so the newest past frame maps to the newest scan
        # and consecutive frames map to consecutive scans (1:1 distinct).
        newest = ts_list[-1]
        now = int(time.time())
        lag_slots = max(0, (now - newest) // SCAN_INTERVAL_SECONDS)
        lag = lag_slots * SCAN_INTERVAL_SECONDS
        if lag > settings.rrqpe_match_tolerance_seconds:
            # Product severely late (>= 2 missed scans): decline the whole
            # source so the models fill; self-heals next fetch cycle.
            return None
        shifted = timestamp - lag
        idx = np.searchsorted(ts_list, shifted)
        if idx == 0:
            nearest = ts_list[0]
        elif idx >= len(ts_list):
            nearest = ts_list[-1]
        else:
            before = ts_list[idx - 1]
            after = ts_list[idx]
            nearest = before if shifted - before <= after - shifted else after
        # One-slot slack: a single missed scan slot falls to its neighbor,
        # never a blink.
        if abs(nearest - shifted) <= SCAN_INTERVAL_SECONDS:
            return nearest
        return None

    def frame_at(self, ts: int) -> np.ndarray | None:
        """The stored scan frame for ``ts``, or None."""
        return self._timesteps.get(ts)

    # ── Fetch loop ────────────────────────────────────────────────────

    def _get_client(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True,
            )
        return self._client

    async def fetch(
        self,
        now_ts: int | None = None,
        history_seconds: int = 0,
        horizon_seconds: int = 0,
    ) -> None:
        """Fetch any missing 10-min scan slots in the recent window.

        Observed-only: slots are only requested up to
        ``now - rrqpe_publish_delay_minutes``, so nothing future is ever
        stored.  ``horizon_seconds`` is accepted for signature parity
        with the fetcher's introspection and ignored.

        Throttled to one pass per ``_REFRESH_THROTTLE_SECONDS`` — the
        radar fetch cycle calls this once per cycle (on the newest-slot
        request), and the startup backfill can fire two live requests
        back-to-back, which should collapse into a single S3 pass.
        """
        now = time.monotonic()
        if now - self._last_refresh_monotonic < _REFRESH_THROTTLE_SECONDS:
            logger.debug("RRQPE: refresh throttled")
            return
        async with self._fetch_lock:
            # Re-check under the lock so two queued callers don't both
            # run the pass back-to-back.
            if (
                time.monotonic() - self._last_refresh_monotonic
                < _REFRESH_THROTTLE_SECONDS
            ):
                return
            try:
                await asyncio.to_thread(self._fetch_sync, now_ts, history_seconds)
                self._last_refresh_monotonic = time.monotonic()
            except Exception:
                logger.exception("Error fetching NOAA RRQPE data")

    def _fetch_sync(self, now_ts: int | None, history_seconds: int) -> None:
        """Synchronous fetch pass — runs in a worker thread.

        Enumerates the 10-min scan-start slots in
        ``[now - history_seconds - tolerance, now - publish_delay]``,
        skips slots already stored, groups the rest by UTC hour, lists
        each hour directory once, matches GLB-5 keys, downloads and
        decodes the missing files, and evicts frames older than the
        window start.  Any individual file's network/parse failure is
        logged and left for the next cycle; a total failure keeps the
        existing frames.
        """
        publish_delay = settings.rrqpe_publish_delay_minutes * 60
        tolerance = settings.rrqpe_match_tolerance_seconds
        if now_ts is None:
            now_ts = int(datetime.now(timezone.utc).timestamp())
        window_start = now_ts - history_seconds - tolerance
        window_end = now_ts - publish_delay
        if window_end < window_start:
            logger.debug("RRQPE: publish delay covers the whole window")
            return

        # 10-min-aligned scan-start slots in the window (ceil on the
        # start so a slot fetched this cycle can't be evicted by the
        # same pass).
        first_slot = -(-window_start // SCAN_INTERVAL_SECONDS) * SCAN_INTERVAL_SECONDS
        needed = set(range(first_slot, window_end + 1, SCAN_INTERVAL_SECONDS))

        # Evict frames older than the window start first (the window
        # slides forward each cycle), keeping the survivors in the new
        # dict so nothing is mutated in place.
        new_frames: dict[int, np.ndarray] = {
            ts: arr for ts, arr in self._timesteps.items() if ts >= window_start
        }
        evicted = [ts for ts in self._timesteps if ts < window_start]
        for ts in evicted:
            try:
                (self._memmap_dir / f"{ts}.dat").unlink(missing_ok=True)
            except OSError:
                pass

        missing = sorted(needed - set(new_frames))
        if not missing:
            if evicted:
                self._timesteps = new_frames
                self._sorted_timestamps = sorted(new_frames.keys())
            logger.debug("RRQPE: no missing slots in window")
            return

        # Group missing slots by UTC hour so each hour directory is
        # listed exactly once.
        by_hour: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
        for slot in missing:
            dt = datetime.fromtimestamp(slot, tz=timezone.utc)
            by_hour[(dt.year, dt.month, dt.day, dt.hour)].append(slot)

        client = self._get_client()
        dbz_offset = settings.rrqpe_dbz_offset
        for hour_key, hour_slots in sorted(by_hour.items()):
            year, month, day, hour = hour_key
            prefix = (
                f"BLEND/RainRate-Blend-INST/{year:04d}/{month:02d}/"
                f"{day:02d}/{hour:02d}/"
            )
            keys = self._list_hour_keys(client, prefix)
            if keys is None:
                logger.warning(
                    "RRQPE: failed to list %s, retrying next cycle", prefix,
                )
                continue
            slot_keys = {scan_ts_from_key(k): k for k in keys}
            for slot in hour_slots:
                key = slot_keys.get(slot)
                if key is None:
                    logger.debug(
                        "RRQPE: no GLB-5 key for slot %s, skipping", _fmt_slot(slot),
                    )
                    continue
                try:
                    content = self._download_file(client, key)
                except Exception:
                    logger.warning(
                        "RRQPE: download failed for %s", key, exc_info=True,
                    )
                    continue
                encoded = self._decode_to_encoded(content, dbz_offset)
                if encoded is None:
                    continue
                new_frames[slot] = self._to_memmap(str(slot), encoded)

        # Build-then-swap: the new dict is fully assembled before the
        # reference is published, so a concurrent reader sees either the
        # old or new snapshot — never a mix.
        new_count = sum(1 for s in missing if s in new_frames)
        self._timesteps = new_frames
        self._sorted_timestamps = sorted(new_frames.keys())
        logger.info(
            "RRQPE: %d new scan(s), %d evicted, %d total (%s)",
            new_count,
            len(evicted),
            len(self._timesteps),
            ", ".join(_fmt_slot(ts) for ts in self._sorted_timestamps[-3:]),
        )

    def _list_hour_keys(self, client: httpx.Client, prefix: str) -> list[str] | None:
        """List one S3 hour directory; return GLB-5 keys or None on failure."""
        url = f"{settings.rrqpe_base_url}/?list-type=2&prefix={prefix}"
        resp = retry_sync(client.get, url, log_name=f"RRQPE listing {prefix}")
        if resp is None:
            return None
        try:
            resp.raise_for_status()
            keys = parse_s3_listing_keys(resp.content)
        except Exception:
            logger.exception("RRQPE: failed to parse listing for %s", prefix)
            return None
        return [k for k in keys if GLB5_KEY_RE.search(k)]

    def _download_file(self, client: httpx.Client, key: str) -> bytes:
        url = f"{settings.rrqpe_base_url}/{key}"
        resp = retry_sync(client.get, url, log_name=f"RRQPE {key}")
        if resp is None:
            raise RuntimeError(f"RRQPE download failed after retries: {key}")
        resp.raise_for_status()
        return resp.content

    def _decode_to_encoded(
        self, data_bytes: bytes, dbz_offset: float,
    ) -> np.ndarray | None:
        """Decode a RRQPE NetCDF4 buffer to an encoded uint8 grid.

        Reads the ``Rows``/``Columns`` dimensions and the RRQPE variable
        attributes from the file itself (so tests can feed small
        synthetic files); the module constants are the fallback when the
        file doesn't carry them.  DQF==3 pixels become NaN before the
        block-nanmean downsample, then Z-R encode.
        """
        import netCDF4

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp.write(data_bytes)
                tmp_path = tmp.name
            with HDF5_LOCK:
                ds = netCDF4.Dataset(tmp_path, "r")
                try:
                    rows = ds.dimensions["Rows"].size
                    cols = ds.dimensions["Columns"].size
                    rvar = ds.variables["RRQPE"]
                    rate_raw = np.asarray(rvar[:], dtype=np.float64)
                    scale = float(getattr(rvar, "scale_factor", 1.0))
                    offset = float(getattr(rvar, "add_offset", 0.0))
                    dqf = np.asarray(ds.variables["DQF"][:])
                finally:
                    ds.close()
            # Collapse a leading singleton time dimension if present.
            if rate_raw.ndim == 3:
                rate_raw = rate_raw[0]
            if dqf.ndim == 3:
                dqf = dqf[0]
            if rate_raw.shape != (rows, cols) or dqf.shape != (rows, cols):
                logger.warning(
                    "RRQPE: unexpected shape rate=%s dqf=%s (want (%d, %d)); "
                    "rejecting",
                    rate_raw.shape, dqf.shape, rows, cols,
                )
                return None
            rate = (rate_raw * scale + offset).astype(np.float32)
            # DQF==3 is no-data; DQF==2 (degraded) stays valid.
            rate = np.where(dqf == 3, np.nan, rate)
            downsampled = block_nanmean_downsample(rate, self._downsample)
            return precip_rate_to_dbz_encoded(downsampled, dbz_offset=dbz_offset)
        except Exception:
            logger.exception("RRQPE: failed to decode NetCDF4 buffer")
            return None
        finally:
            if tmp_path is not None:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def close(self) -> None:
        """Release resources.

        Persistent mode keeps the memmap files on disk so a fresh process
        can re-open them; temp-dir mode wipes the directory.
        """
        if self._client is not None and not self._client.is_closed:
            self._client.close()
        self._client = None
        self._timesteps.clear()
        self._sorted_timestamps.clear()
        if self._persistent:
            logger.info("RRQPE memmaps retained on disk at %s", self._memmap_dir)
        else:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.debug("RRQPE memmap directory cleaned up")
