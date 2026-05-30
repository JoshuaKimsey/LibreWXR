# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""JMA MSM (Mesoscale Model) regional precipitation source.

Implements the NWPSource Protocol for the Japan Meteorological Agency's
MSM at native 5 km resolution (0.0625° longitude × 0.05° latitude),
covering 22.4-47.6°N × 120-150°E.  Eight daily cycles
(00/03/06/09/12/15/18/21 UTC), hourly forecast steps, 78 h horizon
from the main 00Z/12Z runs (39 h from the others — we only fetch the
~3-hour window around now in either case).

Distribution: anonymous AWS Open Data S3 (``s3://openmeteo`` in
``us-west-2``, the same bucket we use for IFS), Open-Meteo's republished
mirror of the JMA GPV.  One Open-Meteo ``.om`` container per (run,
valid-time), ~800 KB each — read directly via ``OmFileReader``.  The
container exposes per-variable children including ``precipitation``
(mm/h, already differenced; no accumulation to undo) and
``temperature_2m`` (°C — Open-Meteo serves this in Celsius natively).

Data attribution: Japan Meteorological Agency, distributed via
Open-Meteo's AWS Open Data mirror.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fsspec
import numpy as np
from omfiles import OmFileReader

from librewxr.config import settings
from librewxr.sources.regional.north_america.usa.nwp.hrrr.grid import compute_snow_mask

logger = logging.getLogger(__name__)


# ── JMA MSM regridded lat/lon grid parameters ─────────────────────────
#
# Source: probed openmeteo S3 .om container 2026-05-30.  Grid is the
# native MSM 5 km lat/lon: 0.0625° lon × 0.05° lat, 481 × 505 cells.
# Open-Meteo writes the array south-up (row 0 = LAT_MIN); decode flips
# vertically so row 0 = north matches grid_indices(), the same
# convention DMI DINI / ICON-EU / HRRR use.

JMA_MSM_LAT_MIN = 22.4
JMA_MSM_LAT_MAX = 47.6
JMA_MSM_LON_MIN = 120.0
JMA_MSM_LON_MAX = 150.0
JMA_MSM_DLAT = 0.05
JMA_MSM_DLON = 0.0625
JMA_MSM_GRID_HEIGHT = int(round((JMA_MSM_LAT_MAX - JMA_MSM_LAT_MIN) / JMA_MSM_DLAT)) + 1   # 505
JMA_MSM_GRID_WIDTH  = int(round((JMA_MSM_LON_MAX - JMA_MSM_LON_MIN) / JMA_MSM_DLON)) + 1   # 481


def grid_indices(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return fractional ``(row, col)`` on the JMA MSM grid (north-up).

    After decode flips the array, row 0 is the NORTHERN edge.  Out-of-
    domain points still return values; callers should test
    ``domain_mask`` first.
    """
    row = (JMA_MSM_LAT_MAX - lat) / JMA_MSM_DLAT
    col = (lon - JMA_MSM_LON_MIN) / JMA_MSM_DLON
    return row, col


def domain_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """``True`` where (lat, lon) falls inside the JMA MSM output grid."""
    row, col = grid_indices(lat, lon)
    return (
        (row >= 0)
        & (row < JMA_MSM_GRID_HEIGHT - 1)
        & (col >= 0)
        & (col < JMA_MSM_GRID_WIDTH - 1)
    )


# ── Boundary feathering ───────────────────────────────────────────────
#
# JMA MSM's domain edges sit over open ocean on three sides (E, S, W)
# and the Sea of Okhotsk / NE Asia mainland on the north.  We feather
# ~1° (~75-110 km) inwards so chain blending hands off cleanly to IFS
# beyond the domain — the surrounding waters of the Western Pacific
# and Sea of Japan aren't covered by any other regional source.

JMA_MSM_FEATHER_DISTANCE_DEG = 1.0


def feather_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Soft taper to 0 at the JMA MSM domain edge (~1° in lat/lon space)."""
    lat_dist = np.minimum(lat - JMA_MSM_LAT_MIN, JMA_MSM_LAT_MAX - lat)
    lon_dist = np.minimum(lon - JMA_MSM_LON_MIN, JMA_MSM_LON_MAX - lon)
    dist_deg = np.minimum(lat_dist, lon_dist)
    weight = np.clip(dist_deg / JMA_MSM_FEATHER_DISTANCE_DEG, 0.0, 1.0)
    return weight.astype(np.float32, copy=False)


# ── Z-R conversion (matches ECMWFGrid / ICONEUGrid / DMIDiniGrid) ─────

ZR_A_RAIN = 200.0
ZR_B_RAIN = 1.6


def precip_rate_to_dbz_encoded(
    precip_mm_per_hour: np.ndarray,
    dbz_offset: float = 0.0,
) -> np.ndarray:
    """Convert mm/h precip rate → uint8 dBZ encoded (pixel = (dBZ+32)*2).

    Marshall-Palmer rain Z-R: Z = 200 * R^1.6.  ``dbz_offset`` shifts
    the resulting dBZ uniformly to compensate for the model-vs-radar
    intensity bias (radar reads the brightest part of the storm column;
    the model gives surface rate).
    """
    rate = np.where(np.isfinite(precip_mm_per_hour), precip_mm_per_hour, 0.0)
    rate = np.maximum(rate, 0.0)
    eps = 1e-6
    z = ZR_A_RAIN * np.power(rate + eps, ZR_B_RAIN)
    dbz = 10.0 * np.log10(np.maximum(z, eps)) + dbz_offset
    encoded = np.clip((dbz + 32.0) * 2.0 + 0.5, 0, 255)
    encoded[rate <= 0.0] = 0
    return encoded.astype(np.uint8)


# ── Run / step timing ─────────────────────────────────────────────────

CYCLE_INTERVAL_SECONDS = 3 * 3600        # JMA MSM runs every 3 hours
SOURCE_STEP_SECONDS = 3600               # forecast steps are 1 hour apart
# Backwards-compatible alias.
BRACKET_INTERVAL_SECONDS = SOURCE_STEP_SECONDS
# Post-interpolation stored cadence.  When ``LIBREWXR_REGIONAL_INTERPOLATION``
# is enabled, the fetch loop runs Farneback warping at the end to fill
# 10-minute synthetic frames between hourly originals.
STORED_INTERVAL_SECONDS = 600
# Hard cap on the lead we'll ever serve from a single run.  The shortest
# JMA MSM runs reach +39 h; main 00Z/12Z runs reach +78 h.  We cap at
# the safe minimum so a 03Z / 06Z run is never asked for a step beyond
# its published horizon.  The fetch window is at most ~3 hours wide
# anyway, so this cap never bites in practice.
MAX_FORECAST_HOURS = 39

# Walk back two cycles (6 h) when looking for a run that can serve a
# given valid time.  Each run reaches +39 h forward, plenty of slack
# for any reasonable active window.
RUN_LOOKBACK_CYCLES = 2


def floor_cycle(ts: int) -> int:
    """Floor a Unix timestamp to the nearest 3-hour cycle boundary."""
    return (ts // CYCLE_INTERVAL_SECONDS) * CYCLE_INTERVAL_SECONDS


def latest_published_run(now_ts: int, publish_delay_seconds: int) -> int:
    """Most recent run we'd expect to be available given a publish delay."""
    return floor_cycle(now_ts - publish_delay_seconds)


def bracket_lead_seconds(
    lead_seconds: int,
    interval_seconds: int = SOURCE_STEP_SECONDS,
) -> tuple[int, int, float]:
    """For a desired lead, return ``(L0, L1, alpha)`` such that L0 ≤ L < L1."""
    if lead_seconds < 0:
        return 0, 0, 0.0
    l0 = (lead_seconds // interval_seconds) * interval_seconds
    l1 = l0 + interval_seconds
    alpha = (lead_seconds - l0) / interval_seconds
    return l0, l1, alpha


# ── Open-Meteo S3 layout ──────────────────────────────────────────────
#
# Per-run directory:
#   s3://openmeteo/data_spatial/jma_msm/{YYYY}/{MM}/{DD}/{HH}{MM}Z/
# Per-step file inside:
#   {YYYY}-{MM}-{DD}T{HH}{MM}.om   (valid-time stamp, hourly)
#
# Each .om file is ~800 KB and contains every published variable as a
# named child (precipitation, temperature_2m, cloud_cover_*, etc.).

def run_dir(run: datetime) -> str:
    """Return the per-run S3 directory key (no leading bucket name)."""
    prefix = settings.jma_msm_s3_prefix.strip("/")
    return (
        f"{prefix}/{run.year}/{run.month:02d}/{run.day:02d}"
        f"/{run.hour:02d}{run.minute:02d}Z"
    )


def file_key(run: datetime, step_hour: int) -> str:
    """Return the S3 key (no bucket prefix) for a (run, step) .om file."""
    valid = run + timedelta(hours=step_hour)
    valid_str = valid.strftime("%Y-%m-%dT%H%M")
    return f"{run_dir(run)}/{valid_str}.om"


# ── .om reader helpers ───────────────────────────────────────────────


def _decode_om_field(
    fs: fsspec.AbstractFileSystem,
    s3_path: str,
    variable: str,
) -> np.ndarray | None:
    """Read one variable out of a (run, step) .om container.

    Returns the decoded float32 array flipped vertically so row 0 is
    the NORTHERN edge — matches ``grid_indices()``.  Returns ``None``
    on decode failure.
    """
    from librewxr.data.retry import retry_sync

    reader = retry_sync(
        OmFileReader.from_fsspec, fs, s3_path,
        log_name=f"JMA MSM {s3_path}",
    )
    if reader is None:
        return None
    try:
        try:
            child = reader.get_child_by_name(variable)
        except Exception:
            logger.warning(
                "JMA MSM: variable %r missing from %s", variable, s3_path,
            )
            return None
        try:
            arr = np.asarray(child[:], dtype=np.float32)
        finally:
            child.close()
    finally:
        reader.close()

    if arr.shape != (JMA_MSM_GRID_HEIGHT, JMA_MSM_GRID_WIDTH):
        logger.warning(
            "JMA MSM %s: unexpected shape %s (expected %s); skipping",
            variable, arr.shape,
            (JMA_MSM_GRID_HEIGHT, JMA_MSM_GRID_WIDTH),
        )
        return None

    # Open-Meteo writes the array south-up; flip so row 0 = north
    # matches the grid_indices() convention used everywhere else.
    return np.ascontiguousarray(np.flipud(arr), dtype=np.float32)


# ── JMAMSMGrid: the public NWPSource implementation ──────────────────


class JMAMSMGrid:
    """JMA MSM as an NWPSource for the East Asia chain slot."""

    name = "jma_msm"

    def __init__(self, cache_dir: Path | None = None):
        # (run_ts, lead_seconds) → uint8 dBZ-encoded array on the lat/lon grid
        self._frames: dict[tuple[int, int], np.ndarray] = {}
        # Per-frame snow mask (1 = snow, 0 = rain) keyed by the same
        # (run_ts, lead_seconds).  Derived from temperature_2m in the
        # same .om container that supplied the precipitation field.
        self._snow_masks: dict[tuple[int, int], np.ndarray] = {}
        self._fs: fsspec.AbstractFileSystem | None = None
        self._latest_run_ts: int | None = None
        self._fetch_lock = asyncio.Lock()

        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "jma_msm"
            self._persistent = True
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_jma_msm_"))
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "JMA MSM memmap directory: %s (persistent=%s)",
            self._memmap_dir, self._persistent,
        )
        if self._persistent:
            self._load_cached_frames()

    # ── Cache management ──────────────────────────────────────────────

    def _frame_path(self, run_ts: int, lead_seconds: int) -> Path:
        return self._memmap_dir / f"r{run_ts}_l{lead_seconds}.dat"

    def _snow_frame_path(self, run_ts: int, lead_seconds: int) -> Path:
        return self._memmap_dir / f"r{run_ts}_l{lead_seconds}_snow.dat"

    def _to_memmap(self, name: str, data: np.ndarray) -> np.ndarray:
        final = self._memmap_dir / f"{name}.dat"
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(tmp, dtype=data.dtype, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        del mm
        os.replace(tmp, final)
        return np.memmap(final, dtype=data.dtype, mode="r", shape=data.shape)

    def _load_cached_frames(self) -> None:
        for path in self._memmap_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        loaded = 0
        pat = re.compile(r"^r(\d+)_l(\d+)$")
        for path in self._memmap_dir.glob("r*_l*.dat"):
            m = pat.match(path.stem)
            if m is None:
                continue
            run_ts = int(m.group(1))
            lead_s = int(m.group(2))
            try:
                mm = np.memmap(
                    path, dtype=np.uint8, mode="r",
                    shape=(JMA_MSM_GRID_HEIGHT, JMA_MSM_GRID_WIDTH),
                )
            except Exception:
                logger.warning("Failed to memmap cached %s, removing", path)
                path.unlink(missing_ok=True)
                continue
            self._frames[(run_ts, lead_s)] = mm
            if self._latest_run_ts is None or run_ts > self._latest_run_ts:
                self._latest_run_ts = run_ts
            loaded += 1
        if loaded:
            logger.info("JMA MSM: loaded %d cached frame(s) from disk", loaded)

        snow_pat = re.compile(r"^r(\d+)_l(\d+)_snow$")
        snow_loaded = 0
        for path in self._memmap_dir.glob("r*_l*_snow.dat"):
            m = snow_pat.match(path.stem)
            if m is None:
                continue
            run_ts = int(m.group(1))
            lead_s = int(m.group(2))
            if (run_ts, lead_s) not in self._frames:
                path.unlink(missing_ok=True)
                continue
            try:
                mm = np.memmap(
                    path, dtype=np.uint8, mode="r",
                    shape=(JMA_MSM_GRID_HEIGHT, JMA_MSM_GRID_WIDTH),
                )
            except Exception:
                logger.warning(
                    "Failed to memmap cached snow %s, removing", path,
                )
                path.unlink(missing_ok=True)
                continue
            self._snow_masks[(run_ts, lead_s)] = mm
            snow_loaded += 1
        if snow_loaded:
            logger.info(
                "JMA MSM: loaded %d cached snow mask(s) from disk",
                snow_loaded,
            )

    def __getstate__(self) -> dict:
        """Serialize state for cross-process reload (multi-worker mode)."""
        return {
            "memmap_dir": str(self._memmap_dir),
            "latest_run_ts": self._latest_run_ts,
            "frame_keys": [[run, lead] for (run, lead) in self._frames.keys()],
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state by rescanning ``memmap_dir`` from disk."""
        self._memmap_dir = Path(state["memmap_dir"])
        self._persistent = True
        self._fs = None
        self._fetch_lock = asyncio.Lock()
        self._frames = {}
        self._snow_masks = {}
        self._latest_run_ts = None
        self._load_cached_frames()

    @property
    def data_bytes(self) -> int:
        return (
            sum(arr.nbytes for arr in self._frames.values())
            + sum(arr.nbytes for arr in self._snow_masks.values())
        )

    @property
    def latest_run_iso(self) -> str | None:
        if self._latest_run_ts is None:
            return None
        return datetime.fromtimestamp(self._latest_run_ts, tz=timezone.utc).isoformat()

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def snow_mask_count(self) -> int:
        return len(self._snow_masks)

    # ── NWPSource Protocol ────────────────────────────────────────────

    def domain_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return domain_mask(lat, lon)

    def feather_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return feather_mask(lat, lon)

    def has_data(self) -> bool:
        return bool(self._frames)

    def has_data_at(self, timestamp: int) -> bool:
        run = self._pick_run(timestamp)
        if run is None:
            return False
        lead = timestamp - run
        l0, l1, _ = bracket_lead_seconds(lead, self._bracket_interval())
        return ((run, l0) in self._frames) and ((run, l1) in self._frames)

    def _bracket_interval(self) -> int:
        """Stored frame spacing — finer when interpolation is enabled."""
        return (
            STORED_INTERVAL_SECONDS
            if settings.regional_interpolation
            else SOURCE_STEP_SECONDS
        )

    @property
    def supports_snow(self) -> bool:
        return True

    def get_snow_mask(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        """Sample the snow / rain classification at each (lat, lon, ts)."""
        if timestamp is None or not self._snow_masks:
            return np.zeros(lat.shape, dtype=bool)
        run = self._pick_run(timestamp)
        if run is None:
            return np.zeros(lat.shape, dtype=bool)
        lead = timestamp - run
        l0, l1, alpha = bracket_lead_seconds(lead, self._bracket_interval())
        s0 = self._snow_masks.get((run, l0))
        s1 = self._snow_masks.get((run, l1))
        if s0 is None or s1 is None:
            return np.zeros(lat.shape, dtype=bool)
        if alpha == 0.0:
            grid = s0
        elif alpha == 1.0:
            grid = s1
        else:
            lerped = (
                (1.0 - alpha) * s0.astype(np.float32)
                + alpha * s1.astype(np.float32)
            )
            grid = (lerped >= 0.5).astype(np.uint8)
        sampled = _sample_grid(grid, lat, lon, bilinear=False)
        return sampled.astype(bool)

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
        bilinear: bool = False,
    ) -> np.ndarray:
        if timestamp is None or not self._frames:
            return np.zeros(lat.shape, dtype=np.uint8)
        run = self._pick_run(timestamp)
        if run is None:
            return np.zeros(lat.shape, dtype=np.uint8)
        lead = timestamp - run
        l0, l1, alpha = bracket_lead_seconds(lead, self._bracket_interval())
        f0 = self._frames.get((run, l0))
        f1 = self._frames.get((run, l1))
        if f0 is None or f1 is None:
            return np.zeros(lat.shape, dtype=np.uint8)
        if alpha == 0.0:
            grid = f0
        elif alpha == 1.0:
            grid = f1
        else:
            grid = (
                (1.0 - alpha) * f0.astype(np.float32)
                + alpha * f1.astype(np.float32)
                + 0.5
            ).astype(np.uint8)
        return _sample_grid(grid, lat, lon, bilinear=bilinear)

    # ── Run selection ─────────────────────────────────────────────────

    def _pick_run(self, timestamp: int) -> int | None:
        """Pick the freshest run whose bracket is loaded for ``timestamp``."""
        interval = self._bracket_interval()
        loaded_runs = sorted({r for (r, _) in self._frames}, reverse=True)
        for run in loaded_runs:
            lead = timestamp - run
            if not (0 <= lead <= MAX_FORECAST_HOURS * 3600):
                continue
            l0, l1, _ = bracket_lead_seconds(lead, interval)
            if (run, l0) in self._frames and (run, l1) in self._frames:
                return run
        return None

    # ── Fetch loop ────────────────────────────────────────────────────

    def _get_fs(self) -> fsspec.AbstractFileSystem:
        if self._fs is None:
            self._fs = fsspec.filesystem(
                "s3", anon=True,
                client_kwargs={"region_name": settings.jma_msm_s3_region},
            )
        return self._fs

    async def fetch(
        self,
        now_ts: int | None = None,
        history_seconds: int = 0,
        horizon_seconds: int = 60 * 60,
    ) -> None:
        """Refresh the in-memory window — same shape as ICONEUGrid.fetch."""
        async with self._fetch_lock:
            if now_ts is None:
                now_ts = int(datetime.now(tz=timezone.utc).timestamp())

            publish_delay = settings.jma_msm_publish_delay_minutes * 60
            latest_run_ts = latest_published_run(now_ts, publish_delay)
            if self._latest_run_ts is None or latest_run_ts > self._latest_run_ts:
                self._latest_run_ts = latest_run_ts

            window_start = now_ts - history_seconds
            window_end = now_ts + horizon_seconds

            earliest_run = max(
                floor_cycle(window_start - CYCLE_INTERVAL_SECONDS),
                latest_run_ts - RUN_LOOKBACK_CYCLES * CYCLE_INTERVAL_SECONDS,
            )
            runs_to_consider = list(range(
                earliest_run, latest_run_ts + 1, CYCLE_INTERVAL_SECONDS,
            ))
            if not runs_to_consider:
                logger.debug("JMA MSM fetch: no runs available for window")
                return

            bucket = settings.jma_msm_s3_bucket
            total_fetched = 0
            total_failed = 0
            for run_ts in runs_to_consider:
                run_dt = datetime.fromtimestamp(run_ts, tz=timezone.utc)
                min_lead = max(0, window_start - run_ts)
                max_lead = min(
                    MAX_FORECAST_HOURS * 3600,
                    window_end - run_ts + BRACKET_INTERVAL_SECONDS,
                )
                if max_lead < min_lead:
                    continue
                min_step = max(0, min_lead // BRACKET_INTERVAL_SECONDS)
                max_step = min(
                    MAX_FORECAST_HOURS,
                    -(-max_lead // BRACKET_INTERVAL_SECONDS),
                )
                for step in range(int(min_step), int(max_step) + 1):
                    added = await asyncio.to_thread(
                        self._fetch_one_step_sync, run_dt, step, bucket,
                    )
                    if added > 0:
                        total_fetched += added
                    elif added < 0:
                        total_failed += 1

            # Optical-flow interpolation per run.  Fills 10-min synthetic
            # frames between hourly originals.  Idempotent.
            total_interpolated = 0
            if settings.regional_interpolation:
                for run_ts in runs_to_consider:
                    total_interpolated += self._interpolate_run_frames(run_ts)

            self._evict_outside_window(window_start, window_end)

            if total_fetched:
                logger.info(
                    "JMA MSM: %d hourly frame(s) ingested + %d interpolated "
                    "across %d run(s); store now holds %d frame(s)",
                    total_fetched, total_interpolated,
                    len(runs_to_consider), len(self._frames),
                )
            elif total_failed:
                logger.warning(
                    "JMA MSM: no frames ingested (%d file(s) failed)",
                    total_failed,
                )

    def _interpolate_run_frames(self, run_ts: int) -> int:
        """Fill 10-min synthetic frames between hourly originals for one run."""
        from librewxr.data.nwp_interpolation import interpolate_run

        frames_by_lead: dict[int, np.ndarray] = {
            lead: arr
            for (r, lead), arr in self._frames.items()
            if r == run_ts
        }
        if len(frames_by_lead) < 2:
            return 0
        snow_by_lead: dict[int, np.ndarray] | None = {
            lead: arr
            for (r, lead), arr in self._snow_masks.items()
            if r == run_ts
        }
        if not snow_by_lead:
            snow_by_lead = None

        aug_frames, aug_snow, _flow = interpolate_run(
            frames_by_lead,
            snow_masks_by_ts=snow_by_lead,
            target_interval_seconds=STORED_INTERVAL_SECONDS,
            log_label=f"JMA MSM interpolation (run {run_ts})",
        )

        added = 0
        for lead, arr in aug_frames.items():
            if (run_ts, lead) in self._frames:
                continue
            mm = self._to_memmap(f"r{run_ts}_l{lead}", arr)
            self._frames[(run_ts, lead)] = mm
            added += 1
        if aug_snow is not None:
            for lead, snow_arr in aug_snow.items():
                if (run_ts, lead) in self._snow_masks:
                    continue
                if (run_ts, lead) not in self._frames:
                    continue
                snow_uint8 = (
                    snow_arr.astype(np.uint8)
                    if snow_arr.dtype != np.uint8
                    else snow_arr
                )
                mm = self._to_memmap(
                    f"r{run_ts}_l{lead}_snow", snow_uint8,
                )
                self._snow_masks[(run_ts, lead)] = mm
        return added

    def _fetch_one_step_sync(
        self, run: datetime, step_hour: int, bucket: str,
    ) -> int:
        """Fetch one (run, step) .om file, decode precip + 2t, store.

        Returns 1 on success, 0 if already loaded, -1 on fetch error.
        Runs synchronously in a worker thread (called via asyncio.to_thread)
        because the omfiles reader + fsspec are blocking.
        """
        run_ts = int(run.timestamp())
        lead_seconds = step_hour * BRACKET_INTERVAL_SECONDS

        if (run_ts, lead_seconds) in self._frames:
            return 0

        fs = self._get_fs()
        s3_path = f"{bucket}/{file_key(run, step_hour)}"

        precip_rate = _decode_om_field(fs, s3_path, "precipitation")
        if precip_rate is None:
            return -1

        encoded = precip_rate_to_dbz_encoded(
            precip_rate,
            dbz_offset=settings.jma_msm_dbz_offset,
        )
        mm = self._to_memmap(f"r{run_ts}_l{lead_seconds}", encoded)
        self._frames[(run_ts, lead_seconds)] = mm

        # Snow side: decode temperature_2m from the SAME .om file and
        # derive the snow mask.  One container, two fields, one fetch.
        # Failures are non-fatal: the precip frame still lands and
        # ``get_snow_mask`` falls through to the next snow-capable
        # source for the affected bracket.
        if (run_ts, lead_seconds) not in self._snow_masks:
            t2_celsius = _decode_om_field(fs, s3_path, "temperature_2m")
            if t2_celsius is not None:
                threshold = settings.regional_snow_temp_threshold
                snow = compute_snow_mask(t2_celsius, threshold)
                snow_mm = self._to_memmap(
                    f"r{run_ts}_l{lead_seconds}_snow", snow,
                )
                self._snow_masks[(run_ts, lead_seconds)] = snow_mm

        return 1

    # ── Eviction ──────────────────────────────────────────────────────

    def _evict_outside_window(self, window_start: int, window_end: int) -> None:
        slack = BRACKET_INTERVAL_SECONDS
        ws = window_start - slack
        we = window_end + slack
        stale_frames = []
        for key in self._frames:
            run_ts, lead = key
            valid_time = run_ts + lead
            if valid_time < ws or valid_time > we:
                stale_frames.append(key)
        for key in stale_frames:
            self._frames.pop(key, None)
            self._snow_masks.pop(key, None)
            try:
                self._frame_path(*key).unlink(missing_ok=True)
            except OSError:
                pass
            try:
                self._snow_frame_path(*key).unlink(missing_ok=True)
            except OSError:
                pass
        stale_orphan_snow = []
        for key in self._snow_masks:
            run_ts, lead = key
            valid_time = run_ts + lead
            if valid_time < ws or valid_time > we:
                stale_orphan_snow.append(key)
        for key in stale_orphan_snow:
            self._snow_masks.pop(key, None)
            try:
                self._snow_frame_path(*key).unlink(missing_ok=True)
            except OSError:
                pass
        if stale_frames:
            logger.info(
                "JMA MSM: evicted %d out-of-window frame(s)", len(stale_frames),
            )

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def close(self) -> None:
        self._frames.clear()
        self._snow_masks.clear()
        self._fs = None
        if not self._persistent:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.info("JMA MSM memmap directory cleaned up")
        else:
            logger.info(
                "JMA MSM cache retained at %s for warm restart",
                self._memmap_dir,
            )


# ── Grid sampling ────────────────────────────────────────────────────


def _sample_grid(
    grid: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    bilinear: bool = False,
) -> np.ndarray:
    """Sample a uint8 lat/lon grid at (lat, lon) points."""
    row_f, col_f = grid_indices(lat, lon)

    if not bilinear:
        row = np.rint(row_f).astype(np.int32)
        col = np.rint(col_f).astype(np.int32)
        in_domain = (
            (row >= 0)
            & (row < JMA_MSM_GRID_HEIGHT)
            & (col >= 0)
            & (col < JMA_MSM_GRID_WIDTH)
        )
        out = np.zeros(lat.shape, dtype=np.uint8)
        if in_domain.any():
            out[in_domain] = grid[row[in_domain], col[in_domain]]
        return out

    r0 = np.floor(row_f).astype(np.int32)
    c0 = np.floor(col_f).astype(np.int32)
    r1 = r0 + 1
    c1 = c0 + 1
    in_domain = (
        (r0 >= 0)
        & (r1 < JMA_MSM_GRID_HEIGHT)
        & (c0 >= 0)
        & (c1 < JMA_MSM_GRID_WIDTH)
    )
    r0c = np.clip(r0, 0, JMA_MSM_GRID_HEIGHT - 1)
    r1c = np.clip(r1, 0, JMA_MSM_GRID_HEIGHT - 1)
    c0c = np.clip(c0, 0, JMA_MSM_GRID_WIDTH - 1)
    c1c = np.clip(c1, 0, JMA_MSM_GRID_WIDTH - 1)
    dr = np.clip(row_f - r0, 0.0, 1.0).astype(np.float32)
    dc = np.clip(col_f - c0, 0.0, 1.0).astype(np.float32)
    v00 = grid[r0c, c0c].astype(np.float32)
    v01 = grid[r0c, c1c].astype(np.float32)
    v10 = grid[r1c, c0c].astype(np.float32)
    v11 = grid[r1c, c1c].astype(np.float32)
    any_zero = (v00 == 0) | (v01 == 0) | (v10 == 0) | (v11 == 0)
    interp = (
        v00 * (1 - dr) * (1 - dc)
        + v01 * (1 - dr) * dc
        + v10 * dr * (1 - dc)
        + v11 * dr * dc
    )
    sampled = np.where(any_zero, v00, interp)
    out = np.clip(sampled + 0.5, 0, 255).astype(np.uint8)
    out[~in_domain] = 0
    return out
