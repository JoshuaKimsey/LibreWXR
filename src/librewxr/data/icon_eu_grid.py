# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""DWD ICON-EU regional precipitation source.

Implements the NWPSource Protocol for Deutscher Wetterdienst's ICON-EU
model — 6.5 km native, regridded to 0.0625° regular lat/lon, covering
Europe (29.5–70.5°N, 23.5°W–62.5°E).  Eight daily cycles
(00/03/06/09/12/15/18/21 UTC) give a ~3-hour effective freshness;
forecast steps are hourly out to +30 h (intermediate runs) or +120 h
(main runs).

ICON-EU does not publish a native composite reflectivity field — only
accumulated precipitation (``tot_prec``).  We difference consecutive
forecast steps to recover the hourly rate and apply the same
Marshall-Palmer Z-R conversion ECMWFGrid uses for IFS.

Distribution: free public access on DWD's open-data file server, no
auth, plain HTTPS.  Files are bzip2-compressed GRIB2, one variable per
file, one forecast step per file.

Data attribution: Deutscher Wetterdienst (DWD), CC-BY-4.0.
"""

from __future__ import annotations

import asyncio
import bz2
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from librewxr.config import settings

logger = logging.getLogger(__name__)


# ── ICON-EU regridded lat/lon grid parameters ─────────────────────────
#
# Source: cfgrib-decoded coordinates of a real DWD ICON-EU GRIB file.
# The output is on a regular lat/lon grid with 0.0625° spacing.  Lat
# increases with row index (row 0 is the SOUTHERN edge), opposite the
# image convention our grid_indices() uses, so decode flips the array
# vertically (same gotcha as HRRR via cfgrib).

ICON_EU_LAT_MIN = 29.5    # southern grid edge (deg N)
ICON_EU_LAT_MAX = 70.5    # northern grid edge (deg N)
ICON_EU_LON_MIN = -23.5   # western grid edge (deg E)
ICON_EU_LON_MAX = 62.5    # eastern grid edge (deg E)
ICON_EU_PIXEL_SIZE = 0.0625
ICON_EU_GRID_HEIGHT = int(round((ICON_EU_LAT_MAX - ICON_EU_LAT_MIN) / ICON_EU_PIXEL_SIZE)) + 1   # 657
ICON_EU_GRID_WIDTH  = int(round((ICON_EU_LON_MAX - ICON_EU_LON_MIN) / ICON_EU_PIXEL_SIZE)) + 1   # 1377


def grid_indices(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return fractional ``(row, col)`` on the ICON-EU grid (north-up).

    After decode flips the array, row 0 is the NORTHERN edge.  Out-of-
    domain points still return values; callers should test
    ``domain_mask`` first.
    """
    row = (ICON_EU_LAT_MAX - lat) / ICON_EU_PIXEL_SIZE
    col = (lon - ICON_EU_LON_MIN) / ICON_EU_PIXEL_SIZE
    return row, col


def domain_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """``True`` where (lat, lon) falls inside the ICON-EU output grid."""
    row, col = grid_indices(lat, lon)
    return (
        (row >= 0)
        & (row < ICON_EU_GRID_HEIGHT - 1)
        & (col >= 0)
        & (col < ICON_EU_GRID_WIDTH - 1)
    )


# ── Boundary feathering ───────────────────────────────────────────────

ICON_EU_FEATHER_DISTANCE_DEG = 1.0  # ~70-110 km depending on latitude


def feather_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Soft taper to 0 at the ICON-EU domain edge (~1° in lat/lon space)."""
    # Distance to nearest edge in degrees
    lat_dist = np.minimum(lat - ICON_EU_LAT_MIN, ICON_EU_LAT_MAX - lat)
    lon_dist = np.minimum(lon - ICON_EU_LON_MIN, ICON_EU_LON_MAX - lon)
    dist_deg = np.minimum(lat_dist, lon_dist)
    weight = np.clip(dist_deg / ICON_EU_FEATHER_DISTANCE_DEG, 0.0, 1.0)
    return weight.astype(np.float32, copy=False)


# ── Z-R conversion (matches ECMWFGrid IFS path) ───────────────────────

ZR_A_RAIN = 200.0
ZR_B_RAIN = 1.6


def precip_rate_to_dbz_encoded(precip_mm_per_hour: np.ndarray) -> np.ndarray:
    """Convert mm/h precip rate → uint8 dBZ encoded (pixel = (dBZ+32)*2).

    Negative or non-finite values map to 0 (no precipitation).  Uses
    Marshall-Palmer rain Z-R: Z = 200 * R^1.6, dBZ = 10 * log10(Z).
    """
    rate = np.where(np.isfinite(precip_mm_per_hour), precip_mm_per_hour, 0.0)
    rate = np.maximum(rate, 0.0)
    # Z = a * R^b — guard log10 against zero rate
    eps = 1e-6
    z = ZR_A_RAIN * np.power(rate + eps, ZR_B_RAIN)
    dbz = 10.0 * np.log10(np.maximum(z, eps))
    encoded = np.clip((dbz + 32.0) * 2.0 + 0.5, 0, 255)
    # Zero-rate sentinel: clamp to 0 so the downstream noise floor and
    # both-zero blend logic both see "no precipitation" rather than the
    # ZR_A * eps^b artifact.
    encoded[rate <= 0.0] = 0
    return encoded.astype(np.uint8)


# ── Run / step timing ─────────────────────────────────────────────────

CYCLE_INTERVAL_SECONDS = 3 * 3600       # ICON-EU runs every 3 hours
BRACKET_INTERVAL_SECONDS = 3600          # forecast steps are 1 hour apart
MAX_FORECAST_HOURS = 30                  # intermediate-run horizon (main runs go further)

# How far back to walk through run cycles when looking for a run that
# can serve a given valid time.  Two cycles (6 hours) is plenty: each
# run reaches +30 h forward, so the latest two cycles cover any active
# window comfortably.  Going further back means falling off the end of
# intermediate runs' published forecast horizons, which produces 404s
# at the request layer.
RUN_LOOKBACK_CYCLES = 2


def floor_cycle(ts: int) -> int:
    """Floor a Unix timestamp to the nearest 3-hour cycle boundary."""
    return (ts // CYCLE_INTERVAL_SECONDS) * CYCLE_INTERVAL_SECONDS


def latest_published_run(now_ts: int, publish_delay_seconds: int) -> int:
    """Most recent run we'd expect to be available given a publish delay."""
    return floor_cycle(now_ts - publish_delay_seconds)


def bracket_lead_seconds(lead_seconds: int) -> tuple[int, int, float]:
    """For a desired lead, return ``(L0, L1, alpha)`` such that L0 ≤ L < L1.

    Both leads are exact hour multiples (multiples of 3600s, ≥ 0).
    Alpha is the lerp weight.

    For ``lead_seconds < 0`` the bracket falls back; the caller selects
    an earlier run instead.
    """
    if lead_seconds < 0:
        return 0, 0, 0.0
    l0 = (lead_seconds // BRACKET_INTERVAL_SECONDS) * BRACKET_INTERVAL_SECONDS
    l1 = l0 + BRACKET_INTERVAL_SECONDS
    alpha = (lead_seconds - l0) / BRACKET_INTERVAL_SECONDS
    return l0, l1, alpha


# ── DWD opendata file URLs ────────────────────────────────────────────

def file_url(run: datetime, step_hour: int, param: str) -> str:
    """Build the opendata.dwd.de URL for a single (run, step, variable) file.

    ``param`` is the lowercase variable name (matches directory name);
    the filename uses upper case.
    """
    base = settings.icon_eu_base_url.rstrip("/")
    hh = run.strftime("%H")
    yyyymmddhh = run.strftime("%Y%m%d%H")
    step = f"{step_hour:03d}"
    upper = param.upper()
    return (
        f"{base}/{hh}/{param}/"
        f"icon-eu_europe_regular-lat-lon_single-level_"
        f"{yyyymmddhh}_{step}_{upper}.grib2.bz2"
    )


# ── GRIB2 decode + bzip2 unpack ───────────────────────────────────────

def _suppress_eccodes_stderr():
    from librewxr.data.sources import _suppress_eccodes_stderr as _s
    return _s()


def decode_tp_message(grib_bytes: bytes) -> np.ndarray | None:
    """Decode an ICON-EU ``tot_prec`` GRIB2 message into a 2D float32 array.

    Returns ``None`` on parse failure.  Output shape is
    ``(ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH)`` with row 0 at the
    NORTHERN edge (the array is flipped vertically because cfgrib
    returns ICON-EU with row 0 = south).
    """
    import xarray as xr

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(grib_bytes)
            tmp_path = tmp.name
        with _suppress_eccodes_stderr():
            ds = xr.open_dataset(
                tmp_path,
                engine="cfgrib",
                backend_kwargs={"indexpath": ""},
            )
        ds = ds.compute()
    except Exception:
        logger.exception("Failed to decode ICON-EU tot_prec GRIB2 message")
        return None
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    # cfgrib names the variable ``tp`` (mm of accumulated precipitation).
    if "tp" in ds.data_vars:
        arr = ds["tp"].values
    else:
        for name, da in ds.data_vars.items():
            if da.ndim == 2 and da.shape == (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH):
                logger.warning(
                    "ICON-EU tp variable not named 'tp' (got %r); using fallback",
                    name,
                )
                arr = da.values
                break
        else:
            logger.warning("ICON-EU GRIB had no recognised tot_prec field")
            return None

    if arr.shape != (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH):
        logger.warning(
            "ICON-EU tp has unexpected shape %s (expected %s); skipping",
            arr.shape, (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH),
        )
        return None

    # cfgrib returns row 0 at the southern edge.  Verify and flip when
    # needed so row 0 = north matches grid_indices().
    if "latitude" in ds.coords:
        lat_first = float(np.asarray(ds["latitude"].values).flat[0])
        lat_last = float(np.asarray(ds["latitude"].values).flat[-1])
        needs_flip = lat_first < lat_last
    else:
        needs_flip = True
    if needs_flip:
        arr = np.flipud(arr)

    return np.ascontiguousarray(arr, dtype=np.float32)


def decompress_bz2(data: bytes) -> bytes:
    """Decompress a bzip2-encoded GRIB2 payload."""
    return bz2.decompress(data)


# ── ICONEUGrid: the public NWPSource implementation ──────────────────


class ICONEUGrid:
    """DWD ICON-EU as an NWPSource for the European chain slot."""

    name = "icon_eu"

    def __init__(self, cache_dir: Path | None = None):
        # (run_ts, lead_seconds) -> uint8 dBZ-encoded array on the lat/lon grid
        self._frames: dict[tuple[int, int], np.ndarray] = {}
        # Raw accumulated tp values keyed by (run_ts, step_hour).  Kept
        # only long enough to compute the rate at the next forecast step;
        # purged on eviction.
        self._accum: dict[tuple[int, int], np.ndarray] = {}
        self._client: httpx.AsyncClient | None = None
        self._latest_run_ts: int | None = None
        self._fetch_lock = asyncio.Lock()

        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "icon_eu"
            self._persistent = True
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_icon_eu_"))
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "ICON-EU memmap directory: %s (persistent=%s)",
            self._memmap_dir, self._persistent,
        )
        if self._persistent:
            self._load_cached_frames()

    # ── Cache management ──────────────────────────────────────────────

    def _frame_path(self, run_ts: int, lead_seconds: int) -> Path:
        return self._memmap_dir / f"r{run_ts}_l{lead_seconds}.dat"

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
                    shape=(ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH),
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
            logger.info("ICON-EU: loaded %d cached frame(s) from disk", loaded)

    @property
    def data_bytes(self) -> int:
        return sum(arr.nbytes for arr in self._frames.values())

    @property
    def latest_run_iso(self) -> str | None:
        if self._latest_run_ts is None:
            return None
        return datetime.fromtimestamp(self._latest_run_ts, tz=timezone.utc).isoformat()

    @property
    def frame_count(self) -> int:
        return len(self._frames)

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
        l0, l1, _ = bracket_lead_seconds(lead)
        return ((run, l0) in self._frames) and ((run, l1) in self._frames)

    def get_snow_mask(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        # Phase 4 v1: defer to next chain source (IFS) for snow classification.
        return np.zeros(lat.shape, dtype=bool)

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
        l0, l1, alpha = bracket_lead_seconds(lead)
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
        loaded_runs = sorted({r for (r, _) in self._frames}, reverse=True)
        for run in loaded_runs:
            lead = timestamp - run
            if not (0 <= lead <= MAX_FORECAST_HOURS * 3600):
                continue
            l0, l1, _ = bracket_lead_seconds(lead)
            if (run, l0) in self._frames and (run, l1) in self._frames:
                return run
        return None

    # ── Fetch loop ────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True,
            )
        return self._client

    async def fetch(
        self,
        now_ts: int | None = None,
        history_seconds: int = 0,
        horizon_seconds: int = 60 * 60,
    ) -> None:
        """Refresh the in-memory window — same shape as HRRRGrid.fetch."""
        async with self._fetch_lock:
            if now_ts is None:
                now_ts = int(datetime.now(tz=timezone.utc).timestamp())

            publish_delay = settings.icon_eu_publish_delay_minutes * 60
            latest_run_ts = latest_published_run(now_ts, publish_delay)
            if self._latest_run_ts is None or latest_run_ts > self._latest_run_ts:
                self._latest_run_ts = latest_run_ts

            window_start = now_ts - history_seconds
            window_end = now_ts + horizon_seconds

            # Enumerate at most a handful of recent cycle boundaries
            # (latest, latest-3h, latest-6h).  Each run reaches +30 h
            # forward so two cycles is enough for any reasonable active
            # window; going further back falls off intermediate runs'
            # forecast horizons and yields 404s.
            earliest_run = max(
                floor_cycle(window_start - CYCLE_INTERVAL_SECONDS),
                latest_run_ts - RUN_LOOKBACK_CYCLES * CYCLE_INTERVAL_SECONDS,
            )
            runs_to_consider = list(range(
                earliest_run, latest_run_ts + 1, CYCLE_INTERVAL_SECONDS,
            ))
            if not runs_to_consider:
                logger.debug("ICON-EU fetch: no runs available for window")
                return

            client = await self._get_client()

            total_fetched = 0
            total_failed = 0
            for run_ts in runs_to_consider:
                run_dt = datetime.fromtimestamp(run_ts, tz=timezone.utc)
                # Bracket-interval slack on each side so the L1 frame at
                # the future edge and the L0 frame at the past edge are
                # both included.
                min_lead = max(0, window_start - run_ts - BRACKET_INTERVAL_SECONDS)
                max_lead = min(
                    MAX_FORECAST_HOURS * 3600,
                    window_end - run_ts + BRACKET_INTERVAL_SECONDS,
                )
                if max_lead < min_lead:
                    continue
                # Need step F-1 to compute the rate at step F via diff,
                # so always start one step earlier than we strictly need.
                min_step = max(0, (min_lead // BRACKET_INTERVAL_SECONDS) - 1)
                max_step = min(
                    MAX_FORECAST_HOURS,
                    -(-max_lead // BRACKET_INTERVAL_SECONDS),
                )
                for step in range(int(min_step), int(max_step) + 1):
                    added = await self._fetch_one_step(run_dt, step, client)
                    if added > 0:
                        total_fetched += added
                    elif added < 0:
                        total_failed += 1

            self._evict_outside_window(window_start, window_end)

            if total_fetched:
                logger.info(
                    "ICON-EU: %d frame(s) ingested across %d run(s); "
                    "store now holds %d frame(s)",
                    total_fetched, len(runs_to_consider), len(self._frames),
                )
            elif total_failed:
                logger.warning(
                    "ICON-EU: no frames ingested (%d file(s) failed)",
                    total_failed,
                )

    async def _fetch_one_step(
        self, run: datetime, step_hour: int, client: httpx.AsyncClient,
    ) -> int:
        """Fetch one step's tot_prec, difference against the previous step,
        encode, and store.  Returns 1 on success, 0 if already loaded,
        -1 on fetch error.
        """
        run_ts = int(run.timestamp())
        lead_seconds = step_hour * BRACKET_INTERVAL_SECONDS

        # Step 0: accumulated precip is zero everywhere (nothing has
        # rained yet at model init).  We still cache it so step 1's diff
        # is straightforward.
        if step_hour == 0:
            if (run_ts, 0) not in self._accum:
                self._accum[(run_ts, 0)] = np.zeros(
                    (ICON_EU_GRID_HEIGHT, ICON_EU_GRID_WIDTH),
                    dtype=np.float32,
                )
            return 0

        # Already have the encoded frame — nothing to do.
        if (run_ts, lead_seconds) in self._frames:
            return 0

        url = file_url(run, step_hour, "tot_prec")
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("ICON-EU fetch failed for %s: %s", url, e)
            return -1

        try:
            grib_bytes = decompress_bz2(resp.content)
        except Exception:
            logger.exception("ICON-EU bz2 decompress failed for %s", url)
            return -1

        accum = decode_tp_message(grib_bytes)
        if accum is None:
            return -1

        # Ensure we have step F-1 cached for the diff.  If not, fetch it.
        prev_key = (run_ts, step_hour - 1)
        prev = self._accum.get(prev_key)
        if prev is None and step_hour - 1 >= 0:
            # Recursive fetch of the previous step.  Step 0 is the
            # zero-baseline; everything else is a real download.
            await self._fetch_one_step(run, step_hour - 1, client)
            prev = self._accum.get(prev_key)
        if prev is None:
            # Couldn't establish a baseline — bail.
            return -1

        rate_mm_per_hour = accum - prev
        encoded = precip_rate_to_dbz_encoded(rate_mm_per_hour)
        mm = self._to_memmap(f"r{run_ts}_l{lead_seconds}", encoded)
        self._frames[(run_ts, lead_seconds)] = mm
        self._accum[(run_ts, step_hour)] = accum

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
            try:
                self._frame_path(*key).unlink(missing_ok=True)
            except OSError:
                pass
        # Drop accumulated cache for runs whose entire window is outside.
        stale_accums = []
        for (run_ts, step_h) in self._accum:
            valid_time = run_ts + step_h * BRACKET_INTERVAL_SECONDS
            if valid_time < ws - BRACKET_INTERVAL_SECONDS or valid_time > we:
                stale_accums.append((run_ts, step_h))
        for k in stale_accums:
            self._accum.pop(k, None)
        if stale_frames:
            logger.info(
                "ICON-EU: evicted %d out-of-window frame(s)", len(stale_frames),
            )

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def close(self) -> None:
        self._frames.clear()
        self._accum.clear()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        if not self._persistent:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.info("ICON-EU memmap directory cleaned up")
        else:
            logger.info(
                "ICON-EU cache retained at %s for warm restart", self._memmap_dir,
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
            & (row < ICON_EU_GRID_HEIGHT)
            & (col >= 0)
            & (col < ICON_EU_GRID_WIDTH)
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
        & (r1 < ICON_EU_GRID_HEIGHT)
        & (c0 >= 0)
        & (c1 < ICON_EU_GRID_WIDTH)
    )
    r0c = np.clip(r0, 0, ICON_EU_GRID_HEIGHT - 1)
    r1c = np.clip(r1, 0, ICON_EU_GRID_HEIGHT - 1)
    c0c = np.clip(c0, 0, ICON_EU_GRID_WIDTH - 1)
    c1c = np.clip(c1, 0, ICON_EU_GRID_WIDTH - 1)
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
