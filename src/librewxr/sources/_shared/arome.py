# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Shared base class for Météo-France AROME overseas NWP variants.

All AROME overseas models (Antilles, Réunion-Mayotte, French Guiana,
New Caledonia, French Polynesia) share:

- Same upstream: ``meteofrance-pnt.s3.rbx.io.cloud.ovh.net`` (OVH
  OpenStack Swift / S3-compatible), anonymous HTTPS, Etalab Open
  Licence v2.0.  Météo-France migrated PNT distribution off the older
  ``object.data.gouv.fr/meteofrance-pnt/`` host around 2026-01; the
  data.gouv.fr API still works but its file links now 302 to OVH.
- Same file layout: single-message GRIB2 per (run, lead hour) with
  ``shortName=tp`` containing accumulated total precipitation
  (kg/m² ≡ mm), regular lat/lon grid (gridType=regular_ll), scan
  mode 0 (row 0 at NORTHERN edge).
- Same cadence: 4 cycles/day (00/06/12/18 UTC), 1-hour forecast
  steps, 48-hour horizon.
- Same dBZ derivation: differenced consecutive cumulative-tp steps
  → mm/h, Marshall-Palmer Z-R (Z = 200 R^1.6) → dBZ → uint8 encoded.

What differs per variant:

- Grid extent and dimensions (``LAT_NORTH``..``GRID_HEIGHT``).
- URL token in the path (``"ANTIL"`` for Antilles, etc.).
- Feather distance (tuned to domain size; small domains feather
  tighter than large ones).
- Settings prefix (``arome_antilles``, ``arome_reunion``, …) — drives
  config lookup for base URL, publish delay, and dBZ offset.

A new overseas variant subclasses ``AROMEOverseasGrid``, overrides the
``ClassVar``-typed attributes near the top of the class, and exposes
itself via the source package's ``nwp_provider``.  No fetch / decode
/ cache code needs to change.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

import httpx
import numpy as np

from librewxr.config import settings
from librewxr.sources._helpers import _suppress_eccodes_stderr

logger = logging.getLogger(__name__)


# ── Marshall-Palmer Z-R (shared with DMI DINI / ICON-EU / HRDPS) ──

ZR_A_RAIN = 200.0
ZR_B_RAIN = 1.6


def precip_rate_to_dbz_encoded(
    precip_mm_per_hour: np.ndarray,
    dbz_offset: float = 0.0,
) -> np.ndarray:
    """Convert mm/h precip rate → uint8 dBZ encoded (pixel = (dBZ+32)*2).

    The optional ``dbz_offset`` shifts the result uniformly to compensate
    for the model-vs-radar intensity bias (radar samples the brightest
    part of the storm column while the model gives surface rate).
    """
    rate = np.where(np.isfinite(precip_mm_per_hour), precip_mm_per_hour, 0.0)
    rate = np.maximum(rate, 0.0)
    eps = 1e-6
    z = ZR_A_RAIN * np.power(rate + eps, ZR_B_RAIN)
    dbz = 10.0 * np.log10(np.maximum(z, eps)) + dbz_offset
    encoded = np.clip((dbz + 32.0) * 2.0 + 0.5, 0, 255)
    encoded[rate <= 0.0] = 0
    return encoded.astype(np.uint8)


# ── Shared timing (AROME-OM deterministic schedule) ──

CYCLE_INTERVAL_SECONDS = 6 * 3600        # runs every 6 h (00/06/12/18 UTC)
BRACKET_INTERVAL_SECONDS = 3600          # 1-hour forecast steps
MAX_FORECAST_HOURS = 48                  # all runs reach +48 h
RUN_LOOKBACK_CYCLES = 2                  # two 6h cycles back is plenty


def floor_cycle(ts: int) -> int:
    """Floor a Unix timestamp to the nearest 6-hour cycle boundary."""
    return (ts // CYCLE_INTERVAL_SECONDS) * CYCLE_INTERVAL_SECONDS


def latest_published_run(now_ts: int, publish_delay_seconds: int) -> int:
    """Most recent run we'd expect to be available given a publish delay."""
    return floor_cycle(now_ts - publish_delay_seconds)


def bracket_lead_seconds(lead_seconds: int) -> tuple[int, int, float]:
    """For a desired lead, return ``(L0, L1, alpha)`` such that L0 ≤ L < L1.

    Both leads are exact hour multiples (multiples of 3600 s, ≥ 0).
    Alpha is the lerp weight: 0 at L0, 1 at L1.
    """
    if lead_seconds < 0:
        return 0, 0, 0.0
    l0 = (lead_seconds // BRACKET_INTERVAL_SECONDS) * BRACKET_INTERVAL_SECONDS
    l1 = l0 + BRACKET_INTERVAL_SECONDS
    alpha = (lead_seconds - l0) / BRACKET_INTERVAL_SECONDS
    return l0, l1, alpha


# ── ISO 8601 with colons (Météo-France filename convention) ──

_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _format_run_ts(dt: datetime) -> str:
    """Format a datetime as Météo-France's ISO 8601 with literal colons."""
    return dt.strftime(_TS_FMT)


# ── AROMEOverseasGrid: NWPSource base for every AROME-OM variant ──


class AROMEOverseasGrid:
    """Common NWPSource implementation for AROME overseas variants.

    Subclasses MUST override the ``ClassVar`` attributes below — grid
    constants, URL token, feather distance, and settings prefix.  All
    other behaviour (fetch loop, cache layout, sample, eviction) is
    inherited unchanged.

    Subclasses MAY override timing class attributes
    (``CYCLE_INTERVAL_SECONDS`` etc.) if a future overseas model
    deviates from the standard AROME-OM schedule.
    """

    # ── Identity (subclass MUST override) ──
    name: ClassVar[str]
    friendly_name: ClassVar[str]        # used in log messages
    settings_prefix: ClassVar[str]      # e.g. "arome_antilles"
    memmap_subdir: ClassVar[str]        # e.g. "arome_antilles"

    # ── URL scheme (subclass MUST override url_token) ──
    url_token: ClassVar[str]            # e.g. "ANTIL"
    resolution_token: ClassVar[str] = "0025"
    package_token: ClassVar[str] = "SP1"

    # ── Grid (subclass MUST override) ──
    LAT_NORTH: ClassVar[float]
    LAT_SOUTH: ClassVar[float]
    LON_WEST_DEG_E: ClassVar[float]
    LON_EAST_DEG_E: ClassVar[float]
    GRID_DLAT: ClassVar[float]
    GRID_DLON: ClassVar[float]
    GRID_WIDTH: ClassVar[int]
    GRID_HEIGHT: ClassVar[int]
    FEATHER_DISTANCE_CELLS: ClassVar[int]

    # ── Timing (override only for unusual schedules) ──
    CYCLE_INTERVAL_SECONDS: ClassVar[int] = CYCLE_INTERVAL_SECONDS
    BRACKET_INTERVAL_SECONDS: ClassVar[int] = BRACKET_INTERVAL_SECONDS
    MAX_FORECAST_HOURS: ClassVar[int] = MAX_FORECAST_HOURS
    RUN_LOOKBACK_CYCLES: ClassVar[int] = RUN_LOOKBACK_CYCLES

    # ── Construction & cache management ──

    def __init__(self, cache_dir: Path | None = None):
        self._frames: dict[tuple[int, int], np.ndarray] = {}
        self._accum: dict[tuple[int, int], np.ndarray] = {}
        self._client: httpx.AsyncClient | None = None
        self._latest_run_ts: int | None = None
        self._fetch_lock = asyncio.Lock()

        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / self.memmap_subdir
            self._persistent = True
        else:
            self._memmap_dir = Path(
                tempfile.mkdtemp(prefix=f"librewxr_{self.memmap_subdir}_")
            )
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "%s memmap directory: %s (persistent=%s)",
            self.friendly_name, self._memmap_dir, self._persistent,
        )
        if self._persistent:
            self._load_cached_frames()

    def _frame_path(self, run_ts: int, lead_seconds: int) -> Path:
        return self._memmap_dir / f"r{run_ts}_l{lead_seconds}.dat"

    def _to_memmap(self, name: str, data: np.ndarray) -> np.ndarray:
        """Atomic-write ``data`` and return a read-only memmap view."""
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
                    shape=(self.GRID_HEIGHT, self.GRID_WIDTH),
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
            logger.info(
                "%s: loaded %d cached frame(s) from disk",
                self.friendly_name, loaded,
            )

    def __getstate__(self) -> dict:
        return {
            "memmap_dir": str(self._memmap_dir),
            "latest_run_ts": self._latest_run_ts,
            "frame_keys": [[run, lead] for (run, lead) in self._frames.keys()],
        }

    def __setstate__(self, state: dict) -> None:
        self._memmap_dir = Path(state["memmap_dir"])
        self._persistent = True
        self._client = None
        self._fetch_lock = asyncio.Lock()
        self._frames = {}
        self._accum = {}
        self._latest_run_ts = None
        self._load_cached_frames()

    @property
    def data_bytes(self) -> int:
        return sum(arr.nbytes for arr in self._frames.values())

    @property
    def latest_run_iso(self) -> str | None:
        if self._latest_run_ts is None:
            return None
        return datetime.fromtimestamp(
            self._latest_run_ts, tz=timezone.utc,
        ).isoformat()

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    # ── Grid math ──

    @classmethod
    def grid_indices(
        cls, lat: np.ndarray, lon: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert (lat, lon) to fractional (row, col) on the variant grid.

        Row 0 is the NORTHERN edge, row HEIGHT-1 the southern.  Column 0
        is the western edge, column WIDTH-1 the eastern.  Inputs may use
        either [-180, 180] or [0, 360) longitude — both are folded onto
        the bucket's [0, 360) convention via modulo.
        """
        lon_e = np.mod(np.asarray(lon, dtype=np.float64), 360.0)
        row = (cls.LAT_NORTH - np.asarray(lat, dtype=np.float64)) / cls.GRID_DLAT
        col = (lon_e - cls.LON_WEST_DEG_E) / cls.GRID_DLON
        return row, col

    @classmethod
    def domain_mask(cls, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """``True`` where (lat, lon) falls inside the variant's grid."""
        row, col = cls.grid_indices(lat, lon)
        return (
            (row >= 0)
            & (row < cls.GRID_HEIGHT - 1)
            & (col >= 0)
            & (col < cls.GRID_WIDTH - 1)
        )

    @classmethod
    def feather_mask(cls, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Float32 weights in [0, 1]: 1 deep inside the variant, 0 outside."""
        row, col = cls.grid_indices(lat, lon)
        dist_cells = np.minimum(
            np.minimum(row, (cls.GRID_HEIGHT - 1) - row),
            np.minimum(col, (cls.GRID_WIDTH - 1) - col),
        )
        weight = np.clip(
            dist_cells / float(cls.FEATHER_DISTANCE_CELLS), 0.0, 1.0,
        )
        return weight.astype(np.float32, copy=False)

    # ── URL construction ──

    @classmethod
    def _get_setting(cls, key: str):
        """Look up ``settings.{settings_prefix}_{key}``."""
        return getattr(settings, f"{cls.settings_prefix}_{key}")

    @classmethod
    def file_url(cls, run: datetime, step_hour: int) -> str:
        """Construct the data.gouv.fr URL for one SP1 GRIB2 file."""
        base = cls._get_setting("base_url").rstrip("/")
        run_str = _format_run_ts(run)
        lll = f"{step_hour:03d}"
        return (
            f"{base}/pnt/{run_str}/arome-om/{cls.url_token}/"
            f"{cls.resolution_token}/{cls.package_token}/"
            f"arome-om-{cls.url_token}__{cls.resolution_token}__"
            f"{cls.package_token}__{lll}H__{run_str}.grib2"
        )

    # ── GRIB2 decode ──

    @classmethod
    def decode_tp_message(cls, grib_bytes: bytes) -> np.ndarray | None:
        """Decode the ``tp`` GRIB2 message into a 2D float32 array.

        Returns ``None`` on parse failure.  Output shape is
        ``(GRID_HEIGHT, GRID_WIDTH)`` with row 0 at the NORTHERN edge.
        Scan mode 0 in the source GRIB already gives row 0 at the north;
        if cfgrib ever permutes the orientation, the latitude coord is
        re-checked and the array is flipped to self-correct.
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
                    backend_kwargs={
                        "indexpath": "",
                        "filter_by_keys": {"shortName": "tp"},
                    },
                )
            ds = ds.compute()
        except Exception:
            logger.exception(
                "Failed to decode %s tp GRIB2 message", cls.friendly_name,
            )
            return None
        finally:
            if tmp_path is not None:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass

        expected_shape = (cls.GRID_HEIGHT, cls.GRID_WIDTH)
        if "tp" in ds.data_vars:
            arr = ds["tp"].values
        else:
            for name, da in ds.data_vars.items():
                if da.ndim == 2 and da.shape == expected_shape:
                    logger.warning(
                        "%s tp variable not named 'tp' (got %r); "
                        "using fallback", cls.friendly_name, name,
                    )
                    arr = da.values
                    break
            else:
                logger.warning(
                    "%s GRIB had no recognised tp field", cls.friendly_name,
                )
                return None

        if arr.shape != expected_shape:
            logger.warning(
                "%s tp has unexpected shape %s (expected %s); skipping",
                cls.friendly_name, arr.shape, expected_shape,
            )
            return None

        if "latitude" in ds.coords:
            lat_arr = np.asarray(ds["latitude"].values)
            if lat_arr.ndim == 1 and lat_arr.size > 1:
                needs_flip = lat_arr[0] < lat_arr[-1]
            elif lat_arr.ndim == 2 and lat_arr.shape[0] > 1:
                needs_flip = lat_arr[0, 0] < lat_arr[-1, 0]
            else:
                needs_flip = False
        else:
            needs_flip = False
        if needs_flip:
            arr = np.flipud(arr)

        return np.ascontiguousarray(arr, dtype=np.float32)

    # ── NWPSource Protocol ──

    def has_data(self) -> bool:
        return bool(self._frames)

    def has_data_at(self, timestamp: int) -> bool:
        run = self._pick_run(timestamp)
        if run is None:
            return False
        lead = timestamp - run
        l0, l1, _ = bracket_lead_seconds(lead)
        return ((run, l0) in self._frames) and ((run, l1) in self._frames)

    @property
    def supports_snow(self) -> bool:
        return False

    def get_snow_mask(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
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
        return self._sample_grid(grid, lat, lon, bilinear=bilinear)

    # ── Run selection ──

    def _pick_run(self, timestamp: int) -> int | None:
        loaded_runs = sorted({r for (r, _) in self._frames}, reverse=True)
        for run in loaded_runs:
            lead = timestamp - run
            if not (0 <= lead <= self.MAX_FORECAST_HOURS * 3600):
                continue
            l0, l1, _ = bracket_lead_seconds(lead)
            if (run, l0) in self._frames and (run, l1) in self._frames:
                return run
        return None

    # ── Fetch loop ──

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_keepalive_connections=4, max_connections=8,
                ),
            )
        return self._client

    async def fetch(
        self,
        now_ts: int | None = None,
        history_seconds: int = 0,
        horizon_seconds: int = 60 * 60,
    ) -> None:
        async with self._fetch_lock:
            if now_ts is None:
                now_ts = int(datetime.now(tz=timezone.utc).timestamp())

            publish_delay = self._get_setting("publish_delay_minutes") * 60
            latest_run_ts = latest_published_run(now_ts, publish_delay)
            if (
                self._latest_run_ts is None
                or latest_run_ts > self._latest_run_ts
            ):
                self._latest_run_ts = latest_run_ts

            window_start = now_ts - history_seconds
            window_end = now_ts + horizon_seconds

            earliest_run = max(
                floor_cycle(window_start - self.CYCLE_INTERVAL_SECONDS),
                latest_run_ts - self.RUN_LOOKBACK_CYCLES * self.CYCLE_INTERVAL_SECONDS,
            )
            runs_to_consider = list(range(
                earliest_run, latest_run_ts + 1, self.CYCLE_INTERVAL_SECONDS,
            ))
            if not runs_to_consider:
                logger.debug("%s fetch: no runs available for window", self.friendly_name)
                return

            client = await self._get_client()

            total_fetched = 0
            total_failed = 0
            for run_ts in runs_to_consider:
                run_dt = datetime.fromtimestamp(run_ts, tz=timezone.utc)
                min_lead = max(
                    0, window_start - run_ts - self.BRACKET_INTERVAL_SECONDS,
                )
                max_lead = min(
                    self.MAX_FORECAST_HOURS * 3600,
                    window_end - run_ts + self.BRACKET_INTERVAL_SECONDS,
                )
                if max_lead < min_lead:
                    continue
                min_step = max(0, (min_lead // self.BRACKET_INTERVAL_SECONDS) - 1)
                max_step = min(
                    self.MAX_FORECAST_HOURS,
                    -(-max_lead // self.BRACKET_INTERVAL_SECONDS),
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
                    "%s: %d frame(s) ingested across %d run(s); "
                    "store now holds %d frame(s)",
                    self.friendly_name, total_fetched, len(runs_to_consider),
                    len(self._frames),
                )
            elif total_failed:
                logger.warning(
                    "%s: no frames ingested (%d file(s) failed)",
                    self.friendly_name, total_failed,
                )

    async def _fetch_one_step(
        self, run: datetime, step_hour: int, client: httpx.AsyncClient,
    ) -> int:
        run_ts = int(run.timestamp())
        lead_seconds = step_hour * self.BRACKET_INTERVAL_SECONDS

        if step_hour == 0:
            if (run_ts, 0) not in self._accum:
                self._accum[(run_ts, 0)] = np.zeros(
                    (self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.float32,
                )
            return 0

        if (run_ts, lead_seconds) in self._frames:
            return 0

        url = self.file_url(run, step_hour)
        from librewxr.data.retry import retry_get
        resp = await retry_get(client, url, log_name=f"{self.friendly_name} data")
        if resp is None:
            return -1
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if getattr(e.response, "status_code", None) == 404:
                logger.debug("%s not yet published for %s", self.friendly_name, url)
            else:
                logger.warning("%s fetch failed for %s: %s", self.friendly_name, url, e)
            return -1
        grib_bytes = resp.content

        accum = self.decode_tp_message(grib_bytes)
        if accum is None:
            return -1

        prev_key = (run_ts, step_hour - 1)
        prev = self._accum.get(prev_key)
        if prev is None and step_hour - 1 >= 0:
            await self._fetch_one_step(run, step_hour - 1, client)
            prev = self._accum.get(prev_key)
        if prev is None:
            return -1

        rate_mm_per_hour = accum - prev
        encoded = precip_rate_to_dbz_encoded(
            rate_mm_per_hour,
            dbz_offset=self._get_setting("dbz_offset"),
        )
        mm = self._to_memmap(f"r{run_ts}_l{lead_seconds}", encoded)
        self._frames[(run_ts, lead_seconds)] = mm
        self._accum[(run_ts, step_hour)] = accum

        return 1

    # ── Eviction ──

    def _evict_outside_window(
        self, window_start: int, window_end: int,
    ) -> None:
        slack = self.BRACKET_INTERVAL_SECONDS
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
        stale_accums = []
        for (run_ts, step_h) in self._accum:
            valid_time = run_ts + step_h * self.BRACKET_INTERVAL_SECONDS
            if valid_time < ws - self.BRACKET_INTERVAL_SECONDS or valid_time > we:
                stale_accums.append((run_ts, step_h))
        for k in stale_accums:
            self._accum.pop(k, None)
        if stale_frames:
            logger.info(
                "%s: evicted %d out-of-window frame(s)",
                self.friendly_name, len(stale_frames),
            )

    # ── Lifecycle ──

    async def close(self) -> None:
        self._frames.clear()
        self._accum.clear()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        if not self._persistent:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.info("%s memmap directory cleaned up", self.friendly_name)
        else:
            logger.info(
                "%s cache retained at %s for warm restart",
                self.friendly_name, self._memmap_dir,
            )

    # ── Grid sampling ──

    @classmethod
    def _sample_grid(
        cls,
        grid: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        bilinear: bool = False,
    ) -> np.ndarray:
        """Sample a uint8 regular lat/lon grid at (lat, lon) points."""
        row_f, col_f = cls.grid_indices(lat, lon)

        if not bilinear:
            row = np.rint(row_f).astype(np.int32)
            col = np.rint(col_f).astype(np.int32)
            in_domain = (
                (row >= 0)
                & (row < cls.GRID_HEIGHT)
                & (col >= 0)
                & (col < cls.GRID_WIDTH)
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
            & (r1 < cls.GRID_HEIGHT)
            & (c0 >= 0)
            & (c1 < cls.GRID_WIDTH)
        )
        r0c = np.clip(r0, 0, cls.GRID_HEIGHT - 1)
        r1c = np.clip(r1, 0, cls.GRID_HEIGHT - 1)
        c0c = np.clip(c0, 0, cls.GRID_WIDTH - 1)
        c1c = np.clip(c1, 0, cls.GRID_WIDTH - 1)
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
