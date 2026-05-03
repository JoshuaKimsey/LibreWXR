# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""DMI HARMONIE-AROME DINI regional precipitation source.

Implements the NWPSource Protocol for the Danish Meteorological Institute's
HARMONIE-AROME DINI run — a 2 km native UWC-West HARMONIE configuration on
a Lambert Conformal Conic grid covering most of populated Europe (UK,
Ireland, France, Benelux, Germany, Switzerland, Austria, northern Italy,
Czechia, Poland, southern Scandinavia, Iceland and surrounding seas).

Eight daily cycles (00/03/06/09/12/15/18/21 UTC), 60 h forecast horizon,
hourly forecast steps.  HARMONIE-AROME's standard surface output has no
native composite reflectivity field, so we derive dBZ from accumulated
``tp`` (Total Precipitation, kg/m²) by differencing consecutive forecast
steps and applying the same Marshall-Palmer Z-R conversion ECMWFGrid /
ICONEUGrid use.

Distribution: anonymous AWS Open Data S3 (``s3://dmi-opendata/`` in
``eu-north-1``), no auth, plain HTTPS.  Each (run, lead) is a single
multi-message GRIB2 file ~600 MB in size containing all surface fields
for that timestep.  DMI publishes no ``.idx`` sidecars, so we do an
on-demand byte-range header walk to locate the ``tp`` message offset
once per run, then use it for every subsequent leadtime download from
that run (offsets are stable across leadtimes within a run).

Data attribution: Danish Meteorological Institute (DMI), distributed via
AWS Open Data.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shutil
import struct
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np

from librewxr.config import settings

logger = logging.getLogger(__name__)


# ── DMI HARMONIE-AROME DINI LCC grid parameters ────────────────────────
#
# Source: GRIB Section 3 of a representative HARMONIE_DINI_SF file
# decoded on 2026-05-03.  Earth radius 6371229 m matches the standard
# spherical sphere used by all HIRLAM / UWC-West models (and HRRR).
#
# Both standard parallels coincide at 55.5°N, so the projection is the
# degenerate single-tangent-parallel case (n = sin φ_0).  Same family
# as HRRR's LCC; we crib the math here with DMI-specific constants.

DMI_DINI_SPHERE_RADIUS = 6371229.0           # metres (sphere, not WGS84)
DMI_DINI_LAT_0 = 55.5                        # latitude of projection origin
DMI_DINI_LON_0 = -8.0                        # central meridian (LoV = 352°E)
DMI_DINI_STD_PARALLEL = 55.5                 # single tangent standard parallel
DMI_DINI_GRID_DX = 2000.0                    # x spacing (m, ~2 km native)
DMI_DINI_GRID_DY = 2000.0                    # y spacing (m)
DMI_DINI_GRID_WIDTH = 1906                   # Ni — points along parallel
DMI_DINI_GRID_HEIGHT = 1606                  # Nj — points along meridian

# Corner [0,0] of the grid, given by GRIB as (La1, Lo1) in geographic
# coordinates: 39.671°N, -25.422°E (= 334.578°E).  We project that into
# LCC space at import time so grid_indices can subtract directly.
DMI_DINI_LA1 = 39.671
DMI_DINI_LO1 = 334.578 - 360.0               # -25.422°E

# Precomputed LCC constants — compiled once at import time.
_PHI_0 = math.radians(DMI_DINI_STD_PARALLEL)
_N = math.sin(_PHI_0)
_F = math.cos(_PHI_0) * math.tan(math.pi / 4 + _PHI_0 / 2) ** _N / _N
_RHO_0 = DMI_DINI_SPHERE_RADIUS * _F / math.tan(math.pi / 4 + _PHI_0 / 2) ** _N
_LON_0_RAD = math.radians(DMI_DINI_LON_0)


def lcc_forward(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project geographic (lat, lon) → DMI DINI LCC (x, y) in metres.

    Spherical Lambert Conformal Conic per Snyder (1987) §15, single
    tangent standard parallel case (n = sin φ_0).
    """
    phi = np.radians(lat)
    lam = np.radians(lon)

    rho = DMI_DINI_SPHERE_RADIUS * _F / np.tan(np.pi / 4 + phi / 2) ** _N
    theta = _N * (lam - _LON_0_RAD)

    x = rho * np.sin(theta)
    y = _RHO_0 - rho * np.cos(theta)
    return x, y


# Project the corner (La1, Lo1) — the grid's southern corner since the
# native scan is south-to-north — to fix the grid's x/y origin in LCC
# space.  After we flip on decode, row 0 lands at the NORTHERN edge,
# whose y-coordinate is the southern origin plus (HEIGHT - 1) cells.
_X0_ARR, _Y0_ARR = lcc_forward(np.array([DMI_DINI_LA1]), np.array([DMI_DINI_LO1]))
DMI_DINI_GRID_X_ORIGIN = float(_X0_ARR[0])
DMI_DINI_GRID_Y_ORIGIN_SOUTH = float(_Y0_ARR[0])
DMI_DINI_GRID_Y_ORIGIN_NORTH = (
    DMI_DINI_GRID_Y_ORIGIN_SOUTH + (DMI_DINI_GRID_HEIGHT - 1) * DMI_DINI_GRID_DY
)
del _X0_ARR, _Y0_ARR


def grid_indices(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (lat, lon) to fractional (row, col) on the DINI grid.

    After decode flips the array, row 0 is the NORTHERN edge (the GRIB
    file scans south-to-north — same gotcha as HRRR / ICON-EU).  Column
    0 is the western edge.  Out-of-domain points still return values;
    callers should test ``domain_mask`` first.
    """
    x, y = lcc_forward(lat, lon)
    col = (x - DMI_DINI_GRID_X_ORIGIN) / DMI_DINI_GRID_DX
    row = (DMI_DINI_GRID_Y_ORIGIN_NORTH - y) / DMI_DINI_GRID_DY
    return row, col


def domain_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """``True`` where (lat, lon) falls inside the DMI DINI LCC grid."""
    row, col = grid_indices(lat, lon)
    return (
        (row >= 0)
        & (row < DMI_DINI_GRID_HEIGHT - 1)
        & (col >= 0)
        & (col < DMI_DINI_GRID_WIDTH - 1)
    )


# ── Boundary feathering ───────────────────────────────────────────────
#
# Width of the soft transition zone at the DINI LCC domain edge in
# metres.  Inside the inner region (≥ FEATHER_DISTANCE_M from any edge)
# DINI is trusted at full weight; over the feather zone the weight
# tapers linearly to 0 at the edge so chain blending hands control to
# the next source (ICON-EU, then IFS) smoothly.

DMI_DINI_FEATHER_DISTANCE_M = 75_000.0  # 75 km ≈ 37 grid cells at 2 km


def feather_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Float32 weights in [0, 1]: 1 deep inside DINI, 0 outside."""
    row, col = grid_indices(lat, lon)
    dist_cells = np.minimum(
        np.minimum(row, (DMI_DINI_GRID_HEIGHT - 1) - row),
        np.minimum(col, (DMI_DINI_GRID_WIDTH - 1) - col),
    )
    dist_m = dist_cells * DMI_DINI_GRID_DX
    weight = np.clip(dist_m / DMI_DINI_FEATHER_DISTANCE_M, 0.0, 1.0)
    return weight.astype(np.float32, copy=False)


# ── Z-R conversion (matches ECMWFGrid / ICONEUGrid) ───────────────────

ZR_A_RAIN = 200.0
ZR_B_RAIN = 1.6


def precip_rate_to_dbz_encoded(
    precip_mm_per_hour: np.ndarray,
    dbz_offset: float = 0.0,
) -> np.ndarray:
    """Convert mm/h precip rate → uint8 dBZ encoded (pixel = (dBZ+32)*2).

    Same Marshall-Palmer rain Z-R as ICON-EU: Z = 200 * R^1.6.  The
    optional ``dbz_offset`` shifts the result uniformly to compensate
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


# ── Run / step timing ─────────────────────────────────────────────────

CYCLE_INTERVAL_SECONDS = 3 * 3600        # DMI DINI deterministic runs every 3 h
BRACKET_INTERVAL_SECONDS = 3600          # forecast steps are 1 hour apart
MAX_FORECAST_HOURS = 60                  # all runs reach +60 h

# Two cycles of lookback (6 h) is plenty: each run covers +60 h, far
# more than any reasonable active window.  We don't have ICON-EU's
# intermediate-run truncation to worry about — every DMI cycle reaches
# full horizon.
RUN_LOOKBACK_CYCLES = 2


def floor_cycle(ts: int) -> int:
    """Floor a Unix timestamp to the nearest 3-hour cycle boundary."""
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


# ── DMI Open Data S3 file URLs ────────────────────────────────────────
#
# Filename scheme:
#   forecastdata/HARMONIE_DINI_SF/HARMONIE_DINI_SF_{run}_{valid}.grib
# where {run} and {valid} are ISO timestamps shaped like
# ``2026-05-03T000000Z`` (no separators inside HHMMSS).  One file per
# (run, leadtime).

_DMI_TS_FMT = "%Y-%m-%dT%H%M%SZ"


def _format_dmi_ts(dt: datetime) -> str:
    """Format a datetime as DMI's ISO-without-HHMMSS-separators stamp."""
    return dt.strftime(_DMI_TS_FMT)


def file_url(run: datetime, step_hour: int) -> str:
    """Construct the S3 URL for a HARMONIE_DINI_SF (run, leadtime) file."""
    bucket = settings.dmi_dini_s3_bucket
    region = settings.dmi_dini_s3_region
    valid = run + timedelta(hours=step_hour)
    run_str = _format_dmi_ts(run)
    valid_str = _format_dmi_ts(valid)
    return (
        f"https://{bucket}.s3.{region}.amazonaws.com/"
        f"forecastdata/HARMONIE_DINI_SF/"
        f"HARMONIE_DINI_SF_{run_str}_{valid_str}.grib"
    )


# ── GRIB2 byte-range header walker ────────────────────────────────────
#
# DMI ships no .idx sidecars and each file is ~600 MB containing ~93
# surface-field messages.  To avoid downloading the whole file just to
# extract precipitation, we walk GRIB Section 0 + Section 4 of each
# message via small Range requests (~512 bytes each).  Section 0 gives
# the message length (so we can hop to the next message); Section 4
# (Product Definition) gives the parameter category & number.
#
# Total Precipitation (accumulated, kg/m²) is discipline=0, category=1
# (Moisture), parameter=52 in HARMONIE's GRIB tables — same paramId
# 228228 cfgrib reports as ``tp``.
#
# Empirically the offset of the tp message is identical across every
# leadtime of a given run (DMI's GRIB packing is deterministic per run),
# so we cache it once per run and reuse for all 60 leadtime downloads.

_TP_DISCIPLINE = 0
_TP_CATEGORY = 1
_TP_PARAMETER = 52


async def find_tp_message_offset(
    url: str,
    client: httpx.AsyncClient,
    *,
    max_messages: int = 200,
) -> tuple[int, int] | None:
    """Walk GRIB headers via byte-range to locate the TP message.

    Returns ``(byte_offset, byte_length)`` of the discipline=0 cat=1
    num=52 message, or ``None`` if not found.  Issues one ~512-byte
    Range request per GRIB message scanned.  An ``httpx`` client with
    keepalive cuts the per-request overhead substantially compared to
    fresh connections.
    """
    offset = 0
    for _ in range(max_messages):
        try:
            resp = await client.get(url, headers={"Range": f"bytes={offset}-{offset + 511}"})
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning("DMI DINI header walk failed at offset %d: %s", offset, e)
            return None
        chunk = resp.content
        if len(chunk) < 16 or chunk[:4] != b"GRIB":
            return None
        if chunk[7] != 2:
            logger.warning("DMI DINI: unexpected GRIB edition %d at offset %d", chunk[7], offset)
            return None
        msg_len = struct.unpack(">Q", chunk[8:16])[0]
        discipline = chunk[6]
        # Walk sections within the chunk to find the Product Definition Section.
        cur = 16
        while cur < len(chunk) - 5:
            if cur + 4 > len(chunk):
                break
            sec_len = struct.unpack(">I", chunk[cur:cur + 4])[0]
            if sec_len == 0 or sec_len > msg_len:
                break
            sec_num = chunk[cur + 4]
            if sec_num == 4:
                if cur + 11 <= len(chunk):
                    cat = chunk[cur + 9]
                    num = chunk[cur + 10]
                    if discipline == _TP_DISCIPLINE and cat == _TP_CATEGORY and num == _TP_PARAMETER:
                        return offset, msg_len
                break
            if sec_num >= 5:
                break  # past PDS without finding it — uninteresting message
            cur += sec_len
        offset += msg_len
    logger.warning(
        "DMI DINI: scanned %d messages without finding tp (discipline=%d cat=%d num=%d)",
        max_messages, _TP_DISCIPLINE, _TP_CATEGORY, _TP_PARAMETER,
    )
    return None


# ── GRIB2 message decode ──────────────────────────────────────────────


def _suppress_eccodes_stderr():
    from librewxr.data.sources import _suppress_eccodes_stderr as _s
    return _s()


def decode_tp_message(grib_bytes: bytes) -> np.ndarray | None:
    """Decode a single GRIB2 ``tp`` message into a 2D float32 array.

    Returns ``None`` on parse failure.  Output shape is
    ``(DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH)`` with row 0 at the
    NORTHERN edge (the array is flipped vertically because cfgrib
    returns the file with row 0 = south, matching the GRIB scan mode).
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
        logger.exception("Failed to decode DMI DINI tp GRIB2 message")
        return None
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    if "tp" in ds.data_vars:
        arr = ds["tp"].values
    else:
        for name, da in ds.data_vars.items():
            if da.ndim == 2 and da.shape == (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH):
                logger.warning(
                    "DMI DINI tp variable not named 'tp' (got %r); using fallback",
                    name,
                )
                arr = da.values
                break
        else:
            logger.warning("DMI DINI GRIB had no recognised tp field")
            return None

    if arr.shape != (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH):
        logger.warning(
            "DMI DINI tp has unexpected shape %s (expected %s); skipping",
            arr.shape, (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH),
        )
        return None

    # cfgrib returns the file in its native scan order (row 0 = south).
    # Flip when needed so row 0 = north matches grid_indices().
    if "latitude" in ds.coords:
        lat_first = float(np.asarray(ds["latitude"].values).flat[0])
        lat_last = float(np.asarray(ds["latitude"].values).flat[-1])
        needs_flip = lat_first < lat_last
    else:
        needs_flip = True
    if needs_flip:
        arr = np.flipud(arr)

    return np.ascontiguousarray(arr, dtype=np.float32)


# ── DMIDiniGrid: the public NWPSource implementation ─────────────────


class DMIDiniGrid:
    """DMI HARMONIE-AROME DINI as an NWPSource for the European chain slot."""

    name = "dmi_dini"

    def __init__(self, cache_dir: Path | None = None):
        # (run_ts, lead_seconds) → uint8 dBZ-encoded array on the LCC grid.
        self._frames: dict[tuple[int, int], np.ndarray] = {}
        # Raw accumulated tp values keyed by (run_ts, step_hour); kept
        # only long enough to compute the rate at the next step.
        self._accum: dict[tuple[int, int], np.ndarray] = {}
        # Per-run cache of (tp_byte_offset, tp_byte_length).  Stable
        # across all leadtimes within a run, so the header walk runs
        # exactly once per new run we encounter.
        self._tp_offsets: dict[int, tuple[int, int]] = {}
        self._client: httpx.AsyncClient | None = None
        self._latest_run_ts: int | None = None
        self._fetch_lock = asyncio.Lock()

        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "dmi_dini"
            self._persistent = True
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_dmi_dini_"))
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "DMI DINI memmap directory: %s (persistent=%s)",
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
                    shape=(DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH),
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
            logger.info("DMI DINI: loaded %d cached frame(s) from disk", loaded)

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
                # Keepalive across the dozens of header-walk requests
                # cuts per-run scan time roughly 10× vs fresh connections.
                limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
            )
        return self._client

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

            publish_delay = settings.dmi_dini_publish_delay_minutes * 60
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
                logger.debug("DMI DINI fetch: no runs available for window")
                return

            client = await self._get_client()

            total_fetched = 0
            total_failed = 0
            for run_ts in runs_to_consider:
                run_dt = datetime.fromtimestamp(run_ts, tz=timezone.utc)
                min_lead = max(0, window_start - run_ts - BRACKET_INTERVAL_SECONDS)
                max_lead = min(
                    MAX_FORECAST_HOURS * 3600,
                    window_end - run_ts + BRACKET_INTERVAL_SECONDS,
                )
                if max_lead < min_lead:
                    continue
                # Need step F-1 to compute the rate at step F via diff,
                # so always start one step earlier than strictly needed.
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
                    "DMI DINI: %d frame(s) ingested across %d run(s); "
                    "store now holds %d frame(s)",
                    total_fetched, len(runs_to_consider), len(self._frames),
                )
            elif total_failed:
                logger.warning(
                    "DMI DINI: no frames ingested (%d file(s) failed)",
                    total_failed,
                )

    async def _fetch_one_step(
        self, run: datetime, step_hour: int, client: httpx.AsyncClient,
    ) -> int:
        """Fetch one step's tp, difference against the previous step,
        encode, and store.  Returns 1 on success, 0 if already loaded,
        -1 on fetch error.
        """
        run_ts = int(run.timestamp())
        lead_seconds = step_hour * BRACKET_INTERVAL_SECONDS

        # Step 0: nothing has rained yet at model init — accumulated
        # precip is zero everywhere.  Cache the zero baseline so step 1
        # can diff against it cleanly.
        if step_hour == 0:
            if (run_ts, 0) not in self._accum:
                self._accum[(run_ts, 0)] = np.zeros(
                    (DMI_DINI_GRID_HEIGHT, DMI_DINI_GRID_WIDTH),
                    dtype=np.float32,
                )
            return 0

        if (run_ts, lead_seconds) in self._frames:
            return 0

        url = file_url(run, step_hour)

        # Resolve (or compute) the byte offset of the tp message in this
        # run's files.  The header walk only happens for the first
        # leadtime we touch from a given run; subsequent leadtimes reuse
        # the cached offset.
        tp_loc = self._tp_offsets.get(run_ts)
        if tp_loc is None:
            tp_loc = await find_tp_message_offset(url, client)
            if tp_loc is None:
                logger.warning(
                    "DMI DINI: unable to locate tp message for run %s step %d",
                    run.isoformat(), step_hour,
                )
                return -1
            self._tp_offsets[run_ts] = tp_loc
            logger.info(
                "DMI DINI: tp message located for run %s at byte offset %d (size %.2f MB)",
                run.isoformat(), tp_loc[0], tp_loc[1] / 1e6,
            )

        offset, size = tp_loc
        try:
            resp = await client.get(
                url,
                headers={"Range": f"bytes={offset}-{offset + size - 1}"},
            )
            resp.raise_for_status()
            grib_bytes = resp.content
        except httpx.HTTPError as e:
            logger.warning("DMI DINI fetch failed for %s: %s", url, e)
            return -1

        accum = decode_tp_message(grib_bytes)
        if accum is None:
            return -1

        # Ensure we have step F-1 cached for the diff.
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
            dbz_offset=settings.dmi_dini_dbz_offset,
        )
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
        stale_accums = []
        for (run_ts, step_h) in self._accum:
            valid_time = run_ts + step_h * BRACKET_INTERVAL_SECONDS
            if valid_time < ws - BRACKET_INTERVAL_SECONDS or valid_time > we:
                stale_accums.append((run_ts, step_h))
        for k in stale_accums:
            self._accum.pop(k, None)
        # Drop tp-offset cache entries for runs whose every frame has
        # been evicted (i.e. that run is no longer in the active window).
        live_runs = {r for (r, _) in self._frames}
        stale_runs = [r for r in self._tp_offsets if r not in live_runs]
        for r in stale_runs:
            self._tp_offsets.pop(r, None)
        if stale_frames:
            logger.info(
                "DMI DINI: evicted %d out-of-window frame(s)", len(stale_frames),
            )

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def close(self) -> None:
        self._frames.clear()
        self._accum.clear()
        self._tp_offsets.clear()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        if not self._persistent:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.info("DMI DINI memmap directory cleaned up")
        else:
            logger.info(
                "DMI DINI cache retained at %s for warm restart", self._memmap_dir,
            )


# ── Grid sampling ────────────────────────────────────────────────────


def _sample_grid(
    grid: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    bilinear: bool = False,
) -> np.ndarray:
    """Sample a uint8 LCC grid at (lat, lon) points."""
    row_f, col_f = grid_indices(lat, lon)

    if not bilinear:
        row = np.rint(row_f).astype(np.int32)
        col = np.rint(col_f).astype(np.int32)
        in_domain = (
            (row >= 0)
            & (row < DMI_DINI_GRID_HEIGHT)
            & (col >= 0)
            & (col < DMI_DINI_GRID_WIDTH)
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
        & (r1 < DMI_DINI_GRID_HEIGHT)
        & (c0 >= 0)
        & (c1 < DMI_DINI_GRID_WIDTH)
    )
    r0c = np.clip(r0, 0, DMI_DINI_GRID_HEIGHT - 1)
    r1c = np.clip(r1, 0, DMI_DINI_GRID_HEIGHT - 1)
    c0c = np.clip(c0, 0, DMI_DINI_GRID_WIDTH - 1)
    c1c = np.clip(c1, 0, DMI_DINI_GRID_WIDTH - 1)
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
