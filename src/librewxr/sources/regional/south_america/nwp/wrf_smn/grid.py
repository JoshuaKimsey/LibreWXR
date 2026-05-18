# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Servicio Meteorológico Nacional Argentina WRF-DET regional precipitation source.

Implements the NWPSource Protocol for SMN's deterministic 4 km WRF run
covering Argentina, Chile, Uruguay, Paraguay, Bolivia, and southern
Brazil — the entire South American Southern Cone, plus surrounding
Atlantic and Pacific waters.

Four daily cycles (00/06/12/18 UTC), 72 h forecast horizon, hourly
forecast steps.  WRF surface output has no native composite
reflectivity field, so we derive dBZ from accumulated ``PP`` (Total
Precipitation, mm) by differencing consecutive forecast steps and
applying the same Marshall-Palmer Z-R conversion ECMWFGrid /
ICONEUGrid / DMIDiniGrid / HRDPS / AROMEAntillesGrid use.

Distribution: anonymous AWS Open Data S3 (``s3://smn-ar-wrf`` in
``us-east-1``), no auth, plain HTTPS.  Each (run, leadtime) is a
single NetCDF4/HDF5 file ~32-36 MB containing about 17 surface fields;
we download the whole file and extract only the ``PP`` variable
(~5 MB after decode).  Range-fetching individual HDF5 chunks would
save bandwidth but adds substantial complexity for marginal benefit
at LibreWXR's window sizes.

Projection: spherical Lambert Conformal Conic, single tangent at
35°S, central meridian 65°W, sphere radius 6,370,000 m (note:
4,229 m smaller than the usual 6,371,229 m WMO sphere — verified
against the file's ``Lambert_Conformal`` grid_mapping attributes
plus the 2D ``lat`` / ``lon`` coord arrays at all four corners).

Data attribution: Servicio Meteorológico Nacional (Argentina),
Creative Commons Attribution 2.5 Argentina Licence; data
distributed via AWS Open Data Registry
(https://registry.opendata.aws/smn-ar-wrf-dataset/).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from librewxr.config import settings
from librewxr.sources.regional.north_america.usa.nwp.hrrr.grid import compute_snow_mask

logger = logging.getLogger(__name__)


# ── WRF-SMN Argentina LCC grid parameters ──────────────────────────────
#
# Source: ``Lambert_Conformal`` grid mapping attributes + ``lat``/``lon``
# 2-D coord arrays of a representative WRFDETAR_01H file decoded on
# 2026-05-08.  Forward projection round-trips all four corners to within
# float precision via the constants below.
#
# Note the unusual sphere radius: SMN's WRF runs against a sphere of
# 6,370,000 m (CF-1.8 ``earth_radius`` attribute), which is 1,229 m
# smaller than the WMO 6,371,229 m sphere most other operational models
# use.  That ~0.02% scale difference matters for cell-precise sampling.

WRF_SMN_SPHERE_RADIUS = 6_370_000.0          # metres
WRF_SMN_LAT_TRUE_SCALE = -35.0               # standard_parallel (single tangent)
WRF_SMN_LAT_PROJ_ORIGIN = -35.0              # latitude_of_projection_origin
WRF_SMN_LON_ORIENT = -65.0                   # longitude_of_central_meridian (LoV)
WRF_SMN_GRID_DX = 4000.0                     # m
WRF_SMN_GRID_DY = 4000.0                     # m
WRF_SMN_GRID_WIDTH = 999                     # Ni — points along the LCC x-axis
WRF_SMN_GRID_HEIGHT = 1249                   # Nj — points along the LCC y-axis

# (row 0, col 0) in the FILE'S native scan order is the SOUTHERN edge
# (lat[0,0] = -54.3868, lon[0,0] = -94.3308).  We flip vertically on
# decode so row 0 in our internal array is the NORTHERN edge, matching
# how every other LibreWXR NWP source stores its data.
WRF_SMN_LA1_SOUTH = -54.3868                 # lat[0, 0] in source file
WRF_SMN_LO1_SOUTH = -94.3308                 # lon[0, 0] in source file


# Precomputed spherical LCC constants (Snyder 1987 §15, single-tangent
# case n = sin φ_0).  Identical math to HRRR/DMI DINI; only the
# constants differ — and for the southern hemisphere n is negative,
# which is fine: the formulas stay the same throughout.

_PHI_0 = math.radians(WRF_SMN_LAT_TRUE_SCALE)
_LON_0_RAD = math.radians(WRF_SMN_LON_ORIENT)
_N = math.sin(_PHI_0)
_F = math.cos(_PHI_0) * math.tan(math.pi / 4 + _PHI_0 / 2) ** _N / _N
_RHO_0 = (
    WRF_SMN_SPHERE_RADIUS * _F
    / math.tan(math.pi / 4 + _PHI_0 / 2) ** _N
)


def lcc_forward(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project geographic (lat, lon) → WRF-SMN LCC (x, y) in metres.

    Spherical Lambert Conformal Conic per Snyder (1987) §15.  The
    standard parallel is in the southern hemisphere (n < 0); the
    formulas work identically — y simply increases southward in the
    natural projected frame.
    """
    phi = np.radians(lat)
    lam = np.radians(lon)

    rho = (
        WRF_SMN_SPHERE_RADIUS * _F
        / np.tan(np.pi / 4 + phi / 2) ** _N
    )
    theta = _N * (lam - _LON_0_RAD)

    x = rho * np.sin(theta)
    y = _RHO_0 - rho * np.cos(theta)
    return x, y


# Project the file's SW corner into LCC space to fix the grid's x/y
# origin.  After we flip on decode, row 0 lands at the NORTHERN edge,
# whose y-coordinate is the southern origin PLUS (HEIGHT-1) cells —
# even though the standard parallel is in the southern hemisphere
# (n < 0), the projected y-axis still *increases going north* in this
# convention because the n < 0 sign already lives in F and ρ_0 in
# the formula y = ρ_0 − ρ cos(θ).  Verified empirically: y at
# (-54.39°S, -94.33°W) ≈ −2.50e6 m; y at (-11.65°S, -82.03°W) ≈
# +2.50e6 m, a 5e6 m span over 1248 cells × 4 km = 4.99e6 m ✓.
_X0_ARR, _Y0_ARR = lcc_forward(
    np.array([WRF_SMN_LA1_SOUTH]),
    np.array([WRF_SMN_LO1_SOUTH]),
)
WRF_SMN_GRID_X_ORIGIN = float(_X0_ARR[0])
WRF_SMN_GRID_Y_ORIGIN_SOUTH = float(_Y0_ARR[0])
WRF_SMN_GRID_Y_ORIGIN_NORTH = (
    WRF_SMN_GRID_Y_ORIGIN_SOUTH + (WRF_SMN_GRID_HEIGHT - 1) * WRF_SMN_GRID_DY
)
del _X0_ARR, _Y0_ARR


def grid_indices(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (lat, lon) to fractional (row, col) on the flipped WRF-SMN grid.

    After decode flips the array, row 0 is the NORTHERN edge and row
    HEIGHT-1 the southern edge.  Column 0 is the western edge.  Out-of-
    domain points still return values; callers should test
    ``domain_mask`` first.
    """
    x, y = lcc_forward(lat, lon)
    col = (x - WRF_SMN_GRID_X_ORIGIN) / WRF_SMN_GRID_DX
    row = (WRF_SMN_GRID_Y_ORIGIN_NORTH - y) / WRF_SMN_GRID_DY
    return row, col


def domain_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """``True`` where (lat, lon) falls inside the WRF-SMN LCC grid."""
    row, col = grid_indices(lat, lon)
    return (
        (row >= 0)
        & (row < WRF_SMN_GRID_HEIGHT - 1)
        & (col >= 0)
        & (col < WRF_SMN_GRID_WIDTH - 1)
    )


# ── Boundary feathering ───────────────────────────────────────────────
#
# 75 km feather, matching HRRR / DMI DINI / HRRR-Alaska — there's
# nothing Argentina-specific about the visual width of the transition
# zone.  At 4 km grid spacing that's ~19 cells of taper.

WRF_SMN_FEATHER_DISTANCE_M = 75_000.0


def feather_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Float32 weights in [0, 1]: 1 deep inside WRF-SMN, 0 outside."""
    row, col = grid_indices(lat, lon)
    dist_cells = np.minimum(
        np.minimum(row, (WRF_SMN_GRID_HEIGHT - 1) - row),
        np.minimum(col, (WRF_SMN_GRID_WIDTH - 1) - col),
    )
    dist_m = dist_cells * WRF_SMN_GRID_DX
    weight = np.clip(dist_m / WRF_SMN_FEATHER_DISTANCE_M, 0.0, 1.0)
    return weight.astype(np.float32, copy=False)


# ── Z-R conversion (matches every accumulation-based source in the chain) ─

ZR_A_RAIN = 200.0
ZR_B_RAIN = 1.6


def precip_rate_to_dbz_encoded(
    precip_mm_per_hour: np.ndarray,
    dbz_offset: float = 0.0,
) -> np.ndarray:
    """Convert mm/h precip rate → uint8 dBZ encoded (pixel = (dBZ+32)*2)."""
    rate = np.where(np.isfinite(precip_mm_per_hour), precip_mm_per_hour, 0.0)
    rate = np.maximum(rate, 0.0)
    eps = 1e-6
    z = ZR_A_RAIN * np.power(rate + eps, ZR_B_RAIN)
    dbz = 10.0 * np.log10(np.maximum(z, eps)) + dbz_offset
    encoded = np.clip((dbz + 32.0) * 2.0 + 0.5, 0, 255)
    encoded[rate <= 0.0] = 0
    return encoded.astype(np.uint8)


# ── Run / step timing ─────────────────────────────────────────────────

CYCLE_INTERVAL_SECONDS = 6 * 3600        # SMN deterministic runs every 6 h
SOURCE_STEP_SECONDS = 3600               # forecast steps are 1 hour apart in the source files
# Backwards-compatible alias — existing call sites read it via ``settings``
# and tests reference the name directly.  Prefer ``SOURCE_STEP_SECONDS``
# in new code.
BRACKET_INTERVAL_SECONDS = SOURCE_STEP_SECONDS
# After optical-flow interpolation runs at the end of each fetch cycle,
# frames are stored at 10-minute spacing within each run (matching the
# radar / nowcast cadence).  Bracket lerp in ``sample`` walks at this
# finer interval so a query 1500 s into the run reads from the
# interpolated frame at 1200 s + small lerp into the frame at 1800 s,
# rather than cross-fading between the hourly bracket pair.
STORED_INTERVAL_SECONDS = 600
MAX_FORECAST_HOURS = 72                  # all runs reach +72 h

# Two cycles of lookback (12 h) is plenty: each run covers +72 h, far
# more than any reasonable active history+horizon window.  Same shape
# as HRDPS / AROME Antilles.
RUN_LOOKBACK_CYCLES = 2

# How many ~34 MB NetCDF files to download in parallel.  Six matches
# typical residential / VPS uplink saturation against AWS us-west-2
# without thrashing the connection pool; raise if you've got Mbps to
# burn and S3 latency is the bottleneck, lower if the cache thrashes
# disk on slow storage.
WRF_SMN_FETCH_CONCURRENCY = 6


def floor_cycle(ts: int) -> int:
    """Floor a Unix timestamp to the nearest 6-hour cycle boundary."""
    return (ts // CYCLE_INTERVAL_SECONDS) * CYCLE_INTERVAL_SECONDS


def latest_published_run(now_ts: int, publish_delay_seconds: int) -> int:
    """Most recent run we'd expect to be available given a publish delay."""
    return floor_cycle(now_ts - publish_delay_seconds)


def bracket_lead_seconds(
    lead_seconds: int,
    interval_seconds: int = SOURCE_STEP_SECONDS,
) -> tuple[int, int, float]:
    """For a desired lead, return ``(L0, L1, alpha)`` such that L0 ≤ L < L1.

    Both leads are exact multiples of ``interval_seconds``, ≥ 0.  Alpha
    is the lerp weight: 0 at L0, 1 at L1.

    Defaults to ``SOURCE_STEP_SECONDS`` (3600 s — the raw hourly source
    cadence).  Pass ``STORED_INTERVAL_SECONDS`` (600 s) to bracket
    against the post-interpolation cadence — that's what the
    ``WRFSMNGrid`` class does internally when ``LIBREWXR_REGIONAL_INTERPOLATION``
    is enabled.
    """
    if lead_seconds < 0:
        return 0, 0, 0.0
    l0 = (lead_seconds // interval_seconds) * interval_seconds
    l1 = l0 + interval_seconds
    alpha = (lead_seconds - l0) / interval_seconds
    return l0, l1, alpha


# ── SMN AWS Open Data file URLs ───────────────────────────────────────
#
# Path scheme:
#   DATA/WRF/DET/{YYYY}/{MM}/{DD}/{HH}/WRFDETAR_01H_{YYYYMMDD}_{HH}_{LLL}.nc
# where {LLL} is the zero-padded 3-digit forecast lead hour (000..072).
# Step 0 (analysis) has no precipitation accumulated since init = 0;
# we treat it as a zero baseline for the diff path.

def file_url(run: datetime, step_hour: int) -> str:
    """Construct the S3 URL for one WRFDETAR (run, leadtime) NetCDF file."""
    bucket = settings.wrf_smn_s3_bucket
    region = settings.wrf_smn_s3_region
    yyyy = run.strftime("%Y")
    mm = run.strftime("%m")
    dd = run.strftime("%d")
    hh = run.strftime("%H")
    lll = f"{step_hour:03d}"
    return (
        f"https://{bucket}.s3.{region}.amazonaws.com/"
        f"DATA/WRF/DET/{yyyy}/{mm}/{dd}/{hh}/"
        f"WRFDETAR_01H_{yyyy}{mm}{dd}_{hh}_{lll}.nc"
    )


# ── NetCDF4/HDF5 decode ───────────────────────────────────────────────
#
# WRF-SMN distributes NetCDF4 files (HDF5 underneath), so we use h5py
# rather than cfgrib.  The ``PP`` dataset is shaped (1, 1249, 999) —
# the leading singleton is the time dimension.


def decode_pp_message(nc_bytes: bytes) -> np.ndarray | None:
    """Decode the ``PP`` field from a WRFDETAR NetCDF4 buffer to float32 mm.

    Returns ``None`` on parse failure.  Output shape is
    ``(WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH)`` with row 0 at the
    NORTHERN edge.  The source file scans south-up (lat[0, ...] is the
    southernmost row), so we flip vertically on decode; this is
    self-correcting against the file's ``lat`` 2D coord array so a
    future SMN format change won't silently flip the grid.
    """
    import h5py

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(nc_bytes)
            tmp_path = tmp.name
        with h5py.File(tmp_path, "r") as f:
            if "PP" not in f:
                logger.warning(
                    "WRF-SMN file has no 'PP' dataset (vars=%s)",
                    list(f.keys())[:10],
                )
                return None
            pp = np.asarray(f["PP"])
            # PP is shape (1, Nj, Ni); collapse the time dim.
            if pp.ndim == 3 and pp.shape[0] == 1:
                pp = pp[0]
            # _FillValue is 1e20 in the source files; clip to NaN-like.
            pp = np.where(np.isfinite(pp) & (pp < 1e10), pp, 0.0).astype(np.float32)

            if pp.shape != (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH):
                logger.warning(
                    "WRF-SMN PP has unexpected shape %s (expected %s); skipping",
                    pp.shape, (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH),
                )
                return None

            # Self-correcting orientation: row 0 should be the NORTHERN
            # edge.  WRF-SMN historically scans south-up, so flip when
            # the lat coord increases with row index.
            needs_flip = True
            if "lat" in f:
                lat_arr = np.asarray(f["lat"])
                if lat_arr.ndim == 2 and lat_arr.shape[0] > 1:
                    needs_flip = lat_arr[0, 0] < lat_arr[-1, 0]
                elif lat_arr.ndim == 1 and lat_arr.size > 1:
                    needs_flip = lat_arr[0] < lat_arr[-1]
            if needs_flip:
                pp = np.flipud(pp)
            return np.ascontiguousarray(pp)
    except Exception:
        logger.exception("Failed to decode WRF-SMN NetCDF4 buffer")
        return None
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass


def decode_t2_message(nc_bytes: bytes) -> np.ndarray | None:
    """Decode the ``T2`` field from a WRFDETAR NetCDF4 buffer to °C float32.

    SMN's WRF-DET output already stores 2-metre temperature in degrees
    Celsius (the per-file ``T2:units`` attribute is ``"C"``), so no
    Kelvin conversion is needed.  Returns ``None`` on parse failure.
    Output shape and orientation match ``decode_pp_message``: shape
    ``(WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH)`` with row 0 = north.
    """
    import h5py

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp.write(nc_bytes)
            tmp_path = tmp.name
        with h5py.File(tmp_path, "r") as f:
            if "T2" not in f:
                logger.warning(
                    "WRF-SMN file has no 'T2' dataset (vars=%s)",
                    list(f.keys())[:10],
                )
                return None
            t2 = np.asarray(f["T2"])
            if t2.ndim == 3 and t2.shape[0] == 1:
                t2 = t2[0]
            t2 = np.where(np.isfinite(t2) & (t2 < 1e10), t2, np.nan).astype(np.float32)

            if t2.shape != (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH):
                logger.warning(
                    "WRF-SMN T2 has unexpected shape %s (expected %s); skipping",
                    t2.shape, (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH),
                )
                return None

            # Self-correcting orientation: row 0 = north.
            needs_flip = True
            if "lat" in f:
                lat_arr = np.asarray(f["lat"])
                if lat_arr.ndim == 2 and lat_arr.shape[0] > 1:
                    needs_flip = lat_arr[0, 0] < lat_arr[-1, 0]
                elif lat_arr.ndim == 1 and lat_arr.size > 1:
                    needs_flip = lat_arr[0] < lat_arr[-1]
            if needs_flip:
                t2 = np.flipud(t2)
            return np.ascontiguousarray(t2)
    except Exception:
        logger.exception("Failed to decode WRF-SMN T2 NetCDF4 buffer")
        return None
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass


# ── WRFSMNGrid: the public NWPSource implementation ───────────────────


class WRFSMNGrid:
    """SMN Argentina WRF as an NWPSource for the South American Cone."""

    name = "wrf_smn"

    def __init__(self, cache_dir: Path | None = None):
        self._frames: dict[tuple[int, int], np.ndarray] = {}
        # Raw accumulated PP values keyed by (run_ts, step_hour); kept
        # only long enough to compute the rate at the next step.
        self._accum: dict[tuple[int, int], np.ndarray] = {}
        # Per-frame snow mask (1 = snow, 0 = rain) keyed by the same
        # (run_ts, lead_seconds) as ``_frames``.  Populated alongside
        # accum decode in ``_fetch_accum`` since T2 lives in the same
        # NetCDF4 file as PP; no separate fetch round-trip.
        self._snow_masks: dict[tuple[int, int], np.ndarray] = {}
        self._client: httpx.AsyncClient | None = None
        self._latest_run_ts: int | None = None
        self._fetch_lock = asyncio.Lock()

        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "wrf_smn"
            self._persistent = True
        else:
            self._memmap_dir = Path(
                tempfile.mkdtemp(prefix="librewxr_wrf_smn_")
            )
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "WRF-SMN memmap directory: %s (persistent=%s)",
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
                # Skip snow files ("rNNN_lNNN_snow") and any other
                # parallel-field files we may add later.
                continue
            run_ts = int(m.group(1))
            lead_s = int(m.group(2))
            try:
                mm = np.memmap(
                    path, dtype=np.uint8, mode="r",
                    shape=(WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH),
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
            logger.info("WRF-SMN: loaded %d cached frame(s) from disk", loaded)

        # Second pass: snow masks.  Orphans (no matching frame) are
        # removed so they don't accumulate.
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
                    shape=(WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH),
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
                "WRF-SMN: loaded %d cached snow mask(s) from disk",
                snow_loaded,
            )

    def __getstate__(self) -> dict:
        """Serialize state for cross-process reload (multi-worker mode).

        The on-disk layout in ``_memmap_dir`` is the canonical state;
        ``__setstate__`` rebuilds the in-memory frame dict by rescanning
        disk via ``_load_cached_frames``.
        """
        return {
            "memmap_dir": str(self._memmap_dir),
            "latest_run_ts": self._latest_run_ts,
            "frame_keys": [[run, lead] for (run, lead) in self._frames.keys()],
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state by rescanning ``memmap_dir`` from disk."""
        self._memmap_dir = Path(state["memmap_dir"])
        self._persistent = True
        self._client = None
        self._fetch_lock = asyncio.Lock()
        self._frames = {}
        self._accum = {}
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
        return datetime.fromtimestamp(
            self._latest_run_ts, tz=timezone.utc,
        ).isoformat()

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
        """Sample the snow / rain classification at each (lat, lon, ts).

        Mirrors ``sample`` for the parallel T2-derived snow mask: same
        run picker, same hourly bracket-lerp, then re-binarise at 0.5.
        Returns False everywhere if no snow mask is loaded for the
        bracket — the chain dispatcher falls through to the next
        snow-capable source for those pixels (IFS, in WRF-SMN's
        region this means falling back to global IFS snow detection).
        """
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

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            # Larger connection pool than the other GRIB sources because
            # WRF-SMN's per-file size (~34 MB NetCDF4) makes the bandwidth-
            # bound fetch the dominant cost; running ``WRF_SMN_FETCH_CONCURRENCY``
            # transfers in parallel keeps the pipeline saturated when a
            # cold start has to download every leadtime in the active
            # window.
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=10.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_keepalive_connections=WRF_SMN_FETCH_CONCURRENCY,
                    max_connections=WRF_SMN_FETCH_CONCURRENCY * 2,
                ),
            )
        return self._client

    async def fetch(
        self,
        now_ts: int | None = None,
        history_seconds: int = 0,
        horizon_seconds: int = 60 * 60,
    ) -> None:
        """Refresh the in-memory window.

        Three-phase pipeline keyed on the SMN file size (~34 MB each):

        1. **Parallel download** — each (run, step) pair that's not
           already cached gets fetched concurrently up to
           ``WRF_SMN_FETCH_CONCURRENCY`` in flight, decoded into the
           cumulative-precip ``self._accum`` dict.  This is where the
           bandwidth time goes for cold starts; serialising it would
           multiply elapsed time by the in-flight count.
        2. **Sequential diff per run** — once the accums are loaded,
           walk steps newest-to-oldest within each run and compute
           ``rate = accum[F] - accum[F-1]``, encode to dBZ, and
           memmap-write the result.  Pure CPU; runs in a few hundred
           milliseconds for the whole window.
        3. **Optical-flow interpolation per run** (if
           ``LIBREWXR_REGIONAL_INTERPOLATION`` is on) — fill in
           10-minute synthetic frames between hourly originals using
           Farneback dense flow.  Without this, cells appear to
           cross-fade between hourly bracket frames at intermediate
           query times; with it, they translate smoothly.
        """
        async with self._fetch_lock:
            if now_ts is None:
                now_ts = int(datetime.now(tz=timezone.utc).timestamp())

            publish_delay = settings.wrf_smn_publish_delay_minutes * 60
            latest_run_ts = latest_published_run(now_ts, publish_delay)
            if (
                self._latest_run_ts is None
                or latest_run_ts > self._latest_run_ts
            ):
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
                logger.debug("WRF-SMN fetch: no runs available for window")
                return

            client = await self._get_client()

            # Build the full (run, step) job list across every cycle in
            # the window.  Each job either already has an accum cached
            # (no-op) or needs a network fetch.
            jobs: list[tuple[datetime, int]] = []
            run_step_ranges: list[tuple[int, datetime, int, int]] = []
            for run_ts in runs_to_consider:
                run_dt = datetime.fromtimestamp(run_ts, tz=timezone.utc)
                min_lead = max(
                    0, window_start - run_ts - BRACKET_INTERVAL_SECONDS,
                )
                max_lead = min(
                    MAX_FORECAST_HOURS * 3600,
                    window_end - run_ts + BRACKET_INTERVAL_SECONDS,
                )
                if max_lead < min_lead:
                    continue
                # Need step F-1 to compute the rate at step F via diff,
                # so always start one step earlier than strictly needed.
                # Step 0 has no actual file (cumulative since init = 0);
                # the diff path treats it as a zero baseline.
                min_step = max(0, (min_lead // BRACKET_INTERVAL_SECONDS) - 1)
                max_step = min(
                    MAX_FORECAST_HOURS,
                    -(-max_lead // BRACKET_INTERVAL_SECONDS),
                )
                run_step_ranges.append((run_ts, run_dt, int(min_step), int(max_step)))
                for step in range(int(min_step), int(max_step) + 1):
                    jobs.append((run_dt, step))

            # ── Phase 1: parallel fetch ─────────────────────────────
            sem = asyncio.Semaphore(WRF_SMN_FETCH_CONCURRENCY)

            async def _bounded_fetch_accum(
                run_dt: datetime, step: int,
            ) -> int:
                async with sem:
                    return await self._fetch_accum(run_dt, step, client)

            fetch_results = await asyncio.gather(
                *[_bounded_fetch_accum(rd, s) for rd, s in jobs],
                return_exceptions=False,
            )
            total_failed = sum(1 for r in fetch_results if r < 0)

            # ── Phase 2: sequential diff per run ────────────────────
            total_fetched = 0
            for run_ts, run_dt, min_step, max_step in run_step_ranges:
                # Skip step ``min_step`` for frame creation — it's only
                # there as the prior-step accum baseline for ``min_step+1``.
                first_frame_step = max(min_step + 1, 1)
                for step in range(first_frame_step, max_step + 1):
                    added = self._compute_frame(run_ts, step)
                    if added > 0:
                        total_fetched += added

            # ── Phase 3: optical-flow interpolation per run ─────────
            total_interpolated = 0
            if settings.regional_interpolation:
                touched_runs = {r for (r, _, _, _) in run_step_ranges}
                for run_ts in touched_runs:
                    total_interpolated += self._interpolate_run_frames(run_ts)

            self._evict_outside_window(window_start, window_end)

            if total_fetched:
                logger.info(
                    "WRF-SMN: %d hourly frame(s) ingested + %d interpolated "
                    "across %d run(s); store now holds %d frame(s)",
                    total_fetched, total_interpolated,
                    len(runs_to_consider), len(self._frames),
                )
            elif total_failed:
                logger.warning(
                    "WRF-SMN: no frames ingested (%d file(s) failed)",
                    total_failed,
                )

    def _interpolate_run_frames(self, run_ts: int) -> int:
        """Fill 10-min synthetic frames between hourly originals for one run.

        Reads the hourly precip + snow frames already in ``self._frames``
        / ``self._snow_masks`` for ``run_ts``, delegates to the shared
        Farneback warper, writes synthetic frames to memmap, and
        registers them in the in-memory dicts at the new ``lead_seconds``
        keys.  Returns the number of synthetic precip frames added.

        Idempotent: if the run already has stored-interval spacing
        (because a prior fetch cycle interpolated it), no work is done.
        """
        from librewxr.data.nwp_interpolation import interpolate_run

        # Pull this run's frames, keyed by lead_seconds.
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
            snow_by_lead,
            target_interval_seconds=STORED_INTERVAL_SECONDS,
            log_label=f"WRF-SMN interpolation (run {run_ts})",
        )

        # Memmap-write any new synthetic frames + snow masks.
        added_precip = 0
        for lead, arr in aug_frames.items():
            if (run_ts, lead) in self._frames:
                continue
            mm = self._to_memmap(f"r{run_ts}_l{lead}", arr)
            self._frames[(run_ts, lead)] = mm
            added_precip += 1
        if aug_snow is not None:
            for lead, snow_arr in aug_snow.items():
                if (run_ts, lead) in self._snow_masks:
                    continue
                # Skip snow files for leads whose precip frame didn't
                # land (couldn't happen given the interpolator output
                # mirrors precip, but defensive).
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

        return added_precip

    async def _fetch_accum(
        self, run: datetime, step_hour: int, client: httpx.AsyncClient,
    ) -> int:
        """Network half of the fetch pipeline: download + decode → ``_accum``.

        Also decodes T2 from the same NetCDF4 file and writes the
        derived snow mask to disk parallel to the accum store — both
        fields come from a single HTTP GET, so there's no extra
        network round-trip vs. the precip-only path.  T2 decode
        failures are non-fatal: the precip frame still lands and
        ``get_snow_mask`` falls through to the next chain source.

        Returns 1 on a fresh successful fetch, 0 if already cached, -1
        on fetch/decode error.  No diff, no frame creation — that's
        ``_compute_frame``'s job.
        """
        run_ts = int(run.timestamp())

        # Step 0 has no file: cumulative precip at model init is zero
        # everywhere.  Cache the zero baseline so step 1's diff can
        # reference it without a network round-trip.
        if step_hour == 0:
            if (run_ts, 0) not in self._accum:
                self._accum[(run_ts, 0)] = np.zeros(
                    (WRF_SMN_GRID_HEIGHT, WRF_SMN_GRID_WIDTH),
                    dtype=np.float32,
                )
            return 0

        # Already have this step's accum (warm restart / partial fetch).
        if (run_ts, step_hour) in self._accum:
            return 0
        # Frame already exists on disk — nothing to do at all.
        lead_seconds = step_hour * BRACKET_INTERVAL_SECONDS
        if (run_ts, lead_seconds) in self._frames:
            return 0

        url = file_url(run, step_hour)
        from librewxr.data.retry import retry_get
        resp = await retry_get(client, url, log_name="WRF-SMN data")
        if resp is None:
            return -1
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if getattr(e.response, "status_code", None) == 404:
                logger.debug("WRF-SMN not yet published for %s", url)
            else:
                logger.warning("WRF-SMN fetch failed for %s: %s", url, e)
            return -1

        accum = decode_pp_message(resp.content)
        if accum is None:
            return -1

        self._accum[(run_ts, step_hour)] = accum

        # Decode T2 from the same buffer and write the snow mask to
        # disk.  T2 is the instantaneous 2-m temperature at the step's
        # lead time, so no diff against prior step is needed.  Already
        # in °C in the source file; threshold matches the other regional
        # NWP sources.
        if (run_ts, lead_seconds) not in self._snow_masks:
            t2 = decode_t2_message(resp.content)
            if t2 is not None:
                threshold = settings.regional_snow_temp_threshold
                snow = compute_snow_mask(t2, threshold)
                mm = self._to_memmap(
                    f"r{run_ts}_l{lead_seconds}_snow", snow,
                )
                self._snow_masks[(run_ts, lead_seconds)] = mm
        return 1

    def _compute_frame(self, run_ts: int, step_hour: int) -> int:
        """CPU half of the fetch pipeline: diff + encode + memmap-write.

        Reads ``self._accum[(run_ts, step_hour)]`` and the prior step's
        accum, computes the windowed precip rate, encodes to dBZ, and
        stores the result as a memmap-backed frame.  Returns 1 on
        success, 0 if the frame already exists or the inputs aren't
        loaded.
        """
        lead_seconds = step_hour * BRACKET_INTERVAL_SECONDS
        if (run_ts, lead_seconds) in self._frames:
            return 0
        accum = self._accum.get((run_ts, step_hour))
        prev = self._accum.get((run_ts, step_hour - 1))
        if accum is None or prev is None:
            return 0  # silently skip — Phase 1 had a fetch failure here

        rate_mm_per_hour = accum - prev
        encoded = precip_rate_to_dbz_encoded(
            rate_mm_per_hour,
            dbz_offset=settings.wrf_smn_dbz_offset,
        )
        mm = self._to_memmap(f"r{run_ts}_l{lead_seconds}", encoded)
        self._frames[(run_ts, lead_seconds)] = mm
        return 1

    # ── Eviction ──────────────────────────────────────────────────────

    def _evict_outside_window(
        self, window_start: int, window_end: int,
    ) -> None:
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
        # Also evict orphan snow masks whose precip frame is gone — can
        # happen if T2 decoded successfully but PP for step F-1 was
        # never fetched (no precip frame at lead F).
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
        stale_accums = []
        for (run_ts, step_h) in self._accum:
            valid_time = run_ts + step_h * BRACKET_INTERVAL_SECONDS
            if valid_time < ws - BRACKET_INTERVAL_SECONDS or valid_time > we:
                stale_accums.append((run_ts, step_h))
        for k in stale_accums:
            self._accum.pop(k, None)
        if stale_frames:
            logger.info(
                "WRF-SMN: evicted %d out-of-window frame(s)",
                len(stale_frames),
            )

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def close(self) -> None:
        self._frames.clear()
        self._accum.clear()
        self._snow_masks.clear()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        if not self._persistent:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.info("WRF-SMN memmap directory cleaned up")
        else:
            logger.info(
                "WRF-SMN cache retained at %s for warm restart",
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
    """Sample a uint8 LCC grid at (lat, lon) points."""
    row_f, col_f = grid_indices(lat, lon)

    if not bilinear:
        row = np.rint(row_f).astype(np.int32)
        col = np.rint(col_f).astype(np.int32)
        in_domain = (
            (row >= 0)
            & (row < WRF_SMN_GRID_HEIGHT)
            & (col >= 0)
            & (col < WRF_SMN_GRID_WIDTH)
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
        & (r1 < WRF_SMN_GRID_HEIGHT)
        & (c0 >= 0)
        & (c1 < WRF_SMN_GRID_WIDTH)
    )
    r0c = np.clip(r0, 0, WRF_SMN_GRID_HEIGHT - 1)
    r1c = np.clip(r1, 0, WRF_SMN_GRID_HEIGHT - 1)
    c0c = np.clip(c0, 0, WRF_SMN_GRID_WIDTH - 1)
    c1c = np.clip(c1, 0, WRF_SMN_GRID_WIDTH - 1)
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
