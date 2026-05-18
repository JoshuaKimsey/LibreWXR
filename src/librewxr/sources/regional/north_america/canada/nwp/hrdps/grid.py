# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""ECCC HRDPS continental high-resolution precipitation source.

Implements the NWPSource Protocol for Environment and Climate Change
Canada's High-Resolution Deterministic Prediction System — a 2.5 km
native rotated-lat/lon HARMONIE-style configuration covering Canada,
the Canadian Arctic, and the northern fringe of CONUS (roughly 27°N–84°N
and 172°W–10°W in geographic coordinates).

Four daily cycles (00/06/12/18 UTC), 48 h forecast horizon, hourly
forecast steps.  HRDPS does not publish a composite reflectivity field;
we derive dBZ from the ``APCP-Accum1h_Sfc`` variable (1-hour windowed
precipitation accumulation in kg/m² ≡ mm) by applying the same
Marshall-Palmer Z-R conversion ECMWFGrid / ICONEUGrid / DMIDiniGrid use.
Unlike DMI DINI we do NOT difference consecutive forecast steps —
``APCP-Accum1h_Sfc`` is already a 1-hour window, not a run-anchored
cumulant — so the value at lead H is directly the precip rate over the
hour ending at H.

Distribution: anonymous Environment Canada / MSC dd.weather.gc.ca
HTTPS.  Each (run, lead) is a single GRIB2 message ~150 KB-1.5 MB
containing one variable / one level / one forecast hour, so we fetch
the whole file every time — no .idx sidecar, no byte-range header walk.

Projection: ECCC migrated HRDPS from polar-stereographic to rotated
lat/lon (RLatLon) in November 2023; pre-2023 example code is wrong.
The grid's "north pole" sits at geographic (36.08852°N, 65.30514°E),
with 0° rotation about the new pole axis.  We hand-roll the forward
transform per the COSMO/WMO ``gridType: rotated_ll`` convention —
geographic → rotated, two trig identities, no pyproj dependency.

Data attribution: Environment and Climate Change Canada (ECCC), Open
Government Licence – Canada.
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

logger = logging.getLogger(__name__)


# ── HRDPS continental rotated lat/lon grid parameters ─────────────────
#
# Source: MSC Open Data HRDPS Datamart README + GRIB Section 3 of a
# representative APCP-Accum1h_Sfc file decoded on first fetch.  The
# Earth radius is informational — the rotated-pole math operates on
# the unit sphere and the grid spacing is in rotated degrees, so
# physical distances (used only for the feather-distance budget)
# come from a domain-mean cos(lat) approximation.

HRDPS_SPHERE_RADIUS = 6371229.0          # metres (informational)

# Rotated north pole (geographic coordinates).  ECCC documents these
# values in the HRDPS Datamart readme; they are stable across runs.
HRDPS_GRID_NORTH_POLE_LAT = 36.08852
HRDPS_GRID_NORTH_POLE_LON = 65.30514
HRDPS_GRID_ROTATION_ANGLE = 0.0

# Grid spacing in rotated degrees (0.0225° ≈ 2.5 km at mid-latitudes).
HRDPS_GRID_DX = 0.0225
HRDPS_GRID_DY = 0.0225
HRDPS_GRID_WIDTH = 2540                  # Ni — points along rotated parallel
HRDPS_GRID_HEIGHT = 1290                 # Nj — points along rotated meridian

# Geographic coordinates of the SW corner (first grid point under the
# native scan order: jScansPositively=1, iScansNegatively=0).  Decoded
# from a 2026-05-07T12Z APCP-Accum1h_Sfc file's GRIB Section 3.  At the
# four corners of the rotated rectangle, geographic coordinates are:
#   [0, 0]   ≈ ( 39.626°N, -133.630°E)   — rotated SW (this point)
#   [0, -1]  ≈ ( 27.285°N,  -66.966°E)   — rotated SE
#   [-1, 0]  ≈ ( 66.569°N, -152.731°E)   — rotated NW
#   [-1, -1] ≈ ( 47.876°N,  -40.709°E)   — rotated NE
HRDPS_LA1 = 39.6260
HRDPS_LO1 = -133.6295

# Precomputed rotated-pole projection constants — compiled once at
# import time.  Geographic → rotated forward transform per the COSMO /
# WMO gridType=rotated_ll convention with the rotated north pole at
# (φ_p, λ_p) = (HRDPS_GRID_NORTH_POLE_LAT, HRDPS_GRID_NORTH_POLE_LON).
_PHI_P = math.radians(HRDPS_GRID_NORTH_POLE_LAT)
_LAM_P = math.radians(HRDPS_GRID_NORTH_POLE_LON)
_SIN_PHI_P = math.sin(_PHI_P)
_COS_PHI_P = math.cos(_PHI_P)


def rotated_forward(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project geographic (lat, lon) → HRDPS rotated (rlat, rlon) in degrees.

    Implements the COSMO / WMO ``gridType: rotated_ll`` north-pole
    convention with the rotated north pole at
    (HRDPS_GRID_NORTH_POLE_LAT, HRDPS_GRID_NORTH_POLE_LON) and zero
    rotation about the new pole axis.  Inputs and outputs both in
    degrees.  Sanity checks: with a (90°, 0°) pole the formulas
    collapse to identity (rlat = lat, rlon = lon), and the geographic
    pole point itself maps to the rotated north pole (rlat = 90°).

    Closed form (two trig identities), where φ_p, λ_p are the rotated
    north pole's geographic coords and Δλ = λ − λ_p:

        sin(φ_r) = sin(φ_p) sin(φ) + cos(φ_p) cos(φ) cos(Δλ)
        cos(φ_r) sin(λ_r) =                  cos(φ) sin(Δλ)
        cos(φ_r) cos(λ_r) = cos(φ_p) sin(φ) − sin(φ_p) cos(φ) cos(Δλ)
    """
    phi = np.radians(lat)
    lam = np.radians(lon)
    dlam = lam - _LAM_P  # cos / sin handle 2π periodicity natively

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_dlam = np.sin(dlam)
    cos_dlam = np.cos(dlam)

    # Defensive clip against floating-point overshoot of arcsin's domain
    # near the rotated pole (where sin_phi_r → ±1).
    sin_phi_r = np.clip(
        _SIN_PHI_P * sin_phi + _COS_PHI_P * cos_phi * cos_dlam,
        -1.0, 1.0,
    )
    phi_r = np.arcsin(sin_phi_r)

    # atan2 yields the correct branch in (−π, π] without explicit
    # quadrant handling; numpy returns 0.0 for the (0, 0) singularity
    # at the pole, which is the conventional choice (rotated longitude
    # is genuinely undefined at the rotated pole).
    y = cos_phi * sin_dlam
    x = _COS_PHI_P * sin_phi - _SIN_PHI_P * cos_phi * cos_dlam
    lam_r = np.arctan2(y, x)

    return np.degrees(phi_r), np.degrees(lam_r)


# Project the SW geographic corner (La1, Lo1) into rotated coordinates
# to fix the grid's rlat/rlon origin.  After we flip the array on
# decode (cfgrib returns row 0 = south because GRIB scan mode 0b0100_0000
# is i+, j+ from the SW corner), row 0 lands at the NORTH (high rlat)
# edge.
_RLAT0_ARR, _RLON0_ARR = rotated_forward(np.array([HRDPS_LA1]), np.array([HRDPS_LO1]))
HRDPS_GRID_RLON_ORIGIN = float(_RLON0_ARR[0])
HRDPS_GRID_RLAT_ORIGIN_SOUTH = float(_RLAT0_ARR[0])
HRDPS_GRID_RLAT_ORIGIN_NORTH = (
    HRDPS_GRID_RLAT_ORIGIN_SOUTH + (HRDPS_GRID_HEIGHT - 1) * HRDPS_GRID_DY
)
del _RLAT0_ARR, _RLON0_ARR


def grid_indices(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (lat, lon) to fractional (row, col) on the HRDPS grid.

    After decode flips the array vertically, row 0 is the NORTHERN edge
    (high rlat) and row HEIGHT-1 is the SOUTHERN edge.  Column 0 is the
    geographic-westernmost edge (which projects to the *highest* rlon
    in our north-pole rotated frame — increasing i scans westward in
    rotated-lon space) and column WIDTH-1 is the easternmost edge.
    Out-of-domain points still return values; callers should test
    ``domain_mask`` first.

    The rotated-longitude branch is folded into the same window as
    ``HRDPS_GRID_RLON_ORIGIN`` to defend against the one geographic edge
    case where atan2's ±π branch cut crosses the grid (the rotated
    dateline, which sits roughly antipodal to the pole and well outside
    the domain — but cheap to handle).
    """
    rlat, rlon = rotated_forward(lat, lon)
    rlon = (
        ((rlon - HRDPS_GRID_RLON_ORIGIN + 180.0) % 360.0) - 180.0
        + HRDPS_GRID_RLON_ORIGIN
    )
    # The grid origin sits at the maximum rlon and increasing i (col)
    # corresponds to decreasing rlon — verified against the four GRIB
    # corners, which span rlon from +14.82° at col 0 to −42.31° at col
    # WIDTH-1.  This is the consequence of using the north-pole
    # convention against an ECCC GRIB that's encoded with a south-pole
    # rotation: the rotated-lon axis flips sign between the two
    # conventions, but staying internally consistent (one convention
    # for both forward and indexing) is what matters.
    col = (HRDPS_GRID_RLON_ORIGIN - rlon) / HRDPS_GRID_DX
    row = (HRDPS_GRID_RLAT_ORIGIN_NORTH - rlat) / HRDPS_GRID_DY
    return row, col


def domain_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """``True`` where (lat, lon) falls inside the HRDPS rotated grid."""
    row, col = grid_indices(lat, lon)
    return (
        (row >= 0)
        & (row < HRDPS_GRID_HEIGHT - 1)
        & (col >= 0)
        & (col < HRDPS_GRID_WIDTH - 1)
    )


# ── Boundary feathering ───────────────────────────────────────────────
#
# Width of the soft transition zone at the HRDPS rotated grid edge,
# measured in grid cells (each cell is 0.0225° in rotated lat/lon ≈
# 2.5 km).  Inside the inner region (≥ FEATHER_DISTANCE_CELLS from any
# edge) HRDPS is trusted at full weight; over the feather zone the
# weight tapers linearly to 0 at the edge so chain blending hands
# control to the next source (IFS) smoothly instead of leaving a
# visible seam.  ~32 cells ≈ 80 km mirrors HRRR / DMI DINI's 75 km
# feather budget.

HRDPS_FEATHER_DISTANCE_CELLS = 32


def feather_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Float32 weights in [0, 1]: 1 deep inside HRDPS, 0 outside."""
    row, col = grid_indices(lat, lon)
    dist_cells = np.minimum(
        np.minimum(row, (HRDPS_GRID_HEIGHT - 1) - row),
        np.minimum(col, (HRDPS_GRID_WIDTH - 1) - col),
    )
    weight = np.clip(
        dist_cells / float(HRDPS_FEATHER_DISTANCE_CELLS), 0.0, 1.0,
    )
    return weight.astype(np.float32, copy=False)


# ── Z-R conversion (matches ECMWFGrid / ICONEUGrid / DMIDiniGrid) ─────

ZR_A_RAIN = 200.0
ZR_B_RAIN = 1.6


def precip_rate_to_dbz_encoded(
    precip_mm_per_hour: np.ndarray,
    dbz_offset: float = 0.0,
) -> np.ndarray:
    """Convert mm/h precip rate → uint8 dBZ encoded (pixel = (dBZ+32)*2).

    Same Marshall-Palmer rain Z-R as DMI DINI: Z = 200 * R^1.6.  The
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

CYCLE_INTERVAL_SECONDS = 6 * 3600        # HRDPS deterministic runs every 6 h
BRACKET_INTERVAL_SECONDS = 3600          # forecast steps are 1 hour apart
MAX_FORECAST_HOURS = 48                  # all runs reach +48 h

# Two cycles of lookback (12 h) is plenty: each run covers +48 h, far
# more than any reasonable active history+horizon window.  Like DMI
# DINI, we don't have ICON-EU's intermediate-run truncation issue —
# every HRDPS cycle reaches full horizon.
RUN_LOOKBACK_CYCLES = 2


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


# ── ECCC dd.weather.gc.ca file URLs ───────────────────────────────────
#
# Use the date-prefixed archive path rather than the ``/today/`` live
# tree because backfill spans up to 12 h of history (RUN_LOOKBACK_CYCLES
# × CYCLE_INTERVAL_SECONDS); ``/today/`` rolls at midnight UTC so the
# previous-day's runs disappear from it within seconds of UTC midnight,
# leaving us mid-fetch with 404s.  The date-prefixed path keeps every
# run available for the standard ECCC retention window (~7 days).
#
# Filename scheme:
#   {base}/{YYYYMMDD}/WXO-DD/model_hrdps/continental/2.5km/{HH}/{hhh}/
#   {YYYYMMDD}T{HH}Z_MSC_HRDPS_APCP-Accum1h_Sfc_RLatLon0.0225_PT{hhh}H.grib2
# where HH is the 2-digit run init hour and hhh is the 3-digit (yes,
# zero-padded to three digits — different from HRRR's 2-digit lead)
# forecast hour.  Step 0 has no file (ECCC reserves it / leaves empty);
# valid leads are 001..048.

HRDPS_VAR = "APCP-Accum1h_Sfc"


def file_url(run: datetime, step_hour: int) -> str:
    """Construct the dd.weather.gc.ca URL for one HRDPS APCP-Accum1h file.

    Step 0 is reserved/empty in ECCC's directory; callers must pass
    ``step_hour >= 1``.  ``HRDPSGrid.fetch`` enforces this.
    """
    base = settings.hrdps_base_url.rstrip("/")
    hh = run.strftime("%H")
    hhh = f"{step_hour:03d}"
    date = run.strftime("%Y%m%d")
    return (
        f"{base}/{date}/WXO-DD/model_hrdps/continental/2.5km/{hh}/{hhh}/"
        f"{date}T{hh}Z_MSC_HRDPS_{HRDPS_VAR}_RLatLon0.0225_PT{hhh}H.grib2"
    )


# ── GRIB2 decode ──────────────────────────────────────────────────────


def _suppress_eccodes_stderr():
    from librewxr.data.sources import _suppress_eccodes_stderr as _s
    return _s()


def decode_apcp_message(grib_bytes: bytes) -> np.ndarray | None:
    """Decode a single GRIB2 ``APCP-Accum1h_Sfc`` message into a 2D float32.

    Returns ``None`` on parse failure.  Output shape is
    ``(HRDPS_GRID_HEIGHT, HRDPS_GRID_WIDTH)`` with row 0 at the NORTHERN
    (high-rlat) edge.  cfgrib returns the file with row 0 at the
    SOUTHERN edge (scan mode 0b0100_0000 = i+, j+ from SW), so we flip
    vertically; verify against the ``latitude`` coord when available so
    the code self-corrects if a future cfgrib release ever normalises
    the orientation upstream.

    Logs the decoded latitude/longitude corners on first call so the
    documented (La1, Lo1) values can be hard-coded once verified live.
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
        logger.exception("Failed to decode HRDPS APCP GRIB2 message")
        return None
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    # ECCC ships HRDPS APCP-Accum1h_Sfc as ECMWF local table 2.196.8
    # which cfgrib's standard paramId table doesn't recognise — the
    # decoded variable comes back as ``unknown`` rather than ``tp``.
    # Pick the first (and only) 2D variable matching the expected
    # shape; this is robust to future paramId table updates.
    arr: np.ndarray | None = None
    for name, da in ds.data_vars.items():
        if da.ndim == 2 and da.shape == (HRDPS_GRID_HEIGHT, HRDPS_GRID_WIDTH):
            arr = da.values
            break
    if arr is None:
        logger.warning("HRDPS GRIB2 had no 2D variable matching expected shape")
        return None

    if arr.shape != (HRDPS_GRID_HEIGHT, HRDPS_GRID_WIDTH):
        logger.warning(
            "HRDPS tp has unexpected shape %s (expected %s); skipping",
            arr.shape, (HRDPS_GRID_HEIGHT, HRDPS_GRID_WIDTH),
        )
        return None

    # One-time corner log so live verification can confirm the
    # documented (La1, Lo1) without hand-decoding a GRIB.
    if not _DECODE_LOGGED["once"] and "latitude" in ds.coords:
        try:
            lat_arr = np.asarray(ds["latitude"].values)
            lon_arr = np.asarray(ds["longitude"].values)
            logger.info(
                "HRDPS GRIB corners (cfgrib pre-flip): "
                "[0,0]=(%.4f,%.4f) [-1,0]=(%.4f,%.4f) "
                "[0,-1]=(%.4f,%.4f) [-1,-1]=(%.4f,%.4f)",
                float(lat_arr.flat[0]), float(lon_arr.flat[0]),
                float(lat_arr[-1, 0]) if lat_arr.ndim == 2 else float(lat_arr.flat[-1]),
                float(lon_arr[-1, 0]) if lon_arr.ndim == 2 else float(lon_arr.flat[-1]),
                float(lat_arr[0, -1]) if lat_arr.ndim == 2 else float(lat_arr.flat[0]),
                float(lon_arr[0, -1]) if lon_arr.ndim == 2 else float(lon_arr.flat[0]),
                float(lat_arr.flat[-1]), float(lon_arr.flat[-1]),
            )
            _DECODE_LOGGED["once"] = True
        except Exception:
            # Logging is best-effort; never block decode on a coord-shape surprise.
            _DECODE_LOGGED["once"] = True

    if "latitude" in ds.coords:
        lat_first = float(np.asarray(ds["latitude"].values).flat[0])
        lat_last = float(np.asarray(ds["latitude"].values).flat[-1])
        needs_flip = lat_first < lat_last
    else:
        # No coord to verify against — trust the documented scan mode.
        needs_flip = True
    if needs_flip:
        arr = np.flipud(arr)

    return np.ascontiguousarray(arr, dtype=np.float32)


# Tracks whether the one-shot corner log has fired in this process.
# Module-level so it survives across multiple HRDPSGrid instances (e.g.
# in tests) but resets on process restart, which is exactly what we
# want for the live-verification log.
_DECODE_LOGGED = {"once": False}


# ── HRDPSGrid: the public NWPSource implementation ────────────────────


class HRDPSGrid:
    """ECCC HRDPS as an NWPSource for the Canadian / northern-NA chain slot.

    Implements the NWPSource Protocol.  Frames are stored at native
    2.5 km rotated-lat/lon resolution as uint8 dBZ-encoded arrays keyed
    by ``(run_unix_ts, lead_seconds)``.  Sampling at a query
    (lat, lon, ts) does:

    1. Pick the freshest run whose forecast covers ``ts`` and has both
       bracket frames loaded.
    2. Lerp between the two bracketing 1-hour frames in time.
    3. Project the query (lat, lon) into the rotated grid and sample.
    """

    name = "hrdps"

    def __init__(self, cache_dir: Path | None = None):
        # (run_ts, lead_seconds) → memmap-backed uint8 array on the rotated grid.
        self._frames: dict[tuple[int, int], np.ndarray] = {}
        self._client: httpx.AsyncClient | None = None
        self._latest_run_ts: int | None = None
        self._fetch_lock = asyncio.Lock()

        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "hrdps"
            self._persistent = True
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_hrdps_"))
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "HRDPS memmap directory: %s (persistent=%s)",
            self._memmap_dir, self._persistent,
        )
        logger.info(
            "HRDPS rotated origin: rlon0=%.4f rlat0_north=%.4f "
            "(from documented La1=%.3f Lo1=%.3f)",
            HRDPS_GRID_RLON_ORIGIN, HRDPS_GRID_RLAT_ORIGIN_NORTH,
            HRDPS_LA1, HRDPS_LO1,
        )
        if self._persistent:
            self._load_cached_frames()

    # ── Cache management ──────────────────────────────────────────────

    def _frame_path(self, run_ts: int, lead_seconds: int) -> Path:
        return self._memmap_dir / f"r{run_ts}_l{lead_seconds}.dat"

    def _to_memmap(self, name: str, data: np.ndarray) -> np.ndarray:
        """Atomic-write ``data`` and return a read-only memmap view.

        ``.tmp`` → ``os.replace`` keeps the previous version intact on
        a mid-write crash.
        """
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
                    shape=(HRDPS_GRID_HEIGHT, HRDPS_GRID_WIDTH),
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
            logger.info("HRDPS: loaded %d cached frame(s) from disk", loaded)

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
        self._latest_run_ts = None
        self._load_cached_frames()

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
                limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
            )
        return self._client

    async def fetch(
        self,
        now_ts: int | None = None,
        history_seconds: int = 0,
        horizon_seconds: int = 60 * 60,
    ) -> None:
        """Refresh the in-memory window — same shape as DMIDiniGrid.fetch.

        Walks back through 6-hourly ECCC cycles to cover the active
        history window; each run's forecast hours that overlap the
        window are downloaded individually (one HTTP GET per (run, hour),
        files are tiny single-message GRIBs ≤ 1.5 MB).
        """
        async with self._fetch_lock:
            if now_ts is None:
                now_ts = int(datetime.now(tz=timezone.utc).timestamp())

            publish_delay = settings.hrdps_publish_delay_minutes * 60
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
                logger.debug("HRDPS fetch: no runs available for window")
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
                # Step 0 has no file (ECCC reserves it); valid leads
                # are 001..048.  Clamp the lower bound accordingly.
                min_step = max(1, (min_lead // BRACKET_INTERVAL_SECONDS))
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
                    "HRDPS: %d frame(s) ingested across %d run(s); "
                    "store now holds %d frame(s)",
                    total_fetched, len(runs_to_consider), len(self._frames),
                )
            elif total_failed:
                logger.warning(
                    "HRDPS: no frames ingested (%d file(s) failed)",
                    total_failed,
                )

    async def _fetch_one_step(
        self, run: datetime, step_hour: int, client: httpx.AsyncClient,
    ) -> int:
        """Fetch one (run, step) APCP-Accum1h file, encode, and store.

        Returns 1 on success, 0 if already loaded, -1 on fetch error.
        Unlike DMI DINI we do NOT need the previous step's accumulation
        — ``APCP-Accum1h_Sfc`` is already a 1-hour windowed rate.
        """
        run_ts = int(run.timestamp())
        lead_seconds = step_hour * BRACKET_INTERVAL_SECONDS

        if (run_ts, lead_seconds) in self._frames:
            return 0

        url = file_url(run, step_hour)
        from librewxr.data.retry import retry_get
        resp = await retry_get(client, url, log_name="HRDPS data")
        if resp is None:
            return -1
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # 404 is expected when polling at the leading edge of a run
            # that hasn't fully published yet — log at debug, not warning.
            if getattr(e.response, "status_code", None) == 404:
                logger.debug("HRDPS not yet published for %s", url)
            else:
                logger.warning("HRDPS fetch failed for %s: %s", url, e)
            return -1
        grib_bytes = resp.content

        rate_mm_per_hour = decode_apcp_message(grib_bytes)
        if rate_mm_per_hour is None:
            return -1

        encoded = precip_rate_to_dbz_encoded(
            rate_mm_per_hour,
            dbz_offset=settings.hrdps_dbz_offset,
        )
        mm = self._to_memmap(f"r{run_ts}_l{lead_seconds}", encoded)
        self._frames[(run_ts, lead_seconds)] = mm

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
        if stale_frames:
            logger.info(
                "HRDPS: evicted %d out-of-window frame(s)", len(stale_frames),
            )

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def close(self) -> None:
        self._frames.clear()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        if not self._persistent:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.info("HRDPS memmap directory cleaned up")
        else:
            logger.info(
                "HRDPS cache retained at %s for warm restart", self._memmap_dir,
            )


# ── Grid sampling ────────────────────────────────────────────────────


def _sample_grid(
    grid: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    bilinear: bool = False,
) -> np.ndarray:
    """Sample a uint8 rotated-lat/lon grid at (lat, lon) points."""
    row_f, col_f = grid_indices(lat, lon)

    if not bilinear:
        row = np.rint(row_f).astype(np.int32)
        col = np.rint(col_f).astype(np.int32)
        in_domain = (
            (row >= 0)
            & (row < HRDPS_GRID_HEIGHT)
            & (col >= 0)
            & (col < HRDPS_GRID_WIDTH)
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
        & (r1 < HRDPS_GRID_HEIGHT)
        & (c0 >= 0)
        & (c1 < HRDPS_GRID_WIDTH)
    )
    r0c = np.clip(r0, 0, HRDPS_GRID_HEIGHT - 1)
    r1c = np.clip(r1, 0, HRDPS_GRID_HEIGHT - 1)
    c0c = np.clip(c0, 0, HRDPS_GRID_WIDTH - 1)
    c1c = np.clip(c1, 0, HRDPS_GRID_WIDTH - 1)
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
