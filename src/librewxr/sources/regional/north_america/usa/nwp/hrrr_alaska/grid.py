# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NOAA HRRR Alaska composite reflectivity source.

Implements the NWPSource Protocol for HRRR's Alaska domain.  Fetches
hourly composite reflectivity (REFC) from the public AWS Open Data S3
bucket and serves output frames at LibreWXR's 10-min cadence by linear
interpolation between bracketing hourly forecast frames.

Differences from the CONUS HRRR class:

- **Native projection is polar stereographic**, not LCC — Alaska's high
  latitudes make a single-tangent-parallel LCC numerically poor near
  the pole, so NCEP runs the Alaska domain on a 60°N-true-scale polar
  stereographic grid oriented to the 225° (= -135°) meridian.
- **Hourly forecast steps**, not 15-min sub-hourly.  No ``wrfsubhf``
  files exist for Alaska — only hourly ``wrfsfcf{FF}`` surface files.
  Lerp interval is 60 min instead of 15 min.
- **3-hourly cycles** (00/03/06/09/12/15/18/21Z), not hourly cycles.
- **0–48 h forecast horizon** (longer than CONUS subh's 18 h).
- **URL infix** is ``.ak.grib2`` (e.g.
  ``hrrr.t00z.wrfsfcf03.ak.grib2``) rather than CONUS's bare ``.grib2``.

Linear lerp between hourly REFC frames is the awkward case for this
source: an instantaneous field at high resolution with a long bracket
interval can ghost noticeably during fast-moving storms (~12-16 grid
cells of motion per hour).  The CONUS pattern justifies linear lerp on
sub-cell motion (subh = 15 min) and the European/Canadian regional
sources do so on time-averaged accumulated fields.  HRRR-Alaska shares
neither escape hatch — if ghosting becomes visible in practice, the
upgrade path is OpenCV Farneback optical flow (the same machinery IFS
already uses), substituted for the linear lerp in ``sample()``.

Data attribution: NOAA HRRR, distributed via AWS Open Data
(s3://noaa-hrrr-bdp-pds/, public domain).
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
from librewxr.sources.regional.north_america.usa.nwp.hrrr.grid import (
    IdxRecord,
    compute_snow_mask,
    fetch_byte_range,
    fetch_idx,
    find_refc_records,
    find_tmp_2m_records,
)

logger = logging.getLogger(__name__)


# ── HRRR Alaska polar stereographic grid parameters ───────────────────
#
# Confirmed by decoding a real ``wrfsfcf00.ak.grib2`` REFC message from
# the noaa-hrrr-bdp-pds bucket on 2026-05-08.  All four corners and the
# centre back-project to within float-precision of cfgrib's reported
# lat/lon coordinates.
#
# WMO polar-stereographic (Template 3.20) parameters reported by GRIB:
#   shapeOfTheEarth=6 (sphere, 6371229 m)
#   LaD=60° (latitude of true scale; the latitude at which Dx, Dy = 3 km
#           hold exactly — at the pole the projected scale is slightly
#           smaller by factor (1 + sin(LaD)) / 2)
#   LoV=225° (orientation of the grid; the meridian parallel to the
#             column axis along which y increases northward)
#   First grid point: (lat=41.612949, lon=185.117126), corresponding to
#                     the SOUTH-WEST corner per scanningMode=64
#                     (i increases first; j increases northward).

HRRR_AK_SPHERE_RADIUS = 6371229.0
HRRR_AK_LAT_TRUE_SCALE = 60.0           # LaD (deg)
HRRR_AK_LON_ORIENT = 225.0              # LoV (deg)
HRRR_AK_GRID_DX = 3000.0                # m
HRRR_AK_GRID_DY = 3000.0                # m
HRRR_AK_GRID_WIDTH = 1299               # cols (Nx)
HRRR_AK_GRID_HEIGHT = 919               # rows (Ny)

# Origin = projection (x, y) at (col=0, row=0) AFTER the cfgrib output
# is flipped vertically so row 0 = north (image convention, matching
# how the CONUS HRRR module stores its grid).  The first grid point in
# the unflipped GRIB stream is the SW corner; after flipud, row 0 = NW
# corner, which lies at projection (x_min, y_max).  x_min computed from
# forward-projecting (41.612949°N, 185.117126°E) and equals the x-coord
# at every row of col 0 thanks to polar-stereo grid alignment.
HRRR_AK_GRID_X_ORIGIN = -3425051.0      # x at col=0 (m)
HRRR_AK_GRID_Y_ORIGIN = -1344804.1      # y at row=0 (north edge, m)

# Precomputed polar-stereographic constants.
_LAT_TRUE_SCALE_RAD = math.radians(HRRR_AK_LAT_TRUE_SCALE)
_LON_ORIENT_RAD = math.radians(HRRR_AK_LON_ORIENT)
_PS_K = (1.0 + math.sin(_LAT_TRUE_SCALE_RAD)) / 2.0   # scale factor at the pole
_PS_2RK = 2.0 * HRRR_AK_SPHERE_RADIUS * _PS_K


def ps_forward(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project geographic (lat, lon) → HRRR-Alaska polar stereo (x, y) in m.

    Spherical polar stereographic, north pole tangent, with grid scale
    set so 1 cell = ``Dx`` at ``LaD`` (Snyder 1987 §21).  The y-axis is
    aligned with the meridian at ``LoV``; +y goes northward (toward the
    pole).  At the LoV meridian, points south of the pole project to
    negative y — same convention as the unflipped GRIB grid.
    """
    phi = np.radians(lat)
    dlon = np.radians(lon - HRRR_AK_LON_ORIENT)
    rho = _PS_2RK * np.tan(np.pi / 4.0 - phi / 2.0)
    x = rho * np.sin(dlon)
    y = -rho * np.cos(dlon)
    return x, y


def grid_indices(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (lat, lon) to fractional (row, col) on the flipped HRRR-AK grid.

    Row 0 is the top of the grid (northernmost row after the cfgrib
    flipud); column 0 is the leftmost.  Out-of-domain points still
    return values, just outside [0, height) × [0, width); callers
    should test ``domain_mask`` first.
    """
    x, y = ps_forward(lat, lon)
    col = (x - HRRR_AK_GRID_X_ORIGIN) / HRRR_AK_GRID_DX
    row = (HRRR_AK_GRID_Y_ORIGIN - y) / HRRR_AK_GRID_DY
    return row, col


def domain_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return ``True`` where (lat, lon) falls inside the HRRR-Alaska grid."""
    row, col = grid_indices(lat, lon)
    return (
        (row >= 0)
        & (row < HRRR_AK_GRID_HEIGHT - 1)
        & (col >= 0)
        & (col < HRRR_AK_GRID_WIDTH - 1)
    )


# ── Boundary feathering ───────────────────────────────────────────────
#
# Width of the soft transition zone at the HRRR-AK domain edge, in
# metres.  Same value as the CONUS module — there's nothing
# Alaska-specific about how a 75 km feather looks at 3 km resolution.

HRRR_AK_FEATHER_DISTANCE_M = 75_000.0


def feather_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return float32 weights in [0, 1]: 1 deep inside HRRR-AK, 0 outside."""
    row, col = grid_indices(lat, lon)
    dist_cells = np.minimum(
        np.minimum(row, (HRRR_AK_GRID_HEIGHT - 1) - row),
        np.minimum(col, (HRRR_AK_GRID_WIDTH - 1) - col),
    )
    dist_m = dist_cells * HRRR_AK_GRID_DX
    weight = np.clip(dist_m / HRRR_AK_FEATHER_DISTANCE_M, 0.0, 1.0)
    return weight.astype(np.float32, copy=False)


# ── URL + idx helpers ──────────────────────────────────────────────────
#
# Reuse the GRIB-byte-range plumbing from the CONUS module so the only
# Alaska-specific differences live in this file: the URL builder, the
# forecast-step parser, and the grid sampler.

def wrfsfcf_url(
    run: datetime, lead_hour: int, *, bucket: str | None = None
) -> str:
    """Construct the S3 URL for a HRRR-Alaska wrfsfcf file.

    ``run`` is the model initialisation time (UTC, hour-aligned, must
    be one of the 3-hourly cycles).  ``lead_hour`` is the forecast hour
    (0..48).  Returned URL points at the ``wrfsfcf{FF}.ak.grib2`` file;
    append ``.idx`` for the sidecar.
    """
    if bucket is None:
        bucket = settings.hrrr_s3_bucket
    date = run.strftime("%Y%m%d")
    hh = run.strftime("%H")
    ff = f"{lead_hour:02d}"
    return (
        f"https://{bucket}.s3.amazonaws.com/hrrr.{date}/alaska/"
        f"hrrr.t{hh}z.wrfsfcf{ff}.ak.grib2"
    )


# Step-label parsing.  HRRR-Alaska's idx uses ``"anl"`` for the analysis
# (lead 0) and ``"N hour fcst"`` for forecast steps — distinct from
# CONUS subh's ``"NN min fcst"`` because the cadence is hourly.

_STEP_HOUR_FCST_RE = re.compile(r"^(\d+) hour fcst$")


def lead_seconds_for_step(step_label: str) -> int | None:
    """Parse an HRRR-AK idx step label to total lead seconds.

    Returns 0 for ``"anl"`` (analysis), N*3600 for ``"N hour fcst"``,
    or ``None`` for any other label (e.g. accumulation buckets, which
    aren't REFC entries anyway).
    """
    if step_label == "anl":
        return 0
    m = _STEP_HOUR_FCST_RE.match(step_label)
    if m is None:
        return None
    return int(m.group(1)) * 3600


# ── GRIB2 decode ───────────────────────────────────────────────────────


def decode_refc_message(grib_bytes: bytes) -> np.ndarray | None:
    """Decode a single GRIB2 message containing REFC into a 2D float32 array.

    Returns ``None`` if cfgrib cannot parse the bytes.  Output shape is
    ``(HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH)`` with row 0 at the
    NORTHERN edge of the grid (cfgrib hands HRRR-AK back with row 0 at
    the south edge, same as CONUS HRRR; we flip on decode).
    """
    import xarray as xr

    from librewxr.data.sources import _suppress_eccodes_stderr

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
        logger.exception("Failed to decode HRRR-Alaska REFC GRIB2 message")
        return None
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    if "refc" in ds.data_vars:
        arr = ds["refc"].values
    else:
        for name, da in ds.data_vars.items():
            if (
                da.ndim == 2
                and da.shape == (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH)
            ):
                logger.warning(
                    "HRRR-AK REFC variable not named 'refc' (got %r); "
                    "using fallback",
                    name,
                )
                arr = da.values
                break
        else:
            logger.warning(
                "HRRR-AK GRIB2 message did not contain a recognisable REFC field"
            )
            return None

    if arr.shape != (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH):
        logger.warning(
            "HRRR-AK REFC has unexpected shape %s (expected %s); skipping",
            arr.shape,
            (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH),
        )
        return None

    # Self-correcting flip: row 0 should be the NORTHERN edge.  cfgrib
    # historically returns HRRR with row 0 at the SOUTH; verify against
    # cfgrib's own latitude coordinate when available.
    if "latitude" in ds.coords:
        lat_top = float(ds["latitude"].values[0, 0])
        lat_bot = float(ds["latitude"].values[-1, 0])
        needs_flip = lat_top < lat_bot
    else:
        needs_flip = True
    if needs_flip:
        arr = np.flipud(arr)

    return np.ascontiguousarray(arr, dtype=np.float32)


def encode_dbz(refc: np.ndarray) -> np.ndarray:
    """Convert raw dBZ float32 → uint8 encoded values (matches radar pipeline)."""
    safe = np.where(np.isfinite(refc), refc, -32.0)
    return np.clip((safe + 32.0) * 2.0 + 0.5, 0, 255).astype(np.uint8)


def decode_tmp_2m_message(grib_bytes: bytes) -> np.ndarray | None:
    """Decode a single GRIB2 message containing 2-m TMP into Celsius float32.

    Returns ``None`` if cfgrib cannot parse the bytes.  Output shape is
    ``(HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH)`` with row 0 = north,
    matching ``decode_refc_message``.
    """
    import xarray as xr

    from librewxr.data.sources import _suppress_eccodes_stderr

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
        logger.exception("Failed to decode HRRR-Alaska TMP:2m GRIB2 message")
        return None
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    if "t2m" in ds.data_vars:
        arr = ds["t2m"].values
    else:
        for name, da in ds.data_vars.items():
            if (
                da.ndim == 2
                and da.shape == (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH)
            ):
                logger.warning(
                    "HRRR-AK TMP:2m variable not named 't2m' (got %r); "
                    "using fallback",
                    name,
                )
                arr = da.values
                break
        else:
            logger.warning(
                "HRRR-AK GRIB2 message did not contain a recognisable "
                "2-m TMP field"
            )
            return None

    if arr.shape != (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH):
        logger.warning(
            "HRRR-AK TMP:2m has unexpected shape %s (expected %s); skipping",
            arr.shape,
            (HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH),
        )
        return None

    # Same self-correcting flip as decode_refc_message: row 0 should be
    # the NORTHERN edge.
    if "latitude" in ds.coords:
        lat_top = float(ds["latitude"].values[0, 0])
        lat_bot = float(ds["latitude"].values[-1, 0])
        needs_flip = lat_top < lat_bot
    else:
        needs_flip = True
    if needs_flip:
        arr = np.flipud(arr)

    # Kelvin → Celsius for the threshold check.
    celsius = np.ascontiguousarray(arr, dtype=np.float32) - 273.15
    return celsius


# ── Run / step timing ─────────────────────────────────────────────────

CYCLE_INTERVAL_SECONDS = 3 * 3600        # HRRR-AK runs every 3 hours
BRACKET_INTERVAL_SECONDS = 3600           # forecast steps are 1 hour apart
MAX_FORECAST_HOURS = 48                   # subh-equivalent horizon for the AK domain


def floor_cycle(ts: int) -> int:
    """Floor a Unix timestamp to the nearest 3-hour cycle boundary."""
    return (ts // CYCLE_INTERVAL_SECONDS) * CYCLE_INTERVAL_SECONDS


def latest_published_run(now_ts: int, publish_delay_seconds: int) -> int:
    """Most recent run we'd expect to be available given a publish delay."""
    return floor_cycle(now_ts - publish_delay_seconds)


def bracket_lead_seconds(lead_seconds: int) -> tuple[int, int, float]:
    """For a desired lead, return ``(L0, L1, alpha)`` where L0 ≤ L < L1.

    Both leads are exact hour multiples (multiples of 3600s, ≥ 0).
    ``alpha`` is the lerp weight between L0 and L1.
    """
    if lead_seconds < 0:
        return 0, 0, 0.0
    l0 = (lead_seconds // BRACKET_INTERVAL_SECONDS) * BRACKET_INTERVAL_SECONDS
    l1 = l0 + BRACKET_INTERVAL_SECONDS
    alpha = (lead_seconds - l0) / BRACKET_INTERVAL_SECONDS
    return l0, l1, alpha


# ── HRRRAlaskaGrid: the public NWPSource implementation ───────────────


class HRRRAlaskaGrid:
    """NOAA HRRR Alaska composite reflectivity, sampled in native polar stereo.

    Implements the NWPSource Protocol.  Frames are stored at native 3 km
    polar-stereo resolution as uint8 dBZ-encoded arrays keyed by
    ``(run_unix_ts, lead_seconds)``.  Sampling at a query (lat, lon, ts)
    does:

    1. Pick the freshest run whose forecast covers ``ts`` and whose
       bracket frames are loaded.
    2. Lerp between the two bracketing hourly REFC frames in time.
    3. Project the query (lat, lon) into the polar-stereo grid and sample.
    """

    name = "hrrr_alaska"

    def __init__(self, cache_dir: Path | None = None):
        # ``_frames`` holds REFC; ``_snow_masks`` holds the parallel 2-m TMP
        # snow classification (1 = snow, 0 = rain) keyed by the same tuple.
        # See HRRRGrid for the storage rationale — same pattern mirrored
        # here for the Alaska domain.
        self._frames: dict[tuple[int, int], np.ndarray] = {}
        self._snow_masks: dict[tuple[int, int], np.ndarray] = {}
        self._client: httpx.AsyncClient | None = None
        self._latest_run_ts: int | None = None
        self._fetch_lock = asyncio.Lock()

        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "hrrr_alaska"
            self._persistent = True
        else:
            self._memmap_dir = Path(
                tempfile.mkdtemp(prefix="librewxr_hrrr_alaska_")
            )
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "HRRR-Alaska memmap directory: %s (persistent=%s)",
            self._memmap_dir, self._persistent,
        )
        if self._persistent:
            self._load_cached_frames()

    # ── Memory management ────────────────────────────────────────────

    def _to_memmap(self, name: str, data: np.ndarray) -> np.ndarray:
        final = self._memmap_dir / f"{name}.dat"
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(tmp, dtype=data.dtype, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        del mm
        os.replace(tmp, final)
        return np.memmap(final, dtype=data.dtype, mode="r", shape=data.shape)

    def _memmap_path_for(self, run_ts: int, lead_seconds: int) -> Path:
        return self._memmap_dir / f"r{run_ts}_l{lead_seconds}.dat"

    def _snow_memmap_path_for(self, run_ts: int, lead_seconds: int) -> Path:
        return self._memmap_dir / f"r{run_ts}_l{lead_seconds}_snow.dat"

    def _load_cached_frames(self) -> None:
        for path in self._memmap_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        loaded = 0
        for path in self._memmap_dir.glob("r*_l*.dat"):
            try:
                stem_parts = path.stem.split("_")
                if len(stem_parts) != 2:
                    # Skip snow files ("rNNN_lNNN_snow") and any other
                    # parallel-field files we may add later.
                    continue
                run_ts = int(stem_parts[0][1:])
                lead_s = int(stem_parts[1][1:])
            except (ValueError, IndexError):
                continue
            try:
                mm = np.memmap(
                    path,
                    dtype=np.uint8,
                    mode="r",
                    shape=(HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH),
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
                "HRRR-Alaska: loaded %d cached frame(s) from disk", loaded,
            )

        # Second pass: parallel snow masks.  Orphan snow files (no
        # matching REFC) are removed so they don't accumulate.
        snow_loaded = 0
        for path in self._memmap_dir.glob("r*_l*_snow.dat"):
            try:
                stem_parts = path.stem.split("_")
                if len(stem_parts) != 3 or stem_parts[2] != "snow":
                    continue
                run_ts = int(stem_parts[0][1:])
                lead_s = int(stem_parts[1][1:])
            except (ValueError, IndexError):
                continue
            if (run_ts, lead_s) not in self._frames:
                path.unlink(missing_ok=True)
                continue
            try:
                mm = np.memmap(
                    path,
                    dtype=np.uint8,
                    mode="r",
                    shape=(HRRR_AK_GRID_HEIGHT, HRRR_AK_GRID_WIDTH),
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
                "HRRR-Alaska: loaded %d cached snow mask(s) from disk",
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
            self._latest_run_ts, tz=timezone.utc
        ).isoformat()

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def snow_mask_count(self) -> int:
        return len(self._snow_masks)

    # ── NWPSource Protocol ───────────────────────────────────────────

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
        return True

    def get_snow_mask(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        """Sample the snow / rain classification at each (lat, lon, ts).

        Mirrors ``sample`` for the parallel 2-m TMP snow mask: same run
        picker, same hourly bracket-lerp, then re-binarize at 0.5.
        Returns False everywhere if no snow mask is loaded for the
        bracket — the chain dispatcher falls through to the next
        snow-capable source for those pixels.
        """
        if timestamp is None or not self._snow_masks:
            return np.zeros(lat.shape, dtype=bool)

        run = self._pick_run(timestamp)
        if run is None:
            return np.zeros(lat.shape, dtype=bool)

        lead = timestamp - run
        l0, l1, alpha = bracket_lead_seconds(lead)
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

    # ── Run selection ────────────────────────────────────────────────

    def _pick_run(self, timestamp: int) -> int | None:
        """Pick the freshest run whose hourly bracket for ``timestamp`` is loaded.

        Walks runs newest-first.  A run R is considered usable iff:

        - ``R ≤ T ≤ R + 48h`` (T inside the forecast horizon)
        - both bracketing hourly frames ``(R, L0)`` and ``(R, L1)`` exist

        Falling back to an older run when the freshest run's bracket
        isn't fully loaded keeps the nowcast loop visually consistent
        across cycle rollovers.
        """
        loaded_runs = sorted({r for (r, _) in self._frames}, reverse=True)
        for run in loaded_runs:
            lead = timestamp - run
            if not (0 <= lead <= MAX_FORECAST_HOURS * 3600):
                continue
            l0, l1, _ = bracket_lead_seconds(lead)
            if (run, l0) in self._frames and (run, l1) in self._frames:
                return run
        return None

    # ── Fetch ────────────────────────────────────────────────────────

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
        """Refresh the in-memory hourly window.

        Determines all 3-hourly runs whose forecasts can cover the
        active window ``[now − history, now + horizon]`` and fetches
        the relevant ``wrfsfcf`` files from each.  Already-loaded
        frames are skipped.  Frames whose valid time falls outside the
        window are evicted.
        """
        async with self._fetch_lock:
            if now_ts is None:
                now_ts = int(datetime.now(tz=timezone.utc).timestamp())

            publish_delay = (
                settings.hrrr_alaska_publish_delay_minutes * 60
            )
            latest_run_ts = latest_published_run(now_ts, publish_delay)
            if (
                self._latest_run_ts is None
                or latest_run_ts > self._latest_run_ts
            ):
                self._latest_run_ts = latest_run_ts

            window_start = now_ts - history_seconds
            window_end = now_ts + horizon_seconds

            # Walk back through cycle boundaries until a run's horizon
            # can no longer reach window_start.  Each run reaches +48h,
            # so two cycles back covers a 6-hour history comfortably.
            earliest_run = floor_cycle(
                window_start - BRACKET_INTERVAL_SECONDS
            )
            runs_to_consider = list(
                range(
                    earliest_run, latest_run_ts + 1, CYCLE_INTERVAL_SECONDS
                )
            )
            if not runs_to_consider:
                logger.debug(
                    "HRRR-Alaska fetch: no runs available for window"
                )
                return

            client = await self._get_client()

            total_fetched = 0
            total_failed = 0
            for run_ts in runs_to_consider:
                run_dt = datetime.fromtimestamp(run_ts, tz=timezone.utc)
                # Forecast hours of this run that overlap the active window.
                # File ``wrfsfcfFF`` covers exactly lead = FF*3600.
                # Extend by one bracket interval on each side so the L1
                # frame at the future edge — and the L0 frame at the past
                # edge — are included.
                min_lead = max(
                    0,
                    window_start - run_ts - BRACKET_INTERVAL_SECONDS,
                )
                max_lead = min(
                    MAX_FORECAST_HOURS * 3600,
                    window_end - run_ts + BRACKET_INTERVAL_SECONDS,
                )
                if max_lead < min_lead:
                    continue
                min_hour = max(0, min_lead // 3600)
                max_hour = min(MAX_FORECAST_HOURS, -(-max_lead // 3600))

                for fh in range(int(min_hour), int(max_hour) + 1):
                    added = await self._fetch_one_wrfsfcf_file(
                        run_dt, fh, client
                    )
                    if added > 0:
                        total_fetched += added
                        logger.debug(
                            "HRRR-Alaska: +%d frame(s) from run %sZ fh=%d",
                            added,
                            run_dt.strftime("%Y%m%d%H"),
                            fh,
                        )
                    elif added < 0:
                        total_failed += 1

            self._evict_outside_window(window_start, window_end)

            if total_fetched:
                logger.info(
                    "HRRR-Alaska: %d frame(s) ingested across %d run(s); "
                    "store now holds %d frame(s)",
                    total_fetched,
                    len(runs_to_consider),
                    len(self._frames),
                )
            elif total_failed:
                logger.warning(
                    "HRRR-Alaska: no frames ingested (%d file(s) failed)",
                    total_failed,
                )

    async def _fetch_one_wrfsfcf_file(
        self, run: datetime, lead_hour: int, client: httpx.AsyncClient
    ) -> int:
        """Fetch the REFC frame from a single wrfsfcf file. Returns frames added.

        Also fetches the parallel 2-m TMP message for the same step and
        stores the derived snow mask alongside.  TMP failures are
        non-fatal — the REFC frame still lands and ``get_snow_mask``
        falls through to the next chain source for that bracket.

        Returns -1 to signal a fetch error (idx unreachable, etc.).
        """
        run_ts = int(run.timestamp())
        url = wrfsfcf_url(run, lead_hour)
        try:
            records = await fetch_idx(url, client)
        except httpx.HTTPError as e:
            logger.warning(
                "HRRR-Alaska idx fetch failed for %s: %s", url, e
            )
            return -1

        added = 0
        for rec, end in find_refc_records(records):
            lead_seconds = lead_seconds_for_step(rec.step)
            if lead_seconds is None:
                continue
            key = (run_ts, lead_seconds)
            if key in self._frames:
                continue

            try:
                grib_bytes = await fetch_byte_range(
                    url, rec.byte_offset, end, client
                )
            except httpx.HTTPError as e:
                logger.warning(
                    "HRRR-Alaska REFC byte-range fetch failed for "
                    "%s lead=%ds: %s",
                    url, lead_seconds, e,
                )
                continue

            arr = decode_refc_message(grib_bytes)
            if arr is None:
                continue

            encoded = encode_dbz(arr)
            mm = self._to_memmap(f"r{run_ts}_l{lead_seconds}", encoded)
            self._frames[key] = mm
            added += 1

        # Parallel pass for 2-m TMP → snow mask.  Same idx, separate
        # byte-range per step.  Skip keys whose snow mask is already
        # loaded (warm restart) or whose REFC didn't land (no point
        # carrying an orphan snow mask).
        threshold = settings.regional_snow_temp_threshold
        for rec, end in find_tmp_2m_records(records):
            lead_seconds = lead_seconds_for_step(rec.step)
            if lead_seconds is None:
                continue
            key = (run_ts, lead_seconds)
            if key in self._snow_masks:
                continue
            if key not in self._frames:
                continue

            try:
                grib_bytes = await fetch_byte_range(
                    url, rec.byte_offset, end, client
                )
            except httpx.HTTPError as e:
                logger.warning(
                    "HRRR-Alaska TMP:2m byte-range fetch failed for "
                    "%s lead=%ds: %s",
                    url, lead_seconds, e,
                )
                continue

            t2m = decode_tmp_2m_message(grib_bytes)
            if t2m is None:
                continue

            snow = compute_snow_mask(t2m, threshold)
            mm = self._to_memmap(f"r{run_ts}_l{lead_seconds}_snow", snow)
            self._snow_masks[key] = mm

        return added

    # ── Eviction ─────────────────────────────────────────────────────

    def _evict_outside_window(
        self, window_start: int, window_end: int
    ) -> None:
        """Drop frames whose valid time falls outside the active window.

        Slack of one bracket interval on each side keeps the bracketing
        L1 frame at the future edge (and L0 frame at the past edge)
        alive long enough for ``sample`` at the window boundary to find
        both halves of its bracket.
        """
        slack = BRACKET_INTERVAL_SECONDS
        ws = window_start - slack
        we = window_end + slack
        stale = []
        for key in self._frames:
            run_ts, lead = key
            valid_time = run_ts + lead
            if valid_time < ws or valid_time > we:
                stale.append(key)
        for key in stale:
            self._frames.pop(key, None)
            self._snow_masks.pop(key, None)
            try:
                self._memmap_path_for(*key).unlink(missing_ok=True)
            except OSError:
                pass
            try:
                self._snow_memmap_path_for(*key).unlink(missing_ok=True)
            except OSError:
                pass
        if stale:
            logger.info(
                "HRRR-Alaska: evicted %d out-of-window frame(s)",
                len(stale),
            )

    # ── Lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        self._frames.clear()
        self._snow_masks.clear()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        if not self._persistent:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.info("HRRR-Alaska memmap directory cleaned up")
        else:
            logger.info(
                "HRRR-Alaska cache retained at %s for warm restart",
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
    """Sample a uint8 polar-stereo grid at (lat, lon) points.

    Returns ``np.ndarray`` of dtype uint8, shape ``lat.shape``.  Points
    outside the grid yield 0.
    """
    row_f, col_f = grid_indices(lat, lon)

    if not bilinear:
        row = np.rint(row_f).astype(np.int32)
        col = np.rint(col_f).astype(np.int32)
        in_domain = (
            (row >= 0)
            & (row < HRRR_AK_GRID_HEIGHT)
            & (col >= 0)
            & (col < HRRR_AK_GRID_WIDTH)
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
        & (r1 < HRRR_AK_GRID_HEIGHT)
        & (c0 >= 0)
        & (c1 < HRRR_AK_GRID_WIDTH)
    )

    r0c = np.clip(r0, 0, HRRR_AK_GRID_HEIGHT - 1)
    r1c = np.clip(r1, 0, HRRR_AK_GRID_HEIGHT - 1)
    c0c = np.clip(c0, 0, HRRR_AK_GRID_WIDTH - 1)
    c1c = np.clip(c1, 0, HRRR_AK_GRID_WIDTH - 1)

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
