# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NOAA HRRR sub-hourly precipitation source.

Implements the NWPSource Protocol for the High-Resolution Rapid Refresh
model.  Fetches 15-min sub-hourly composite reflectivity (REFC) from the
public AWS Open Data S3 bucket, samples by reprojecting query lat/lon
into HRRR's native Lambert Conformal Conic grid, and serves output frames
at LibreWXR's 10-min cadence by linear interpolation between bracketing
subh frames.

Why this layout (vs. the IFS/ECMWFGrid layout):

- HRRR is native LCC at 3 km — we project the query points into the grid
  rather than regridding the whole grid into lat/lon.
- ``wrfsubhf`` files give us 15-min frames natively; lerp between adjacent
  frames is sufficient and far cheaper than the optical-flow interpolation
  IFS needs from its hourly cadence.
- HRRR's ``composite_reflectivity`` field is already in dBZ from the
  model's microphysics — no Marshall-Palmer Z-R conversion required.

Data attribution: NOAA HRRR, distributed via AWS Open Data
(s3://noaa-hrrr-bdp-pds/, public domain).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

from librewxr.config import settings

logger = logging.getLogger(__name__)


# ── HRRR CONUS LCC grid parameters ─────────────────────────────────────
#
# Source: GeoTransform metadata read from a representative HRRR GRIB2
# file by the dynamical.org/reformatters project.  The WKT declares a
# 2SP Lambert Conformal Conic with both standard parallels at 38.5°N,
# which is the degenerate case equivalent to a single tangent parallel.
# HRRR uses a 6371229m sphere (not WGS84) — this is the Earth-radius
# convention NCEP uses for most operational models.

HRRR_SPHERE_RADIUS = 6371229.0           # metres (sphere, not WGS84)
HRRR_LAT_0 = 38.5                        # latitude of projection origin (deg)
HRRR_LON_0 = -97.5                       # central meridian (deg)
HRRR_STD_PARALLEL = 38.5                 # single tangent standard parallel (deg)
HRRR_GRID_X_ORIGIN = -2699020.142521929  # x of (col=0, row=0) in projection metres
HRRR_GRID_Y_ORIGIN = 1588193.847443335   # y of (col=0, row=0) in projection metres
HRRR_GRID_DX = 3000.0                    # x spacing (m, increases left→right)
HRRR_GRID_DY = 3000.0                    # y spacing (m, decreases top→bottom)
HRRR_GRID_WIDTH = 1799                   # CONUS columns
HRRR_GRID_HEIGHT = 1059                  # CONUS rows

# Precomputed LCC constants — compiled once at import time.
_PHI_0 = math.radians(HRRR_STD_PARALLEL)
_N = math.sin(_PHI_0)
_F = math.cos(_PHI_0) * math.tan(math.pi / 4 + _PHI_0 / 2) ** _N / _N
_RHO_0 = HRRR_SPHERE_RADIUS * _F / math.tan(math.pi / 4 + _PHI_0 / 2) ** _N
_LON_0_RAD = math.radians(HRRR_LON_0)


def lcc_forward(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project geographic (lat, lon) → HRRR LCC (x, y) in metres.

    Spherical Lambert Conformal Conic per Snyder (1987) §15, single
    tangent standard parallel case (n = sin φ_0).
    """
    phi = np.radians(lat)
    lam = np.radians(lon)

    rho = HRRR_SPHERE_RADIUS * _F / np.tan(np.pi / 4 + phi / 2) ** _N
    theta = _N * (lam - _LON_0_RAD)

    x = rho * np.sin(theta)
    y = _RHO_0 - rho * np.cos(theta)
    return x, y


def grid_indices(
    lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (lat, lon) to fractional (row, col) on the HRRR grid.

    Row 0 is the top of the grid (northernmost projected y); column 0 is
    the leftmost (westernmost projected x).  Out-of-domain points still
    return values, just outside [0, height) × [0, width); callers should
    test ``domain_mask`` first.
    """
    x, y = lcc_forward(lat, lon)
    col = (x - HRRR_GRID_X_ORIGIN) / HRRR_GRID_DX
    row = (HRRR_GRID_Y_ORIGIN - y) / HRRR_GRID_DY
    return row, col


def domain_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return ``True`` where (lat, lon) falls inside the HRRR CONUS grid.

    Uses the inclusive-of-edges convention for cells but excludes the
    last row/column to leave room for bilinear sampling without boundary
    fixup.  At HRRR's 3 km resolution this drops ~3 km of fringe; not
    visible at any tile zoom.
    """
    row, col = grid_indices(lat, lon)
    return (
        (row >= 0)
        & (row < HRRR_GRID_HEIGHT - 1)
        & (col >= 0)
        & (col < HRRR_GRID_WIDTH - 1)
    )


# ── Boundary feathering ───────────────────────────────────────────────
#
# Width of the soft transition zone at the HRRR LCC domain edge, in
# metres.  Inside the inner region (≥ FEATHER_DISTANCE_M from any edge)
# HRRR is trusted at full weight; over the feather zone the weight
# tapers linearly to 0 at the edge so chain blending hands control to
# IFS smoothly instead of leaving a visible seam.

HRRR_FEATHER_DISTANCE_M = 75_000.0  # 75 km ≈ 25 grid cells


def feather_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return float32 weights in [0, 1]: 1 deep inside HRRR, 0 outside.

    Tapers linearly over ``HRRR_FEATHER_DISTANCE_M`` from the nearest
    LCC grid edge.  Out-of-domain points return 0.  Used by ``NWPChain``
    to soft-blend HRRR with the next source in the chain (typically IFS).
    """
    row, col = grid_indices(lat, lon)
    # Distance to nearest edge in grid cells (negative if outside).
    dist_cells = np.minimum(
        np.minimum(row, (HRRR_GRID_HEIGHT - 1) - row),
        np.minimum(col, (HRRR_GRID_WIDTH - 1) - col),
    )
    # Convert to metres using HRRR's grid spacing.
    dist_m = dist_cells * HRRR_GRID_DX
    weight = np.clip(dist_m / HRRR_FEATHER_DISTANCE_M, 0.0, 1.0)
    return weight.astype(np.float32, copy=False)


# ── GRIB2 byte-range fetch ─────────────────────────────────────────────
#
# HRRR is published as multi-message GRIB2 files alongside an ``.idx``
# sidecar that lists each message's variable, level, forecast step, and
# byte offset.  Range requests against the bucket let us pull just the
# REFC message (~5-10 MB) instead of the whole file (~80-200 MB),
# turning a one-cycle fetch into ~200 MB total instead of ~3 GB.

# Each idx line:
#   record_num : byte_offset : d=YYYYMMDDHH : VAR : LEVEL : STEP :
import re  # noqa: E402  (kept inline so the projection block is self-contained)

_IDX_LINE_RE = re.compile(
    r"^(?P<num>\d+):"
    r"(?P<offset>\d+):"
    r"d=(?P<date>\d+):"
    r"(?P<var>[^:]+):"
    r"(?P<level>[^:]+):"
    r"(?P<step>[^:]*):"
)


class IdxRecord:
    __slots__ = ("record_num", "byte_offset", "var", "level", "step")

    def __init__(self, record_num: int, byte_offset: int, var: str, level: str, step: str):
        self.record_num = record_num
        self.byte_offset = byte_offset
        self.var = var
        self.level = level
        self.step = step

    def __repr__(self) -> str:
        return (
            f"IdxRecord(num={self.record_num}, offset={self.byte_offset}, "
            f"var={self.var!r}, level={self.level!r}, step={self.step!r})"
        )


def parse_idx(idx_text: str) -> list[IdxRecord]:
    """Parse a GRIB2 ``.idx`` sidecar file into a list of records.

    Lines that don't match the expected format are silently skipped.
    """
    records: list[IdxRecord] = []
    for line in idx_text.strip().splitlines():
        m = _IDX_LINE_RE.match(line)
        if m is None:
            continue
        records.append(
            IdxRecord(
                record_num=int(m.group("num")),
                byte_offset=int(m.group("offset")),
                var=m.group("var"),
                level=m.group("level"),
                step=m.group("step"),
            )
        )
    return records


def find_refc_records(records: list[IdxRecord]) -> list[tuple[IdxRecord, int]]:
    """Return [(refc_record, end_byte)] entries for every REFC message in the file.

    A subh file contains multiple sub-hourly steps; each step has its
    own REFC message.  We return all of them so the caller can pick
    by step (e.g. ``"15 min fcst"``).

    ``end_byte`` is the byte offset of the next record minus one
    (inclusive HTTP Range), or ``-1`` if the REFC record is the last
    one in the file (caller should issue ``Range: bytes=START-``).
    """
    out: list[tuple[IdxRecord, int]] = []
    for i, rec in enumerate(records):
        if rec.var != "REFC":
            continue
        if i + 1 < len(records):
            end = records[i + 1].byte_offset - 1
        else:
            end = -1
        out.append((rec, end))
    return out


def find_tmp_2m_records(records: list[IdxRecord]) -> list[tuple[IdxRecord, int]]:
    """Return [(tmp_2m_record, end_byte)] entries for every 2-metre TMP message.

    Parallel to ``find_refc_records`` but for the 2-metre temperature
    field used by the snow-mask classifier.  ``TMP`` appears at many
    levels in HRRR's surface bundle (e.g. ``surface``, ``2 m above
    ground``, isobaric layers); we want only the 2-m one.
    """
    out: list[tuple[IdxRecord, int]] = []
    for i, rec in enumerate(records):
        if rec.var != "TMP" or rec.level != "2 m above ground":
            continue
        if i + 1 < len(records):
            end = records[i + 1].byte_offset - 1
        else:
            end = -1
        out.append((rec, end))
    return out


def subh_url(run: datetime, lead_hour: int, *, bucket: str | None = None) -> str:
    """Construct the S3 URL for a HRRR subh file.

    ``run`` is the model initialisation time (UTC, hour-aligned).
    ``lead_hour`` is the forecast hour (1..18).  Returned URL points
    at the ``wrfsubhf{FF}.grib2`` file; append ``.idx`` for the sidecar.
    """
    if bucket is None:
        bucket = settings.hrrr_s3_bucket
    date = run.strftime("%Y%m%d")
    hh = run.strftime("%H")
    ff = f"{lead_hour:02d}"
    return (
        f"https://{bucket}.s3.amazonaws.com/hrrr.{date}/conus/"
        f"hrrr.t{hh}z.wrfsubhf{ff}.grib2"
    )


async def fetch_idx(
    url: str, client: httpx.AsyncClient
) -> list[IdxRecord]:
    """Fetch and parse a ``.idx`` sidecar."""
    from librewxr.data.retry import retry_get
    resp = await retry_get(client, url + ".idx", log_name="HRRR idx")
    if resp is None:
        raise httpx.TransportError("failed to fetch idx after retries")
    resp.raise_for_status()
    return parse_idx(resp.text)


async def fetch_byte_range(
    url: str, start: int, end: int, client: httpx.AsyncClient
) -> bytes:
    """Fetch a byte range from a URL.  ``end == -1`` means to EOF."""
    from librewxr.data.retry import retry_get
    if end == -1:
        range_header = f"bytes={start}-"
    else:
        range_header = f"bytes={start}-{end}"
    resp = await retry_get(
        client, url, headers={"Range": range_header}, log_name="HRRR data",
    )
    if resp is None:
        raise httpx.TransportError("failed to fetch byte range after retries")
    resp.raise_for_status()
    return resp.content


# Lazily import the eccodes-stderr suppressor so this module's import
# graph mirrors the IFS / MRMS pattern (sources.py owns the helper).
def _suppress_eccodes_stderr():
    from librewxr.data.sources import _suppress_eccodes_stderr as _s
    return _s()


def decode_refc_message(grib_bytes: bytes) -> np.ndarray | None:
    """Decode a single GRIB2 message containing REFC into a 2D float32 array.

    Returns ``None`` if cfgrib cannot parse the bytes; callers retry.
    The shape is ``(HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH)``.
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
                backend_kwargs={"indexpath": ""},  # don't write a .idx next to tmp
            )
        ds = ds.compute()
    except Exception:
        logger.exception("Failed to decode HRRR REFC GRIB2 message")
        return None
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    # cfgrib names HRRR's composite reflectivity ``refc`` (lowercase).
    # If for any reason the variable comes back under a different key
    # (e.g. a future codetable change), fall back to the first 2D var.
    if "refc" in ds.data_vars:
        arr = ds["refc"].values
    else:
        for name, da in ds.data_vars.items():
            if da.ndim == 2 and da.shape == (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH):
                logger.warning(
                    "HRRR REFC variable not named 'refc' (got %r); using fallback",
                    name,
                )
                arr = da.values
                break
        else:
            logger.warning(
                "HRRR GRIB2 message did not contain a recognisable REFC field"
            )
            return None

    if arr.shape != (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH):
        logger.warning(
            "HRRR REFC has unexpected shape %s (expected %s); skipping",
            arr.shape,
            (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH),
        )
        return None

    # cfgrib returns HRRR with the j-axis (rows) flipped relative to the
    # image-convention our grid_indices() assumes: row 0 lands at the
    # SOUTH edge of the grid (lat ~21°N), not the north (lat ~52°N).
    # Flip vertically so row 0 = north.  Verify against cfgrib's own
    # latitude coordinate when available so the code self-corrects if a
    # future cfgrib release ever normalises the orientation upstream.
    if "latitude" in ds.coords:
        lat_top = float(ds["latitude"].values[0, 0])
        lat_bot = float(ds["latitude"].values[-1, 0])
        needs_flip = lat_top < lat_bot
    else:
        # No coord to verify against — assume the historically-observed
        # HRRR-on-cfgrib orientation (row 0 = south) and flip.
        needs_flip = True
    if needs_flip:
        arr = np.flipud(arr)

    return np.ascontiguousarray(arr, dtype=np.float32)


def encode_dbz(refc: np.ndarray) -> np.ndarray:
    """Convert raw dBZ float32 → uint8 encoded values matching the radar pipeline.

    Encoding: pixel = clip((dBZ + 32) * 2, 0, 255).  Non-finite cells
    become 0 (no precipitation) so they integrate cleanly with the rest
    of the rendering pipeline.
    """
    safe = np.where(np.isfinite(refc), refc, -32.0)
    return np.clip((safe + 32.0) * 2.0 + 0.5, 0, 255).astype(np.uint8)


def decode_tmp_2m_message(grib_bytes: bytes) -> np.ndarray | None:
    """Decode a single GRIB2 message containing 2-m TMP into Celsius float32.

    Returns ``None`` if cfgrib cannot parse the bytes.  The shape is
    ``(HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH)`` and the row orientation is
    normalised to row 0 = north, matching ``decode_refc_message``.
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
        logger.exception("Failed to decode HRRR TMP:2m GRIB2 message")
        return None
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    # cfgrib names 2-m temperature ``t2m``.  Fall back to first
    # matching 2D var if a future codetable change moves it.
    if "t2m" in ds.data_vars:
        arr = ds["t2m"].values
    else:
        for name, da in ds.data_vars.items():
            if da.ndim == 2 and da.shape == (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH):
                logger.warning(
                    "HRRR TMP:2m variable not named 't2m' (got %r); using fallback",
                    name,
                )
                arr = da.values
                break
        else:
            logger.warning(
                "HRRR GRIB2 message did not contain a recognisable 2-m TMP field"
            )
            return None

    if arr.shape != (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH):
        logger.warning(
            "HRRR TMP:2m has unexpected shape %s (expected %s); skipping",
            arr.shape,
            (HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH),
        )
        return None

    # Match the row-orientation handling in ``decode_refc_message``.
    if "latitude" in ds.coords:
        lat_top = float(ds["latitude"].values[0, 0])
        lat_bot = float(ds["latitude"].values[-1, 0])
        needs_flip = lat_top < lat_bot
    else:
        needs_flip = True
    if needs_flip:
        arr = np.flipud(arr)

    # cfgrib returns TMP in Kelvin; convert to Celsius for the threshold check.
    celsius = np.ascontiguousarray(arr, dtype=np.float32) - 273.15
    return celsius


def compute_snow_mask(t2m_celsius: np.ndarray, threshold: float) -> np.ndarray:
    """Classify each HRRR cell as snow (1) or rain (0) by 2-m temperature.

    Non-finite cells become 0 (treated as no-snow so they don't paint
    the snow palette over decode glitches).
    """
    safe = np.where(np.isfinite(t2m_celsius), t2m_celsius, threshold + 100.0)
    return (safe < threshold).astype(np.uint8)


# ── Subh frame timing helpers ─────────────────────────────────────────

# Subh files publish four 15-minute steps per forecast hour and label
# each step's idx entry with the *absolute* lead time in minutes
# ("75 min fcst" in subh02, not "15 min fcst").  ``wrfsubhf01`` covers
# R+0:15..R+1:00, ``wrfsubhf02`` covers R+1:15..R+2:00, etc.

SUBH_INTERVAL_SECONDS = 15 * 60

_STEP_FCST_RE = re.compile(r"^(\d+) min fcst$")


def lead_seconds_for_step(step_label: str) -> int | None:
    """Parse an idx step label ("NN min fcst") to total lead seconds.

    Returns ``None`` for non-instantaneous steps (e.g. ``"70-75 min ave
    fcst"``) — those aren't REFC entries anyway, but checking keeps the
    parse defensive.
    """
    m = _STEP_FCST_RE.match(step_label)
    if m is None:
        return None
    return int(m.group(1)) * 60


def floor_hour(ts: int) -> int:
    """Floor a Unix timestamp to the start of its UTC hour."""
    return (ts // 3600) * 3600


def latest_published_run(now_ts: int, publish_delay_seconds: int) -> int:
    """Most recent run we'd expect to be available given a publish delay."""
    return floor_hour(now_ts - publish_delay_seconds)


def bracket_subh_leads(lead_seconds: int) -> tuple[int, int, float]:
    """For a desired lead, return ``(L0, L1, alpha)`` where L0 ≤ L < L1.

    Both L0 and L1 are exact subh frame leads (multiples of 900s, ≥ 900s).
    Alpha is the lerp weight: 0 at L0, 1 at L1.

    For ``lead_seconds < 900`` (before the first subh frame), the
    bracket falls back to the previous run's last subh frames; the
    caller (``HRRRGrid._pick_run``) handles that by selecting an
    earlier run and recomputing.
    """
    if lead_seconds < SUBH_INTERVAL_SECONDS:
        # Caller should have rolled to a previous run; signal by returning
        # a clamped bracket at L0=L1=900 so a ``has_data_at`` check fails
        # cleanly without raising.
        return SUBH_INTERVAL_SECONDS, SUBH_INTERVAL_SECONDS, 0.0
    l0 = (lead_seconds // SUBH_INTERVAL_SECONDS) * SUBH_INTERVAL_SECONDS
    l1 = l0 + SUBH_INTERVAL_SECONDS
    alpha = (lead_seconds - l0) / SUBH_INTERVAL_SECONDS
    return l0, l1, alpha


# ── HRRRGrid: the public NWPSource implementation ─────────────────────


class HRRRGrid:
    """NOAA HRRR sub-hourly composite reflectivity, sampled in native LCC.

    Implements the NWPSource Protocol.  Frames are stored at native 3 km
    LCC resolution as uint8 dBZ-encoded arrays keyed by ``(run_unix_ts,
    lead_seconds)``.  Sampling at a query (lat, lon, ts) does:

    1. Pick the freshest run whose subh forecast covers ``ts``.
    2. Lerp between the two bracketing 15-min subh frames in time.
    3. Project the query (lat, lon) into the LCC grid and sample.
    """

    name = "hrrr"

    def __init__(self, cache_dir: Path | None = None):
        # (run_ts, lead_seconds) -> memmap-backed uint8 array on the LCC grid.
        # ``_frames`` holds REFC; ``_snow_masks`` holds the parallel 2-m TMP
        # snow classification (1 = snow, 0 = rain) keyed by the same tuple.
        # Snow masks are populated opportunistically — if TMP:2m is missing
        # for a given step ``get_snow_mask`` falls through to all-False there
        # and the chain dispatcher reaches the next source.
        self._frames: dict[tuple[int, int], np.ndarray] = {}
        self._snow_masks: dict[tuple[int, int], np.ndarray] = {}
        self._client: httpx.AsyncClient | None = None
        self._latest_run_ts: int | None = None
        self._fetch_lock = asyncio.Lock()

        # When ``cache_dir`` is given, store memmap files under
        # ``<cache_dir>/hrrr/`` so they survive process restarts; otherwise
        # use a temp directory that's cleaned up on close.  The on-disk
        # layout is identical in both modes (one file per ``(run, lead)``
        # key) so the eviction logic doesn't need to know the difference.
        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "hrrr"
            self._persistent = True
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_hrrr_"))
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "HRRR memmap directory: %s (persistent=%s)",
            self._memmap_dir,
            self._persistent,
        )

        if self._persistent:
            self._load_cached_frames()

    # ── Memory management ────────────────────────────────────────────

    def _to_memmap(self, name: str, data: np.ndarray) -> np.ndarray:
        """Write ``data`` to disk atomically and return a read-only memmap.

        Atomic write (``.tmp`` → ``os.replace``) means a crash mid-write
        leaves the previous version intact, never a half-written file.
        """
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
        """Memmap-load every previously-cached frame and snow mask from disk.

        Called at startup when ``cache_dir`` is configured.  Frames that
        the next fetch cycle ends up not needing will be evicted on the
        first call to ``_evict_outside_window``; everything still inside
        the active window is served immediately, no cold-start fetch.
        Stale ``.tmp`` artifacts left behind by a crash mid-write are
        removed on the way through.

        Snow masks are loaded in a separate pass so a missing snow file
        for a given ``(run, lead)`` doesn't drop the REFC frame — the
        chain just falls through to the next snow-capable source for
        that bracket.
        """
        # Stale .tmp files from a crashed atomic-write — drop them.
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
                run_ts = int(stem_parts[0][1:])    # strip "r"
                lead_s = int(stem_parts[1][1:])    # strip "l"
            except (ValueError, IndexError):
                continue
            try:
                mm = np.memmap(
                    path,
                    dtype=np.uint8,
                    mode="r",
                    shape=(HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH),
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
                "HRRR: loaded %d cached subh frame(s) from disk", loaded,
            )

        # Second pass: parallel snow masks.  Only kept for (run, lead)
        # keys whose REFC frame is also present — orphan snow masks are
        # removed since they'd never be queried.
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
                # No matching REFC — orphan, drop it.
                path.unlink(missing_ok=True)
                continue
            try:
                mm = np.memmap(
                    path,
                    dtype=np.uint8,
                    mode="r",
                    shape=(HRRR_GRID_HEIGHT, HRRR_GRID_WIDTH),
                )
            except Exception:
                logger.warning("Failed to memmap cached snow %s, removing", path)
                path.unlink(missing_ok=True)
                continue
            self._snow_masks[(run_ts, lead_s)] = mm
            snow_loaded += 1
        if snow_loaded:
            logger.info(
                "HRRR: loaded %d cached snow mask(s) from disk", snow_loaded,
            )

    def __getstate__(self) -> dict:
        """Serialize state for cross-process reload (multi-worker mode).

        The on-disk layout in ``_memmap_dir`` is the canonical state.
        ``__setstate__`` rebuilds the in-memory frame dict by rescanning
        disk via ``_load_cached_frames`` — the per-frame numpy memmaps
        are file-backed so the kernel page cache is shared across all
        processes that map the same files.
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
        return datetime.fromtimestamp(self._latest_run_ts, tz=timezone.utc).isoformat()

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
        l0, l1, _ = bracket_subh_leads(lead)
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

        Mirrors ``sample`` for the parallel 2-m TMP snow mask: same
        run picker, same subh bracket-lerp, then re-binarize at 0.5.
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
        l0, l1, alpha = bracket_subh_leads(lead)
        s0 = self._snow_masks.get((run, l0))
        s1 = self._snow_masks.get((run, l1))
        if s0 is None or s1 is None:
            return np.zeros(lat.shape, dtype=bool)

        if alpha == 0.0:
            grid = s0
        elif alpha == 1.0:
            grid = s1
        else:
            # Lerp the two 0/1 frames and threshold at 0.5 — equivalent
            # to nearest-neighbour selection of the closer subh frame
            # at the half-step, picking the L1 frame from alpha == 0.5
            # onwards.
            lerped = (1.0 - alpha) * s0.astype(np.float32) + alpha * s1.astype(np.float32)
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
        l0, l1, alpha = bracket_subh_leads(lead)
        f0 = self._frames.get((run, l0))
        f1 = self._frames.get((run, l1))
        if f0 is None or f1 is None:
            return np.zeros(lat.shape, dtype=np.uint8)

        # Lerp the two subh frames in float, then sample at fractional grid index.
        if alpha == 0.0:
            grid = f0
        elif alpha == 1.0:
            grid = f1
        else:
            grid = ((1.0 - alpha) * f0.astype(np.float32) + alpha * f1.astype(np.float32) + 0.5).astype(np.uint8)

        return _sample_grid(grid, lat, lon, bilinear=bilinear)

    # ── Run selection ────────────────────────────────────────────────

    def _pick_run(self, timestamp: int) -> int | None:
        """Pick the freshest run whose subh bracket for ``timestamp`` is loaded.

        Walks runs newest-first.  A run R is considered usable iff:

        - ``R + 0:15 ≤ T ≤ R + 18:00`` (T inside the subh forecast horizon)
        - both bracketing subh frames ``(R, L0)`` and ``(R, L1)`` exist in
          the store right now.

        The "bracket loaded" check matters during run rollover: when a
        fresh run starts publishing and we've fetched only some of its
        subh files, ``_pick_run`` would otherwise return the new run even
        for queries whose bracket isn't loaded yet — causing
        ``has_data_at`` to return False and the chain to fall through to
        IFS for those frames.  Falling back to the previous run keeps the
        nowcast loop visually consistent across hour rollovers.
        """
        loaded_runs = sorted({r for (r, _) in self._frames}, reverse=True)
        for run in loaded_runs:
            lead = timestamp - run
            if not (SUBH_INTERVAL_SECONDS <= lead <= 18 * 3600):
                continue
            l0, l1, _ = bracket_subh_leads(lead)
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
        """Refresh the in-memory subh window.

        Called by RadarFetcher each cycle.  Determines all runs whose
        forecasts can cover the active window ``[now − history, now +
        horizon]`` and fetches the relevant subh files from each, so
        every valid time in the window is served by a recent forecast.

        Multiple runs are needed because each HRRR run only forecasts
        forward from its init time: the freshest run can't cover ``ts``
        values earlier than ``run + 15min``.  For radar-history times
        we walk back through hourly runs as needed; ``_pick_run`` then
        chooses the freshest loaded run that has the bracket frames in
        store at sample time.

        Already-loaded frames are skipped.  Frames whose valid time
        falls outside the active window are evicted.
        """
        async with self._fetch_lock:
            if now_ts is None:
                now_ts = int(datetime.now(tz=timezone.utc).timestamp())

            publish_delay = settings.hrrr_publish_delay_minutes * 60
            latest_run_ts = latest_published_run(now_ts, publish_delay)
            if self._latest_run_ts is None or latest_run_ts > self._latest_run_ts:
                self._latest_run_ts = latest_run_ts

            window_start = now_ts - history_seconds
            window_end = now_ts + horizon_seconds

            # Enumerate runs that could cover any ts in the window.  A
            # run R covers valid times in [R + 15min, R + 18h], so the
            # earliest run we'd need is the one whose horizon barely
            # reaches window_start (R + 18h ≈ window_start).  In
            # practice for a 2h history we need the latest run plus 1-2
            # earlier hourly runs.  Cap on the new side at the latest
            # published run.
            earliest_run = floor_hour(window_start - SUBH_INTERVAL_SECONDS)
            runs_to_consider = list(range(earliest_run, latest_run_ts + 1, 3600))
            if not runs_to_consider:
                logger.debug("HRRR fetch: no runs available for window")
                return

            client = await self._get_client()

            total_fetched = 0
            total_failed = 0
            for run_ts in runs_to_consider:
                run_dt = datetime.fromtimestamp(run_ts, tz=timezone.utc)
                # Forecast hours of this run that overlap the active window.
                # Each subh file H covers leads (H-1)*3600+900 .. H*3600.
                # Extend the search by one bracket interval on each side so
                # the L1 frame for queries at the future edge of the window
                # — and the L0 frame at the past edge — are included.
                min_lead = max(
                    SUBH_INTERVAL_SECONDS,
                    window_start - run_ts - SUBH_INTERVAL_SECONDS,
                )
                max_lead = min(
                    18 * 3600,
                    window_end - run_ts + SUBH_INTERVAL_SECONDS,
                )
                if max_lead < min_lead:
                    continue
                min_hour = max(1, -(-min_lead // 3600))
                max_hour = min(18, -(-max_lead // 3600))

                for fh in range(int(min_hour), int(max_hour) + 1):
                    added = await self._fetch_one_subh_file(run_dt, fh, client)
                    if added > 0:
                        total_fetched += added
                        logger.debug(
                            "HRRR: +%d frame(s) from run %sZ fh=%d",
                            added, run_dt.strftime("%Y%m%d%H"), fh,
                        )
                    elif added < 0:
                        total_failed += 1

            self._evict_outside_window(window_start, window_end)

            if total_fetched:
                logger.info(
                    "HRRR: %d subh frame(s) ingested across %d run(s); "
                    "store now holds %d frame(s)",
                    total_fetched, len(runs_to_consider), len(self._frames),
                )
            elif total_failed:
                logger.warning(
                    "HRRR: no frames ingested (%d file(s) failed)",
                    total_failed,
                )

    async def _fetch_one_subh_file(
        self, run: datetime, lead_hour: int, client: httpx.AsyncClient
    ) -> int:
        """Fetch every REFC frame from a single subh file. Returns frames added.

        Also fetches the parallel 2-m TMP message for each step and stores
        the derived snow mask alongside the REFC frame.  TMP failures are
        non-fatal — the REFC frame still lands and ``get_snow_mask`` falls
        through to the next chain source for that bracket.

        Returns -1 to signal a fetch error (idx unreachable, etc.).
        """
        run_ts = int(run.timestamp())
        url = subh_url(run, lead_hour)
        try:
            records = await fetch_idx(url, client)
        except httpx.HTTPError as e:
            logger.warning("HRRR idx fetch failed for %s: %s", url, e)
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
                    "HRRR REFC byte-range fetch failed for %s lead=%ds: %s",
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
                    "HRRR TMP:2m byte-range fetch failed for %s lead=%ds: %s",
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

    def _evict_outside_window(self, window_start: int, window_end: int) -> None:
        """Drop frames whose valid time falls outside the active window.

        Slack is one full bracket interval (15 min) on each side so the
        bracketing L1 frame at the future edge — and the L0 frame at the
        past edge — survive long enough for ``sample`` at the window
        boundary to find both halves of its bracket.  Parallel snow
        masks for the same keys are evicted in lockstep.
        """
        slack = SUBH_INTERVAL_SECONDS
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
            logger.info("HRRR: evicted %d out-of-window subh frame(s)", len(stale))

    # ── Lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        self._frames.clear()
        self._snow_masks.clear()
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        if not self._persistent:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.info("HRRR memmap directory cleaned up")
        else:
            logger.info("HRRR cache retained at %s for warm restart", self._memmap_dir)


# ── Grid sampling ────────────────────────────────────────────────────


def _sample_grid(
    grid: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    bilinear: bool = False,
) -> np.ndarray:
    """Sample a uint8 LCC grid at (lat, lon) points.

    Returns ``np.ndarray`` of dtype uint8, shape ``lat.shape``.  Points
    outside the grid yield 0.
    """
    row_f, col_f = grid_indices(lat, lon)

    if not bilinear:
        row = np.rint(row_f).astype(np.int32)
        col = np.rint(col_f).astype(np.int32)
        in_domain = (
            (row >= 0)
            & (row < HRRR_GRID_HEIGHT)
            & (col >= 0)
            & (col < HRRR_GRID_WIDTH)
        )
        out = np.zeros(lat.shape, dtype=np.uint8)
        if in_domain.any():
            out[in_domain] = grid[row[in_domain], col[in_domain]]
        return out

    # Bilinear path mirrors ECMWFGrid.sample's bilinear branch.
    r0 = np.floor(row_f).astype(np.int32)
    c0 = np.floor(col_f).astype(np.int32)
    r1 = r0 + 1
    c1 = c0 + 1

    in_domain = (
        (r0 >= 0)
        & (r1 < HRRR_GRID_HEIGHT)
        & (c0 >= 0)
        & (c1 < HRRR_GRID_WIDTH)
    )

    r0c = np.clip(r0, 0, HRRR_GRID_HEIGHT - 1)
    r1c = np.clip(r1, 0, HRRR_GRID_HEIGHT - 1)
    c0c = np.clip(c0, 0, HRRR_GRID_WIDTH - 1)
    c1c = np.clip(c1, 0, HRRR_GRID_WIDTH - 1)

    dr = np.clip(row_f - r0, 0.0, 1.0).astype(np.float32)
    dc = np.clip(col_f - c0, 0.0, 1.0).astype(np.float32)

    v00 = grid[r0c, c0c].astype(np.float32)
    v01 = grid[r0c, c1c].astype(np.float32)
    v10 = grid[r1c, c0c].astype(np.float32)
    v11 = grid[r1c, c1c].astype(np.float32)

    # Don't bleed precipitation into adjacent zero cells (matches ECMWF
    # bilinear sampling's clear-sky guard).
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
