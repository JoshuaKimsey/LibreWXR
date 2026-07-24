# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""NOAA GMGSI (Global Mosaic of Geostationary Satellite Imagery) source.

GMGSI is a hourly pre-composited global mosaic that NESDIS builds from
GOES-East + GOES-West + Meteosat-9 + Meteosat-10 + Himawari-9.  It ships
on a regular equirectangular lat/lon grid (3000 yc x 4999 xc at 0.0722°
between ±72.74° lat), so we get all the seam blending and reprojection
for free — there is nothing geostationary about the data once it lands.

Each channel (LW, VIS, …) is a separate S3 product path and a separate
``GMGSISource`` subclass:

    s3://noaa-gmgsi-pds/{product}/{YYYY}/{MM}/{DD}/{HH}/
        GLOBCOMPLIR_v3r0_blend_s{...}_e{...}_c{...}.nc

One file per hour per channel, ~7.5 MB compressed.  See the project
plan at ``docs/satellite-implementation-plan.md`` for the channel
reference, encoding direction analysis, and the per-phase scope.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar

import fsspec
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Grid constants.  Verified against a live LW file 2026-05-23 and
# documented in docs/satellite-implementation-plan.md.  Re-verified
# 2026-07-24 — NESDIS shaved one column (xc 5000 -> 4999) with no
# change to the lat/lon bounding box, so coverage is unchanged and
# LAT_VEC / LON_VEC still sample the same geographic points.
#
# An incoming frame whose height doesn't match GRID_HEIGHT, or whose
# width is off by more than +-1 from GRID_WIDTH, is rejected rather
# than silently mis-rendered — if NESDIS ever re-grids GMGSI, we want
# loud failure on first ingest, not subtle pixel drift.  A +-1 column
# wobble is tolerated by padding/cropping to the canonical width so a
# future single-column twiddle doesn't take the satellite layer down
# again; height stays strict (lat re-grids are never benign).
GRID_HEIGHT = 3000
GRID_WIDTH = 4999
GRID_SHAPE = (GRID_HEIGHT, GRID_WIDTH)
GRID_DTYPE = np.uint8

# On-disk cache layout version.  Bumped to "v2" when the grid width
# dropped (5000 -> 4999) so stale 5000-wide memmaps from prior runs
# are never re-opened at the new shape — numpy would silently
# truncate the oversized file and shear each row by one column,
# producing a subtly broken render rather than an error.  Old
# gmgsi/{channel}/ dirs are ignored on upgrade; space is reclaimed
# via `docker compose run --rm clear-cache`.
CACHE_LAYOUT_VERSION = "v2"

# Coordinate vectors — also stored verbatim from the upstream file's
# geospatial_lat_max / lat_min / lon_min / lon_max attrs.  These are
# the same for every channel because GMGSI uses one global grid.
LAT_MAX = 72.7154
LAT_MIN = -72.7368
LON_MIN = -179.9284
LON_MAX = 179.9996

# GMGSI rows are spaced uniformly in the Mercator y coordinate
# ``y = atanh(sin(lat))``, NOT uniformly in latitude.  Verified
# empirically against the live NetCDF: the local row-step in latitude
# is ~0.021°/row at row 0 (lat ≈ 72.7°), ~0.038°/row at row 500
# (lat ≈ 58°), ~0.060°/row at row 1000 (lat ≈ 34°), and ~0.072°/row
# at the equator — cos-of-lat compressed exactly the way Mercator
# prescribes.  This caught us once already: an earlier implementation
# assumed equirectangular spacing and the resulting renders shifted
# every land mass ~600 miles south near mid-latitudes.  Treat ``LAT_VEC``
# as authoritative and never assume row count alone gives a latitude.
_Y_MAX = float(np.arctanh(np.sin(np.deg2rad(LAT_MAX))))
_Y_MIN = float(np.arctanh(np.sin(np.deg2rad(LAT_MIN))))
_Y_STEP = (_Y_MAX - _Y_MIN) / (GRID_HEIGHT - 1)
_Y_VEC = np.linspace(_Y_MAX, _Y_MIN, GRID_HEIGHT, dtype=np.float64)
LAT_VEC = np.rad2deg(np.arcsin(np.tanh(_Y_VEC))).astype(np.float32)

# Longitude is plate carrée (linear) — verified by walking columns in
# the live file.  Step is exactly ~0.072°/col across the full disk.
LON_VEC = np.linspace(LON_MIN, LON_MAX, GRID_WIDTH, dtype=np.float32)

# S3 bucket — anonymous, NOAA Open Data Dissemination Program.
S3_BUCKET = "noaa-gmgsi-pds"

# Filename prefix shared by every GMGSI product (LW, VIS, WV, SW).
# The product path under the bucket changes per channel; the leading
# filename token does not.
_FILENAME_PREFIX = "GLOBCOMP"


class GMGSISource:
    """Base class for one GMGSI channel.

    Subclasses pin ``channel`` and ``s3_product_path``; everything else
    is shared.  The class is both fetcher and store: ``fetch()`` walks
    the recent S3 window, downloads + decodes new hourly files, and
    drops them into the in-memory ``_frames`` dict (mirrored to disk
    under ``cache_dir / "gmgsi" / channel`` so render workers can
    memmap them via the cross-worker snapshot).
    """

    # Subclasses set these.
    channel: ClassVar[str]
    s3_product_path: ClassVar[str]
    friendly_name: ClassVar[str]

    # Filename anchor token unique to this channel's product (e.g.
    # ``GLOBCOMPLIR`` for longwave IR).  Used to filter listings within
    # an hour directory; bare ``GLOBCOMP`` would over-match if NESDIS
    # ever co-locates multiple products in one bucket prefix.
    s3_filename_prefix: ClassVar[str]

    def __init__(self, cache_dir: Path | None = None, max_frames: int = 12) -> None:
        self.name = self.friendly_name
        self._frames: dict[int, np.ndarray] = {}
        self._sorted_timestamps: list[int] = []
        self._fs: fsspec.AbstractFileSystem | None = None
        self._max_frames = max_frames

        resolved_cache_root = cache_dir
        self._cache_root: Path | None = (
            Path(resolved_cache_root) if resolved_cache_root else None
        )
        self._channel_cache_dir: Path | None = (
            self._cache_root / "gmgsi" / CACHE_LAYOUT_VERSION / self.channel
            if self._cache_root else None
        )
        if self._channel_cache_dir is not None:
            self._channel_cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cached_frames()

    # ── Public state ──

    @property
    def timestamps(self) -> list[int]:
        return list(self._sorted_timestamps)

    @property
    def loaded(self) -> bool:
        return bool(self._sorted_timestamps)

    @property
    def data_bytes(self) -> int:
        """Total bytes across all loaded frames (for the /health payload)."""
        return sum(arr.nbytes for arr in self._frames.values())

    # ── Fetch / decode ──

    def _get_fs(self) -> fsspec.AbstractFileSystem:
        if self._fs is None:
            self._fs = fsspec.filesystem("s3", anon=True)
        return self._fs

    async def fetch(self) -> bool:
        """Fetch any new hourly slots in the retention window.

        Returns True when at least one new frame was ingested (used by
        the pipeline for downstream invalidation).  All I/O happens in
        a thread so the asyncio loop stays responsive.
        """
        try:
            return await asyncio.to_thread(self._fetch_sync)
        except Exception:
            logger.exception("%s: fetch failed", self.friendly_name)
            return False

    def _fetch_sync(self) -> bool:
        fs = self._get_fs()
        now = datetime.now(timezone.utc)
        # Walk every hour bucket in the retention window backwards.
        # Earliest first when we ingest so the sorted_timestamps list
        # builds monotonically — listing is cheap (LIST on one hour
        # directory returns one file).
        window_start = now - timedelta(hours=self._max_frames)
        keys = self._list_recent_keys(fs, window_start, now)
        if not keys:
            logger.warning("%s: no S3 keys in retention window", self.friendly_name)
            return False

        new_count = 0
        for unix_ts, s3_key in keys:
            if unix_ts in self._frames:
                continue
            arr = self._download_and_decode(fs, s3_key)
            if arr is None:
                continue
            self._frames[unix_ts] = arr
            new_count += 1
            if self._channel_cache_dir is not None:
                self._write_cache(unix_ts, arr)

        # Trim retention.
        self._sorted_timestamps = sorted(self._frames)
        while len(self._sorted_timestamps) > self._max_frames:
            oldest = self._sorted_timestamps.pop(0)
            self._frames.pop(oldest, None)
            if self._channel_cache_dir is not None:
                cache_path = self._cache_path_for(oldest)
                cache_path.unlink(missing_ok=True)

        if new_count:
            logger.info(
                "%s: ingested %d new frame(s); store now holds %d (%s)",
                self.friendly_name, new_count, len(self._sorted_timestamps),
                ", ".join(
                    datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%MZ")
                    for ts in self._sorted_timestamps[-3:]
                ),
            )
        return new_count > 0

    def _list_recent_keys(
        self,
        fs: fsspec.AbstractFileSystem,
        window_start: datetime,
        window_end: datetime,
    ) -> list[tuple[int, str]]:
        """List one S3 key per hour in the window.

        GMGSI layout puts exactly one file under each
        ``{product}/{YYYY}/{MM}/{DD}/{HH}/`` directory.  We walk
        hour-by-hour to keep each LIST call narrow.  Returns
        ``[(unix_ts, key), …]`` sorted ascending by timestamp.
        """
        results: list[tuple[int, str]] = []
        cursor = window_start.replace(minute=0, second=0, microsecond=0)
        while cursor <= window_end:
            prefix = (
                f"{S3_BUCKET}/{self.s3_product_path}/"
                f"{cursor.year:04d}/{cursor.month:02d}/{cursor.day:02d}/"
                f"{cursor.hour:02d}/"
            )
            try:
                entries = fs.ls(prefix, detail=False)
            except FileNotFoundError:
                entries = []
            except Exception:
                logger.exception(
                    "%s: failed to list %s", self.friendly_name, prefix,
                )
                entries = []
            for entry in entries:
                name = entry.rsplit("/", 1)[-1]
                if not name.startswith(self.s3_filename_prefix):
                    continue
                unix_ts = self._parse_start_timestamp(name)
                if unix_ts is None:
                    continue
                results.append((unix_ts, entry))
                break  # one file per hour — stop after the first match
            cursor += timedelta(hours=1)
        return sorted(set(results))

    @staticmethod
    def _parse_start_timestamp(filename: str) -> int | None:
        """Parse the ``_s{YYYYMMDDHHMMSSt}`` token to a Unix timestamp.

        Floors to the hour: GMGSI files cover a 10-min window inside
        their nominal hour but we treat the hour as the canonical slot
        for animation alignment (see open question 1 in the plan).
        """
        try:
            tok = filename.split("_s", 1)[1].split("_", 1)[0]
        except IndexError:
            return None
        if len(tok) < 14:
            return None
        try:
            yr = int(tok[0:4])
            mo = int(tok[4:6])
            da = int(tok[6:8])
            hh = int(tok[8:10])
        except ValueError:
            return None
        dt = datetime(yr, mo, da, hh, 0, 0, tzinfo=timezone.utc)
        return int(dt.timestamp())

    def _download_and_decode(
        self, fs: fsspec.AbstractFileSystem, s3_key: str,
    ) -> np.ndarray | None:
        """Pull one NetCDF file to a temp path, decode, return uint8 grid."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
                fs.get(s3_key, tmp.name)
                return self._decode_netcdf(tmp.name)
        except Exception:
            logger.exception(
                "%s: download/decode failed for %s", self.friendly_name, s3_key,
            )
            return None

    def _decode_netcdf(self, path: str) -> np.ndarray | None:
        """Open the NetCDF, mask via dqf, return shape-checked uint8 grid.

        The data variable carries float32 values that are already
        ``0-255 Brightness Temperature`` per the upstream long_name.
        We mask pixels where ``dqf != 0`` (per CF convention 0=good)
        to 0, clip to [0, 255], and cast to uint8.

        Width is coerced to ``GRID_WIDTH`` when within +-1 column
        (NESDIS shaved the grid by one column on 2026-07-24; the guard
        tolerates a future single-column twiddle without taking the
        satellite layer down).  Anything further off, or any height
        mismatch, is rejected outright — see ``_coerce_to_canonical``.
        """
        ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
        try:
            data = ds["data"].values
            # Strip time axis if present.
            if data.ndim == 3 and data.shape[0] == 1:
                data = data[0]

            dqf = None
            if "dqf" in ds.variables:
                dqf = ds["dqf"].values
                if dqf.ndim == 3 and dqf.shape[0] == 1:
                    dqf = dqf[0]

            data = self._coerce_to_canonical(data)
            if data is None:
                return None
            if dqf is not None:
                dqf = self._coerce_to_canonical(dqf)
                if dqf is not None and dqf.shape == data.shape:
                    data = np.where(dqf == 0, data, 0.0)

            # Treat NaN / out-of-range as no-data sentinel (0).
            data = np.where(np.isfinite(data), data, 0.0)
            data = np.clip(data, 0, 255).astype(GRID_DTYPE)
            return data
        finally:
            ds.close()

    @staticmethod
    def _coerce_to_canonical(arr: np.ndarray) -> np.ndarray | None:
        """Coerce a decoded GMGSI plane to the canonical ``GRID_SHAPE``.

        Height must always equal ``GRID_HEIGHT`` — a latitude re-grid is
        never benign (the Mercator-spaced ``LAT_VEC`` assumes exact row
        count) so it stays loud-fail.  Width is allowed to wobble by one
        column in either direction: pads narrow frames by repeating the
        last column (edge mode — keeps the right-hand disk edge intact
        rather than zero-filling) and crops wide frames by dropping the
        extra trailing column(s).  Anything further off returns ``None``
        so the caller rejects the frame rather than mis-rendering it.
        """
        if arr.ndim != 2 or arr.shape[0] != GRID_HEIGHT:
            logger.warning(
                "GMGSI: height mismatch %s, want %s — rejecting",
                arr.shape, GRID_HEIGHT,
            )
            return None
        width = arr.shape[1]
        if width == GRID_WIDTH:
            return arr
        if width == GRID_WIDTH - 1:
            return np.pad(arr, ((0, 0), (0, 1)), mode="edge")
        if width == GRID_WIDTH + 1:
            return arr[:, :GRID_WIDTH]
        logger.warning(
            "GMGSI: width %s out of tolerance (want %s +-1) — rejecting",
            width, GRID_WIDTH,
        )
        return None

    # ── Cache (disk persistence + cross-worker snapshot) ──

    def _cache_path_for(self, unix_ts: int) -> Path:
        assert self._channel_cache_dir is not None
        return self._channel_cache_dir / f"frame_{unix_ts}.dat"

    def _write_cache(self, unix_ts: int, arr: np.ndarray) -> None:
        final = self._cache_path_for(unix_ts)
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(tmp, dtype=GRID_DTYPE, mode="w+", shape=GRID_SHAPE)
        mm[:] = arr
        mm.flush()
        del mm
        os.replace(tmp, final)

    def _read_cache(self, unix_ts: int) -> np.ndarray | None:
        path = self._cache_path_for(unix_ts)
        if not path.exists():
            return None
        try:
            return np.memmap(path, dtype=GRID_DTYPE, mode="r", shape=GRID_SHAPE)
        except Exception:
            logger.warning(
                "%s: failed to memmap %s, removing", self.friendly_name, path,
            )
            path.unlink(missing_ok=True)
            return None

    def _load_cached_frames(self) -> None:
        """Populate ``_frames`` from any existing on-disk frames.

        Lets the pipeline restart and serve immediately from prior-cycle
        data without waiting for the first fetch; also how render workers
        materialize the store after ``__setstate__`` re-creates the
        instance shape but not the in-memory data.
        """
        assert self._channel_cache_dir is not None
        for entry in self._channel_cache_dir.glob("frame_*.dat"):
            try:
                unix_ts = int(entry.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            arr = self._read_cache(unix_ts)
            if arr is not None:
                self._frames[unix_ts] = arr
        self._sorted_timestamps = sorted(self._frames)

    # ── Sampling (renderer hook) ──

    def _nearest_timestamp(self, timestamp: int | None) -> int | None:
        if not self._sorted_timestamps:
            return None
        if timestamp is None:
            return self._sorted_timestamps[-1]
        ts_list = self._sorted_timestamps
        idx = np.searchsorted(ts_list, timestamp)
        if idx == 0:
            return ts_list[0]
        if idx >= len(ts_list):
            return ts_list[-1]
        before = ts_list[idx - 1]
        after = ts_list[idx]
        return before if timestamp - before <= after - timestamp else after

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> np.ndarray:
        """Sample encoded uint8 values at the given lat/lon points.

        Nearest-neighbour for Phase 1 — bilinear is a Phase 4 polish.
        Returns 0 (no data) outside the global ±72.74° latitude band.
        Always returns a uint8 array shaped like ``lat``/``lon``.
        """
        out = np.zeros(lat.shape, dtype=GRID_DTYPE)
        ts = self._nearest_timestamp(timestamp)
        if ts is None:
            return out

        grid = self._frames[ts]

        # Mercator row inversion: rows are uniform in y=atanh(sin(lat)),
        # not in lat.  Clip sin(lat) shy of ±1 so atanh stays finite at
        # the geometric poles even though the in_bounds mask below will
        # zero out anything past LAT_MAX/LAT_MIN anyway.
        sin_lat = np.clip(
            np.sin(np.deg2rad(lat.astype(np.float64))), -0.9999, 0.9999,
        )
        y_query = np.arctanh(sin_lat)
        row = ((_Y_MAX - y_query) / _Y_STEP).astype(np.int32)

        lon_step = (LON_MAX - LON_MIN) / (GRID_WIDTH - 1)
        col = ((lon - LON_MIN) / lon_step).astype(np.int32)

        # Mask points outside the global band — leave them at the
        # zero sentinel rather than wrapping or clamping silently.
        in_bounds = (
            (lat <= LAT_MAX) & (lat >= LAT_MIN)
            & (lon >= LON_MIN) & (lon <= LON_MAX)
        )
        row = np.clip(row, 0, GRID_HEIGHT - 1)
        col = np.clip(col, 0, GRID_WIDTH - 1)

        sampled = grid[row, col]
        out = np.where(in_bounds, sampled, 0).astype(GRID_DTYPE)
        return out

    # ── Lifecycle ──

    async def close(self) -> None:
        # No long-lived connections — fsspec's S3 client is recreated
        # each fetch.  Provided to satisfy the SatelliteSource protocol.
        return None

    # ── Cross-process snapshot (pickle for multi-worker) ──

    def __getstate__(self) -> dict:
        """Serialize for cross-worker reload via master_state.

        Render workers don't repeat the S3 fetch — they re-open the
        cached memmaps that the pipeline already wrote to disk.  So
        the snapshot only carries the cache root, channel, and a list
        of known timestamps; ``__setstate__`` re-memmaps each one.
        """
        return {
            "cache_root": str(self._cache_root) if self._cache_root else None,
            "channel": self.channel,
            "timestamps": list(self._sorted_timestamps),
            "max_frames": self._max_frames,
        }

    def __setstate__(self, state: dict) -> None:
        cache_root = state.get("cache_root")
        self._cache_root = Path(cache_root) if cache_root else None
        self._max_frames = state.get("max_frames", 12)
        self._frames = {}
        self._sorted_timestamps = []
        self._fs = None
        self.name = self.friendly_name

        if self._cache_root is None:
            self._channel_cache_dir = None
            return
        self._channel_cache_dir = (
            self._cache_root / "gmgsi" / CACHE_LAYOUT_VERSION / self.channel
        )
        # The directory may not exist yet on render-worker cold start
        # if pipeline hasn't created it — handle gracefully.
        if not self._channel_cache_dir.exists():
            return

        for unix_ts in state.get("timestamps", []):
            arr = self._read_cache(unix_ts)
            if arr is not None:
                self._frames[unix_ts] = arr
        self._sorted_timestamps = sorted(self._frames)


# ── Concrete channel subclasses ──


class GMGSILWSource(GMGSISource):
    """GMGSI Longwave Infrared (~12 µm).

    The "IR window" channel — cold (high cloud tops) renders bright,
    warm (ground / ocean) renders dim.  Works 24/7.  Primary baseline
    for the satellite composite; VIS is overlaid on top during the day.
    """

    channel: ClassVar[str] = "LW"
    s3_product_path: ClassVar[str] = "GMGSI_LW"
    friendly_name: ClassVar[str] = "GMGSI LW"
    s3_filename_prefix: ClassVar[str] = "GLOBCOMPLIR"


class GMGSIVISSource(GMGSISource):
    """GMGSI Visible (~0.6 µm).

    Reflected sunlight — bright clouds and bright continents on the
    sunlit side, encoded value 0 on the night side because there is
    no sun to reflect.  The renderer uses the encoded value directly
    as both luminance and alpha, so the day/night terminator emerges
    naturally from the underlying reflectance field.
    """

    channel: ClassVar[str] = "VIS"
    s3_product_path: ClassVar[str] = "GMGSI_VIS"
    friendly_name: ClassVar[str] = "GMGSI VIS"
    s3_filename_prefix: ClassVar[str] = "GLOBCOMPVIS"
