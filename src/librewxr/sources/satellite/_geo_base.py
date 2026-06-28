# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Shared base for geostationary satellite sources (GOES, Himawari).

``GeoSatSource`` handles everything common to any geostationary imager
that publishes fixed-grid NetCDF files on anonymous S3:

- S3 listing + download via fsspec
- Frame retention ring buffer
- Geostationary projection for ``sample()``
- Disk cache / memmap / cross-worker pickle

Concrete subclasses (``GOESIRSource``, ``HimawariIRSource``, …) pin the
satellite-specific parameters (bucket, product path, band, sat_lon,
sat_height) and override ``_decode_netcdf`` for sensor-specific value
mapping.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from abc import abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import ClassVar

import fsspec
import numpy as np
import xarray as xr

from librewxr.tiles.geostationary import forward as geo_forward

logger = logging.getLogger(__name__)


class GeoSatSource:
    """Abstract base for one channel of a geostationary satellite."""

    # ── Subclass-defined class variables ──

    # Satellite orbital parameters
    sat_lon: ClassVar[float]  # sub-satellite longitude (degrees)
    sat_height: ClassVar[float]  # orbital height above ellipsoid (metres)

    # S3 addressing
    s3_bucket: ClassVar[str]
    s3_product_path: ClassVar[str]  # e.g. "ABI-L2-CMIPC"
    s3_filename_token: ClassVar[str]  # e.g. "CMIPC-M6C13" for filtering listings

    # Display
    friendly_name: ClassVar[str]
    channel: ClassVar[str]  # "IR" or "VIS"

    # Cadence in minutes (5 for GOES CONUS, 10 for Himawari full-disk)
    cadence_minutes: ClassVar[int] = 5

    def __init__(
        self,
        cache_dir: Path | None = None,
        max_frames: int = 36,
    ) -> None:
        self.name = self.friendly_name
        self._frames: dict[int, np.ndarray] = {}
        self._sorted_timestamps: list[int] = []
        self._fs: fsspec.AbstractFileSystem | None = None
        self._max_frames = max_frames

        # Per-frame grid metadata (set on first decode)
        self._x_vec: np.ndarray | None = None  # 1-D scan-angle x coords
        self._y_vec: np.ndarray | None = None  # 1-D scan-angle y coords
        self._grid_height: int = 0
        self._grid_width: int = 0

        self._cache_root: Path | None = (
            Path(cache_dir) if cache_dir else None
        )
        self._channel_cache_dir: Path | None = None
        if self._cache_root is not None:
            self._channel_cache_dir = (
                self._cache_root / self._cache_subdir() / self.channel
            )
            self._channel_cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cached_frames()

    def _cache_subdir(self) -> str:
        """Subdirectory name under the cache root (e.g. ``goes18``)."""
        return self.s3_bucket.replace("-", "_").replace("noaa_", "")

    # ── Public state ──

    @property
    def timestamps(self) -> list[int]:
        return list(self._sorted_timestamps)

    @property
    def loaded(self) -> bool:
        return bool(self._sorted_timestamps)

    @property
    def data_bytes(self) -> int:
        return sum(arr.nbytes for arr in self._frames.values())

    # ── Fetch / decode ──

    def _get_fs(self) -> fsspec.AbstractFileSystem:
        if self._fs is None:
            self._fs = fsspec.filesystem("s3", anon=True)
        return self._fs

    async def fetch(self) -> bool:
        try:
            return await asyncio.to_thread(self._fetch_sync)
        except Exception:
            logger.exception("%s: fetch failed", self.friendly_name)
            return False

    def _fetch_sync(self) -> bool:
        fs = self._get_fs()
        now = datetime.now(timezone.utc)
        window_hours = max(1, (self._max_frames * self.cadence_minutes) // 60 + 1)
        window_start = now - timedelta(hours=window_hours)
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

        self._sorted_timestamps = sorted(self._frames)
        while len(self._sorted_timestamps) > self._max_frames:
            oldest = self._sorted_timestamps.pop(0)
            self._frames.pop(oldest, None)
            if self._channel_cache_dir is not None:
                self._cache_path_for(oldest).unlink(missing_ok=True)

        if new_count:
            logger.info(
                "%s: ingested %d new frame(s); store holds %d",
                self.friendly_name, new_count, len(self._sorted_timestamps),
            )
        return new_count > 0

    def _list_recent_keys(
        self,
        fs: fsspec.AbstractFileSystem,
        window_start: datetime,
        window_end: datetime,
    ) -> list[tuple[int, str]]:
        """List S3 keys for the retention window.

        Subclasses may override to handle different path layouts.
        Default walks GOES-style ``{product}/{year}/{doy}/{hour}/``.
        """
        results: list[tuple[int, str]] = []
        cursor = window_start.replace(minute=0, second=0, microsecond=0)
        while cursor <= window_end:
            doy = cursor.timetuple().tm_yday
            prefix = (
                f"{self.s3_bucket}/{self.s3_product_path}/"
                f"{cursor.year:04d}/{doy:03d}/{cursor.hour:02d}/"
            )
            try:
                entries = fs.ls(prefix, detail=False)
            except FileNotFoundError:
                entries = []
            except Exception:
                logger.exception("%s: failed to list %s", self.friendly_name, prefix)
                entries = []
            for entry in entries:
                name = entry.rsplit("/", 1)[-1]
                if self.s3_filename_token not in name:
                    continue
                unix_ts = self._parse_start_timestamp(name)
                if unix_ts is None:
                    continue
                results.append((unix_ts, entry))
            cursor += timedelta(hours=1)
        results = sorted(set(results))
        return results

    @staticmethod
    def _parse_start_timestamp(filename: str) -> int | None:
        """Parse ``_s{YYYYDDDHHMMSSt}`` token to Unix timestamp.

        GOES/Himawari filenames use day-of-year format, unlike GMGSI's
        ``YYYYMMDDHHMMSS``.  We keep minute-level precision (not floored
        to the hour) since the cadence is 5–10 minutes.
        """
        try:
            tok = filename.split("_s", 1)[1].split("_", 1)[0]
        except IndexError:
            return None
        if len(tok) < 13:
            return None
        try:
            yr = int(tok[0:4])
            doy = int(tok[4:7])
            hh = int(tok[7:9])
            mm = int(tok[9:11])
            ss = int(tok[11:13])
        except ValueError:
            return None
        try:
            dt = datetime(yr, 1, 1, hh, mm, ss, tzinfo=timezone.utc) + timedelta(days=doy - 1)
        except (ValueError, OverflowError):
            return None
        return int(dt.timestamp())

    def _download_and_decode(
        self, fs: fsspec.AbstractFileSystem, s3_key: str,
    ) -> np.ndarray | None:
        try:
            with tempfile.NamedTemporaryFile(suffix=".nc") as tmp:
                fs.get(s3_key, tmp.name)
                return self._decode_netcdf(tmp.name)
        except Exception:
            logger.exception(
                "%s: download/decode failed for %s", self.friendly_name, s3_key,
            )
            return None

    @abstractmethod
    def _decode_netcdf(self, path: str) -> np.ndarray | None:
        """Open NetCDF, decode sensor-specific values, return uint8 grid.

        Must also set ``self._x_vec``, ``self._y_vec``, ``self._grid_height``,
        ``self._grid_width`` on first successful decode.
        """
        ...

    def _init_grid_vectors(self, ds: xr.Dataset) -> None:
        """Extract 1-D scan-angle coordinate vectors from the dataset.

        Only runs once — subsequent frames reuse the stored vectors
        (the fixed-grid coordinates never change for a given product).
        """
        if self._x_vec is not None:
            return
        self._x_vec = ds["x"].values.astype(np.float64)
        self._y_vec = ds["y"].values.astype(np.float64)
        self._grid_width = len(self._x_vec)
        self._grid_height = len(self._y_vec)

    # ── Sampling ──

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

        Uses the geostationary forward projection to convert lat/lon to
        scan angles, then nearest-neighbour lookup into the stored grid.
        Returns 0 (no data) for points not visible to the satellite.
        """
        out = np.zeros(lat.shape, dtype=np.uint8)
        ts = self._nearest_timestamp(timestamp)
        if ts is None:
            return out
        if self._x_vec is None or self._y_vec is None:
            return out

        grid = self._frames[ts]

        # Forward-project lat/lon to scan angles
        x_ang, y_ang = geo_forward(
            lat.astype(np.float64),
            lon.astype(np.float64),
            self.sat_lon,
            self.sat_height,
        )

        # Map scan angles to pixel indices via the stored coordinate vectors
        x_step = (self._x_vec[-1] - self._x_vec[0]) / (self._grid_width - 1)
        y_step = (self._y_vec[0] - self._y_vec[-1]) / (self._grid_height - 1)

        col = ((x_ang - self._x_vec[0]) / x_step).astype(np.int32)
        row = ((self._y_vec[0] - y_ang) / y_step).astype(np.int32)

        visible = ~(np.isnan(x_ang) | np.isnan(y_ang))
        in_bounds = (
            visible
            & (row >= 0) & (row < self._grid_height)
            & (col >= 0) & (col < self._grid_width)
        )

        row_safe = np.clip(row, 0, self._grid_height - 1)
        col_safe = np.clip(col, 0, self._grid_width - 1)

        sampled = grid[row_safe, col_safe]
        out = np.where(in_bounds, sampled, 0).astype(np.uint8)
        return out

    # ── Cache (disk persistence + cross-worker snapshot) ──

    def _cache_path_for(self, unix_ts: int) -> Path:
        assert self._channel_cache_dir is not None
        return self._channel_cache_dir / f"frame_{unix_ts}.dat"

    def _meta_cache_path(self) -> Path:
        assert self._channel_cache_dir is not None
        return self._channel_cache_dir / "grid_meta.npz"

    def _write_cache(self, unix_ts: int, arr: np.ndarray) -> None:
        final = self._cache_path_for(unix_ts)
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(
            tmp, dtype=np.uint8, mode="w+",
            shape=(self._grid_height, self._grid_width),
        )
        mm[:] = arr
        mm.flush()
        del mm
        os.replace(tmp, final)
        # Persist grid vectors alongside frames so render workers can
        # reconstruct sample() without a fetch.
        if self._x_vec is not None:
            meta_path = self._meta_cache_path()
            np.savez_compressed(
                meta_path,
                x_vec=self._x_vec,
                y_vec=self._y_vec,
            )

    def _read_cache(self, unix_ts: int) -> np.ndarray | None:
        path = self._cache_path_for(unix_ts)
        if not path.exists():
            return None
        try:
            return np.memmap(
                path, dtype=np.uint8, mode="r",
                shape=(self._grid_height, self._grid_width),
            )
        except Exception:
            logger.warning(
                "%s: failed to memmap %s, removing", self.friendly_name, path,
            )
            path.unlink(missing_ok=True)
            return None

    def _load_grid_meta(self) -> bool:
        """Load grid vectors from disk cache.  Returns True if successful."""
        meta_path = self._meta_cache_path()
        if not meta_path.exists():
            return False
        try:
            data = np.load(meta_path)
            self._x_vec = data["x_vec"]
            self._y_vec = data["y_vec"]
            self._grid_width = len(self._x_vec)
            self._grid_height = len(self._y_vec)
            return True
        except Exception:
            logger.warning("%s: failed to load grid meta", self.friendly_name)
            return False

    def _load_cached_frames(self) -> None:
        assert self._channel_cache_dir is not None
        if not self._load_grid_meta():
            return
        for entry in self._channel_cache_dir.glob("frame_*.dat"):
            try:
                unix_ts = int(entry.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            arr = self._read_cache(unix_ts)
            if arr is not None:
                self._frames[unix_ts] = arr
        self._sorted_timestamps = sorted(self._frames)

    # ── Lifecycle ──

    async def close(self) -> None:
        return None

    # ── Cross-process snapshot (pickle for multi-worker) ──

    def __getstate__(self) -> dict:
        return {
            "cache_root": str(self._cache_root) if self._cache_root else None,
            "channel": self.channel,
            "timestamps": list(self._sorted_timestamps),
            "max_frames": self._max_frames,
            "bucket": self.s3_bucket,
        }

    def __setstate__(self, state: dict) -> None:
        cache_root = state.get("cache_root")
        self._cache_root = Path(cache_root) if cache_root else None
        self._max_frames = state.get("max_frames", 36)
        self._frames = {}
        self._sorted_timestamps = []
        self._fs = None
        self._x_vec = None
        self._y_vec = None
        self._grid_height = 0
        self._grid_width = 0
        self.name = self.friendly_name

        if self._cache_root is None:
            self._channel_cache_dir = None
            return
        self._channel_cache_dir = (
            self._cache_root / self._cache_subdir() / self.channel
        )
        if not self._channel_cache_dir.exists():
            return
        if not self._load_grid_meta():
            return
        for unix_ts in state.get("timestamps", []):
            arr = self._read_cache(unix_ts)
            if arr is not None:
                self._frames[unix_ts] = arr
        self._sorted_timestamps = sorted(self._frames)
