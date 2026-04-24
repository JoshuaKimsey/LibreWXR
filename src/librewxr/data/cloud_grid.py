# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import numpy as np
from earthkit.regrid import interpolate
from omfiles import OmFileReader

from librewxr.config import settings

logger = logging.getLogger(__name__)

# Regridded output at 0.1deg resolution (same grid as ECMWFGrid)
PIXEL_SIZE = 0.1
WEST = -180.0
NORTH = 90.0
GRID_WIDTH = 3600
GRID_HEIGHT = 1801

S3_LATEST_PATH = "data_spatial/ecmwf_ifs/latest.json"
CLOUD_VARS = ("cloud_cover_high", "cloud_cover_mid", "cloud_cover_low")


class CloudGrid:
    """ECMWF IFS cloud cover grid for satellite-like visualization.

    Fetches cloud_cover_high, cloud_cover_mid, and cloud_cover_low from
    Open-Meteo's S3-hosted ECMWF IFS data at native 9km resolution
    (O1280 reduced Gaussian grid, regridded to 0.1deg lat/lon).

    Designed to run in the background so it never blocks radar startup
    or the main fetch cycle.  Uses the same S3 bucket and file format
    as ECMWFGrid but reads different variables.

    When ``cache_dir`` is configured, processed grids are persisted to
    disk as memory-mapped files.  On startup, cached data loads instantly
    so satellite tiles are available without waiting for S3 downloads.

    Data attribution: ECMWF IFS, provided by Open-Meteo.com (CC-BY-4.0)
    """

    def __init__(self):
        self._timesteps: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._sorted_timestamps: list[int] = []
        self._reference_time: str | None = None
        self._fs: fsspec.AbstractFileSystem | None = None
        self._cache = None  # CloudGridCache | None

        # Initialize disk cache and eagerly load existing data
        if settings.cache_dir:
            from librewxr.data.cloud_cache import CloudGridCache

            self._cache = CloudGridCache(Path(settings.cache_dir))
            ref_time, cached = self._cache.load_all()
            if cached:
                self._timesteps = cached
                self._sorted_timestamps = sorted(cached.keys())
                self._reference_time = ref_time
                logger.info(
                    "Cloud cache: loaded %d timesteps from disk (%s)",
                    len(cached),
                    ", ".join(
                        datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%MZ")
                        for ts in self._sorted_timestamps
                    ),
                )

    @property
    def timestamps(self) -> list[int]:
        return list(self._sorted_timestamps)

    @property
    def reference_time(self) -> str | None:
        return self._reference_time

    @property
    def timestep_count(self) -> int:
        return len(self._timesteps)

    @property
    def loaded(self) -> bool:
        return len(self._timesteps) > 0

    def _get_fs(self) -> fsspec.AbstractFileSystem:
        if self._fs is None:
            self._fs = fsspec.filesystem(
                "s3", anon=True,
                client_kwargs={"region_name": settings.ecmwf_s3_region},
            )
        return self._fs

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

    async def fetch(self) -> bool:
        """Fetch cloud cover data from S3 in a background thread."""
        try:
            return await asyncio.to_thread(self._fetch_sync)
        except Exception:
            logger.exception("Error fetching cloud cover data")
            return False

    def _fetch_sync(self) -> bool:
        fs = self._get_fs()
        bucket = settings.ecmwf_s3_bucket

        latest_raw = fs.cat(f"{bucket}/{S3_LATEST_PATH}")
        latest = json.loads(latest_raw)

        if not latest.get("completed", False):
            logger.warning("ECMWF IFS not complete, skipping cloud fetch")
            return False

        ref_time = latest["reference_time"]
        valid_times = latest.get("valid_times", [])
        variables = latest.get("variables", [])

        missing = [v for v in CLOUD_VARS if v not in variables]
        if missing:
            logger.warning("ECMWF IFS missing cloud variables: %s", missing)
            return False

        if not valid_times:
            return False

        max_ts = settings.satellite_max_frames
        vt_to_fetch = self._select_valid_times(valid_times, max_ts)

        ref_dt = datetime.fromisoformat(ref_time.replace("Z", "+00:00"))
        run_prefix = (
            f"{bucket}/{settings.ecmwf_s3_prefix}"
            f"/{ref_dt.year}/{ref_dt.month:02d}/{ref_dt.day:02d}"
            f"/{ref_dt.hour:02d}{ref_dt.minute:02d}Z"
        )

        # Convert valid_times to unix timestamps and check cache
        new_timesteps: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        vt_to_download: list[tuple[str, int]] = []  # (vt_string, unix_ts)

        for vt in vt_to_fetch:
            unix_ts = self._vt_to_unix(vt)
            if self._cache is not None and self._cache.has(unix_ts):
                cached = self._cache.read(unix_ts)
                if cached is not None:
                    new_timesteps[unix_ts] = cached
                    continue
            vt_to_download.append((vt, unix_ts))

        total = len(vt_to_fetch)
        cached_count = len(new_timesteps)
        download_count = len(vt_to_download)

        if download_count == 0:
            logger.info(
                "Cloud cover: all %d timesteps cached, no downloads needed", total
            )
        else:
            logger.info(
                "Cloud cover: %d total, %d cached, %d to download",
                total, cached_count, download_count,
            )

        # Download only uncached timesteps
        for i, (vt, unix_ts) in enumerate(vt_to_download):
            try:
                high, mid, low = self._fetch_one_timestep(fs, run_prefix, vt)

                if self._cache is not None:
                    self._cache.write(unix_ts, high, mid, low)
                    # Re-read as memmap so we serve from page cache
                    cached = self._cache.read(unix_ts)
                    if cached is not None:
                        new_timesteps[unix_ts] = cached
                    else:
                        new_timesteps[unix_ts] = (high, mid, low)
                else:
                    new_timesteps[unix_ts] = (high, mid, low)

                logger.info(
                    "Cloud cover: %d/%d downloaded (%s)",
                    i + 1, download_count, vt,
                )
            except Exception:
                logger.warning(
                    "Failed to fetch cloud timestep %s (%d/%d)",
                    vt, i + 1, download_count, exc_info=True,
                )

        # Backfill from cache: if the current model run doesn't cover
        # enough past hours, pull older timestamps from disk (previous runs)
        if self._cache is not None and len(new_timesteps) < max_ts:
            all_cached = self._cache.get_cached_timestamps()
            earliest = min(new_timesteps.keys()) if new_timesteps else float("inf")
            for ts in sorted(all_cached, reverse=True):
                if len(new_timesteps) >= max_ts:
                    break
                if ts not in new_timesteps and ts < earliest:
                    cached = self._cache.read(ts)
                    if cached is not None:
                        new_timesteps[ts] = cached
                        logger.info(
                            "Cloud cover: backfilled %s from cache (previous run)",
                            datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                                "%Y-%m-%dT%H:%MZ"
                            ),
                        )

        if not new_timesteps:
            logger.warning("No cloud timesteps fetched successfully")
            return False

        # Save metadata and clean up old files
        if self._cache is not None:
            self._cache.save_metadata(ref_time, sorted(new_timesteps.keys()))
            self._cache.cleanup(list(new_timesteps.keys()))

        self._timesteps = new_timesteps
        self._sorted_timestamps = sorted(new_timesteps.keys())
        self._reference_time = ref_time

        logger.info(
            "Cloud cover ready: %d timesteps (%s)",
            len(new_timesteps),
            ", ".join(
                datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%MZ")
                for ts in self._sorted_timestamps
            ),
        )
        return True

    @staticmethod
    def _vt_to_unix(vt: str) -> int:
        """Convert a valid_time ISO string to a Unix timestamp."""
        vt_dt = datetime.fromisoformat(vt.replace("Z", "+00:00"))
        if vt_dt.tzinfo is None:
            vt_dt = vt_dt.replace(tzinfo=timezone.utc)
        return int(vt_dt.timestamp())

    @staticmethod
    def _select_valid_times(valid_times: list[str], max_ts: int) -> list[str]:
        """Select up to ``max_ts`` valid times, capped at the current hour.

        Unlike ECMWF precipitation (which needs future hours for nowcast
        blending), satellite imagery should only show past/present cloud
        positions.  Future forecast hours are excluded — the backfill
        logic in ``_fetch_sync`` fills any remaining slots from cached
        data of previous model runs.
        """
        now_ts = int(datetime.now(timezone.utc).timestamp())
        vt_unix = []
        for vt in valid_times:
            vt_dt = datetime.fromisoformat(vt.replace("Z", "+00:00"))
            if vt_dt.tzinfo is None:
                vt_dt = vt_dt.replace(tzinfo=timezone.utc)
            vt_unix.append(int(vt_dt.timestamp()))

        # Only keep valid times up to the current hour (no future forecast)
        past_vts = [vt for vt, ts in zip(valid_times, vt_unix) if ts <= now_ts]
        if not past_vts:
            # All valid times are in the future — take just the first one
            return valid_times[:1]

        # Take up to max_ts most recent past timestamps
        if len(past_vts) <= max_ts:
            return past_vts
        return past_vts[-max_ts:]

    def _fetch_one_timestep(
        self,
        fs: fsspec.AbstractFileSystem,
        run_prefix: str,
        vt: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vt_clean = vt.replace("Z", "").replace(":", "")
        om_path = f"{run_prefix}/{vt_clean}.om"

        # Download the entire .om file to a temp file, then read locally.
        # Cloud cover variables are stored at byte offsets that cause
        # pathologically slow random-access via fsspec Range requests
        # (precipitation reads fine via fsspec, but cloud vars hang
        # indefinitely).  A bulk download (~35s for 113 MB) followed by
        # instant local reads is dramatically faster.
        with tempfile.NamedTemporaryFile(suffix=".om") as tmp:
            t0 = time.monotonic()
            fs.get(om_path, tmp.name)
            logger.info("Cloud .om download: %s (%.1fs)", vt, time.monotonic() - t0)
            reader = OmFileReader.from_path(tmp.name)
            try:
                layers = []
                for var_name in CLOUD_VARS:
                    var = reader.get_child_by_name(var_name)
                    raw = var[:].flatten().astype(np.float32)
                    var.close()
                    layers.append(raw)
            finally:
                reader.close()

        results = []
        for raw in layers:
            grid = interpolate(
                raw,
                in_grid={"grid": "O1280"},
                out_grid={"grid": [PIXEL_SIZE, PIXEL_SIZE]},
                method="linear",
            )
            grid = np.roll(grid, GRID_WIDTH // 2, axis=1)
            results.append(np.clip(grid, 0, 100).astype(np.uint8))

        return results[0], results[1], results[2]

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (high, mid, low) cloud cover percentages for the given coordinates."""
        ts = self._nearest_timestamp(timestamp)
        if ts is None:
            z = np.zeros(lat.shape, dtype=np.uint8)
            return z, z.copy(), z.copy()

        high, mid, low = self._timesteps[ts]

        row = ((NORTH - lat) / PIXEL_SIZE).astype(np.int32)
        col = ((lon - WEST) / PIXEL_SIZE).astype(np.int32)
        row = np.clip(row, 0, GRID_HEIGHT - 1)
        col = np.clip(col, 0, GRID_WIDTH - 1)

        return high[row, col], mid[row, col], low[row, col]

    async def close(self) -> None:
        self._timesteps.clear()
        self._sorted_timestamps.clear()
        self._fs = None
