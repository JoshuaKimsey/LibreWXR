# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import fsspec
import numpy as np
from earthkit.regrid import interpolate
from omfiles import OmFileReader

from librewxr.config import settings

logger = logging.getLogger(__name__)

# Regridded output at 0.1° resolution
PIXEL_SIZE = 0.1
WEST = -180.0
EAST = 180.0
NORTH = 90.0
SOUTH = -90.0
GRID_WIDTH = int((EAST - WEST) / PIXEL_SIZE)    # 3600
GRID_HEIGHT = int((NORTH - SOUTH) / PIXEL_SIZE) + 1  # 1801

# Z-R relationship constants (Marshall-Palmer)
ZR_A_RAIN = 200.0
ZR_B_RAIN = 1.6
ZR_A_SNOW = 2000.0
ZR_B_SNOW = 2.0

# S3 path construction
S3_LATEST_PATH = "data_spatial/ecmwf_ifs/latest.json"


class ECMWFGrid:
    """ECMWF IFS 9km precipitation grid for global fallback coverage.

    Replaces both GFSReflectivityGrid and TemperatureGrid with a single
    data source from Open-Meteo's S3-hosted ECMWF IFS at native 9km
    resolution (O1280 reduced Gaussian grid, regridded to 0.1° lat/lon).

    Stores multiple hourly timesteps so ECMWF data animates across radar
    frames. Each radar frame is matched to the nearest available IFS
    timestep.

    Provides:
    - Pseudo-reflectivity derived from precipitation rate via Z-R relationship
    - Snow/rain classification from snowfall vs total precipitation ratio

    Data attribution: ECMWF IFS, provided by Open-Meteo.com (CC-BY-4.0)
    """

    def __init__(self):
        # dict mapping Unix timestamp -> (precip_dbz uint8, snow_mask bool)
        self._timesteps: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._sorted_timestamps: list[int] = []
        self._reference_time: str | None = None
        self._fs: fsspec.AbstractFileSystem | None = None

    @property
    def data(self) -> np.ndarray | None:
        """The latest precipitation dBZ grid, or None if not yet loaded."""
        if not self._sorted_timestamps:
            return None
        return self._timesteps[self._sorted_timestamps[-1]][0]

    @property
    def reference_time(self) -> str | None:
        return self._reference_time

    @property
    def timestep_count(self) -> int:
        return len(self._timesteps)

    def _get_fs(self) -> fsspec.AbstractFileSystem:
        if self._fs is None:
            self._fs = fsspec.filesystem(
                "s3", anon=True,
                client_kwargs={"region_name": settings.ecmwf_s3_region},
            )
        return self._fs

    def _nearest_timestamp(self, timestamp: int | None) -> int | None:
        """Find the stored timestep closest to the given Unix timestamp."""
        if not self._sorted_timestamps:
            return None
        if timestamp is None:
            return self._sorted_timestamps[-1]

        # Binary search for nearest
        ts_list = self._sorted_timestamps
        idx = np.searchsorted(ts_list, timestamp)
        if idx == 0:
            return ts_list[0]
        if idx >= len(ts_list):
            return ts_list[-1]
        # Check which neighbor is closer
        before = ts_list[idx - 1]
        after = ts_list[idx]
        if timestamp - before <= after - timestamp:
            return before
        return after

    @staticmethod
    def _select_valid_times(valid_times: list[str], max_ts: int) -> list[str]:
        """Pick valid_times that best bracket the current radar window.

        Radar frames span roughly (now - radar_history) to now.  We pick
        ``max_ts`` consecutive IFS hours such that the latest hour is
        the smallest valid_time >= now (i.e. the first hour at or after
        the current time).  This ensures the most recent radar frames
        fall *between* IFS hours rather than off the trailing edge.

        Example: now=06:30, IFS hours=[01..12], max_ts=3
        → first vt >= 06:30 is 07Z (idx 6) → window [05, 06, 07]

        Note: this only ever fetches from a single model run.  If the
        latest run was just released (e.g. now=06:30 and the only run
        available is 06Z with valid_times starting at 07Z), all stored
        IFS hours will be in the future of the radar window and the
        oldest radar frames will all snap to the earliest stored hour
        until the next fetch picks up later valid_times.
        """
        if len(valid_times) <= max_ts:
            return valid_times

        now_ts = int(datetime.now(timezone.utc).timestamp())

        # Parse valid_time strings to Unix timestamps
        vt_unix = []
        for vt in valid_times:
            vt_dt = datetime.fromisoformat(vt.replace("Z", "+00:00"))
            if vt_dt.tzinfo is None:
                vt_dt = vt_dt.replace(tzinfo=timezone.utc)
            vt_unix.append(int(vt_dt.timestamp()))

        # Find the first vt that is at or after "now".  That hour
        # becomes the trailing edge of our window so it brackets the
        # current time rather than ending one hour early.
        anchor_idx = None
        for i, t in enumerate(vt_unix):
            if t >= now_ts:
                anchor_idx = i
                break
        if anchor_idx is None:
            # All valid_times are in the past — take the most recent ones.
            anchor_idx = len(valid_times) - 1

        end = anchor_idx + 1  # exclusive
        start = max(end - max_ts, 0)
        # If we hit the start of the list, shift forward to fill the window.
        end = min(start + max_ts, len(valid_times))

        return valid_times[start:end]

    async def fetch(self) -> bool:
        """Fetch the latest ECMWF IFS precipitation data from S3."""
        try:
            return await asyncio.to_thread(self._fetch_sync)
        except Exception:
            logger.exception("Error fetching ECMWF IFS data")
            return False

    def _fetch_sync(self) -> bool:
        """Synchronous fetch — runs in a thread to avoid blocking the event loop."""
        fs = self._get_fs()
        bucket = settings.ecmwf_s3_bucket

        # Read latest.json to find current model run
        latest_raw = fs.cat(f"{bucket}/{S3_LATEST_PATH}")
        latest = json.loads(latest_raw)

        if not latest.get("completed", False):
            logger.warning("ECMWF IFS model run not yet complete, skipping")
            return False

        ref_time = latest["reference_time"]
        valid_times = latest.get("valid_times", [])
        variables = latest.get("variables", [])

        if "precipitation" not in variables:
            logger.warning("ECMWF IFS data missing precipitation variable")
            return False

        # Select IFS timesteps that overlap the radar frame window.
        # Radar frames span roughly (now - 2h) to now, so we pick the
        # IFS valid_times closest to the current time.  Skip index 0
        # (analysis T+0 with no accumulated precip).
        max_ts = settings.get_ecmwf_max_timesteps()
        if len(valid_times) < 2:
            logger.warning("ECMWF IFS has fewer than 2 valid times")
            return False

        vt_to_fetch = self._select_valid_times(valid_times[1:], max_ts)
        has_snow = "snowfall_water_equivalent" in variables
        ref_dt = datetime.fromisoformat(ref_time.replace("Z", "+00:00"))
        run_prefix = (
            f"{bucket}/{settings.ecmwf_s3_prefix}"
            f"/{ref_dt.year}/{ref_dt.month:02d}/{ref_dt.day:02d}"
            f"/{ref_dt.hour:02d}{ref_dt.minute:02d}Z"
        )

        logger.info(
            "Fetching ECMWF IFS: %d timesteps from %s to %s (ref=%s, max_ts=%d)",
            len(vt_to_fetch), vt_to_fetch[0], vt_to_fetch[-1], ref_time, max_ts,
        )

        new_timesteps: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Fetch timesteps concurrently — each fetch is an independent S3
        # read + regrid, so they parallelize cleanly.  fsspec readers and
        # earthkit interpolate are thread-safe for read-only operations.
        with ThreadPoolExecutor(max_workers=len(vt_to_fetch)) as executor:
            future_to_vt = {
                executor.submit(
                    self._fetch_one_timestep,
                    fs, run_prefix, vt, has_snow, variables,
                ): vt
                for vt in vt_to_fetch
            }
            for future in as_completed(future_to_vt):
                vt = future_to_vt[future]
                try:
                    precip_dbz, snow_mask = future.result()
                    vt_dt = datetime.fromisoformat(vt.replace("Z", "+00:00"))
                    if vt_dt.tzinfo is None:
                        vt_dt = vt_dt.replace(tzinfo=timezone.utc)
                    unix_ts = int(vt_dt.timestamp())
                    new_timesteps[unix_ts] = (precip_dbz, snow_mask)
                except Exception:
                    logger.warning("Failed to fetch ECMWF timestep %s", vt, exc_info=True)

        if not new_timesteps:
            logger.warning("No ECMWF timesteps fetched successfully")
            return False

        # Optionally interpolate between hourly frames to produce 10-min steps
        if settings.ecmwf_interpolation and len(new_timesteps) >= 2:
            from librewxr.data.ecmwf_interpolation import interpolate_timesteps

            new_timesteps = interpolate_timesteps(new_timesteps)

        self._timesteps = new_timesteps
        self._sorted_timestamps = sorted(new_timesteps.keys())
        self._reference_time = ref_time

        logger.info(
            "ECMWF IFS updated: ref=%s, %d timesteps loaded (%s)",
            ref_time,
            len(new_timesteps),
            ", ".join(
                datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%MZ")
                for ts in self._sorted_timestamps
            ),
        )
        return True

    def _fetch_one_timestep(
        self,
        fs: fsspec.AbstractFileSystem,
        run_prefix: str,
        vt: str,
        has_snow: bool,
        variables: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fetch, regrid, and convert a single IFS timestep."""
        vt_clean = vt.replace("Z", "").replace(":", "")
        om_path = f"{run_prefix}/{vt_clean}.om"

        reader = OmFileReader.from_fsspec(fs, om_path)
        try:
            precip_var = reader.get_child_by_name("precipitation")
            precip_raw = precip_var[:].flatten().astype(np.float32)
            precip_var.close()

            if has_snow:
                snow_var = reader.get_child_by_name("snowfall_water_equivalent")
                snow_raw = snow_var[:].flatten().astype(np.float32)
                snow_var.close()
            else:
                snow_raw = np.zeros_like(precip_raw)
        finally:
            reader.close()

        # Regrid from O1280 reduced Gaussian to regular 0.1° lat/lon
        precip_grid = interpolate(
            precip_raw,
            in_grid={"grid": "O1280"},
            out_grid={"grid": [PIXEL_SIZE, PIXEL_SIZE]},
            method="linear",
        )
        if has_snow:
            snow_grid = interpolate(
                snow_raw,
                in_grid={"grid": "O1280"},
                out_grid={"grid": [PIXEL_SIZE, PIXEL_SIZE]},
                method="linear",
            )
        else:
            snow_grid = np.zeros_like(precip_grid)

        # Shift from 0-360 to -180..180
        precip_grid = np.roll(precip_grid, GRID_WIDTH // 2, axis=1)
        snow_grid = np.roll(snow_grid, GRID_WIDTH // 2, axis=1)

        # Accumulated precip for this timestep is the 1-hour total (mm)
        rate = np.maximum(precip_grid, 0.0)

        # Determine snow ratio for classification
        with np.errstate(divide="ignore", invalid="ignore"):
            snow_ratio = np.where(
                rate > 1e-6,
                np.clip(snow_grid / rate, 0.0, 1.0),
                0.0,
            )
        is_snow = snow_ratio > settings.ecmwf_snow_ratio_threshold

        # Apply Z-R relationship: Z = a * R^b
        z_values = np.where(
            is_snow,
            ZR_A_SNOW * np.power(np.maximum(rate, 1e-10), ZR_B_SNOW),
            ZR_A_RAIN * np.power(np.maximum(rate, 1e-10), ZR_B_RAIN),
        )

        # Convert Z to dBZ: dBZ = 10 * log10(Z)
        dbz = np.where(
            rate > 0.01,
            10.0 * np.log10(np.maximum(z_values, 1e-10)),
            0.0,
        )

        # Encode as uint8: pixel = clamp((dBZ + 32) * 2, 0, 255)
        result = np.clip((dbz + 32.0) * 2.0, 0, 255).astype(np.uint8)
        result[rate <= 0.01] = 0

        valid_pixels = rate > 0.01
        logger.debug(
            "  Timestep %s: %.1f-%.1f dBZ, %d precip pixels, %.1f%% snow",
            vt,
            dbz[valid_pixels].min() if valid_pixels.any() else 0,
            dbz[valid_pixels].max() if valid_pixels.any() else 0,
            int(valid_pixels.sum()),
            100.0 * is_snow.sum() / max(1, valid_pixels.sum()),
        )

        return result, is_snow

    def sample(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        timestamp: int | None = None,
        bilinear: bool = False,
    ) -> np.ndarray:
        """Return uint8 dBZ-encoded values for the given lat/lon arrays.

        Uses the same encoding as radar composites (pixel = (dBZ + 32) * 2)
        so values can be fed directly into the color scheme pipeline.

        Args:
            lat: Latitude array in degrees (any shape).
            lon: Longitude array in degrees (same shape as lat).
            timestamp: Unix timestamp to select nearest IFS timestep.
                       If None, uses the latest available timestep.
            bilinear: If True, use bilinear interpolation between source
                      pixels (smoother appearance at high zoom levels).
                      Falls back to nearest-neighbor at any boundary
                      where one of the four neighbors is zero, to avoid
                      ghosting precip into clear-sky pixels.
        """
        ts = self._nearest_timestamp(timestamp)
        if ts is None:
            return np.zeros(lat.shape, dtype=np.uint8)

        precip_dbz = self._timesteps[ts][0]

        if not bilinear:
            row = ((NORTH - lat) / PIXEL_SIZE).astype(np.int32)
            col = ((lon - WEST) / PIXEL_SIZE).astype(np.int32)
            row = np.clip(row, 0, GRID_HEIGHT - 1)
            col = np.clip(col, 0, GRID_WIDTH - 1)
            return precip_dbz[row, col]

        # Bilinear sampling
        row_f = (NORTH - lat) / PIXEL_SIZE
        col_f = (lon - WEST) / PIXEL_SIZE

        r0 = np.floor(row_f).astype(np.int32)
        c0 = np.floor(col_f).astype(np.int32)
        r1 = r0 + 1
        c1 = c0 + 1

        r0 = np.clip(r0, 0, GRID_HEIGHT - 1)
        c0 = np.clip(c0, 0, GRID_WIDTH - 1)
        r1 = np.clip(r1, 0, GRID_HEIGHT - 1)
        c1 = np.clip(c1, 0, GRID_WIDTH - 1)

        dr = np.clip(row_f - np.floor(row_f), 0.0, 1.0).astype(np.float32)
        dc = np.clip(col_f - np.floor(col_f), 0.0, 1.0).astype(np.float32)

        v00 = precip_dbz[r0, c0].astype(np.float32)
        v01 = precip_dbz[r0, c1].astype(np.float32)
        v10 = precip_dbz[r1, c0].astype(np.float32)
        v11 = precip_dbz[r1, c1].astype(np.float32)

        # Don't bleed precipitation into adjacent zero (clear-sky) cells.
        any_zero = (v00 == 0) | (v01 == 0) | (v10 == 0) | (v11 == 0)

        interp = (
            v00 * (1 - dr) * (1 - dc)
            + v01 * (1 - dr) * dc
            + v10 * dr * (1 - dc)
            + v11 * dr * dc
        )
        result = np.where(any_zero, v00, interp)
        return np.clip(result + 0.5, 0, 255).astype(np.uint8)

    def get_snow_mask(
        self, lat: np.ndarray, lon: np.ndarray, timestamp: int | None = None,
    ) -> np.ndarray:
        """Return boolean mask: True where precipitation is classified as snow.

        Replaces TemperatureGrid.get_freezing_mask() with direct snow
        classification from ECMWF IFS snowfall vs total precipitation.

        Args:
            lat: Latitude array in degrees (any shape).
            lon: Longitude array in degrees (same shape as lat).
            timestamp: Unix timestamp to select nearest IFS timestep.
                       If None, uses the latest available timestep.
        """
        ts = self._nearest_timestamp(timestamp)
        if ts is None:
            return np.zeros(lat.shape, dtype=bool)

        snow_mask = self._timesteps[ts][1]

        row = ((NORTH - lat) / PIXEL_SIZE).astype(np.int32)
        col = ((lon - WEST) / PIXEL_SIZE).astype(np.int32)

        row = np.clip(row, 0, GRID_HEIGHT - 1)
        col = np.clip(col, 0, GRID_WIDTH - 1)

        return snow_mask[row, col]

    async def close(self) -> None:
        """Clean up resources."""
        self._fs = None
