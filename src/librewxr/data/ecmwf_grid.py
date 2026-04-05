# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import json
import logging
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

    Provides:
    - Pseudo-reflectivity derived from precipitation rate via Z-R relationship
    - Snow/rain classification from snowfall vs total precipitation ratio

    Data attribution: ECMWF IFS, provided by Open-Meteo.com (CC-BY-4.0)
    """

    def __init__(self):
        self._precip_dbz: np.ndarray | None = None  # uint8, (1801, 3600)
        self._snow_mask: np.ndarray | None = None    # bool, (1801, 3600)
        self._reference_time: str | None = None
        self._fs: fsspec.AbstractFileSystem | None = None

    @property
    def data(self) -> np.ndarray | None:
        """The precipitation dBZ grid, or None if not yet loaded."""
        return self._precip_dbz

    @property
    def reference_time(self) -> str | None:
        return self._reference_time

    def _get_fs(self) -> fsspec.AbstractFileSystem:
        if self._fs is None:
            self._fs = fsspec.filesystem(
                "s3", anon=True,
                client_kwargs={"region_name": settings.ecmwf_s3_region},
            )
        return self._fs

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

        # Pick the first forecast hour (T+1) for current conditions
        # Index 0 is the analysis (T+0) which has no accumulated precip
        if len(valid_times) < 2:
            logger.warning("ECMWF IFS has fewer than 2 valid times")
            return False

        vt = valid_times[1]

        # Build the S3 path
        ref_dt = datetime.fromisoformat(ref_time.replace("Z", "+00:00"))
        # valid_times format: "2026-04-05T07:00Z" -> filename "2026-04-05T0700"
        vt_clean = vt.replace("Z", "").replace(":", "")
        om_path = (
            f"{bucket}/{settings.ecmwf_s3_prefix}"
            f"/{ref_dt.year}/{ref_dt.month:02d}/{ref_dt.day:02d}"
            f"/{ref_dt.hour:02d}{ref_dt.minute:02d}Z"
            f"/{vt_clean}.om"
        )

        logger.info("Fetching ECMWF IFS: %s (ref=%s)", vt, ref_time)

        # Read precipitation and snowfall from the OM file
        reader = OmFileReader.from_fsspec(fs, om_path)
        try:
            precip_var = reader.get_child_by_name("precipitation")
            precip_raw = precip_var[:].flatten().astype(np.float32)
            precip_var.close()

            has_snow = "snowfall_water_equivalent" in variables
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

        # precip_grid shape is (1801, 3600) — 90N to 90S, 0E to 359.9E
        # We need to shift to -180 to 180 (the western half needs to wrap)
        precip_grid = np.roll(precip_grid, GRID_WIDTH // 2, axis=1)
        snow_grid = np.roll(snow_grid, GRID_WIDTH // 2, axis=1)

        # Accumulated precip for this timestep is already the 1-hour total (mm)
        # since it's the first step of the forecast
        rate = np.maximum(precip_grid, 0.0)  # mm/h (1-hour accumulation)

        # Determine snow ratio for classification
        with np.errstate(divide="ignore", invalid="ignore"):
            snow_ratio = np.where(
                rate > 1e-6,
                np.clip(snow_grid / rate, 0.0, 1.0),
                0.0,
            )
        is_snow = snow_ratio > settings.ecmwf_snow_ratio_threshold

        # Apply Z-R relationship: Z = a * R^b
        # Use snow Z-R where snow ratio exceeds threshold
        z_values = np.where(
            is_snow,
            ZR_A_SNOW * np.power(np.maximum(rate, 1e-10), ZR_B_SNOW),
            ZR_A_RAIN * np.power(np.maximum(rate, 1e-10), ZR_B_RAIN),
        )

        # Convert Z to dBZ: dBZ = 10 * log10(Z)
        dbz = np.where(
            rate > 0.01,  # Only compute for non-trivial precipitation
            10.0 * np.log10(np.maximum(z_values, 1e-10)),
            0.0,
        )

        # Encode as uint8 using the same encoding as radar composites:
        # pixel = clamp((dBZ + 32) * 2, 0, 255)
        result = np.clip((dbz + 32.0) * 2.0, 0, 255).astype(np.uint8)
        result[rate <= 0.01] = 0  # Clear sky

        self._precip_dbz = result
        self._snow_mask = is_snow
        self._reference_time = ref_time

        valid_pixels = rate > 0.01
        logger.info(
            "ECMWF IFS updated: ref=%s, %.1f-%.1f dBZ range, "
            "%d pixels with precip, %.1f%% snow",
            ref_time,
            dbz[valid_pixels].min() if valid_pixels.any() else 0,
            dbz[valid_pixels].max() if valid_pixels.any() else 0,
            int(valid_pixels.sum()),
            100.0 * is_snow.sum() / max(1, valid_pixels.sum()),
        )
        return True

    def sample(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Return uint8 dBZ-encoded values for the given lat/lon arrays.

        Uses the same encoding as radar composites (pixel = (dBZ + 32) * 2)
        so values can be fed directly into the color scheme pipeline.
        """
        if self._precip_dbz is None:
            return np.zeros(lat.shape, dtype=np.uint8)

        row = ((NORTH - lat) / PIXEL_SIZE).astype(np.int32)
        col = ((lon - WEST) / PIXEL_SIZE).astype(np.int32)

        row = np.clip(row, 0, GRID_HEIGHT - 1)
        col = np.clip(col, 0, GRID_WIDTH - 1)

        return self._precip_dbz[row, col]

    def get_snow_mask(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Return boolean mask: True where precipitation is classified as snow.

        Replaces TemperatureGrid.get_freezing_mask() with direct snow
        classification from ECMWF IFS snowfall vs total precipitation.
        """
        if self._snow_mask is None:
            return np.zeros(lat.shape, dtype=bool)

        row = ((NORTH - lat) / PIXEL_SIZE).astype(np.int32)
        col = ((lon - WEST) / PIXEL_SIZE).astype(np.int32)

        row = np.clip(row, 0, GRID_HEIGHT - 1)
        col = np.clip(col, 0, GRID_WIDTH - 1)

        return self._snow_mask[row, col]

    async def close(self) -> None:
        """Clean up resources."""
        self._fs = None
