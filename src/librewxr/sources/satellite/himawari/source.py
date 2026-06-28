# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Himawari-9 AHI satellite source.

Himawari-9 (140.7°E) covers the Asia-Pacific region with the Advanced
Himawari Imager (AHI), spectrally equivalent to GOES ABI.  NOAA mirrors
the data to anonymous S3 at ``s3://noaa-himawari9/``.

S3 layout::

    s3://noaa-himawari9/AHI-L2-FLDK-ISatSS/{year}/{month}/{day}/{hour}/
      OR_AHI-L2-FLDK-ISatSS-M1C13_v1r0_G09_s{start}_e{end}_c{create}.nc

AHI uses the same geostationary fixed-grid coordinate system as ABI,
with different satellite parameters (140.7°E sub-satellite point).
Band numbering differs: AHI Band 13 (10.4 µm) is the IR window
equivalent to ABI Band 13; AHI Band 3 (0.64 µm) is the VIS equivalent
to ABI Band 2.

Value encoding is the same as GOES: CMI in Kelvin (IR) or reflectance
factor (VIS), DQF quality flag, x/y scan-angle coordinate arrays.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import ClassVar

import fsspec
import numpy as np
import xarray as xr

from librewxr.sources.satellite._geo_base import GeoSatSource

logger = logging.getLogger(__name__)

HIMAWARI_LON = 140.7  # sub-satellite longitude (degrees East)
HIMAWARI_HEIGHT = 35785831.0  # metres above ellipsoid

# Same IR temperature range as GOES for consistent rendering.
_IR_T_MIN = 170.0
_IR_T_MAX = 320.0
_IR_RANGE = _IR_T_MAX - _IR_T_MIN


class HimawariSource(GeoSatSource):
    """Abstract base for one Himawari-9 AHI channel."""

    sat_lon: ClassVar[float] = HIMAWARI_LON
    sat_height: ClassVar[float] = HIMAWARI_HEIGHT
    cadence_minutes: ClassVar[int] = 10

    band: ClassVar[int]

    def _list_recent_keys(
        self,
        fs: fsspec.AbstractFileSystem,
        window_start: datetime,
        window_end: datetime,
    ) -> list[tuple[int, str]]:
        """List S3 keys for Himawari's path layout.

        Himawari uses ``{product}/{year}/{month}/{day}/{hour}/`` (unlike
        GOES which uses day-of-year).
        """
        results: list[tuple[int, str]] = []
        cursor = window_start.replace(minute=0, second=0, microsecond=0)
        while cursor <= window_end:
            prefix = (
                f"{self.s3_bucket}/{self.s3_product_path}/"
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
                if self.s3_filename_token not in name:
                    continue
                unix_ts = self._parse_start_timestamp(name)
                if unix_ts is None:
                    continue
                results.append((unix_ts, entry))
            cursor += timedelta(hours=1)
        return sorted(set(results))

    def _decode_netcdf(self, path: str) -> np.ndarray | None:
        ds = xr.open_dataset(path, engine="netcdf4", decode_times=False)
        try:
            self._init_grid_vectors(ds)

            cmi = ds["CMI"].values
            if cmi.ndim == 3 and cmi.shape[0] == 1:
                cmi = cmi[0]
            if cmi.shape != (self._grid_height, self._grid_width):
                logger.warning(
                    "%s: unexpected grid shape %s vs (%d, %d)",
                    self.friendly_name, cmi.shape,
                    self._grid_height, self._grid_width,
                )
                return None

            if "DQF" in ds.variables:
                dqf = ds["DQF"].values
                if dqf.ndim == 3 and dqf.shape[0] == 1:
                    dqf = dqf[0]
                if dqf.shape == cmi.shape:
                    cmi = np.where(dqf == 0, cmi, np.nan)

            return self._map_to_uint8(cmi)
        finally:
            ds.close()

    def _map_to_uint8(self, cmi: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HimawariIRSource(HimawariSource):
    """Himawari-9 Band 13 (10.4 µm longwave IR), full-disk."""

    s3_bucket: ClassVar[str] = "noaa-himawari9"
    s3_product_path: ClassVar[str] = "AHI-L2-FLDK-ISatSS"
    s3_filename_token: ClassVar[str] = "ISatSS-M1C13"
    friendly_name: ClassVar[str] = "Himawari-9 IR"
    channel: ClassVar[str] = "IR"
    band: ClassVar[int] = 13

    def _map_to_uint8(self, cmi: np.ndarray) -> np.ndarray:
        encoded = np.where(
            np.isfinite(cmi),
            255.0 * (_IR_T_MAX - cmi) / _IR_RANGE,
            0.0,
        )
        return np.clip(encoded, 0, 255).astype(np.uint8)


class HimawariVISSource(HimawariSource):
    """Himawari-9 Band 3 (0.64 µm visible), full-disk."""

    s3_bucket: ClassVar[str] = "noaa-himawari9"
    s3_product_path: ClassVar[str] = "AHI-L2-FLDK-ISatSS"
    s3_filename_token: ClassVar[str] = "ISatSS-M1C03"
    friendly_name: ClassVar[str] = "Himawari-9 VIS"
    channel: ClassVar[str] = "VIS"
    band: ClassVar[int] = 3

    def _map_to_uint8(self, cmi: np.ndarray) -> np.ndarray:
        encoded = np.where(
            np.isfinite(cmi),
            np.clip(cmi, 0.0, 1.0) * 255.0,
            0.0,
        )
        return np.clip(encoded, 0, 255).astype(np.uint8)
