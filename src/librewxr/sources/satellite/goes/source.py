# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""GOES-18 / GOES-19 ABI satellite source.

GOES ABI (Advanced Baseline Imager) publishes Cloud and Moisture Imagery
(CMI) on anonymous S3 at 5-minute cadence for the CONUS sector (CMIPC)
and 10-minute for the full-disk (CMIPF).  We use CONUS for best cadence
and resolution over the Americas.

S3 layout (GOES-18 example)::

    s3://noaa-goes18/ABI-L2-CMIPC/{year}/{doy}/{hour}/
      OR_ABI-L2-CMIPC-M6C13_G18_s{start}_e{end}_c{create}.nc

NetCDF variables:
  - ``CMI``: float32, brightness temperature (K) for IR bands,
    reflectance factor (0–1) for VIS bands.
  - ``DQF``: uint8 data quality flag (0 = good).
  - ``x``, ``y``: 1-D scan-angle coordinate arrays (radians).
  - ``goes_imager_projection``: projection metadata.

Value mapping to uint8 (renderer-compatible):
  - IR (Band 13, 10.3 µm): Kelvin 170–320 K → uint8 0–255 with
    cold=high matching GMGSI convention.
  - VIS (Band 2, 0.64 µm): reflectance 0–1 → uint8 0–255.
"""
from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np
import xarray as xr

from librewxr.sources.satellite._geo_base import GeoSatSource

logger = logging.getLogger(__name__)

# GOES orbital parameters (identical for GOES-16/17/18/19)
GOES_HEIGHT = 35786023.0  # metres above ellipsoid

# Brightness-temperature range for IR → uint8 mapping.
# GMGSI convention: cold (high cloud) = high uint8, warm (ground) = low.
# We invert: encode = 255 * (T_MAX - T) / (T_MAX - T_MIN), clamped.
_IR_T_MIN = 170.0  # K — coldest cloud tops (~-103°C)
_IR_T_MAX = 320.0  # K — warmest ground/ocean (~47°C)
_IR_RANGE = _IR_T_MAX - _IR_T_MIN


class GOESSource(GeoSatSource):
    """Abstract base for one GOES ABI channel.

    Subclasses set ``sat_lon``, ``band``, and the S3 addressing class vars.
    The sat_lon determines which GOES satellite (East vs West).
    """

    sat_height: ClassVar[float] = GOES_HEIGHT
    cadence_minutes: ClassVar[int] = 5

    # Overridden by concrete satellite+channel subclasses
    band: ClassVar[int]

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

            # Apply quality mask
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
        """Map CMI float values to uint8.  Overridden per channel type."""
        raise NotImplementedError


# ── GOES-18 (West, 137.0°W) ──


class GOES18IRSource(GOESSource):
    """GOES-18 Band 13 (10.3 µm longwave IR), CONUS sector."""

    sat_lon: ClassVar[float] = -137.0
    s3_bucket: ClassVar[str] = "noaa-goes18"
    s3_product_path: ClassVar[str] = "ABI-L2-CMIPC"
    s3_filename_token: ClassVar[str] = "CMIPC-M6C13"
    friendly_name: ClassVar[str] = "GOES-18 IR"
    channel: ClassVar[str] = "IR"
    band: ClassVar[int] = 13

    def _map_to_uint8(self, cmi: np.ndarray) -> np.ndarray:
        # Cold = high uint8 (matching GMGSI convention)
        encoded = np.where(
            np.isfinite(cmi),
            255.0 * (_IR_T_MAX - cmi) / _IR_RANGE,
            0.0,
        )
        return np.clip(encoded, 0, 255).astype(np.uint8)


class GOES18VISSource(GOESSource):
    """GOES-18 Band 2 (0.64 µm visible), CONUS sector."""

    sat_lon: ClassVar[float] = -137.0
    s3_bucket: ClassVar[str] = "noaa-goes18"
    s3_product_path: ClassVar[str] = "ABI-L2-CMIPC"
    s3_filename_token: ClassVar[str] = "CMIPC-M6C02"
    friendly_name: ClassVar[str] = "GOES-18 VIS"
    channel: ClassVar[str] = "VIS"
    band: ClassVar[int] = 2

    def _map_to_uint8(self, cmi: np.ndarray) -> np.ndarray:
        # Reflectance factor 0–1 → uint8 0–255 (bright = high)
        encoded = np.where(
            np.isfinite(cmi),
            np.clip(cmi, 0.0, 1.0) * 255.0,
            0.0,
        )
        return np.clip(encoded, 0, 255).astype(np.uint8)


# ── GOES-19 (East, 75.2°W) ──


class GOES19IRSource(GOESSource):
    """GOES-19 Band 13 (10.3 µm longwave IR), CONUS sector."""

    sat_lon: ClassVar[float] = -75.2
    s3_bucket: ClassVar[str] = "noaa-goes19"
    s3_product_path: ClassVar[str] = "ABI-L2-CMIPC"
    s3_filename_token: ClassVar[str] = "CMIPC-M6C13"
    friendly_name: ClassVar[str] = "GOES-19 IR"
    channel: ClassVar[str] = "IR"
    band: ClassVar[int] = 13

    def _map_to_uint8(self, cmi: np.ndarray) -> np.ndarray:
        encoded = np.where(
            np.isfinite(cmi),
            255.0 * (_IR_T_MAX - cmi) / _IR_RANGE,
            0.0,
        )
        return np.clip(encoded, 0, 255).astype(np.uint8)


class GOES19VISSource(GOESSource):
    """GOES-19 Band 2 (0.64 µm visible), CONUS sector."""

    sat_lon: ClassVar[float] = -75.2
    s3_bucket: ClassVar[str] = "noaa-goes19"
    s3_product_path: ClassVar[str] = "ABI-L2-CMIPC"
    s3_filename_token: ClassVar[str] = "CMIPC-M6C02"
    friendly_name: ClassVar[str] = "GOES-19 VIS"
    channel: ClassVar[str] = "VIS"
    band: ClassVar[int] = 2

    def _map_to_uint8(self, cmi: np.ndarray) -> np.ndarray:
        encoded = np.where(
            np.isfinite(cmi),
            np.clip(cmi, 0.0, 1.0) * 255.0,
            0.0,
        )
        return np.clip(encoded, 0, 255).astype(np.uint8)
