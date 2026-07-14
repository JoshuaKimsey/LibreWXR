# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Shared utility helpers for source packages.

Two helpers contributors reach for when implementing a new source:

- ``_dbz_float_to_uint8`` — the canonical float-dBZ-to-uint8 encoder.
  Every radar source converts its native reflectivity (mm/h palette,
  raw float dBZ, RGB hue, etc.) into this 8-bit encoding so the
  renderer and tile pipeline see a single shape.
- ``_suppress_eccodes_stderr`` — a context manager that muzzles the
  eccodes C library's non-actionable ``dataTime`` truncation noise.
  Used by every NWP source that opens GRIB2 (HRRR, HRRR-Alaska, HRDPS,
  ICON-EU, DMI DINI, AROME Antilles) and by the MRMS radar source.
- ``HDF5_LOCK`` — a process-wide lock guarding every call into the
  HDF5 C library (h5py, and xarray's ``engine="netcdf4"``).  HDF5 is
  not thread-safe, and each of h5py/netCDF4's wheels bundles its own
  private ``libhdf5`` build, so concurrent access from more than one
  Python thread (e.g. one source parsing on the event loop while
  another parses inside ``asyncio.to_thread``) corrupts the library's
  internal state and segfaults the process. Used by OPERA (radar) and
  WRF-SMN + GMGSI (NWP/satellite) — the only sources that touch HDF5.

Both intentionally live outside any one source package so a new source
can pick them up without importing from a sibling source's internals.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager

import numpy as np

HDF5_LOCK = threading.Lock()


def _dbz_float_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert float32 dBZ values to uint8 using IEM's encoding.

    Formula: pixel = clamp((dBZ + 32) * 2, 0, 255)
    NODATA (anything <= -32) maps to 0 (transparent in all color schemes).
    """
    nodata_mask = arr <= -32.0
    result = np.clip((arr + 32.0) * 2.0, 0, 255).astype(np.uint8)
    result[nodata_mask] = 0
    return result


@contextmanager
def _suppress_eccodes_stderr():
    """Redirect OS-level stderr to /dev/null during the block.

    The eccodes C library (used by cfgrib) writes non-actionable
    ``dataTime`` truncation messages directly to stderr.  This silences
    them without affecting Python logging or other error reporting.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    original = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(original, 2)
        os.close(devnull)
        os.close(original)
