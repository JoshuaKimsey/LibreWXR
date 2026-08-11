# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Shared utility helpers for source packages.

Two helpers contributors reach for when implementing a new source:

- ``_dbz_float_to_uint8`` — the canonical float-dBZ-to-uint8 encoder.
  Every radar source converts its native reflectivity (mm/h palette,
  raw float dBZ, RGB hue, etc.) into this 8-bit encoding so the
  renderer and tile pipeline see a single shape.
- ``HDF5_LOCK`` — a process-wide lock guarding every call into the
  HDF5 C library (h5py, and xarray's ``engine="netcdf4"``).  HDF5 is
  not thread-safe, and each of h5py/netCDF4's wheels bundles its own
  private ``libhdf5`` build, so concurrent access from more than one
  Python thread (e.g. one source parsing on the event loop while
  another parses inside ``asyncio.to_thread``) corrupts the library's
  internal state and segfaults the process. Used by OPERA (radar) and
  WRF-SMN + GMGSI (NWP/satellite) — the only sources that touch HDF5.

eccodes stderr noise: the eccodes C library (via cfgrib) writes
non-actionable ``dataTime`` truncation messages directly to stderr.
This module instead points the C library's default-context logging at
``/dev/null`` once at import time via the supported
``codes_context_set_logging`` binding (``grib_context_set_logging_file``
in the C API).  This is thread-safe: the redirect is installed before
any decode runs, and concurrent writes to the ``/dev/null`` FILE* from
GRIB decodes running inside ``asyncio.to_thread`` are invisible.
(The previous approach — an ``os.dup2(devnull, 2)`` context manager —
redirected the process-global fd 2 and would clobber another thread's
stderr once decodes moved off the event loop.)

Both intentionally live outside any one source package so a new source
can pick them up without importing from a sibling source's internals.
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)

HDF5_LOCK = threading.Lock()

# Redirect the eccodes C library's default-context logging to /dev/null
# so its non-actionable "dataTime truncation" warnings stop hitting
# stderr.  ``codes_context_set_logging`` hands the Python file object's
# FILE* to the C library (cffi), so the object must stay alive for the
# whole process — hence the module-level name.
_ECCODES_DEVNULL = open(os.devnull, "wb")
try:
    import eccodes
except Exception:
    logger.warning(
        "eccodes not importable; GRIB decode warnings will reach stderr",
        exc_info=True,
    )
else:
    try:
        eccodes.codes_context_set_logging(_ECCODES_DEVNULL)
    except Exception:
        logger.warning(
            "eccodes logging redirect failed; GRIB decode warnings will "
            "reach stderr",
            exc_info=True,
        )


def _dbz_float_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert float32 dBZ values to uint8 using IEM's encoding.

    Formula: pixel = clamp((dBZ + 32) * 2, 0, 255)
    NODATA (anything <= -32) maps to 0 (transparent in all color schemes).

    In-place formulation: one pre-sized float32 working buffer reused via
    ``out=`` across the add/multiply/clip chain, then a single uint8 cast.
    """
    result = np.empty(arr.shape, dtype=np.float32)
    np.add(arr, 32.0, out=result)
    result *= 2.0
    np.clip(result, 0, 255, out=result)
    out = result.astype(np.uint8)
    out[arr <= -32.0] = 0
    return out
