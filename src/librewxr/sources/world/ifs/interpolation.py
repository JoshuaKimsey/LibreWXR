# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Optical-flow temporal interpolation for ECMWF IFS hourly grids.

Adapter around the shared ``nwp_interpolation`` helper that the
regional NWP sources also use.  IFS stores its frames as ``(precip,
snow)`` tuples keyed by unix valid time; this module splits the
tuple-dict into parallel dicts, delegates to the shared interpolator,
and re-packs the result.  No IFS-specific logic remains here — the
warp pipeline lives in :mod:`librewxr.data.nwp_interpolation`.
"""
from __future__ import annotations

import logging

import numpy as np

from librewxr.data.nwp_interpolation import interpolate_run

logger = logging.getLogger(__name__)


def interpolate_timesteps(
    timesteps: dict[int, tuple[np.ndarray, np.ndarray]],
    interval_seconds: int = 600,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Create sub-hourly IFS frames by optical-flow interpolation.

    Adapter shim: splits the IFS ``(precip, snow)`` tuple-dict into
    parallel dicts, calls the shared interpolator, and re-packs.

    Args:
        timesteps: Original hourly dict ``{unix_ts: (precip_dbz, snow_mask)}``.
        interval_seconds: Target interval between frames (default 600
            = 10 min, matching the radar cadence).

    Returns:
        New dict containing both original and interpolated timesteps.
        The warp's internal flow field is no longer surfaced — the
        hybrid arrow path builds a composite NWP flow raster instead
        (see ``NowcastGenerator._compute_nwp_flow_sync``), so IFS no
        longer needs to carry its own flow for arrow rendering.
    """
    if len(timesteps) < 2:
        return dict(timesteps)

    precip_by_ts = {ts: t[0] for ts, t in timesteps.items()}
    snow_by_ts = {ts: t[1] for ts, t in timesteps.items()}

    aug_precip, aug_snow, _last_flow = interpolate_run(
        precip_by_ts,
        snow_by_ts,
        target_interval_seconds=interval_seconds,
        log_label="ECMWF interpolation",
    )

    # Re-pack into the IFS-native tuple-dict format.
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ts in aug_precip:
        snow = aug_snow[ts] if aug_snow is not None else None
        # IFS expects snow to be bool; the shared helper preserves bool
        # dtype if input was bool, so this is just defensive.
        if snow is not None and snow.dtype != np.bool_:
            snow = snow.astype(bool)
        result[ts] = (aug_precip[ts], snow)
    return result
