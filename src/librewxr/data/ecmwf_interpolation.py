# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Optical-flow temporal interpolation for ECMWF IFS hourly grids.

Generates synthetic sub-hourly frames by computing motion vectors between
consecutive IFS hours and warping precipitation fields along those vectors.
This makes global fallback areas animate smoothly at the same ~10-minute
cadence as radar data, instead of jumping hour-to-hour.

Uses OpenCV's Farneback dense optical flow, computed at reduced resolution
for speed, then upscaled for full-resolution warping.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Downscale factor for optical flow computation.  Flow is computed on a
# smaller grid for speed (~450×900 from 1801×3600), then upscaled.
_DOWNSCALE = 4

# Farneback optical flow parameters tuned for weather system motion.
# At 0.1° resolution, typical weather systems move 1-10 px/hr.
_FARNEBACK = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def interpolate_timesteps(
    timesteps: dict[int, tuple[np.ndarray, np.ndarray]],
    interval_seconds: int = 600,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Create sub-hourly frames by optical-flow interpolation between IFS hours.

    For each consecutive pair of hourly timesteps, computes a dense motion
    field, then warps and blends precipitation and snow masks at 10-minute
    intervals.  The original hourly frames are preserved unchanged.

    Args:
        timesteps: Original hourly dict ``{unix_ts: (precip_dbz, snow_mask)}``.
        interval_seconds: Target interval between frames (default 600 = 10 min).

    Returns:
        New dict containing both original and interpolated timesteps.
    """
    if len(timesteps) < 2:
        return dict(timesteps)

    sorted_ts = sorted(timesteps.keys())
    result = dict(timesteps)

    # Pre-compute coordinate grids (shared across all warp operations)
    h, w = next(iter(timesteps.values()))[0].shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    t0_wall = time.monotonic()
    total_interpolated = 0

    for i in range(len(sorted_ts) - 1):
        ts0 = sorted_ts[i]
        ts1 = sorted_ts[i + 1]
        gap = ts1 - ts0

        if gap <= interval_seconds:
            continue

        precip0, snow0 = timesteps[ts0]
        precip1, snow1 = timesteps[ts1]

        flow = _compute_flow(precip0, precip1)

        n_steps = gap // interval_seconds
        for step in range(1, n_steps):
            t = step / n_steps
            interp_ts = ts0 + step * interval_seconds

            interp_precip = _interpolate_frame(
                precip0, precip1, flow, t, xs, ys,
            )
            interp_snow = _interpolate_snow(
                snow0, snow1, flow, t, xs, ys,
            )

            result[interp_ts] = (interp_precip, interp_snow)
            total_interpolated += 1

    elapsed = time.monotonic() - t0_wall
    logger.info(
        "ECMWF interpolation: %d synthetic frames from %d hourly originals (%.1fs)",
        total_interpolated, len(sorted_ts), elapsed,
    )

    return result


def _compute_flow(frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
    """Compute dense optical flow between two ``precip_dbz`` grids.

    Downscales both frames for efficiency, runs Farneback, and upscales the
    resulting flow vectors back to full resolution.
    """
    h, w = frame0.shape
    small_h = h // _DOWNSCALE
    small_w = w // _DOWNSCALE

    small0 = cv2.resize(frame0, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small1 = cv2.resize(frame1, (small_w, small_h), interpolation=cv2.INTER_AREA)

    flow_small = cv2.calcOpticalFlowFarneback(
        small0, small1, flow=None, **_FARNEBACK,
    )

    # Upscale flow field and multiply vectors by scale factor so
    # displacements are correct at full resolution.
    flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
    flow *= _DOWNSCALE

    return flow


def _interpolate_frame(
    frame0: np.ndarray,
    frame1: np.ndarray,
    flow: np.ndarray,
    t: float,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Warp and blend two precipitation grids at time fraction *t* ∈ (0, 1).

    Forward-warps *frame0* by *t* along the flow and backward-warps *frame1*
    by *(1 − t)* (using negated flow as an approximation of backward flow),
    then linearly blends the two warped frames.
    """
    # Forward warp of frame0: for output pixel p, sample frame0 at p − t·flow(p)
    map0_x = xs - t * flow[..., 0]
    map0_y = ys - t * flow[..., 1]
    warped0 = cv2.remap(
        frame0, map0_x, map0_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Backward warp of frame1: sample frame1 at p + (1−t)·flow(p)
    map1_x = xs + (1 - t) * flow[..., 0]
    map1_y = ys + (1 - t) * flow[..., 1]
    warped1 = cv2.remap(
        frame1, map1_x, map1_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Time-weighted blend
    blended = (1 - t) * warped0.astype(np.float32) + t * warped1.astype(np.float32)

    # Don't hallucinate precipitation where neither warped frame has any
    both_zero = (warped0 == 0) & (warped1 == 0)
    result = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    result[both_zero] = 0

    return result


def _interpolate_snow(
    snow0: np.ndarray,
    snow1: np.ndarray,
    flow: np.ndarray,
    t: float,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Warp and blend snow classification masks at time fraction *t*.

    Warps boolean masks as floats using the same flow field, then thresholds
    back to boolean at 0.5.
    """
    s0_f = snow0.astype(np.float32)
    s1_f = snow1.astype(np.float32)

    map0_x = xs - t * flow[..., 0]
    map0_y = ys - t * flow[..., 1]
    warped0 = cv2.remap(
        s0_f, map0_x, map0_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    map1_x = xs + (1 - t) * flow[..., 0]
    map1_y = ys + (1 - t) * flow[..., 1]
    warped1 = cv2.remap(
        s1_f, map1_x, map1_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    blended = (1 - t) * warped0 + t * warped1
    return blended > 0.5
