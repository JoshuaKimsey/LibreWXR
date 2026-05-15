# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Generic optical-flow temporal interpolation for NWP grids.

Used by both the global ECMWF IFS path (which sees one continuous
timeline of unix timestamps) and the regional NWP sources that
interpolate within a single model run (lead_seconds-keyed).  The
helper is projection-agnostic and grid-shape-agnostic — it just
needs frames keyed by sortable integers in consistent units and a
target interval to fill in.

Uses OpenCV's Farneback dense optical flow, computed at reduced
resolution for speed, then upscaled for full-resolution warping.
The same machinery handles both precipitation (uint8 dBZ-encoded)
and snow-mask (uint8 0/1) fields.

Historical note: this code was extracted from the IFS-specific
``ecmwf_interpolation.py`` so the regional NWP sources can share
the warp pipeline.  The original module is now a thin wrapper that
adapts the IFS ``(precip, snow)`` tuple-dict format to this
helper's parallel-dict signature.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Downscale factor for optical flow computation.  Flow is computed on a
# smaller grid for speed (~450×900 from 1801×3600 on IFS, scaled down
# accordingly for regional sources), then upscaled.
_DEFAULT_DOWNSCALE = 4

# Farneback optical flow parameters tuned for weather system motion.
# These work across a wide range of resolutions (9 km IFS through 2 km
# DMI DINI) because Farneback's multi-scale pyramid adapts to the
# per-pixel displacement magnitude.
_FARNEBACK = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


def interpolate_run(
    frames_by_ts: dict[int, np.ndarray],
    snow_masks_by_ts: dict[int, np.ndarray] | None = None,
    target_interval_seconds: int = 600,
    downscale: int = _DEFAULT_DOWNSCALE,
    log_label: str = "NWP interpolation",
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray] | None,
    np.ndarray | None,
]:
    """Fill gaps in a sorted timeline of NWP frames with optical-flow synthetics.

    For each consecutive pair of source frames whose gap exceeds the
    target interval, computes a dense motion field and warps + blends
    the bracket frames at sub-interval positions.  Originals are
    preserved unchanged.  Idempotent: if the timeline already has
    target-interval spacing, no work is done.

    Args:
        frames_by_ts: Source precip frames keyed by sortable int
            (unix timestamps for IFS, lead_seconds within a single
            run for regional sources).  Values are uint8 dBZ-encoded.
        snow_masks_by_ts: Optional parallel snow-mask dict with the
            same keys.  Values are uint8 0/1 (or bool).  If provided,
            the function fills in interpolated snow masks at the same
            synthetic keys.
        target_interval_seconds: The desired stored interval between
            frames.  Pairs with gap ≤ this value are skipped (no work).
        downscale: Resolution factor for the Farneback compute pass.
            Higher = faster but coarser flow.  4 has worked well from
            9 km down to 2 km.
        log_label: Human-readable label for the summary log line.

    Returns:
        Tuple of:
        - augmented frames dict (originals + synthetics)
        - augmented snow_masks dict (None if input was None)
        - last computed flow field, or None if no interpolation
          happened (caller may use this for motion-arrow rendering).
    """
    if len(frames_by_ts) < 2:
        return (
            dict(frames_by_ts),
            None if snow_masks_by_ts is None else dict(snow_masks_by_ts),
            None,
        )

    sorted_ts = sorted(frames_by_ts.keys())
    result_frames = dict(frames_by_ts)
    result_snow = (
        dict(snow_masks_by_ts) if snow_masks_by_ts is not None else None
    )

    # Pre-compute coordinate grids (shared across all warp ops in this run)
    h, w = next(iter(frames_by_ts.values())).shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    t0_wall = time.monotonic()
    total_interpolated = 0
    last_flow = None

    for i in range(len(sorted_ts) - 1):
        ts0 = sorted_ts[i]
        ts1 = sorted_ts[i + 1]
        gap = ts1 - ts0

        if gap <= target_interval_seconds:
            continue

        precip0 = frames_by_ts[ts0]
        precip1 = frames_by_ts[ts1]

        flow = _compute_flow(precip0, precip1, downscale)
        last_flow = flow

        # Optional snow side: only if both endpoints have masks.
        snow0 = (
            snow_masks_by_ts.get(ts0) if snow_masks_by_ts is not None else None
        )
        snow1 = (
            snow_masks_by_ts.get(ts1) if snow_masks_by_ts is not None else None
        )
        do_snow = snow0 is not None and snow1 is not None

        n_steps = gap // target_interval_seconds
        for step in range(1, n_steps):
            t = step / n_steps
            interp_ts = ts0 + step * target_interval_seconds

            result_frames[interp_ts] = _interpolate_precip(
                precip0, precip1, flow, t, xs, ys,
            )
            if do_snow:
                result_snow[interp_ts] = _interpolate_snow(
                    snow0, snow1, flow, t, xs, ys,
                )
            total_interpolated += 1

    elapsed = time.monotonic() - t0_wall
    if total_interpolated:
        logger.info(
            "%s: %d synthetic frames from %d originals (%.1fs)",
            log_label, total_interpolated, len(sorted_ts), elapsed,
        )

    return result_frames, result_snow, last_flow


def interpolate_pair_at_fraction(
    frame0: np.ndarray,
    frame1: np.ndarray,
    t: float,
    flow: np.ndarray | None = None,
    downscale: int = _DEFAULT_DOWNSCALE,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a single intermediate frame between two native frames.

    Companion to :func:`interpolate_run` for callers that have one pair
    of frames and need one synthetic frame at a known time fraction
    (e.g. radar sources whose native cadence is wider than the stored
    cadence).  Returns both the warped frame and the computed flow
    field, so callers caching the flow for adjacent fractions can pass
    it back in via the ``flow`` argument and skip recomputation.

    Args:
        frame0: Earlier native frame (uint8 dBZ-encoded).
        frame1: Later native frame (uint8 dBZ-encoded).
        t: Position between the two, in (0, 1).  ``t=0`` returns
            ``frame0`` unchanged; ``t=1`` returns ``frame1`` unchanged.
        flow: Optional pre-computed Farneback flow field from a prior
            call on the same pair.  Skips the compute pass if supplied.
        downscale: Same as :func:`interpolate_run`.

    Returns:
        ``(interp_frame, flow)``.  Caller may reuse ``flow`` for other
        fractions of the same pair.
    """
    if t <= 0.0:
        return frame0.copy(), flow if flow is not None else np.zeros(
            frame0.shape + (2,), dtype=np.float32,
        )
    if t >= 1.0:
        return frame1.copy(), flow if flow is not None else np.zeros(
            frame0.shape + (2,), dtype=np.float32,
        )

    if flow is None:
        flow = _compute_flow(frame0, frame1, downscale)

    h, w = frame0.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    interp = _interpolate_precip(frame0, frame1, flow, t, xs, ys)
    return interp, flow


def _compute_flow(
    frame0: np.ndarray,
    frame1: np.ndarray,
    downscale: int = _DEFAULT_DOWNSCALE,
) -> np.ndarray:
    """Compute dense optical flow between two precip grids.

    Downscales both frames for efficiency, runs Farneback, and
    upscales the resulting flow vectors back to full resolution.
    """
    h, w = frame0.shape
    small_h = max(1, h // downscale)
    small_w = max(1, w // downscale)

    small0 = cv2.resize(
        frame0, (small_w, small_h), interpolation=cv2.INTER_AREA,
    )
    small1 = cv2.resize(
        frame1, (small_w, small_h), interpolation=cv2.INTER_AREA,
    )

    flow_small = cv2.calcOpticalFlowFarneback(
        small0, small1, flow=None, **_FARNEBACK,
    )

    # Upscale flow field and multiply vectors by scale so
    # displacements are correct at full resolution.
    flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
    flow *= downscale

    return flow


def _interpolate_precip(
    frame0: np.ndarray,
    frame1: np.ndarray,
    flow: np.ndarray,
    t: float,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    """Warp and blend two precipitation grids at time fraction t ∈ (0, 1).

    Forward-warps frame0 by t along the flow and backward-warps frame1
    by (1 − t) (using negated flow as an approximation of backward
    flow), then linearly blends the two warped frames.
    """
    map0_x = xs - t * flow[..., 0]
    map0_y = ys - t * flow[..., 1]
    warped0 = cv2.remap(
        frame0, map0_x, map0_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    map1_x = xs + (1 - t) * flow[..., 0]
    map1_y = ys + (1 - t) * flow[..., 1]
    warped1 = cv2.remap(
        frame1, map1_x, map1_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    blended = (1 - t) * warped0.astype(np.float32) + t * warped1.astype(np.float32)

    # Don't hallucinate precipitation where neither warped frame has any.
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
    """Warp and blend snow classification masks at time fraction t.

    Warps masks (0/1 uint8 or bool) as floats with the same flow
    field, then thresholds back at 0.5.  Returns uint8 0/1 to match
    the regional sources' on-disk storage convention.
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
    # Preserve bool dtype if both inputs were bool, otherwise uint8.
    if snow0.dtype == np.bool_ and snow1.dtype == np.bool_:
        return blended > 0.5
    return (blended > 0.5).astype(np.uint8)
