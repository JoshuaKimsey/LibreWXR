# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Precipitation nowcasting via radar extrapolation and IFS blending.

Generates short-range forecast frames (default 60 minutes) by:

1. Computing optical flow between the two most recent radar frames
   (per region, with adaptive downscaling for speed).
2. Extrapolating the latest radar forward along the motion vectors.
3. Storing extrapolated frames in a lightweight ``NowcastStore`` with
   per-frame blend weights that tell the renderer how much to trust
   the extrapolation vs the ECMWF IFS forecast.

The renderer handles the actual blending — this module only produces
the extrapolated radar data and the temporal blend weight for each step.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from librewxr.config import settings

logger = logging.getLogger(__name__)

# Target longest dimension for optical flow computation.
# Larger grids are downscaled to this for speed, then flow vectors are
# upscaled back to full resolution.
_TARGET_FLOW_DIM = 1000

# Farneback optical flow parameters (same tuning as ecmwf_interpolation).
_FARNEBACK = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0,
)


# ---------------------------------------------------------------------------
# NowcastStore
# ---------------------------------------------------------------------------

@dataclass
class NowcastFrame:
    """A single nowcast frame with per-region extrapolated radar data."""
    timestamp: int
    regions: dict[str, np.ndarray] = field(default_factory=dict)
    blend_weight: float = 1.0  # 1.0 = trust radar, 0.0 = trust IFS


class NowcastStore:
    """Lightweight store for nowcast frames.

    Nowcast frames are regenerated every fetch cycle, so no persistence
    or max-frames eviction is needed — just an atomic swap of the frame
    dict each cycle.
    """

    def __init__(self):
        self._frames: dict[int, NowcastFrame] = {}
        self._lock = asyncio.Lock()

    async def replace_all(
        self, frames: list[NowcastFrame],
    ) -> list[int]:
        """Atomically replace all nowcast frames.

        Returns the timestamps of the old frames that were removed
        (for tile cache invalidation).
        """
        async with self._lock:
            old_timestamps = list(self._frames.keys())
            self._frames = {f.timestamp: f for f in frames}
            return old_timestamps

    async def get_frame(
        self, timestamp: int,
    ) -> tuple[NowcastFrame | None, float]:
        """Return ``(frame, blend_weight)`` or ``(None, 0.0)``."""
        async with self._lock:
            frame = self._frames.get(timestamp)
            if frame is None:
                return None, 0.0
            return frame, frame.blend_weight

    async def get_timestamps(self) -> list[int]:
        async with self._lock:
            return sorted(self._frames.keys())

    def clear(self) -> None:
        self._frames.clear()


# ---------------------------------------------------------------------------
# NowcastGenerator
# ---------------------------------------------------------------------------

class NowcastGenerator:
    """Generates nowcast frames from the latest radar data."""

    def __init__(self, store, nowcast_store: NowcastStore, cache=None):
        self._store = store          # FrameStore (past radar)
        self._nowcast_store = nowcast_store
        self._cache = cache          # TileCache (for invalidation)

    async def generate(self) -> None:
        """Generate nowcast frames from the two most recent radar frames.

        Called after each fetch cycle.  Runs the optical flow computation
        in a thread to avoid blocking the event loop.
        """
        if not settings.nowcast_enabled:
            return

        timestamps = await self._store.get_timestamps()
        if len(timestamps) < 2:
            logger.debug("Nowcast: need at least 2 frames, have %d", len(timestamps))
            return

        latest_ts = timestamps[-1]
        prev_ts = timestamps[-2]

        latest_frame = await self._store.get_frame(latest_ts)
        prev_frame = await self._store.get_frame(prev_ts)
        if latest_frame is None or prev_frame is None:
            return

        n_steps = settings.nowcast_frames
        interval = settings.fetch_interval

        # Run CPU-heavy extrapolation in a thread
        nowcast_frames = await asyncio.to_thread(
            self._generate_sync,
            prev_frame.regions, latest_frame.regions,
            latest_ts, n_steps, interval,
        )

        # Swap into the store and invalidate old tile cache entries
        old_timestamps = await self._nowcast_store.replace_all(nowcast_frames)
        if self._cache is not None:
            for ts in old_timestamps:
                self._cache.invalidate_timestamp(ts)

        if nowcast_frames:
            logger.info(
                "Nowcast updated: %d frames (T+%d to T+%d min)",
                len(nowcast_frames),
                interval // 60,
                n_steps * interval // 60,
            )

    @staticmethod
    def _generate_sync(
        prev_regions: dict[str, np.ndarray],
        latest_regions: dict[str, np.ndarray],
        latest_ts: int,
        n_steps: int,
        interval: int,
    ) -> list[NowcastFrame]:
        """Synchronous nowcast generation (runs in a thread)."""
        t0 = time.monotonic()

        # Pre-compute flow per region
        flows: dict[str, np.ndarray] = {}
        for region_name in latest_regions:
            data0 = prev_regions.get(region_name)
            data1 = latest_regions.get(region_name)
            if data0 is None or data1 is None:
                continue
            flows[region_name] = _compute_flow(data0, data1)

        if not flows:
            return []

        # Generate extrapolated frames for each step
        frames: list[NowcastFrame] = []
        for step in range(1, n_steps + 1):
            nowcast_ts = latest_ts + step * interval
            # Blend curve: radar-dominant near-term, fading to 30% radar
            # at 60 min.  Beyond 60 min, radar extrapolation is too
            # inaccurate so we fall back to pure IFS.
            max_blend_steps = max(1, 3600 // interval)  # 6 at 10-min cadence
            if step <= max_blend_steps:
                t = step / max_blend_steps
                blend_weight = 0.30 + 0.70 * (1.0 - t) ** 1.1
            else:
                blend_weight = 0.0

            regions: dict[str, np.ndarray] = {}
            for region_name, flow in flows.items():
                data = latest_regions[region_name]
                regions[region_name] = _extrapolate_forward(data, flow, step)

            frames.append(NowcastFrame(
                timestamp=nowcast_ts,
                regions=regions,
                blend_weight=blend_weight,
            ))

        elapsed = time.monotonic() - t0
        logger.info(
            "Nowcast generation: %d frames × %d regions (%.1fs)",
            len(frames), len(flows), elapsed,
        )
        return frames


# ---------------------------------------------------------------------------
# Optical flow helpers
# ---------------------------------------------------------------------------

def _compute_flow(frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
    """Compute dense optical flow with adaptive downscaling.

    Downscales so the longest dimension is ~``_TARGET_FLOW_DIM`` pixels,
    computes Farneback flow, then upscales the flow vectors to the
    original resolution.
    """
    h, w = frame0.shape
    scale = min(_TARGET_FLOW_DIM / max(h, w), 1.0)

    if scale < 1.0:
        small_h = max(1, int(h * scale))
        small_w = max(1, int(w * scale))
        small0 = cv2.resize(frame0, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small1 = cv2.resize(frame1, (small_w, small_h), interpolation=cv2.INTER_AREA)

        flow_small = cv2.calcOpticalFlowFarneback(
            small0, small1, flow=None, **_FARNEBACK,
        )

        flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
        flow *= 1.0 / scale  # scale vectors to full resolution
    else:
        flow = cv2.calcOpticalFlowFarneback(
            frame0, frame1, flow=None, **_FARNEBACK,
        )

    return flow


def _extrapolate_forward(
    frame: np.ndarray, flow: np.ndarray, steps: int,
) -> np.ndarray:
    """Warp *frame* forward by *steps* × flow using inverse remap.

    For each output pixel p, samples *frame* at ``p − steps·flow(p)``.
    After warping, rescales the result to preserve the total precipitation
    energy of the source frame — bilinear interpolation in cv2.remap
    tends to smooth peak values, causing artificial intensity loss.
    """
    h, w = frame.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    map_x = xs - steps * flow[..., 0]
    map_y = ys - steps * flow[..., 1]

    warped = cv2.remap(
        frame, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Intensity preservation: rescale so the mean of non-zero pixels
    # matches the source frame, preventing artificial decay from
    # bilinear interpolation smoothing.
    src_mask = frame > 0
    warp_mask = warped > 0
    if src_mask.any() and warp_mask.any():
        src_mean = frame[src_mask].astype(np.float32).mean()
        warp_mean = warped[warp_mask].astype(np.float32).mean()
        if warp_mean > 0:
            scale = src_mean / warp_mean
            # Clamp scale to avoid amplifying noise (allow up to 30% boost)
            scale = min(scale, 1.3)
            if scale > 1.0:
                corrected = warped.astype(np.float32) * scale
                warped = np.clip(corrected + 0.5, 0, 255).astype(np.uint8)

    return warped
