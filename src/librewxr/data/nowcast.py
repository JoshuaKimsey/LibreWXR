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
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

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
    dict each cycle.  Region arrays and flow fields are backed by
    memory-mapped temp files so the OS page cache manages physical RAM.
    """

    def __init__(self, cache_dir: Path | None = None):
        self._frames: dict[int, NowcastFrame] = {}
        self._flows: dict[str, np.ndarray] = {}
        self._lock = asyncio.Lock()
        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "nowcast"
            self._persistent = True
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_nowcast_"))
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        for path in self._memmap_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        logger.info(
            "Nowcast memmap directory: %s (persistent=%s)",
            self._memmap_dir, self._persistent,
        )

    def _to_memmap(self, name: str, data: np.ndarray) -> np.ndarray:
        """Write array to disk atomically and return a read-only memory-mapped view."""
        final = self._memmap_dir / f"{name}.dat"
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(tmp, dtype=data.dtype, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        del mm
        os.replace(tmp, final)
        return np.memmap(final, dtype=data.dtype, mode="r", shape=data.shape)

    async def replace_all(
        self, frames: list[NowcastFrame],
    ) -> list[int]:
        """Atomically replace all nowcast frames.

        Returns the timestamps of the old frames that were removed
        (for tile cache invalidation).
        """
        async with self._lock:
            old_timestamps = list(self._frames.keys())

            # Clean up old frame memmap files
            for path in self._memmap_dir.glob("frame_*.dat"):
                try:
                    path.unlink()
                except OSError:
                    pass

            # Convert region arrays to memmaps
            for frame in frames:
                for name, data in list(frame.regions.items()):
                    frame.regions[name] = self._to_memmap(
                        f"frame_{frame.timestamp}_{name}", data
                    )

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

    async def replace_flows(self, flows: dict[str, np.ndarray]) -> None:
        """Update the latest optical flow vectors."""
        async with self._lock:
            # Clean up old flow memmap files
            for path in self._memmap_dir.glob("flow_*.dat"):
                try:
                    path.unlink()
                except OSError:
                    pass

            for name, data in list(flows.items()):
                flows[name] = self._to_memmap(f"flow_{name}", data)
            self._flows = flows

    async def get_flows(self) -> dict[str, np.ndarray]:
        """Return the latest per-region optical flow vectors."""
        async with self._lock:
            return dict(self._flows)

    @property
    def data_bytes(self) -> int:
        """Total bytes across all nowcast frame arrays and flow fields."""
        total = 0
        for frame in self._frames.values():
            for arr in frame.regions.values():
                total += arr.nbytes
        for arr in self._flows.values():
            total += arr.nbytes
        return total

    def clear(self) -> None:
        self._frames.clear()
        self._flows.clear()

    def __getstate__(self) -> dict:
        """Serialize state for cross-process reload (multi-worker mode)."""
        frames_state: list[dict] = []
        for ts, frame in self._frames.items():
            regions: dict[str, list] = {}
            for name, arr in frame.regions.items():
                regions[name] = [
                    os.path.basename(str(arr.filename)),
                    arr.dtype.str,
                    list(arr.shape),
                ]
            frames_state.append({
                "timestamp": ts,
                "blend_weight": frame.blend_weight,
                "regions": regions,
            })
        flows_state: dict[str, list] = {}
        for name, arr in self._flows.items():
            flows_state[name] = [
                os.path.basename(str(arr.filename)),
                arr.dtype.str,
                list(arr.shape),
            ]
        return {
            "memmap_dir": str(self._memmap_dir),
            "frames": frames_state,
            "flows": flows_state,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state from the dict produced by ``__getstate__``."""
        memmap_dir = Path(state["memmap_dir"])
        new_frames: dict[int, NowcastFrame] = {}
        for f_info in state["frames"]:
            ts = int(f_info["timestamp"])
            frame = NowcastFrame(
                timestamp=ts,
                blend_weight=float(f_info["blend_weight"]),
            )
            for name, (basename, dtype_str, shape) in f_info["regions"].items():
                frame.regions[name] = np.memmap(
                    memmap_dir / basename,
                    dtype=np.dtype(dtype_str), mode="r",
                    shape=tuple(shape),
                )
            new_frames[ts] = frame

        new_flows: dict[str, np.ndarray] = {}
        for name, (basename, dtype_str, shape) in state["flows"].items():
            new_flows[name] = np.memmap(
                memmap_dir / basename,
                dtype=np.dtype(dtype_str), mode="r",
                shape=tuple(shape),
            )

        self._memmap_dir = memmap_dir
        self._frames = new_frames
        self._flows = new_flows
        self._persistent = True
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()

    def cleanup(self) -> None:
        """Clear data; remove the memmap dir only when non-persistent."""
        self.clear()
        if self._persistent:
            logger.info("Nowcast memmaps retained on disk at %s", self._memmap_dir)
        else:
            shutil.rmtree(self._memmap_dir, ignore_errors=True)
            logger.info("Nowcast memmap directory cleaned up")


# ---------------------------------------------------------------------------
# NowcastGenerator
# ---------------------------------------------------------------------------

class NowcastGenerator:
    """Generates nowcast frames from the latest radar data.

    May consult external ``NowcastContribution`` sources (region-keyed)
    for regions where an upstream agency publishes their own forecast
    leg directly (e.g. JMA HRPN for JPCOMP).  Where an external source
    returns data for the expected validtime, it's used as-is.  Where it
    doesn't (or returns ``None``), the region falls back to internal
    optical-flow extrapolation.
    """

    def __init__(
        self,
        store,
        nowcast_store: NowcastStore,
        cache=None,
        nowcast_contributions: list | None = None,
    ):
        self._store = store          # FrameStore (past radar)
        self._nowcast_store = nowcast_store
        self._cache = cache          # TileCache (for invalidation)
        self._contributions = list(nowcast_contributions or [])
        self._by_region = {
            c.region_name: c for c in self._contributions
        }

    async def generate(self) -> None:
        """Generate nowcast frames from the two most recent radar frames.

        Called after each fetch cycle.  Runs the optical flow computation
        in a thread to avoid blocking the event loop.  External nowcast
        contributions are fetched first (async) and passed into the sync
        path; regions without a contribution fall through to optical-flow
        extrapolation.
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

        # Fetch external nowcast contributions in parallel.  Each call
        # returns a list of (validtime_unix, frame_data) or None.  We
        # iterate every registered contribution, NOT just regions present
        # in latest_frame — the external source publishes its forecast
        # independently of our analysis fetch, and may carry the only
        # signal for the region if the latest analysis slot was missed.
        from librewxr.data.regions import REGIONS as _ALL_REGIONS
        external_by_region: dict[str, dict[int, np.ndarray]] = {}
        fetch_tasks = []
        fetch_region_names = []
        for contrib in self._contributions:
            region_def = _ALL_REGIONS.get(contrib.region_name)
            if region_def is None:
                continue
            fetch_tasks.append(contrib.instance.fetch_forecast(region_def))
            fetch_region_names.append(contrib.region_name)
        if fetch_tasks:
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for region_name, result in zip(fetch_region_names, results):
                if isinstance(result, Exception):
                    logger.warning(
                        "External nowcast for %s raised: %s", region_name, result,
                    )
                    continue
                if result is None:
                    continue
                external_by_region[region_name] = {
                    ts: frame for (ts, frame) in result
                }

        # Run CPU-heavy extrapolation in a thread
        nowcast_frames, flows = await asyncio.to_thread(
            self._generate_sync,
            prev_frame.regions, latest_frame.regions,
            latest_ts, n_steps, interval,
            settings.nowcast_blend_mode,
            external_by_region,
        )

        # Swap into the store and invalidate old tile cache entries
        old_timestamps = await self._nowcast_store.replace_all(nowcast_frames)
        await self._nowcast_store.replace_flows(flows)
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
        blend_mode: str = "blended",
        external_by_region: dict[str, dict[int, np.ndarray]] | None = None,
    ) -> list[NowcastFrame]:
        """Synchronous nowcast generation (runs in a thread).

        For regions present in ``external_by_region``, the external
        validtime → frame mapping is consulted first for each forecast
        step; only when an external frame is missing for the target
        validtime does the optical-flow extrapolation run.  Regions
        not in the mapping at all run optical-flow for every step,
        matching the pre-contribution behaviour exactly.
        """
        t0 = time.monotonic()
        external_by_region = external_by_region or {}

        # Pre-compute flow per region.  Regions fully served by an
        # external contribution still get a flow computed because the
        # extrapolation path may be needed for any step where the
        # external source returned no frame (transient miss).
        flows: dict[str, np.ndarray] = {}
        for region_name in latest_regions:
            data0 = prev_regions.get(region_name)
            data1 = latest_regions.get(region_name)
            if data0 is None or data1 is None:
                continue
            flows[region_name] = _compute_flow(data0, data1)

        # Regions to forecast = anything we can extrapolate internally
        # OR anything an external source published.  External-only is
        # the common JPCOMP case when JMA's analysis fetch for the
        # latest slot is partial — the external N2 forecast still
        # carries authoritative model output for the region.
        forecast_regions: set[str] = set(flows.keys()) | set(external_by_region.keys())
        if not forecast_regions:
            return [], {}

        # Generate extrapolated frames for each step
        frames: list[NowcastFrame] = []
        for step in range(1, n_steps + 1):
            nowcast_ts = latest_ts + step * interval
            # Blend weight controls how much radar extrapolation vs IFS
            # forecast is used.  1.0 = pure radar, 0.0 = pure IFS.
            # Beyond 60 min, always fall back to pure IFS regardless of
            # mode because radar extrapolation becomes too inaccurate.
            max_blend_steps = max(1, 3600 // interval)  # 6 at 10-min cadence
            if step > max_blend_steps:
                blend_weight = 0.0
            elif blend_mode == "radar":
                blend_weight = 1.0
            elif blend_mode == "model":
                blend_weight = 0.0
            else:  # "blended" (default)
                # Tuned for HRRR's native-resolution, dBZ-matched output.
                # Starts at ~82% radar at T+10 for a smooth transition off
                # the live frame, crosses to model-dominant by T+40, lands
                # at 80% HRRR by T+60.  When the chain falls back to IFS
                # (outside HRRR's CONUS domain) the same curve still
                # applies — radar dominance early, model dominance late.
                t = step / max_blend_steps
                blend_weight = 0.20 + 0.80 * (1.0 - t) ** 1.4

            regions: dict[str, np.ndarray] = {}
            for region_name in forecast_regions:
                external = external_by_region.get(region_name)
                external_frame = external.get(nowcast_ts) if external else None
                # No per-pixel boundary feathering: the internal
                # extrapolation seeds from the same upstream analysis
                # as the external contribution (e.g. JMA HRPN N1 →
                # FrameStore → both nowcast paths), so both produce
                # zeros wherever the upstream has no coverage.  Per-
                # step replacement is sufficient; mixing them per
                # pixel would add noise rather than fill a real gap.
                if external_frame is not None:
                    regions[region_name] = external_frame
                    continue
                flow = flows.get(region_name)
                data = latest_regions.get(region_name)
                if flow is not None and data is not None:
                    regions[region_name] = _extrapolate_forward(data, flow, step)
                # else: no external for this step, no internal flow —
                # skip this region for this step.  Renderer falls back
                # to NWP fill which is the correct behaviour for an
                # uncovered region.

            frames.append(NowcastFrame(
                timestamp=nowcast_ts,
                regions=regions,
                blend_weight=blend_weight,
            ))

        elapsed = time.monotonic() - t0
        logger.info(
            "Nowcast generation: %d frames × %d regions "
            "(%d internal, %d external) (%.1fs)",
            len(frames), len(forecast_regions),
            len(flows), len(external_by_region), elapsed,
        )
        return frames, flows


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

    # Note: intensity preservation (rescaling warped pixels to match
    # source mean) was removed because bilinear interpolation only
    # loses ~1-2% per step, while new low-value boundary pixels from
    # spreading inflated the correction ratio and caused a visible
    # intensity jump on the first forecast frame.

    return warped
