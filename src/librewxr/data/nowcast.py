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

Phase 1 (optical-flow computation) is also reused by the
``/v2/radar`` motion-arrow overlay when ``nowcast_enabled=false`` but
``arrow_flow_enabled=true``.  In that state ``generate()`` runs Phase 1
only — at a lower target resolution (tuned for the arrow draw grid) —
skipping extrapolation entirely.  The Farneback vectors land in
``NowcastStore._flows`` exactly as in the nowcast-on path, so the arrow
renderer at ``routes.radar_tile`` reads them transparently.  See
``LIBREWXR_ARROW_FLOW_ENABLED`` / ``LIBREWXR_ARROW_FLOW_TARGET_DIM``
in ``config.py`` for the tunables.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from librewxr.config import settings

logger = logging.getLogger(__name__)

# Target longest dimension for optical flow computation.
# Larger grids are downscaled to this for speed.  The flow is stored
# (and memmapped) at this reduced resolution — the full-resolution
# field is only materialised transiently at warp time in
# ``_extrapolate_forward``, or sampled at reduced resolution by the
# arrow overlay.  Persisted flow storage for USCOMP drops from
# ~527 MB to ~3.5 MB this way (see ``_compute_flow_low`` /
# ``_upscale_flow``).
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

# Coverage-degradation guard: if the latest radar frame for a region has
# lost more than this fraction of its non-zero pixels vs the previous
# frame, skip optical-flow extrapolation for that region this cycle.
# Belt-and-braces alongside the flow-magnitude clamp below — catches
# catastrophic partial-frame scenarios the clamp can't see.
_COVERAGE_DEGRADATION_RATIO = 0.7

# Don't apply the degradation guard when the previous frame had only
# a small smattering of precipitation — natural variation can swing
# small counts by large fractions without anything being wrong.
_MIN_PREV_NONZERO_PX = 1000

# Physical cap on flow magnitude: 200 km/h is the upper extreme of
# storm motion (severe convective cells embedded in a strong jet).
# Any per-pixel flow vector implying motion faster than this is an
# optical-flow artifact at a data/no-data boundary — the Farneback
# polynomial fit gets unreliable where bright pixels sit next to zero
# pixels, and the algorithm reports vectors of 50-200+ px/step.  When
# ``_extrapolate_forward`` inverse-warps with those vectors it samples
# boundary brightness into the no-data region, producing the vertical
# streak artifact visible in CACOMP at MRMS/MSC boundaries and
# anywhere else with irregular coverage.  Clamping in km/h units (vs
# raw pixels) makes the cap scale correctly across radar grids that
# range from 0.01° to 0.05° per pixel.
_MAX_FLOW_KM_PER_HOUR = 200.0

# Wrap-padding width (in columns) for full-longitude (global) regions
# whose grid crosses the ±180° seam — RRQPE's observed-precip tier.
# The seam must be *periodic* for both Farneback and the inverse-warp:
# content that advects across the dateline re-enters on the other side
# instead of being zeroed at a hard edge.  The pad needs to exceed the
# maximum advection distance, which is bounded by the flow clamp
# (``_max_flow_pixels`` ≈ 7.5 px/step at 0.04° → ≤ ~45 px over a 6-step
# nowcast) — 256 px covers that at any plausible region resolution and
# stays a small fraction of a global grid's width; ``width // 8`` is
# the effective cap for tiny synthetic grids so the pad never dominates.
_WRAP_FLOW_PAD = 256

# ── Composite NWP flow raster geometry ────────────────────────────────
#
# The arrow overlay outside radar coverage historically fell through to
# a single IFS optical-flow field (``ecmwf_grid._flow``).  The hybrid
# arrow path replaces that with a *composite* flow raster built from
# ``NWPChain.sample()`` over a fixed lat/lon grid at T and T−1, so it
# reflects whichever regional NWP source is active at each point
# (HRRR over CONUS, ICON-EU over Europe, JMA MSM over Japan, ...) and
# not just IFS.  The per-region radar flow (``NowcastStore._flows``)
# still wins inside radar coverage; this raster only fills NWP-only
# pixels, so the radar-coverage boundary-artifact problem that
# ``_clamp_flow`` / the coverage-degradation guard exist to contain
# does not apply here — NWP grids have full-domain coverage with no
# station-range holes, and feather seams are smooth precip transitions.
#
# North/South/West are fixed; the resolution comes from
# ``settings.arrow_nwp_flow_resolution_deg`` (default 0.25°).  The
# renderer derives the per-pixel step from the loaded array's shape
# so a knob change between cycles can't drift out of sync.
NWP_FLOW_NORTH = 90.0
NWP_FLOW_SOUTH = -90.0
NWP_FLOW_WEST = -180.0

# Process-lifetime nowcast worker pool.  Phase A (per-region Farneback
# optical flow) and Phase B (per-step × per-region cv2.remap warp) are
# embarrassingly parallel and cv2 releases the GIL, so a 4-thread pool
# scales real speedup.  Lazily created on first use so deployments that
# never run nowcast (or the arrow-flow path) don't pay for the pool;
# never shut down — it lives for the process lifetime.
_NOWCAST_POOL: concurrent.futures.ThreadPoolExecutor | None = None


def _nowcast_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Return the process-lifetime nowcast thread pool (lazily created)."""
    global _NOWCAST_POOL
    if _NOWCAST_POOL is None:
        _NOWCAST_POOL = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="nowcast",
        )
    return _NOWCAST_POOL


def _coverage_degraded(prev: np.ndarray, latest: np.ndarray) -> tuple[bool, int, int]:
    """Detect partial-coverage degradation between two consecutive frames.

    Returns ``(is_degraded, prev_nonzero, latest_nonzero)``.  Caller
    logs the counts on degradation so the rack log shows exactly which
    region tripped the guard and by how much.
    """
    prev_nz = int(np.count_nonzero(prev))
    latest_nz = int(np.count_nonzero(latest))
    if prev_nz < _MIN_PREV_NONZERO_PX:
        return False, prev_nz, latest_nz
    return latest_nz < _COVERAGE_DEGRADATION_RATIO * prev_nz, prev_nz, latest_nz


def _max_flow_pixels(
    pixel_size_lat_deg: float, interval_seconds: int,
    max_km_per_hour: float = _MAX_FLOW_KM_PER_HOUR,
) -> float:
    """Convert the km/h flow cap to a pixel magnitude bound.

    Uses latitude pixel size (uniform ~111 km/°, independent of
    latitude) as the reference axis.  Lon pixels are narrower toward
    the poles (cos(lat) factor), so the resulting clamp slightly
    over-restricts east-west motion at high latitudes — acceptable
    because the artifact vectors we target exceed any physical motion
    by 5-10x regardless.
    """
    max_km_per_step = max_km_per_hour * interval_seconds / 3600.0
    km_per_px = pixel_size_lat_deg * 111.0
    return max_km_per_step / km_per_px


def _clamp_flow(flow: np.ndarray, max_magnitude_px: float) -> np.ndarray:
    """Cap per-pixel flow magnitudes at ``max_magnitude_px``.

    Vectors with magnitude below the cap pass through unchanged;
    over-cap vectors are scaled down to the cap while preserving
    direction.  Returns the original array when nothing needs clamping
    (the common path) so we avoid an allocation per cycle on regions
    with well-behaved flow fields.
    """
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    over = mag > max_magnitude_px
    if not np.any(over):
        return flow
    scale = np.where(
        over, max_magnitude_px / np.maximum(mag, 1e-9), 1.0,
    ).astype(np.float32)
    clamped = flow.copy()
    clamped[..., 0] = (dx * scale).astype(np.float32)
    clamped[..., 1] = (dy * scale).astype(np.float32)
    return clamped


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

    def __init__(
        self, cache_dir: Path | None = None, *, cleanup_tmp: bool = True,
    ):
        self._frames: dict[int, NowcastFrame] = {}
        # Per-region optical flow, stored at the resolution it was
        # COMPUTED at (longest dim ≤ _TARGET_FLOW_DIM, vectors in
        # full-res pixel units) — consumers upscale at the point of use.
        self._flows: dict[str, np.ndarray] = {}
        # Composite NWP optical-flow field (one global raster) used by
        # the arrow overlay outside radar coverage.  Distinct filename
        # prefix (``nwp_flow.dat``) keeps it clear of the radar-flow
        # ``flow_*.dat`` cleanup glob and its own replace path below.
        self._nwp_flow: np.ndarray | None = None
        # Monotonic content version for the flow fields (per-region radar
        # flows + composite NWP raster).  Bumped on every replace swap and
        # shipped via state.json so render workers can key shared-store
        # overlay tiles by flow content identity.
        self._flow_version: int = 0
        self._lock = asyncio.Lock()
        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "nowcast"
            self._persistent = True
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_nowcast_"))
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        # The ``*.tmp`` unlink is a stale-leftover sweep for the store's
        # OWN dir.  A render worker resurrecting a shared (multi-mode)
        # store mid-run must skip it — the pipeline process may be
        # concurrently writing ``.dat.tmp`` files it is about to rename.
        if cleanup_tmp:
            for path in self._memmap_dir.glob("*.tmp"):
                path.unlink(missing_ok=True)
        logger.debug(
            "Nowcast memmap directory: %s (persistent=%s)",
            self._memmap_dir, self._persistent,
        )

    def _to_memmap(self, name: str, data: np.ndarray) -> np.ndarray:
        """Write array to disk atomically and return a read-only memory-mapped view."""
        final = self._memmap_dir / f"{name}.dat"
        # The nowcast dir is shared across processes in multi mode (the
        # pipeline writes it, render workers read it via state.json).  A
        # deterministic tmp name lets a concurrent writer's rename steal
        # the file out from under this writer's os.replace (production
        # incident); pid+uuid makes writers independent — both succeed,
        # and the last replace wins the final name atomically.  The
        # constructor's stale-``*.tmp`` sweep still matches these names.
        tmp = final.with_name(
            f"{final.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
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
        """Update the latest optical flow vectors.

        Flows are stored at their computed (reduced, ≤ target-dim)
        resolution in full-res pixel units — see ``_compute_flow_low``.
        Consumers that need full-resolution coordinates upscale at the
        point of use (``_upscale_flow`` for the warp path; coordinate-
        mapped sampling in the arrow overlay and storm-cell detection).
        """
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
            self._flow_version += 1

    async def get_flows(self) -> dict[str, np.ndarray]:
        """Return the latest per-region optical flow vectors."""
        async with self._lock:
            return dict(self._flows)

    async def replace_nwp_flow(self, flow: np.ndarray | None) -> None:
        """Update the single composite NWP optical-flow raster (or clear it).

        The NWP flow is a single global field, so unlike ``replace_flows``
        there's no per-region dict to swap — just one memmap-backed
        array stored as ``nwp_flow.dat``.  Passing ``None`` clears it
        (arrows outside radar coverage then simply don't render until
        the next cycle rebuilds it).
        """
        async with self._lock:
            old = self._nwp_flow
            self._nwp_flow = None
            if old is not None and hasattr(old, "filename"):
                try:
                    Path(str(old.filename)).unlink(missing_ok=True)
                except OSError:
                    pass
            if flow is None:
                return
            self._nwp_flow = self._to_memmap("nwp_flow", flow)
            self._flow_version += 1

    async def get_nwp_flow(self) -> np.ndarray | None:
        """Return the composite NWP optical-flow raster, or ``None``."""
        async with self._lock:
            return self._nwp_flow

    @property
    def flow_version(self) -> int:
        """Content version for the flow fields (radar + NWP composite).

        Bumped on every ``replace_flows`` / ``replace_nwp_flow`` swap.
        Synchronous and lock-free on purpose: the version is a plain
        attribute read that only the pipeline mutates (under the async
        lock), so a read under the GIL always sees a consistent value;
        render workers receive it via the state.json snapshot.
        """
        return self._flow_version

    @property
    def data_bytes(self) -> int:
        """Total bytes across all nowcast frame arrays and flow fields."""
        total = 0
        for frame in self._frames.values():
            for arr in frame.regions.values():
                total += arr.nbytes
        for arr in self._flows.values():
            total += arr.nbytes
        if self._nwp_flow is not None:
            total += int(self._nwp_flow.nbytes)
        return total

    def clear(self) -> None:
        self._frames.clear()
        self._flows.clear()
        self._nwp_flow = None

    def __getstate__(self) -> dict | None:
        """Serialize state for cross-process reload (multi-worker mode).

        Returns ``None`` while the store holds no content — an all-empty
        store means the first generation is still in flight after a
        pipeline boot, and dumping it would null populated nowcast stores
        on serving render workers (production incident).  ``dump_state``
        skips stores whose ``__getstate__`` returns ``None``, so the
        worker keeps its current frames until the first real generation
        lands.  The all-three-empty condition keeps arrow-flow-only
        configurations correct: a store with flows but no frames is a
        valid, dumpable state.
        """
        if not self._frames and not self._flows and self._nwp_flow is None:
            return None
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
        nwp_flow_state = None
        if self._nwp_flow is not None:
            nwp_flow_state = [
                os.path.basename(str(self._nwp_flow.filename)),
                self._nwp_flow.dtype.str,
                list(self._nwp_flow.shape),
            ]
        return {
            "memmap_dir": str(self._memmap_dir),
            "frames": frames_state,
            "flows": flows_state,
            "nwp_flow": nwp_flow_state,
            "flow_version": self._flow_version,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state from the dict produced by ``__getstate__``.

        Stale memmap files are tolerated — a snapshot can reference files
        the pipeline has since deleted (dump/generation ordering window),
        so missing files degrade the store instead of failing it: a frame
        with any missing region file is skipped wholesale (a partial frame
        would render misleading partial tiles), a missing flow file skips
        just that region's flow (arrows for the region suppress until the
        next cycle), and a missing ``nwp_flow`` file becomes ``None``.
        Genuine corruption (other exceptions) still propagates so
        ``apply_state`` can log it.
        """
        # Belt-and-suspenders apply-side guard: an all-empty payload
        # carries no information and historically meant "first generation
        # in flight" — keep serving the current frames/flows.  With
        # ``__getstate__`` returning ``None`` for empty stores, current
        # pipelines never emit such payloads; this only defends against
        # snapshots from older builds or hand-crafted payloads.  A store
        # that is itself empty applies the payload normally (a no-op),
        # and a payload with frames=[] but non-empty flows (arrow-only
        # path) still applies — it is a valid, information-bearing state.
        incoming_empty = (
            not state.get("frames")
            and not state.get("flows")
            and state.get("nwp_flow") is None
        )
        holding_content = (
            bool(self._frames) or bool(self._flows) or self._nwp_flow is not None
        )
        if incoming_empty and holding_content:
            return
        memmap_dir = Path(state["memmap_dir"])
        new_frames: dict[int, NowcastFrame] = {}
        for f_info in state["frames"]:
            ts = int(f_info["timestamp"])
            frame = NowcastFrame(
                timestamp=ts,
                blend_weight=float(f_info["blend_weight"]),
            )
            try:
                for name, (basename, dtype_str, shape) in f_info["regions"].items():
                    frame.regions[name] = np.memmap(
                        memmap_dir / basename,
                        dtype=np.dtype(dtype_str), mode="r",
                        shape=tuple(shape),
                    )
            except FileNotFoundError:
                # Stale frame file race — the pipeline replaced the set of
                # frames between dump and this read.  Skip the WHOLE frame:
                # a partial frame would render misleading partial tiles.
                logger.debug("Nowcast: skipping stale frame %d", ts)
                continue
            new_frames[ts] = frame

        new_flows: dict[str, np.ndarray] = {}
        for name, (basename, dtype_str, shape) in state["flows"].items():
            try:
                new_flows[name] = np.memmap(
                    memmap_dir / basename,
                    dtype=np.dtype(dtype_str), mode="r",
                    shape=tuple(shape),
                )
            except FileNotFoundError:
                # Stale flow file race — skip just this region's flow;
                # arrows for the region suppress until the next cycle.
                logger.debug("Nowcast: skipping stale flow %s", name)

        new_nwp_flow = None
        # Older snapshots written before the hybrid arrow path landed
        # omit ``nwp_flow``; treat absence the same as "not computed".
        nwp_state = state.get("nwp_flow")
        if nwp_state is not None:
            nw_basename, nw_dtype, nw_shape = nwp_state
            try:
                new_nwp_flow = np.memmap(
                    memmap_dir / nw_basename,
                    dtype=np.dtype(nw_dtype), mode="r",
                    shape=tuple(nw_shape),
                )
            except FileNotFoundError:
                # Stale nwp_flow file race — the arrow overlay outside
                # radar coverage simply doesn't render until the next cycle.
                new_nwp_flow = None

        self._memmap_dir = memmap_dir
        self._frames = new_frames
        self._flows = new_flows
        self._nwp_flow = new_nwp_flow
        if "flow_version" in state:
            self._flow_version = int(state["flow_version"])
        else:
            # Legacy snapshot from an older pipeline (pre ``flow_version``).
            # Conservative fallback: this store's payload always differs
            # per cycle, so ``apply_state`` reloads it every cycle, and
            # bumping on every apply matches the flow regeneration cadence.
            self._flow_version = self._flow_version + 1
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
            logger.debug("Nowcast memmap directory cleaned up")


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
        nwp_chain=None,
    ):
        self._store = store          # FrameStore (past radar)
        self._nowcast_store = nowcast_store
        self._cache = cache          # TileCache (for invalidation)
        self._contributions = list(nowcast_contributions or [])
        self._by_region = {
            c.region_name: c for c in self._contributions
        }
        # NWPChain used by the hybrid arrow path to build the composite
        # flow raster outside radar coverage.  ``None`` in tests or in
        # deployments with every NWP source disabled; the generator
        # then simply skips Phase A-NWP and arrows outside radar
        # coverage don't render.
        self._nwp_chain = nwp_chain

    async def generate(self) -> None:
        """Generate nowcast frames from the two most recent radar frames.

        Called after each fetch cycle.  Runs the optical flow computation
        in a thread to avoid blocking the event loop.  External nowcast
        contributions are fetched first (async) and passed into the sync
        path; regions without a contribution fall through to optical-flow
        extrapolation.

        Two configurably-independent paths share this entry point:

        * ``nowcast_enabled=true``  — full path: Farneback optical flow
          + per-step extrapolation + IFS blend, swapped into
          ``NowcastStore._frames`` for the nowcast tile endpoint.
        * ``nowcast_enabled=false, arrow_flow_enabled=true`` — flow
          only: Farneback (at a reduced target resolution — see
          ``LIBREWXR_ARROW_FLOW_TARGET_DIM``) computed for every
          region with ≥2 frames, swapped into ``NowcastStore._flows``,
          extrapolation skipped entirely.  The ``/v2/radar`` motion-
          arrow overlay reads the flows the same way it does in the
          nowcast-on state, so arrow direction is correct regardless
          of the nowcast toggle.

        Disabling both flags makes this method a no-op (no optical-
        flow CPU, no arrow vectors — the arrow renderer then suppresses
        the overlay via its own ``arrow_style`` gate).
        """
        if not settings.nowcast_enabled and not settings.arrow_flow_enabled:
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
        extrapolate = settings.nowcast_enabled

        # External nowcast contributions are part of the extrapolation
        # phase only — the arrow-flow path doesn't need them (it never
        # produces forecast frames), so skip the async fetch entirely
        # and let ``external_by_region`` stay empty.
        external_by_region: dict[str, dict[int, np.ndarray]] = {}
        if extrapolate:
            # Fetch external nowcast contributions in parallel.  Each call
            # returns a list of (validtime_unix, frame_data) or None.  We
            # iterate every registered contribution, NOT just regions present
            # in latest_frame — the external source publishes its forecast
            # independently of our analysis fetch, and may carry the only
            # signal for the region if the latest analysis slot was missed.
            from librewxr.data.regions import REGIONS as _ALL_REGIONS
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

        # Run CPU-heavy work in a thread.  ``extrapolate`` controls both
        # the target flow resolution (full for the nowcast feed to
        # cv2.remap; reduced for arrows which downsample flow ~10-30x
        # while drawing) and whether Phase B (per-step extrapolation)
        # runs at all.
        nowcast_frames, flows = await asyncio.to_thread(
            self._generate_sync,
            prev_frame.regions, latest_frame.regions,
            latest_ts, n_steps, interval,
            settings.nowcast_blend_mode,
            external_by_region,
            extrapolate,
        )

        # The flows swap is unconditional — the arrow overlay depends on
        # it in both the nowcast-on and arrow-flow-only paths.
        await self._nowcast_store.replace_flows(flows)

        # Phase A-NWP — composite global flow raster for the hybrid
        # arrow path.  Only the arrow overlay reads this (nowcast
        # extrapolation doesn't), so gate on ``arrow_flow_enabled``.
        # When the flag is off, clear any prior raster so a stale field
        # can't leak arrows in a deployment that just disabled it.
        if settings.arrow_flow_enabled and self._nwp_chain is not None:
            nwp_flow = await asyncio.to_thread(
                self._compute_nwp_flow_sync, prev_ts, latest_ts, interval,
            )
            await self._nowcast_store.replace_nwp_flow(nwp_flow)
        else:
            await self._nowcast_store.replace_nwp_flow(None)

        # The frames swap only matters in the nowcast-on path; skipping
        # it on the arrow-flow-only path leaves NowcastStore._frames
        # empty, which is exactly what ``routes.radar_tile`` expects for
        # a nowcast-disabled deployment (no nowcast tiles to serve).
        if extrapolate and nowcast_frames:
            old_timestamps = await self._nowcast_store.replace_all(nowcast_frames)
            if self._cache is not None:
                for ts in old_timestamps:
                    self._cache.invalidate_timestamp(ts)
            logger.debug(
                "Nowcast updated: %d frames (T+%d to T+%d min)",
                len(nowcast_frames),
                interval // 60,
                n_steps * interval // 60,
            )
        elif flows:
            logger.debug(
                "Arrow flow updated: %d region%s (nowcast disabled)",
                len(flows), "s" if len(flows) != 1 else "",
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
        extrapolate: bool = True,
    ) -> tuple[list[NowcastFrame], dict[str, np.ndarray]]:
        """Synchronous nowcast generation (runs in a thread).

        Phase A — optical flow: for every region present in both
        ``prev_regions`` and ``latest_regions``, compute Farneback flow
        between the two frames, apply the coverage-degradation guard
        and the magnitude clamp, and accumulate into ``flows``.  The
        target flow resolution is ``_TARGET_FLOW_DIM`` when
        ``extrapolate`` is true or ``settings.arrow_flow_target_dim``
        when false (the arrow overlay downsamples flow ~10-30x while
        drawing, so a high-resolution field is wasted work).  Flows are
        stored at the resolution they were COMPUTED at (≤ target dim)
        with vectors pre-scaled to full-resolution pixel units — the
        full-res field is only materialised transiently at warp time
        (``_extrapolate_forward``) or sampled at reduced resolution by
        the arrow overlay, never persisted.

        Phase B — extrapolation: for each forecast step, prefer an
        external ``NowcastContribution`` frame for the validtime when
        one was returned; otherwise inverse-warp the latest radar
        forward along the precomputed flow.  Skipped entirely when
        ``extrapolate`` is false (the arrow-flow-only path), in which
        case this method returns ``( [], flows )`` after Phase A.

        For regions present in ``external_by_region``, the external
        validtime → frame mapping is consulted first for each forecast
        step; only when an external frame is missing for the target
        validtime does the optical-flow extrapolation run.  Regions
        not in the mapping at all run optical-flow for every step,
        matching the pre-contribution behaviour exactly.
        """
        t0 = time.monotonic()
        external_by_region = external_by_region or {}
        # Flow target resolution: _TARGET_FLOW_DIM for the nowcast
        # extrapolation feed (stored at this reduced resolution and
        # upscaled only at warp time); reduced for the arrow-only path
        # (arrows draw on a 32/48px grid and downsample the flow
        # anyway).
        flow_target_dim = (
            _TARGET_FLOW_DIM if extrapolate else settings.arrow_flow_target_dim
        )

        # Pre-compute flow per region.  Regions fully served by an
        # external contribution still get a flow computed because the
        # extrapolation path may be needed for any step where the
        # external source returned no frame (transient miss).
        flows: dict[str, np.ndarray] = {}
        # Unclamped low-res flows for the extrapolation phase (bit-exact
        # warp path — see the clamp comment in the helper) plus the
        # per-region clamp bound.  Both are tiny (≤ target-dim arrays).
        warp_flows: dict[str, np.ndarray] = {}
        flow_clamps: dict[str, float] = {}
        # Phase A runs one optical-flow task per region over the nowcast
        # pool — cv2 releases the GIL, so Farneback work scales across
        # threads.  ``executor.map`` preserves input order, so results
        # come back in the same order as the ``latest_regions``
        # iteration; assembly happens here in the driving thread only
        # (task bodies never mutate shared dicts).
        region_results = list(_nowcast_pool().map(
            lambda r: _compute_region_flow(
                r, prev_regions.get(r), latest_regions.get(r),
                flow_target_dim, interval,
            ),
            latest_regions,
        ))
        for result in region_results:
            if result is None:
                continue  # missing prev/latest frame or coverage-degraded skip
            region_name, clamped, flow_unclamped, clamp_bound = result
            flows[region_name] = clamped
            if clamp_bound is not None:
                flow_clamps[region_name] = clamp_bound
            warp_flows[region_name] = flow_unclamped

        # Arrow-flow-only path: Phase A is the whole job.  Return an
        # empty frame list so the caller skips the nowcast store swap
        # (``replace_all``) but still publishes ``flows`` via
        # ``replace_flows``.  Mirrors the existing empty-set precedent
        # below (no forecast regions → return [], {}).
        if not extrapolate:
            elapsed = time.monotonic() - t0
            logger.info(
                "Arrow flow generation: %d region%s (%.1fs, target_dim=%d)",
                len(flows), "s" if len(flows) != 1 else "",
                elapsed, flow_target_dim,
            )
            return [], flows

        # Regions to forecast = anything we can extrapolate internally
        # OR anything an external source published.  External-only is
        # the common JPCOMP case when JMA's analysis fetch for the
        # latest slot is partial — the external N2 forecast still
        # carries authoritative model output for the region.
        forecast_regions: set[str] = set(flows.keys()) | set(external_by_region.keys())
        if not forecast_regions:
            return [], {}

        # Precompute the float32 mgrid coordinate grids ONCE per region
        # instead of rebuilding them for every forecast step — xs/ys are
        # identical across all steps (only map_x/map_y, which are
        # steps·flow away from them, vary per step).  Built up-front for
        # every region that has both a warp flow and data so the
        # parallel Phase B tasks below only read them (the old lazy
        # per-step build mutated a shared dict, which the pool forbids).
        coord_grids: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for region_name in forecast_regions:
            flow = warp_flows.get(region_name)
            data = latest_regions.get(region_name)
            if flow is not None and data is not None:
                h, w = data.shape
                ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
                coord_grids[region_name] = (ys, xs)

        # Generate extrapolated frames for each step.  Phase B submits
        # one task per (step, region) pair over the nowcast pool and
        # assembles the final frames strictly in step order 1..n_steps
        # here in the driving thread; within each step, regions are
        # inserted in ``sorted(forecast_regions)`` order for
        # determinism.  Task bodies read shared dicts only.
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

            results = list(_nowcast_pool().map(
                lambda r: _extrapolate_region_step(
                    r, nowcast_ts, step,
                    external_by_region.get(r),
                    warp_flows, latest_regions, coord_grids, flow_clamps,
                ),
                sorted(forecast_regions),
            ))
            regions: dict[str, np.ndarray] = {}
            for result in results:
                if result is None:
                    # No external frame for this step and no internal
                    # flow — skip this region for this step.  Renderer
                    # falls back to NWP fill which is the correct
                    # behaviour for an uncovered region.
                    continue
                region_name, frame_data = result
                regions[region_name] = frame_data

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

    def _compute_nwp_flow_sync(
        self, prev_ts: int, latest_ts: int, interval: int,
    ) -> np.ndarray | None:
        """Build the composite NWP optical-flow raster for the hybrid arrow path.

        Samples ``NWPChain.sample()`` over the fixed lat/lon grid (extent
        set by ``NWP_FLOW_NORTH/SOUTH/WEST``, resolution by
        ``settings.arrow_nwp_flow_resolution_deg``) at ``prev_ts`` and
        ``latest_ts`` to get two composite precip snapshots reflecting
        whichever regional NWP source is active at each point, then runs
        Farneback between them.  One pass replaces the IFS-only flow
        field that the arrow overlay previously special-cased — every
        regional source (HRRR / ICON-EU / JMA MSM / ...) draws arrows
        automatically once it joins the chain, with no per-source
        ``sample_flow`` plumbing.

        Units are raster-pixels per ``interval``-second step, matching
        how the per-region radar flow is also per-step.  The renderer
        finite-diffs the (lat, lon) → raster (row, col) mapping for the
        Jacobian, just as it does for radar regions.

        Returns ``None`` when the chain has no data at either timestamp
        — the arrow overlay then simply doesn't render outside radar
        coverage until the next cycle rebuilds it.
        """
        chain = self._nwp_chain
        if chain is None or not chain.has_data():
            return None

        res = settings.arrow_nwp_flow_resolution_deg
        lat_count = int(round((NWP_FLOW_NORTH - NWP_FLOW_SOUTH) / res)) + 1
        lon_count = int(round(360.0 / res))
        lats = np.linspace(
            NWP_FLOW_NORTH, NWP_FLOW_SOUTH, lat_count, dtype=np.float32,
        )
        lons = np.linspace(
            NWP_FLOW_WEST, NWP_FLOW_WEST + 360.0 - res, lon_count,
            dtype=np.float32,
        )
        # np.meshgrid(indexing='xy') → lon_grid, lat_grid both shape
        # (lat_count, lon_count); row index runs N→S, col index W→E.
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        prev = chain.sample(lat_grid, lon_grid, prev_ts, bilinear=False)
        latest = chain.sample(lat_grid, lon_grid, latest_ts, bilinear=False)

        # If the second snapshot is all-zero (NWP fetch gap),
        # Farneback would happily produce spurious flow from noise; bail.
        if not prev.any() or not latest.any():
            return None

        flow = _compute_flow(
            prev, latest, target_dim=settings.arrow_flow_target_dim,
        )
        return flow


# ---------------------------------------------------------------------------
# Optical flow helpers
# ---------------------------------------------------------------------------

def _compute_flow_low(
    frame0: np.ndarray, frame1: np.ndarray,
    target_dim: int = _TARGET_FLOW_DIM,
) -> tuple[np.ndarray, float]:
    """Compute dense optical flow at reduced resolution.

    Downscales so the longest dimension is ~``target_dim`` pixels,
    computes Farneback flow, and returns ``(flow_small, scale)`` where
    ``flow_small`` is the flow on the small grid (vector units are
    small-grid pixels per step — NOT yet scaled to full resolution) and
    ``scale`` is the downscale factor
    (``min(target_dim / max(h, w), 1.0)``; ``1.0`` when no downscaling
    happened).  ``target_dim`` defaults to the module constant
    ``_TARGET_FLOW_DIM`` (the nowcast extrapolation feed); the
    arrow-only path passes ``settings.arrow_flow_target_dim`` (lower —
    arrows draw on a coarse grid, so a high-resolution field is wasted
    work).

    Callers choose where to upscale: ``_compute_flow`` materialises the
    full-resolution field (legacy one-shot contract, still used by the
    composite NWP flow raster and tests), while the nowcast pipeline
    stores the small field and only upscales it transiently at warp
    time — see ``_upscale_flow`` and the ``NowcastStore`` storage note.
    """
    h, w = frame0.shape
    scale = min(target_dim / max(h, w), 1.0)

    if scale < 1.0:
        small_h = max(1, int(h * scale))
        small_w = max(1, int(w * scale))
        small0 = cv2.resize(frame0, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small1 = cv2.resize(frame1, (small_w, small_h), interpolation=cv2.INTER_AREA)

        flow_small = cv2.calcOpticalFlowFarneback(
            small0, small1, flow=None, **_FARNEBACK,
        )
        return flow_small, scale

    flow = cv2.calcOpticalFlowFarneback(
        frame0, frame1, flow=None, **_FARNEBACK,
    )
    return flow, 1.0


def _upscale_flow(
    flow_low: np.ndarray, target_shape: tuple[int, int],
) -> np.ndarray:
    """Upscale a reduced-resolution flow field to ``target_shape``.

    ``flow_low`` must be in *full-resolution pixel units* — callers
    pre-multiply the small-grid vectors by ``1/scale`` when the field is
    stored (see ``_generate_sync``).  Because ``cv2.resize`` is a
    linear operation, ``resize(v * c) == resize(v) * c`` bitwise for
    float32, so upscaling a pre-scaled low-res field reproduces the
    legacy ``_compute_flow`` result (which resized first and scaled
    afterwards) exactly — no further magnitude scaling is applied here.

    Returns ``flow_low`` unchanged when the shapes already match (the
    region was small enough to never downscale).
    """
    h, w = target_shape
    if flow_low.shape[0] == h and flow_low.shape[1] == w:
        return flow_low
    return cv2.resize(flow_low, (w, h), interpolation=cv2.INTER_LINEAR)


def _compute_flow(
    frame0: np.ndarray, frame1: np.ndarray,
    target_dim: int = _TARGET_FLOW_DIM,
) -> np.ndarray:
    """Compute dense optical flow, upscaled to the input resolution.

    Downscales so the longest dimension is ~``target_dim`` pixels,
    computes Farneback flow, then upscales the flow vectors to the
    original resolution (resize first, then scale by ``1/scale`` — the
    operation ``_upscale_flow`` reproduces at warp time).  Kept as a
    convenience for callers that need the full-resolution field in one
    shot (the composite NWP flow raster and tests); the nowcast
    pipeline itself stores the reduced-resolution field instead.
    """
    h, w = frame0.shape
    flow_small, scale = _compute_flow_low(frame0, frame1, target_dim)
    if scale < 1.0:
        flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
        flow *= 1.0 / scale  # scale vectors to full resolution
        return flow
    return flow_small


def _compute_region_flow(
    region_name: str,
    data0: np.ndarray | None,
    data1: np.ndarray | None,
    flow_target_dim: int,
    interval: int,
) -> tuple[str, np.ndarray, np.ndarray, float | None] | None:
    """Phase A per-region optical-flow task (runs on the nowcast pool).

    Returns ``None`` when the region must be skipped this cycle —
    either a prev/latest frame is missing (silent skip, as before) or
    the coverage-degradation guard tripped (the warning is logged here;
    logging from worker threads is fine).  Otherwise returns
    ``(region_name, clamped_flow, unclamped_flow, clamp_bound)`` where
    ``clamped_flow`` is the stored field (identical to the unclamped
    field when nothing needed clamping or the region has no ``REGIONS``
    entry), ``unclamped_flow`` feeds the bit-exact warp path, and
    ``clamp_bound`` is the per-region km/h-derived pixel cap, set only
    when the low-res clamp actually fired (``None`` otherwise).
    """
    if data0 is None or data1 is None:
        return None
    degraded, prev_nz, latest_nz = _coverage_degraded(data0, data1)
    if degraded:
        logger.warning(
            "Nowcast: %s coverage degraded (%d → %d non-zero px) — "
            "skipping optical-flow extrapolation to avoid streak "
            "artifacts from partial-frame motion estimation",
            region_name, prev_nz, latest_nz,
        )
        return None

    from librewxr.data.regions import REGIONS as _ALL_REGIONS  # local import: avoid circular at module load
    region_def = _ALL_REGIONS.get(region_name)
    wrap_pad = 0
    if region_def is not None and region_def.is_global:
        # Full-longitude grid: wrap-pad the column axis so Farneback sees
        # the content across the ±180° seam as adjacent (periodic), not as
        # a hard edge.  Pixels away from the seam are untouched — their
        # neighbourhoods are unchanged — so the flow for every non-global
        # region is bit-identical to the unpadded path.
        wrap_pad = min(_WRAP_FLOW_PAD, max(1, data0.shape[1] // 8))
        data0 = np.pad(data0, ((0, 0), (wrap_pad, wrap_pad)), mode="wrap")
        data1 = np.pad(data1, ((0, 0), (wrap_pad, wrap_pad)), mode="wrap")

    flow_small, scale = _compute_flow_low(
        data0, data1, target_dim=flow_target_dim,
    )
    if wrap_pad:
        # Slice the padding back off the computed flow so the stored /
        # warp-time fields keep the unpadded grid shape.  The central
        # region's vectors were computed against the wrapped content and
        # are correct; only the pad columns are discarded.
        pad_small = max(1, int(round(wrap_pad * scale)))
        flow_small = flow_small[:, pad_small:-pad_small, :]

    # Store the flow at the resolution it was computed at, with
    # vectors pre-multiplied by 1/scale so they stay in
    # full-resolution pixel units.  Consumers upscale with
    # ``_upscale_flow`` at the point of use (warp time / arrow
    # sampling); because cv2.resize is linear this is bitwise
    # identical to the legacy store-full-res pipeline.  USCOMP
    # flow storage drops from ~527 MB to ~3.5 MB.
    flow_unclamped = (
        flow_small * (1.0 / scale)
        if scale < 1.0 else flow_small
    )
    # Cap unphysical motion vectors before extrapolation.  Without
    # this, Farneback's polynomial fit at data/no-data boundaries
    # reports 50-200+ px/step magnitudes, which the inverse-warp
    # then renders as vertical streaks of fake precipitation.
    #
    # Clamp placement (see also ``_extrapolate_forward``):
    #  * The STORED field (arrows, storm-cell detection) is
    #    clamped at low resolution.  Upscaling is a linear
    #    convex-combination operation, so every upscaled vector
    #    has magnitude ≤ the cap whenever the low-res source is
    #    capped — the max-displacement guarantee for consumers
    #    holds exactly as before, and when no vector exceeds the
    #    cap the clamp is a no-op identical to the old order.
    #  * The WARP path stays bit-identical to the old pipeline
    #    by clamping the UPSCALED field at warp time (the old
    #    code clamped the full-res field once after upscaling).
    #    Clamping before upscaling would NOT reproduce it: an
    #    over-cap vector spreads over a ~1/scale² full-res
    #    neighbourhood, and capping it before the spread changes
    #    the interpolated directions/magnitudes in that band.
    #    ``warp_flows`` therefore carries the unclamped field and
    #    ``flow_clamps`` the per-region cap; when the low-res
    #    clamp below is a no-op (no over-cap vector anywhere),
    #    the upscaled field is provably ≤ the cap too, so the
    #    warp-time clamp is skipped and both paths share one
    #    array.  The stored clamped field is what the arrow
    #    overlay / storm cells sample, keeping them bounded.
    clamp_bound = None
    if region_def is not None:
        ps_y = (
            region_def.pixel_size_y
            if region_def.pixel_size_y > 0
            else region_def.pixel_size
        )
        max_px = _max_flow_pixels(ps_y, interval)
        clamped = _clamp_flow(flow_unclamped, max_px)
        if clamped is not flow_unclamped:
            clamp_bound = max_px
    else:
        clamped = flow_unclamped
    return region_name, clamped, flow_unclamped, clamp_bound


def _extrapolate_forward(
    frame: np.ndarray, flow: np.ndarray, steps: int,
    xs: np.ndarray | None = None,
    ys: np.ndarray | None = None,
    max_px: float | None = None,
    wrap: bool = False,
) -> np.ndarray:
    """Warp *frame* forward by *steps* × flow using inverse remap.

    For each output pixel p, samples *frame* at ``p − steps·flow(p)``.
    After warping, rescales the result to preserve the total precipitation
    energy of the source frame — bilinear interpolation in cv2.remap
    tends to smooth peak values, causing artificial intensity loss.

    ``flow`` is normally the store's reduced-resolution field (see
    ``_compute_flow_low``); it is upscaled to the frame grid here at
    warp time (``_upscale_flow`` — the same resize math the old code
    applied at storage time, so results are numerically identical).
    Full-resolution flows (small regions, tests) pass through
    untouched.  ``xs``/``ys`` are optional precomputed float32
    coordinate grids — ``_generate_sync`` builds them once per region
    and reuses them across forecast steps (they are identical for every
    step); when omitted they are built here (direct-call / test path).

    ``max_px`` optionally applies ``_clamp_flow`` to the upscaled field
    right here — the old pipeline clamped the full-res field once after
    upscaling, and clamping after upscale (not before) is what keeps
    the warp output bit-identical.  ``_generate_sync`` passes the
    per-region cap only when the stored low-res clamp actually fired;
    otherwise the upscaled field is provably within the cap already
    (bilinear upscaling is a convex combination) and the clamp would be
    a no-op, so it is skipped to avoid a full-res magnitude pass per
    step.

    ``wrap`` selects the periodic-seam path for full-longitude (global)
    regions: the frame and flow are wrap-padded on the column axis
    (``_WRAP_FLOW_PAD`` — larger than the clamp-bounded maximum
    displacement, so central samples never leave the padded frame), the
    map is built on the padded grid, and the warp uses
    ``cv2.BORDER_WRAP`` so any sample that does fall outside the padded
    frame still lands on the periodic continuation rather than zero.
    The result is cropped back to the unpadded width.  ``False`` (every
    regional radar composite) reproduces the legacy behaviour exactly —
    no padding, BORDER_CONSTANT, no crop.
    """
    h, w = frame.shape
    if flow.shape[0] != h or flow.shape[1] != w:
        flow = _upscale_flow(flow, (h, w))
    if max_px is not None:
        flow = _clamp_flow(flow, max_px)
    if wrap:
        pad = min(_WRAP_FLOW_PAD, w // 8)
        frame = np.pad(frame, ((0, 0), (pad, pad)), mode="wrap")
        flow = np.pad(flow, ((0, 0), (pad, pad), (0, 0)), mode="wrap")
        # Coordinate grid over the PADDED frame (column indices are P-space
        # already — the crop back below restores the unwrapped view).  The
        # row axis never wraps.
        ys, xs = np.mgrid[0:h, 0 : w + 2 * pad].astype(np.float32)
    elif xs is None or ys is None:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    map_x = xs - steps * flow[..., 0]
    map_y = ys - steps * flow[..., 1]

    warped = cv2.remap(
        frame, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=(
            cv2.BORDER_WRAP if wrap else cv2.BORDER_CONSTANT
        ),
        borderValue=0,
    )
    if wrap:
        warped = warped[:, pad : pad + w]

    # Note: intensity preservation (rescaling warped pixels to match
    # source mean) was removed because bilinear interpolation only
    # loses ~1-2% per step, while new low-value boundary pixels from
    # spreading inflated the correction ratio and caused a visible
    # intensity jump on the first forecast frame.

    return warped


def _extrapolate_region_step(
    region_name: str,
    nowcast_ts: int,
    step: int,
    external: dict[int, np.ndarray] | None,
    warp_flows: dict[str, np.ndarray],
    latest_regions: dict[str, np.ndarray],
    coord_grids: dict[str, tuple[np.ndarray, np.ndarray]],
    flow_clamps: dict[str, float],
) -> tuple[str, np.ndarray] | None:
    """Phase B per-(step, region) extrapolation task (runs on the pool).

    External ``NowcastContribution`` frames take precedence for the
    validtime and are returned as-is; otherwise the latest radar is
    inverse-warped forward along the precomputed flow using the
    prebuilt coordinate grids.  Returns ``None`` when neither applies —
    the no-region marker; the renderer falls back to NWP fill, which is
    the correct behaviour for an uncovered region.  Reads shared dicts
    only — never mutates them.
    """
    external_frame = external.get(nowcast_ts) if external else None
    # No per-pixel boundary feathering: the internal
    # extrapolation seeds from the same upstream analysis
    # as the external contribution (e.g. JMA HRPN N1 →
    # FrameStore → both nowcast paths), so both produce
    # zeros wherever the upstream has no coverage.  Per-
    # step replacement is sufficient; mixing them per
    # pixel would add noise rather than fill a real gap.
    if external_frame is not None:
        return region_name, external_frame
    flow = warp_flows.get(region_name)
    data = latest_regions.get(region_name)
    if flow is not None and data is not None:
        ys, xs = coord_grids[region_name]
        # Full-longitude regions wrap at the ±180° seam — the warp pads
        # the column axis periodically so seam-crossing advection
        # re-enters on the other side instead of zeroing.
        from librewxr.data.regions import REGIONS as _ALL_REGIONS
        region_def = _ALL_REGIONS.get(region_name)
        wrap = bool(region_def is not None and region_def.is_global)
        return region_name, _extrapolate_forward(
            data, flow, step, xs=xs, ys=ys,
            max_px=flow_clamps.get(region_name),
            wrap=wrap,
        )
    # No external for this step, no internal flow — skip this region
    # for this step.  Renderer falls back to NWP fill which is the
    # correct behaviour for an uncovered region.
    return None
