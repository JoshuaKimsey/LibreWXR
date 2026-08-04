# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Per-cycle storm-cell detection on radar frames.  Stores detected cells
(centroids + motion vectors) for the renderer's ?cells= overlay and the
MCP get_storm_cells tool (deferred)."""

import asyncio
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from librewxr.config import settings
from librewxr.data.regions import REGIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Fixed-size cell array per region.  Memmap serialization needs a bounded
# size, and 256 is generous -- hitting the cap would require ~500+ distinct
# convective cells in a single 1000x1000 region frame at the 40 dBZ cutoff,
# which is meteorologically implausible.  Cells beyond the cap are silently
# dropped (an acceptable degradation in extreme outbreaks).
MAX_CELLS_PER_REGION = 256

# Structured dtype for one detected cell.  Stored as a fixed-size numpy
# array per region for memmap-friendly serialization.  Field access via
# ``arr['centroid_row'][i]`` or ``arr[i]['centroid_row']`` (numpy void
# scalar).  NaN in centroid_row indicates padding (unused slots beyond
# the actual count); consumers iterate until they hit NaN.
_CELL_DTYPE = np.dtype([
    ("centroid_row", np.float32),     # row of cell centroid in region pixels
    ("centroid_col", np.float32),     # col of cell centroid in region pixels
    ("area_px", np.float32),          # number of pixels in the cell
    ("area_km2", np.float32),         # area converted to km^2 (approximation)
    ("max_dbz", np.float32),          # max decoded dBZ within the cell
    ("motion_dx_px", np.float32),     # flow x-component at centroid (pixels/cycle)
    ("motion_dy_px", np.float32),     # flow y-component at centroid (pixels/cycle)
    ("motion_speed_kmh", np.float32), # derived speed in km/h (NaN if no flow)
    ("motion_heading_deg", np.float32),  # compass heading 0=N, 90=E (NaN if no flow)
])


# ---------------------------------------------------------------------------
# StormCellStore
# ---------------------------------------------------------------------------

class StormCellStore:
    """Lightweight store for detected storm cells.

    Cells are regenerated every fetch cycle (the radar frame moves and
    cells appear/disappear), so no max-cells eviction is needed -- just an
    atomic swap of the per-region cell arrays each cycle.  Cell arrays are
    backed by memory-mapped temp files so the OS page cache manages
    physical RAM and the multi-worker render-only path can memmap the same
    files via state.json.
    """

    def __init__(self, cache_dir: Path | None = None):
        self._cells: dict[str, np.ndarray] = {}
        self._counts: dict[str, int] = {}  # actual cell count per region (vs MAX cap)
        self._last_updated: float = 0.0
        self._detected_at_timestamp: int = 0
        self._lock = asyncio.Lock()
        if cache_dir is not None:
            self._memmap_dir = Path(cache_dir) / "storm_cells"
            self._persistent = True
        else:
            self._memmap_dir = Path(tempfile.mkdtemp(prefix="librewxr_stormcells_"))
            self._persistent = False
        self._memmap_dir.mkdir(parents=True, exist_ok=True)
        for path in self._memmap_dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        logger.info(
            "Storm-cell memmap directory: %s (persistent=%s)",
            self._memmap_dir, self._persistent,
        )

    def _to_memmap(self, name: str, data: np.ndarray) -> np.ndarray:
        """Write array to disk atomically and return a read-only memmap view."""
        final = self._memmap_dir / f"{name}.dat"
        tmp = final.with_suffix(".dat.tmp")
        mm = np.memmap(tmp, dtype=data.dtype, mode="w+", shape=data.shape)
        mm[:] = data
        mm.flush()
        del mm
        os.replace(tmp, final)
        return np.memmap(final, dtype=data.dtype, mode="r", shape=data.shape)

    async def replace_cells(
        self,
        cells_by_region: dict[str, np.ndarray],
        detected_at_timestamp: int = 0,
    ) -> None:
        """Atomically replace all detected cells.

        ``cells_by_region`` maps region name -> structured ndarray of shape
        ``(count,)`` with dtype ``_CELL_DTYPE``.  Arrays shorter than
        ``MAX_CELLS_PER_REGION`` are zero-padded with NaN rows to a fixed
        memmap size so reload via ``__setstate__`` is straightforward;
        ``self._counts`` records the actual count.
        """
        async with self._lock:
            # Clean up old cell memmap files
            for path in self._memmap_dir.glob("cells_*.dat"):
                try:
                    path.unlink()
                except OSError:
                    pass

            new_cells: dict[str, np.ndarray] = {}
            new_counts: dict[str, int] = {}
            for region_name, arr in cells_by_region.items():
                count = len(arr)
                # Pad to fixed size with NaN rows for stable memmap shape.
                padded = np.full((MAX_CELLS_PER_REGION,), np.nan, dtype=_CELL_DTYPE)
                if count > 0:
                    if count > MAX_CELLS_PER_REGION:
                        logger.warning(
                            "Region %s produced %d cells (cap=%d); truncating.",
                            region_name, count, MAX_CELLS_PER_REGION,
                        )
                        count = MAX_CELLS_PER_REGION
                    padded[:count] = arr[:count]
                memmap = self._to_memmap(f"cells_{region_name}", padded)
                new_cells[region_name] = memmap
                new_counts[region_name] = count

            self._cells = new_cells
            self._counts = new_counts
            self._last_updated = time.time()
            self._detected_at_timestamp = detected_at_timestamp

    async def get_cells(self) -> dict[str, np.ndarray]:
        """Return the latest per-region cell arrays (fixed-size, NaN-padded).

        Each array has shape ``(MAX_CELLS_PER_REGION,)`` and dtype
        ``_CELL_DTYPE``.  Consumers should iterate the first
        ``self._counts[region]`` rows (or scan until NaN in
        centroid_row) -- rows beyond the count are NaN-padded.
        """
        async with self._lock:
            return dict(self._cells)

    async def get_counts(self) -> dict[str, int]:
        """Return the actual cell count per region (vs the MAX cap)."""
        async with self._lock:
            return dict(self._counts)

    @property
    def last_updated(self) -> float:
        """Unix timestamp of the last ``replace_cells`` call."""
        return self._last_updated

    @property
    def detected_at_timestamp(self) -> int:
        """The radar frame timestamp the current detection was run on."""
        return self._detected_at_timestamp

    @property
    def total_count(self) -> int:
        """Total detected cells across all regions (sum of per-region counts)."""
        return sum(self._counts.values())

    def __getstate__(self) -> dict[str, Any]:
        """Serialize state for cross-process reload (multi-worker mode)."""
        cells_state: dict[str, list] = {}
        for name, arr in self._cells.items():
            # Use descr for structured dtypes so field names survive the
            # round-trip (arr.dtype.str returns "V36" which loses fields).
            cells_state[name] = [
                os.path.basename(str(arr.filename)),
                arr.dtype.descr,
                list(arr.shape),
            ]
        return {
            "memmap_dir": str(self._memmap_dir),
            "cells": cells_state,
            "counts": dict(self._counts),
            "last_updated": self._last_updated,
            "detected_at_timestamp": self._detected_at_timestamp,
        }

    def __setstate__(self, state: dict) -> None:
        """Restore state from the dict produced by ``__getstate__``."""
        memmap_dir = Path(state["memmap_dir"])
        new_cells: dict[str, np.ndarray] = {}
        for name, (basename, dtype_info, shape) in state["cells"].items():
            # dtype_info comes from dtype.descr as a list of (name, format)
            # tuples, but JSON serialization (dump_state -> json.dumps ->
            # json.loads -> apply_state) converts tuples to lists.  numpy
            # requires tuples for structured dtype field specs, so convert
            # each inner list back to a tuple.  tuple(tuple(x)) is a no-op
            # when x is already a tuple (the in-memory test path).
            dtype = np.dtype([tuple(item) for item in dtype_info])
            new_cells[name] = np.memmap(
                memmap_dir / basename,
                dtype=dtype, mode="r",
                shape=tuple(shape),
            )
        self._memmap_dir = memmap_dir
        self._cells = new_cells
        self._counts = dict(state.get("counts", {}))
        self._last_updated = float(state.get("last_updated", 0.0))
        self._detected_at_timestamp = int(state.get("detected_at_timestamp", 0))
        self._persistent = True
        if not hasattr(self, "_lock"):
            self._lock = asyncio.Lock()

    def cleanup(self) -> None:
        """Release memmap file handles (called on shutdown)."""
        self._cells.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# Detection algorithm
# ---------------------------------------------------------------------------

def detect_storm_cells(
    latest_frame_regions: dict[str, np.ndarray],
    enabled_regions: list[str],
    flows_by_region: dict[str, np.ndarray] | None,
    min_dbz: int,
    min_area_km2: float,
    fetch_interval_s: int,
) -> dict[str, np.ndarray]:
    """Detect storm cells on the latest radar frames.

    For each region in ``latest_frame_regions``, threshold the uint8 frame
    at ``min_dbz`` (decoded via the inverse of ``_dbz_float_to_uint8``),
    run ``cv2.connectedComponentsWithStats`` to label contiguous cells,
    filter by minimum area, optionally sample the optical flow at each
    cell's centroid for a motion vector, and return a structured ndarray
    per region.

    Args:
        latest_frame_regions: region name -> uint8 (H, W) dBZ-encoded frame.
        enabled_regions: region names to consider (others are skipped).
        flows_by_region: optional region name -> (H, W, 2) float32 optical
            flow field (vectors in full-res pixel units; may be stored at
            reduced resolution, ≤ 1000 px target dim).  When ``None`` or
            a region is absent, motion fields are set to 0 / NaN.
        min_dbz: minimum dBZ threshold for a pixel to be part of a cell.
        min_area_km2: minimum cell area in km^2.
        fetch_interval_s: the flow field's time interval in seconds (used
            to convert pixel displacement to km/h).

    Returns:
        Region name -> structured ndarray of shape ``(count,)`` with dtype
        ``_CELL_DTYPE``.  Empty regions produce an empty (0-row) array.
    """
    # Convert dBZ threshold to uint8 pixel threshold (inverse of
    # _dbz_float_to_uint8: pixel = (dBZ + 32) * 2).
    pixel_threshold = int(round((min_dbz + 32.0) * 2.0))
    pixel_threshold = max(1, min(pixel_threshold, 255))  # ignore pixel 0 (NODATA)

    results: dict[str, np.ndarray] = {}

    for region_name, frame_uint8 in latest_frame_regions.items():
        if region_name not in enabled_regions:
            continue
        region = REGIONS.get(region_name)
        if region is None:
            continue

        # Threshold: pixels >= threshold are part of candidate cells.
        # ascontiguousarray guards against memmap-backed frames that may
        # not be C-contiguous -- cv2 requires a contiguous uint8 input.
        binary = np.ascontiguousarray(
            (frame_uint8 >= pixel_threshold).astype(np.uint8)
        )

        # cv2.connectedComponentsWithStats returns:
        #   num_labels (int), labels (H, W int32), stats (N, 5 int32),
        #   centroids (N, 2 float64) where each row is (cx, cy) --
        #   cx is the column-mean, cy is the row-mean.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8,
        )
        if num_labels <= 1:
            # Only the background label -- no cells.
            results[region_name] = np.empty(0, dtype=_CELL_DTYPE)
            continue

        # Per-pixel km^2 conversion (approximation -- 1 deg lat ~ 111 km;
        # cos(lat) factor for lon ignored at midlatitudes for simplicity).
        # region.pixel_size is degrees per pixel; _ps_y is the y-axis
        # equivalent.  Cell area in km^2 = area_px * (ps_lon_km * ps_lat_km).
        ps_lon_km = region.pixel_size * 111.0
        ps_lat_km = region._ps_y * 111.0
        px_to_km2 = ps_lon_km * ps_lat_km
        min_area_px = min_area_km2 / px_to_km2 if px_to_km2 > 0 else 0

        flow = None
        if flows_by_region is not None:
            flow = flows_by_region.get(region_name)

        cells_list: list[tuple] = []
        for label in range(1, num_labels):  # skip label 0 (background)
            area_px = int(stats[label, cv2.CC_STAT_AREA])
            if area_px < min_area_px:
                continue

            # cv2 returns (cx, cy) where cx is column-mean, cy is row-mean.
            centroid_col = float(centroids[label][0])
            centroid_row = float(centroids[label][1])

            # Max dBZ within the cell -- decode from uint8 pixel.
            cell_mask = labels == label
            cell_pixels = frame_uint8[cell_mask]
            max_pixel = int(cell_pixels.max())
            max_dbz = float(max_pixel) / 2.0 - 32.0  # decode_dbz formula

            area_km2 = area_px * px_to_km2

            # Motion vector from optical flow at the centroid (if available).
            motion_dx_px = 0.0
            motion_dy_px = 0.0
            motion_speed_kmh = float("nan")
            motion_heading_deg = float("nan")
            if flow is not None:
                # Sample flow at the centroid pixel.  Flows are stored
                # at the resolution they were computed at (≤ 1000 px
                # target dim, vectors in full-res pixel units — see
                # nowcast._compute_flow_low), so full-res centroid
                # coordinates are mapped into the stored grid with the
                # same center mapping cv2.resize uses when it upscales a
                # low-res field.  Full-res fields (small regions, tests)
                # sample directly as before.
                fh, fw = flow.shape[0], flow.shape[1]
                frame_h, frame_w = frame_uint8.shape
                if fh == frame_h and fw == frame_w:
                    fr = int(round(centroid_row))
                    fc = int(round(centroid_col))
                else:
                    fr = int(round((centroid_row + 0.5) * fh / frame_h - 0.5))
                    fc = int(round((centroid_col + 0.5) * fw / frame_w - 0.5))
                fr = max(0, min(fr, fh - 1))
                fc = max(0, min(fc, fw - 1))
                fx = float(flow[fr, fc, 0])  # pixel displacement over fetch_interval
                fy = float(flow[fr, fc, 1])

                motion_dx_px = fx
                motion_dy_px = fy

                # Convert pixel displacement to km/h.  Flow represents
                # displacement over ``fetch_interval_s`` seconds.
                dx_km = fx * ps_lon_km
                dy_km = fy * ps_lat_km
                speed_km_per_fetch = math.sqrt(dx_km * dx_km + dy_km * dy_km)
                if fetch_interval_s > 0:
                    motion_speed_kmh = speed_km_per_fetch * (3600.0 / fetch_interval_s)
                else:
                    motion_speed_kmh = float("nan")

                # Compass heading: 0=N, 90=E, 180=S, 270=W.
                # Row 0 is north (top of frame), so -dy_km is "northward".
                # atan2(east_km, north_km) -> radians, convert to deg, mod 360.
                if speed_km_per_fetch > 1e-9:
                    east_km = dx_km
                    north_km = -dy_km
                    heading_rad = math.atan2(east_km, north_km)
                    motion_heading_deg = (math.degrees(heading_rad) + 360.0) % 360.0

            cells_list.append((
                centroid_row, centroid_col,
                float(area_px), area_km2, max_dbz,
                motion_dx_px, motion_dy_px,
                motion_speed_kmh, motion_heading_deg,
            ))

        if not cells_list:
            results[region_name] = np.empty(0, dtype=_CELL_DTYPE)
        else:
            arr = np.array(cells_list, dtype=_CELL_DTYPE)
            results[region_name] = arr

    return results


# ---------------------------------------------------------------------------
# StormCellGenerator
# ---------------------------------------------------------------------------

class StormCellGenerator:
    """Cycle driver that runs storm-cell detection on the latest radar frame.

    Called once per fetch cycle (after ``_run_nowcast`` so it can reuse the
    just-computed optical flow).  Runs the CPU-bound detection work in a
    thread via ``asyncio.to_thread`` so the event loop isn't blocked.
    Stores the result via ``StormCellStore.replace_cells``.  Cells missing
    the flow (nowcast disabled, or region outside any flow field) get NaN
    motion -- the cells are still detected, just drawn without arrows by
    the renderer.
    """

    def __init__(
        self,
        frame_store,
        storm_cell_store: StormCellStore,
        nowcast_store=None,
    ):
        self._frame_store = frame_store
        self._storm_cell_store = storm_cell_store
        self._nowcast_store = nowcast_store  # optional -- for flow sampling

    async def generate(self) -> None:
        """Detect storm cells on the latest frame and store them.

        Silently no-ops if the frame store is empty (cold start) -- the
        previous cycle's cells (if any) stay in the store until the next
        successful cycle.
        """
        latest = None
        try:
            latest = await self._frame_store.get_latest_frame()
            if latest is None:
                return

            # Read the latest flow (if available) so the sync worker has it.
            flows_by_region = None
            if self._nowcast_store is not None:
                flows_by_region = await self._nowcast_store.get_flows() or None

            # Run the CPU-bound detection in a thread.
            cells_by_region = await asyncio.to_thread(
                _detect_sync,
                latest.regions,
                settings.get_enabled_regions(),
                flows_by_region,
                settings_fetch_interval(),
                settings_min_dbz(),
                settings_min_area_km2(),
            )

            await self._storm_cell_store.replace_cells(
                cells_by_region,
                detected_at_timestamp=latest.timestamp,
            )
            total = sum(len(arr) for arr in cells_by_region.values())
            logger.info(
                "Storm-cell detection: %d cells across %d region(s)",
                total, len(cells_by_region),
            )
        except Exception:
            logger.exception(
                "StormCellGenerator.generate() failed -- "
                "latest=%s, frame_regions=%s, nowcast_store=%s",
                latest is not None,
                list(latest.regions.keys()) if latest is not None else [],
                type(self._nowcast_store).__name__ if self._nowcast_store else None,
            )
            raise


def _detect_sync(
    latest_regions: dict[str, np.ndarray],
    enabled_regions: list[str],
    flows_by_region: dict[str, np.ndarray] | None,
    fetch_interval_s: int,
    min_dbz: int,
    min_area_km2: float,
) -> dict[str, np.ndarray]:
    """Sync worker -- thin wrapper around ``detect_storm_cells`` for ``to_thread``."""
    return detect_storm_cells(
        latest_regions, enabled_regions, flows_by_region,
        min_dbz, min_area_km2, fetch_interval_s,
    )


# ---------------------------------------------------------------------------
# Settings indirection functions (Phase A defaults; Phase B promotes to config)
# ---------------------------------------------------------------------------

def settings_fetch_interval() -> int:
    """Indirection so tests can monkeypatch without touching real config."""
    return settings.fetch_interval


def settings_min_dbz() -> int:
    """Minimum dBZ for a pixel to be part of a storm cell."""
    return settings.storm_cells_min_dbz


def settings_min_area_km2() -> float:
    """Minimum area in km^2 for a detected cell."""
    return settings.storm_cells_min_area_km2
