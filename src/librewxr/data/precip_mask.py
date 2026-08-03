# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Per-timestamp global precip mask for the multi-worker empty-tile fast path.

The old Tier 2 gate probed the IFS precip envelope per source and per tile
bbox; on live traffic it returned 0 events because the IFS envelope is far
wider than the actual precipitation (the bbox covers the whole storm
system footprint).  This module replaces that with a single coarse boolean
grid per radar/nowcast timestamp, built by the pipeline from EVERY
source's combined contribution:

- radar region arrays forward-sampled onto the coarse grid,
- all NWP chain sources sampled on the coarse grid (via
  ``nwp_chain.sample``), and
- nowcast region arrays (Tier 3 folded in — one mechanism covers the
  past-radar, no-radar-overlap, AND nowcast paths).

The grid is 720x360 cells over the full globe.  Each cell is 0.5 deg
(720 cells span 360 deg of longitude, 360 cells span 180 deg of
latitude); at one bool per cell a full mask is ~260 KB.  Masks are
dilated by 1 coarse cell (with antimeridian wrap) so precip that lands
between coarse cell centers still trips the gate.

Render workers query ``has_precip_in_bbox(timestamp, tile_bbox)`` in
O(1) (a memmap slice + ``any()``) to decide whether to skip the
expensive ``nwp_chain.sample`` + blend.  Multi-mode only — not
instantiated in single mode.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import numpy as np

from librewxr.data.regions import REGIONS, RegionDef
from librewxr.tiles.coordinates import _laea_pixel_coords, _tmerc_pixel_coords

logger = logging.getLogger(__name__)


class PrecipMaskStore:
    """Per-timestamp global precip mask, coarse boolean grid (720x360).

    Built by the pipeline from every source's combined contribution (radar
    region arrays OR'd with all NWP source samples via ``nwp_chain.sample``
    OR'd with nowcast region arrays), then dilated by 1 coarse cell with
    antimeridian wrap to handle center-sampling aliasing and
    antimeridian-straddling tile bboxes.

    Render workers query ``has_precip_in_bbox(timestamp, tile_bbox)`` in O(1)
    to decide whether to skip ``nwp_chain.sample``.  Multi-mode only; not
    instantiated in single mode.
    """

    GRID_WIDTH = 720   # 720 cells = 360 deg of longitude at 0.5 deg/cell
    GRID_HEIGHT = 360  # 360 cells = 180 deg of latitude at 0.5 deg/cell
    # One cell per 0.5 deg: the meshgrid built in ``build`` spans
    # -180..180 in GRID_WIDTH cells and -90..90 in GRID_HEIGHT cells, so
    # the cell-index math ``(x - WEST) / PIXEL_SIZE`` lines up with the
    # array dimensions exactly.
    PIXEL_SIZE = 0.5
    WEST = -180.0
    NORTH = 90.0

    def __init__(self, cache_dir: Path | None = None):
        self._masks: dict[int, np.ndarray] = {}
        self._version: int = 0
        self._memmap_dir: Path | None = (Path(cache_dir) / "mask") if cache_dir else None
        self._persistent = bool(cache_dir)
        # Per-timestamp NWP mask cache (pipeline side only; never
        # serialized).  Keyed by the NWP signature below so an unchanged
        # NWP state skips the expensive chain sample entirely.
        self._nwp_cache: dict[int, np.ndarray] | None = None
        self._nwp_signature: tuple | None = None
        # Coarse global meshgrid (built once per ``build`` call; used by
        # the region projection + NWP sampling).
        self._latlon_meshgrid: tuple[np.ndarray, np.ndarray] | None = None
        if self._memmap_dir is not None:
            self._memmap_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    async def build(self, stores: dict, nwp_chain, settings) -> None:
        """Rebuild all per-timestamp masks from the current store contents.

        ``stores`` is the pipeline's store dict (``frame_store``,
        ``nowcast_store`` + the NWP grids).  ``nwp_chain`` supplies the
        NWP half of the OR; ``settings`` supplies the noise floor used by
        the renderer (must match so the gate never masks a visible pixel).
        """
        frame_store = stores.get("frame_store")
        if frame_store is None:
            # Nothing to build from — empty the mask table so stale
            # masks can't produce false "has precip" hits.
            self._masks = {}
            self._cleanup_stale_files()
            return

        nowcast_store = stores.get("nowcast_store")
        radar_ts = await frame_store.get_timestamps()
        if nowcast_store is not None:
            nowcast_ts = await nowcast_store.get_timestamps()
        else:
            nowcast_ts = []
        timestamps = sorted(set(radar_ts) | set(nowcast_ts))
        if not timestamps:
            self._masks = {}
            self._cleanup_stale_files()
            return

        pixel_threshold = (
            int((settings.noise_floor_dbz + 32) * 2)
            if settings.noise_floor_dbz > -32 else 0
        )

        # Coarse global meshgrid, cell centers at half-cell offsets so the
        # index math ``int((x - WEST) / PIXEL_SIZE)`` buckets correctly.
        self._ensure_meshgrid()
        lat_grid, lon_grid = self._latlon_meshgrid

        # NWP half of the OR, gated by the chain's content signature: an
        # unchanged NWP state (same timestep counts + same runs) reuses
        # the cached per-timestamp masks instead of re-sampling the whole
        # coarse grid from every source.
        nwp_masks = self._nwp_cache
        if nwp_chain is not None and nwp_chain.has_data():
            sig = self._nwp_signature_of(nwp_chain)
            reuse = (
                sig == self._nwp_signature
                and nwp_masks is not None
                and all(ts in nwp_masks for ts in timestamps)
            )
            if not reuse:
                if nwp_masks is None:
                    nwp_masks = {}
                for ts in timestamps:
                    nwp_values = await asyncio.to_thread(
                        nwp_chain.sample, lat_grid, lon_grid, ts,
                        bilinear=False,
                    )
                    nwp_masks[ts] = nwp_values >= pixel_threshold
                self._nwp_cache = nwp_masks
                self._nwp_signature = sig
        elif nwp_chain is not None:
            # Chain present but empty — no NWP contribution; drop the
            # cache so a later data arrival can't reuse stale masks.
            self._nwp_cache = None
            self._nwp_signature = None
            nwp_masks = None
        else:
            nwp_masks = None

        new_masks: dict[int, np.ndarray] = {}
        for ts in timestamps:
            frame = await frame_store.get_frame(ts)
            region_arrays: dict[str, np.ndarray] = {}
            if frame is not None:
                region_arrays = dict(frame.regions)
            elif nowcast_store is not None:
                nc_frame, _blend = await nowcast_store.get_frame(ts)
                if nc_frame is not None:
                    region_arrays = dict(nc_frame.regions)
            nwp_mask = nwp_masks.get(ts) if nwp_masks is not None else None
            mask = await asyncio.to_thread(
                self._build_timestamp_mask_sync,
                ts, region_arrays, nwp_mask, pixel_threshold,
            )
            self._save_mask(new_masks, ts, mask)

        self._masks = new_masks
        self._version += 1
        self._cleanup_stale_files()

    def _build_timestamp_mask_sync(
        self,
        ts: int,
        region_arrays: dict[str, np.ndarray],
        nwp_mask: np.ndarray | None,
        pixel_threshold: int,
    ) -> np.ndarray:
        """Build one timestamp's mask from pre-fetched region arrays.

        Sync helper (no async scaffolding) so tests can exercise the
        projection / OR / dilation math directly.  ORs the radar/nowcast
        region contributions with the precomputed NWP mask, then dilates
        by 1 coarse cell with antimeridian wrap.
        """
        mask = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=bool)
        self._ensure_meshgrid()
        for region_name, region_array in region_arrays.items():
            region = REGIONS.get(region_name)
            if region is None:
                logger.debug(
                    "precip mask: unknown region %r for ts=%d, skipping",
                    region_name, ts,
                )
                continue
            self._project_region(mask, region, region_array, pixel_threshold)
        if nwp_mask is not None:
            mask |= nwp_mask
        return self._dilate(mask)

    def _ensure_meshgrid(self) -> None:
        """Build (once per instance) the coarse cell-center meshgrid."""
        if self._latlon_meshgrid is not None:
            return
        lat = np.linspace(
            self.NORTH - 0.125, -self.NORTH + 0.125, self.GRID_HEIGHT,
            dtype=np.float32,
        )
        lon = np.linspace(
            self.WEST + 0.125, -self.WEST - 0.125, self.GRID_WIDTH,
            dtype=np.float32,
        )
        self._latlon_meshgrid = np.meshgrid(lat, lon, indexing="ij")

    def _project_region(
        self, mask: np.ndarray, region: RegionDef, region_array: np.ndarray,
        pixel_threshold: int,
    ) -> None:
        """Sample ``region_array`` on the coarse meshgrid and OR hits in.

        Uses the same lat/lon -> row/col projection math as the renderer
        (``region_pixel_indices_fractional`` family), so a radar pixel the
        renderer would draw at threshold lands in the same coarse cell
        here.  Handles latlon (rectilinear), LAEA (OPERA), and tmerc
        (DPC Italy) regions uniformly.
        """
        lat_grid, lon_grid = self._latlon_meshgrid
        if region.proj == "laea":
            # Helpers meshgrid internally; pass the 1D axes.
            col_grid, row_grid = _laea_pixel_coords(
                lon_grid[0, :], lat_grid[:, 0], region,
            )
        elif region.proj == "tmerc":
            col_grid, row_grid = _tmerc_pixel_coords(
                lon_grid[0, :], lat_grid[:, 0], region,
            )
        else:
            col_grid = (lon_grid - region.west) / region.pixel_size
            row_grid = (region.north - lat_grid) / region._ps_y

        in_bounds = (
            (row_grid >= 0.0) & (row_grid < region.height)
            & (col_grid >= 0.0) & (col_grid < region.width)
        )
        if not in_bounds.any():
            return
        row_i = np.clip(np.rint(row_grid).astype(np.int32), 0, region.height - 1)
        col_i = np.clip(np.rint(col_grid).astype(np.int32), 0, region.width - 1)
        sampled = region_array[row_i[in_bounds], col_i[in_bounds]]
        hits = sampled >= pixel_threshold
        mask[in_bounds] |= hits

    @staticmethod
    def _dilate(mask: np.ndarray) -> np.ndarray:
        """Return ``mask`` dilated by 1 cell in all 8 directions.

        Top/bottom rows clip naturally (no wrap); left/right columns wrap
        across the antimeridian (column 0 and column GRID_WIDTH-1 are
        neighbours).
        """
        out = mask.copy()
        out[1:, :] |= mask[:-1, :]        # north
        out[:-1, :] |= mask[1:, :]        # south
        out[:, 1:] |= mask[:, :-1]        # west
        out[:, :-1] |= mask[:, 1:]        # east
        out[1:, 1:] |= mask[:-1, :-1]     # north-west
        out[1:, :-1] |= mask[:-1, 1:]     # north-east
        out[:-1, 1:] |= mask[1:, :-1]     # south-west
        out[:-1, :-1] |= mask[1:, 1:]     # south-east
        # Antimeridian wrap for the east/west and diagonal neighbours.
        out[:, 0] |= mask[:, -1]
        out[:, -1] |= mask[:, 0]
        out[1:, 0] |= mask[:-1, -1]
        out[:-1, 0] |= mask[1:, -1]
        out[1:, -1] |= mask[:-1, 0]
        out[:-1, -1] |= mask[1:, 0]
        return out

    def _save_mask(
        self, masks: dict[int, np.ndarray], ts: int, mask: np.ndarray,
    ) -> None:
        """Persist ``mask`` (memmap + atomic replace) or keep on the heap."""
        if self._persistent:
            final = self._memmap_dir / f"{ts}.dat"
            tmp = final.with_suffix(".dat.tmp")
            mm = np.memmap(tmp, dtype=mask.dtype, mode="w+", shape=mask.shape)
            mm[:] = mask
            mm.flush()
            del mm
            os.replace(tmp, final)
            masks[ts] = np.memmap(final, dtype=mask.dtype, mode="r", shape=mask.shape)
        else:
            masks[ts] = mask

    def _cleanup_stale_files(self) -> None:
        """Delete memmap files for timestamps no longer in ``_masks``."""
        if not self._persistent:
            return
        for path in self._memmap_dir.glob("*.dat"):
            try:
                ts = int(path.stem)
            except ValueError:
                continue
            if ts not in self._masks:
                try:
                    path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _nwp_signature_of(nwp_chain) -> tuple:
        """Content signature of the NWP chain for mask-cache reuse.

        Per source: name, timestep count (``_sorted_timestamps`` list or
        ``_timesteps`` dict, whichever the source exposes), and latest run
        timestamp (``latest_run_ts`` or ``_latest_run_ts``).  IFS is the
        only source that exposes ``reference_time`` — fold that in plus
        its sorted timestep keys so hourly window slides bust the cache
        even when the run stays put.
        """
        items: list[tuple] = []
        for src in nwp_chain.sources:
            name = getattr(src, "name", "?")
            count = len(getattr(src, "_sorted_timestamps", ()))
            if not count:
                count = len(getattr(src, "_timesteps", {}))
            latest = getattr(src, "latest_run_ts", None)
            if latest is None:
                latest = getattr(src, "_latest_run_ts", None)
            items.append((name, count, latest))
            ref_time = getattr(src, "reference_time", None)
            if ref_time is not None:
                ts_keys = sorted(getattr(src, "_timesteps", {}).keys())
                items.append(("ecmwf", ref_time, tuple(ts_keys)))
        return tuple(items)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def has_precip_in_bbox(self, timestamp: int, tile_bbox: tuple) -> bool:
        """Return whether the mask may have precip inside ``tile_bbox``.

        Conservative by design: unknown timestamps, antimeridian-wrapping
        bboxes, and degenerate slices all return True so the renderer
        falls through to the expensive compute instead of risking a
        wrongly-transparent tile.
        """
        west, south, east, north = tile_bbox
        if timestamp not in self._masks:
            return True
        if west > east:
            return True
        row_north = int((self.NORTH - north) / self.PIXEL_SIZE)
        row_south = int((self.NORTH - south) / self.PIXEL_SIZE) + 1
        col_west = int((west - self.WEST) / self.PIXEL_SIZE)
        col_east = int((east - self.WEST) / self.PIXEL_SIZE) + 1
        row_north = min(max(row_north, 0), self.GRID_HEIGHT - 1)
        row_south = min(max(row_south, 0), self.GRID_HEIGHT)
        col_west = min(max(col_west, 0), self.GRID_WIDTH - 1)
        col_east = min(max(col_east, 0), self.GRID_WIDTH)
        if row_north >= row_south or col_west >= col_east:
            return True
        region = self._masks[timestamp][row_north:row_south, col_west:col_east]
        return bool(region.any())

    # ------------------------------------------------------------------
    # State round-trip (multi-worker snapshot)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Serialize mask locations for cross-process reload.

        Only memmap-backed masks are serializable (heap arrays lack
        ``.filename``); render workers re-mmap the basenames read-only.
        """
        if self._persistent:
            return {
                "memmap_dir": str(self._memmap_dir),
                "version": self._version,
                "masks": {
                    str(ts): [
                        os.path.basename(str(mask.filename)),
                        mask.dtype.name,
                        [self.GRID_HEIGHT, self.GRID_WIDTH],
                    ]
                    for ts, mask in self._masks.items()
                    if hasattr(mask, "filename")
                },
            }
        return {"version": self._version, "masks": {}}

    def __setstate__(self, state: dict) -> None:
        """Restore masks from the snapshot dict (render-worker side).

        Re-opens each mask file read-only; stale files (cleaned up by the
        pipeline mid-cycle) are skipped so the renderer falls back to
        conservative True for that timestamp.
        """
        memmap_dir = state.get("memmap_dir")
        self._memmap_dir = Path(memmap_dir) if memmap_dir else None
        self._persistent = self._memmap_dir is not None
        self._version = state.get("version", 0)
        # The NWP mask cache lives only on the pipeline side; render
        # workers never rebuild masks, so it is not serialized.
        self._nwp_cache = None
        self._nwp_signature = None
        self._masks = {}
        if "masks" not in state:
            # Pre-fix state.json (Tier 2 era) has no mask section — every
            # query falls back to conservative True.
            self._version = 0
            return
        for ts_str, (basename, dtype_str, shape) in state["masks"].items():
            ts = int(ts_str)
            path = self._memmap_dir / basename
            try:
                self._masks[ts] = np.memmap(
                    path, dtype=np.dtype(dtype_str), mode="r",
                    shape=tuple(shape),
                )
            except FileNotFoundError:
                # Stale mask file race — the pipeline replaced the set of
                # timestamps between dump and this read.  Skip; queries
                # for this ts return conservative True.
                continue
