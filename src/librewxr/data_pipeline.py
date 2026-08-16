# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Standalone data-pipeline process for the multi-worker tile-server split.

This is the "master" half of the split: a single asyncio process that
fetches radar frames + auxiliary NWP grids, runs the nowcast generator,
and dumps a cross-process state snapshot under ``LIBREWXR_CACHE_DIR``
after every cycle.

The companion render-only tile-server workers (``librewxr.main`` with
``LIBREWXR_RENDER_ONLY=1``) memory-map the same files and refresh their
in-memory store views by polling ``state.json``'s mtime.

Run with::

    python -m librewxr.data_pipeline

``LIBREWXR_CACHE_DIR`` is required.  Without it there is nothing to
share with the render workers and the script exits immediately.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

import cv2

from librewxr.config import settings
from librewxr.data.alerts_fetcher import WMOAlertsFetcher
from librewxr.data.alerts_store import AlertsStore
from librewxr.data.coverage import (
    build_coverage_masks,
    build_feather_masks,
    persist_masks_in_background,
)
from librewxr.data.fetcher import RadarFetcher
from librewxr.data.master_state import snapshot_state, write_state_snapshot
from librewxr.data.nowcast import NowcastGenerator, NowcastStore
from librewxr.data.storm_cells import StormCellGenerator, StormCellStore
from librewxr.data.nwp_source import NWPChain
from librewxr.data.pagecache import prime_fresh_memmaps
from librewxr.data.precip_mask import PrecipMaskStore
from librewxr.data.radar_cache import RadarFrameCache
from librewxr.data.regions import REGIONS
from librewxr.data.store import FrameStore
from librewxr.logging_setup import setup_logging
from librewxr.sources import (
    collect_nowcast_contributions,
    collect_nwp_contributions,
    collect_radar_coverage_metadata,
    collect_satellite_contributions,
    enabled_regions_with_always_on,
    nwp_grid_slug,
    satellite_source_slug,
)
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import prune_shared_coord_store
from librewxr.tiles.shared_tile_store import SharedTileStore

# The pipeline writes no tiles itself, but RadarFetcher invalidates a
# TileCache on frame eviction.  A shared one here would be useless to
# the render workers (different process, no cross-process invalidation),
# so we hand it a tiny no-op-effect cache and rely on the render workers
# to invalidate their own caches when they pick up a new state.json.

logger = logging.getLogger(__name__)

# The mask-persistence background task.  Held at module scope for the
# process lifetime so it can't be garbage-collected mid-write (the write
# is tens of MB); ``run_pipeline`` assigns it once at startup.
_mask_save_task: asyncio.Task | None = None


async def run_pipeline() -> None:
    """Construct the pipeline, run until signalled, then shut down cleanly."""
    if not settings.cache_dir:
        raise SystemExit(
            "LIBREWXR_CACHE_DIR must be set when running the data pipeline — "
            "render-only workers need a shared directory to read snapshots from."
        )
    cache_dir = Path(settings.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # The enabled set includes every always-on contribution region (the
    # coarse global observed tier stays fetchable even under a narrow
    # region spec).
    enabled = enabled_regions_with_always_on(settings)
    logger.info(
        "Pipeline starting (cache_dir=%s, regions=%s, fetch_interval=%ds)",
        cache_dir, ", ".join(enabled), settings.fetch_interval,
    )

    # All persistent stores share the same cache_dir so render workers can
    # memmap the same files and __setstate__ from a single state.json.
    store = FrameStore(max_frames=settings.max_frames, cache_dir=cache_dir)
    tile_cache = TileCache(max_mb=1)  # noop-effect (see module docstring)

    # Walk the auto-discovered NWP providers under ``librewxr.sources``;
    # each returns a contribution (or ``None`` when its config flag is
    # off).  Same chain order as ``main.py`` — see that module for the
    # priority assignments.
    nwp_contribs = collect_nwp_contributions(settings, cache_dir)
    nwp_grids_by_slug = {nwp_grid_slug(c): c.instance for c in nwp_contribs}
    nwp_chain = NWPChain([c.instance for c in nwp_contribs])
    logger.info("NWP chain: [%s]", ", ".join(s.name for s in nwp_chain.sources))

    # GMGSI satellite sources — one contribution per enabled channel.
    # collect_satellite_contributions short-circuits to [] when
    # satellite_enabled is False, mirroring the radar / NWP toggles.
    satellite_contribs = collect_satellite_contributions(settings, cache_dir)
    satellite_grids_by_slug = {
        satellite_source_slug(c): c.instance for c in satellite_contribs
    }
    if satellite_contribs:
        logger.info(
            "Satellite chain: [%s]",
            ", ".join(c.name for c in satellite_contribs),
        )

    station_map, range_overrides, coverage_polygons = collect_radar_coverage_metadata(settings)
    build_coverage_masks(
        station_map,
        range_overrides=range_overrides,
        coverage_polygons=coverage_polygons,
    )
    build_feather_masks()
    # Persist the built masks once so render workers memmap them read-only
    # instead of rebuilding ~30 MB each at boot.  Background thread: the
    # write is tens of MB and must not delay the first fetch cycle.  Keep
    # the task referenced at module scope for the process lifetime so it
    # can't be garbage-collected mid-write.
    global _mask_save_task
    _mask_save_task = persist_masks_in_background(
        cache_dir, enabled, station_map, range_overrides, coverage_polygons,
    )

    nowcast_store = None
    nowcast_generator = None
    # Constructed whenever nowcast is on OR arrow_flow is on — the
    # latter reuses NowcastGenerator's Phase A (optical flow) to
    # populate the arrow overlay's flow vectors without running the
    # extrapolation phase, so arrows show real storm motion even
    # with nowcast disabled (see LIBREWXR_ARROW_FLOW_ENABLED).
    if settings.nowcast_enabled or settings.arrow_flow_enabled:
        nowcast_store = NowcastStore(cache_dir=cache_dir)
        # External nowcast contributions are only relevant to the
        # extrapolation path; skip the fetch when nowcast is off.
        nowcast_contribs = (
            collect_nowcast_contributions(settings)
            if settings.nowcast_enabled
            else []
        )
        nowcast_generator = NowcastGenerator(
            store, nowcast_store, cache=tile_cache,
            nowcast_contributions=nowcast_contribs,
            nwp_chain=nwp_chain,
        )
        external_names = [c.region_name for c in nowcast_contribs]
        if settings.nowcast_enabled:
            if external_names:
                logger.info(
                    "Nowcast enabled: %d frames (external sources: %s)",
                    settings.nowcast_frames, ", ".join(external_names),
                )
            else:
                logger.info("Nowcast enabled: %d frames", settings.nowcast_frames)
        else:
            logger.info(
                "Arrow flow enabled (nowcast off): target_dim=%d",
                settings.arrow_flow_target_dim,
            )

    storm_cell_store = None
    storm_cell_generator = None
    if settings.storm_cells_enabled:
        storm_cell_store = StormCellStore(cache_dir=cache_dir)
        storm_cell_generator = StormCellGenerator(
            store, storm_cell_store, nowcast_store=nowcast_store,
        )
        logger.info("Storm-cell detection enabled (min_dbz=%d, min_area=%.1f km^2)",
                     settings.storm_cells_min_dbz, settings.storm_cells_min_area_km2)

    # RadarFrameCache lets the pipeline restart and re-populate its
    # FrameStore from the prior session's frames before the first fetch
    # completes — without it, render workers would see an empty
    # state.json on cold start.
    radar_cache = RadarFrameCache(cache_dir)
    regions_by_name = {name: REGIONS[name] for name in enabled}
    restored = radar_cache.load_frames(regions_by_name)
    if restored:
        for frame in restored:
            await store.add_frame(frame)
        logger.info(
            "Restored %d radar frame(s) from disk cache (%d → %d)",
            len(restored), restored[0].timestamp, restored[-1].timestamp,
        )

    # AlertsStore is constructed up-front (even before the fetcher starts)
    # so it can ride along in the master_state snapshot.  The render-only
    # workers don't run their own alerts ingest — they read this store
    # via apply_state instead.
    alerts_store = AlertsStore() if settings.alerts_enabled else None

    # Per-timestamp global precip mask (the Tier 2 empty-tile gate's
    # replacement).  Built from every source's combined contribution
    # (radar + all NWP samples + nowcast) each cycle, then snapshotted
    # into state.json so render workers query it via memmap without
    # touching the NWP chain.  Multi-mode only.
    precip_mask_store = PrecipMaskStore(cache_dir=cache_dir)

    # Mirrors the render-worker budget resolution in main.py; the
    # pipeline holds the handle solely to own pruning (it never reads
    # or publishes tiles).
    mb = settings.shared_tile_store_mb
    mb = 2048 if mb is None else mb
    shared_tiles = SharedTileStore(cache_dir, max_mb=mb) if mb > 0 else None

    # Stores keyed by slug — render-only workers consume the same keys
    # via ``apply_state``.  None entries are skipped by snapshot_state.
    stores = {
        "frame_store": store,
        **nwp_grids_by_slug,
        **satellite_grids_by_slug,
        "nowcast_store": nowcast_store,
        "storm_cell_store": storm_cell_store,
        "alerts_store": alerts_store,
        "precip_mask": precip_mask_store,
    }

    async def on_cycle_complete() -> None:
        try:
            await precip_mask_store.build(stores, nwp_chain, settings)
        except Exception:
            logger.exception("Failed to build precip mask")
        # snapshot_state is pure in-memory dict building (fast) — take the
        # consistent snapshot on the loop, then hand the JSON encode +
        # atomic rename to a worker thread so the serialisation never
        # blocks the event loop.
        payload = snapshot_state(stores)
        try:
            await asyncio.to_thread(write_state_snapshot, payload, cache_dir)
        except Exception:
            logger.exception("Failed to dump master state snapshot")
        # Prime freshly written memmap frames into the host page cache so
        # render workers don't cold-fault on a slow backing disk (the host
        # page cache is shared between the pipeline and renderer
        # containers).  Best-effort — a failure must never break the cycle.
        if settings.pagecache_prime_enabled:
            try:
                primed = await asyncio.to_thread(
                    prime_fresh_memmaps, payload, cache_dir
                )
                if primed:
                    logger.info("Primed page cache for %d bytes of frame files", primed)
            except Exception:
                logger.exception("Failed to prime page cache")
        # The pipeline owns coord-store maintenance in multi mode (render
        # workers never prune).  Guard-free: _get_store()'s gate covers
        # enabled/cache_dir and the helper never raises.  The directory
        # scans run in a worker thread so they never block the loop.
        await asyncio.to_thread(prune_shared_coord_store)
        # The pipeline is the sole pruner of the shared tile store (render
        # workers only invalidate by timestamp / sweep on a full clear);
        # prune full-scans the shard tree so it stays off the event loop.
        if shared_tiles is not None:
            await asyncio.to_thread(shared_tiles.prune)

    fetcher = RadarFetcher(
        store, tile_cache,
        nwp_contributions=nwp_contribs,
        satellite_contributions=satellite_contribs,
        nowcast_generator=nowcast_generator,
        storm_cell_generator=storm_cell_generator,
        warmer=None,  # tile warming is single-mode only; multi mode has no warmer (the empty-tile fast path + per-worker LRU cover it)
        radar_cache=radar_cache,
        on_cycle_complete=on_cycle_complete,
    )

    alerts_fetcher = None
    if alerts_store is not None:
        alerts_cache = (
            cache_dir if settings.cache_dir
            else (Path(settings.alerts_cache_dir) if settings.alerts_cache_dir else None)
        )
        alerts_fetcher = WMOAlertsFetcher(
            store=alerts_store,
            cache_dir=str(alerts_cache) if alerts_cache else None,
            interval=settings.alerts_fetch_interval,
            concurrency=settings.alerts_concurrency,
        )
        await alerts_fetcher.start()
        logger.info(
            "Alerts: WMO ingest started (interval=%ds)",
            settings.alerts_fetch_interval,
        )

    await fetcher.start()
    logger.info("Pipeline running — Ctrl-C / SIGTERM to stop")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_stop(signame: str) -> None:
        logger.info("Received %s, shutting down…", signame)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_stop, sig.name)
        except NotImplementedError:
            # Windows: signal handlers via the loop aren't supported, but
            # Ctrl-C still raises KeyboardInterrupt at the top level.
            pass

    try:
        await stop_event.wait()
    finally:
        await fetcher.stop()
        if alerts_fetcher is not None:
            await alerts_fetcher.close()
        if nowcast_store is not None:
            nowcast_store.cleanup()
        if storm_cell_store is not None:
            storm_cell_store.cleanup()
        store.cleanup()
        logger.info("Pipeline shutdown complete")


def main() -> None:
    setup_logging()
    # The pipeline's heavy cv2 work (Farneback nowcast flow + cv2.remap
    # warps) runs once per fetch cycle, now overlapped across the 4-worker
    # nowcast pool (regions and steps run in parallel).  4 workers × 2 cv2
    # threads keeps the aggregate cv2 thread count at the previous 8,
    # respecting the pipeline container's CPU cap.
    cv2.setNumThreads(2)
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        # Belt-and-braces: signal_handler path is the normal route, but on
        # platforms where add_signal_handler is unavailable we still exit
        # cleanly on Ctrl-C.
        sys.exit(0)


if __name__ == "__main__":
    main()
