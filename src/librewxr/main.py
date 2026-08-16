# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import logging
import os
import random
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from librewxr.api import routes
from librewxr.config import settings
from librewxr.data.coverage import (
    build_coverage_masks,
    build_feather_masks,
    load_masks,
    persist_masks_in_background,
)
from librewxr.data.fetcher import RadarFetcher
from librewxr.data.master_state import (
    _load_and_apply_state,
    apply_state,
    dump_state,
    load_state,
    state_mtime,
)
from librewxr.data.nowcast import NowcastGenerator, NowcastStore
from librewxr.data.storm_cells import StormCellGenerator, StormCellStore
from librewxr.data.nwp_source import NWPChain
from librewxr.data.precip_mask import PrecipMaskStore
from librewxr.data.store import FrameStore
from librewxr.data.worker_pulse import PULSE_INTERVAL_S, write_worker_pulse
from librewxr.sources import (
    collect_nowcast_contributions,
    collect_nwp_contributions,
    collect_radar_coverage_metadata,
    collect_satellite_contributions,
    enabled_regions_with_always_on,
    nwp_grid_slug,
    satellite_source_slug,
)
from librewxr.data.alerts_store import AlertsStore
from librewxr.data.alerts_fetcher import WMOAlertsFetcher
from librewxr.memory import MemoryMonitor, detect_memory_limit_mb
from librewxr.logging_setup import setup_logging
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import (
    ALL_CACHES,
    coord_store_cold,
    prune_shared_coord_store,
    warm_coordinate_caches,
)
from librewxr.tiles.request_tracker import TileRequestTracker
from librewxr.tiles.shared_tile_store import SharedTileStore
from librewxr.tiles.warmer import TileWarmer

# Centralized logging: Rich-tagged handler at LIBREWXR_LOG_LEVEL (default
# INFO).  Called at module scope so import-time logging is configured,
# exactly as the old inline setup was.
setup_logging()
logger = logging.getLogger(__name__)

# The background mask-persistence task, held for the process lifetime so it
# can't be garbage-collected mid-write (see ``_hold_mask_save_task``).  Only
# one lifespan runs per process, so a single module-level slot suffices.
_mask_save_task: asyncio.Task | None = None

_WARM_JITTER_MAX_S = 15.0  # multi-mode warm-start de-synchronisation; patched to 0 in tests


def _hold_mask_save_task(app: FastAPI, task: asyncio.Task) -> None:
    """Keep the background mask-save task referenced for the process lifetime.

    Real FastAPI apps expose ``app.state``; the minimal stubs used by the
    lifespan tests don't, so fall back to a module-level reference.  Either
    way the task survives until process exit and can't be garbage-collected
    mid-write (the write is tens of MB).
    """
    state = getattr(app, "state", None)
    if state is not None:
        state.mask_save_task = task
    else:
        global _mask_save_task
        _mask_save_task = task


def _clear_coord_caches() -> None:
    """Clear all coordinate LRU caches."""
    for fn in ALL_CACHES:
        fn.cache_clear()
    logger.info("Coordinate caches cleared by memory monitor")


async def _worker_pulse_loop(stop: asyncio.Event, cache_dir: Path) -> None:
    """Periodically publish this process's pulse to the shared cache dir.

    Every PULSE_INTERVAL_S seconds (jittered like ``_poll_state`` so the
    16 render workers don't write in lockstep) this worker writes its
    compact /health payload to ``<cache_dir>/workers/worker_<pid>.json``
    via ``asyncio.to_thread`` (``write_worker_pulse`` never raises, and
    ``collect_worker_pulse`` is guarded, so the loop survives transient
    store/cache hiccups).  Other workers aggregate these files for the
    ``/health`` ``cluster`` section; the files are mtime-filtered and
    swept when stale, so a crashed worker simply stops refreshing.
    """
    while not stop.is_set():
        try:
            await asyncio.wait_for(
                stop.wait(),
                timeout=PULSE_INTERVAL_S * (0.5 + random.random()),
            )
            return
        except asyncio.TimeoutError:
            pass
        try:
            payload = routes.collect_worker_pulse()
            await asyncio.to_thread(write_worker_pulse, cache_dir, payload)
        except Exception:
            logger.exception("Worker pulse write failed")


async def _warm_coord_caches_background(
    executor: ThreadPoolExecutor,
    regions: list[str],
    zoom: int,
    *,
    jitter: bool,
) -> None:
    """Background coordinate-cache warm; never lets a failure escape.

    Runs the trig-heavy warm on ``executor`` (off the event loop) and
    swallows every exception: the coordinate wrappers already handle
    unwarmed entries by computing on demand and publishing to the shared
    on-disk store, so a failed warm must never take the worker down.
    The caller holds the returned task so it can be cancelled at
    shutdown.

    ``jitter`` is the multi-mode start de-synchronisation: with 16 render
    workers booting against a cold shared store, each would otherwise
    compute + publish the same entries simultaneously (correct but
    wasteful; converges on the first replace).  Once the store is warm
    the pass is cheap mmap hits and the jitter is dead time.  Single
    mode has no sibling workers to de-synchronise against.
    """
    try:
        if jitter and coord_store_cold():
            await asyncio.sleep(random.uniform(0.0, _WARM_JITTER_MAX_S))
        start = time.time()
        loop = asyncio.get_running_loop()
        warmed = await loop.run_in_executor(
            executor,
            warm_coordinate_caches,
            regions,
            zoom,
        )
        logger.info(
            "Coordinate caches warmed: %d entries up to zoom %d (%.2fs)",
            warmed, zoom, time.time() - start,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(
            "Coordinate cache warm failed; serving continues "
            "(entries load lazily via the store)"
        )


async def _cancel_coord_warm_task(task: asyncio.Task | None) -> None:
    """Cancel a background coordinate warm at shutdown; never raises."""
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Coordinate cache warm task error during shutdown")


async def _wait_for_state(cache_dir, timeout: float) -> None:
    """Block until state.json exists under cache_dir, or fail loudly.

    A render-only worker is useless without a snapshot to read.  Polling
    rather than inotify keeps the implementation portable to Docker
    bind-mounts and shared NFS volumes.

    During cold start the pipeline takes minutes to complete its first
    fetch cycle, so we poll lazily (every 2 s) and only log on entry +
    every 30 s — N workers each polling at 1 Hz produces real log spam.
    """
    deadline = time.time() + timeout if timeout > 0 else None
    poll = max(settings.state_poll_interval, 2.0)
    log_every = 30.0
    started = time.time()
    last_logged = 0.0
    if state_mtime(cache_dir) is not None:
        return
    logger.info("Waiting for pipeline state.json under %s …", cache_dir)
    while True:
        await asyncio.sleep(poll)
        if state_mtime(cache_dir) is not None:
            logger.info(
                "Pipeline state.json appeared after %.0fs",
                time.time() - started,
            )
            return
        if deadline is not None and time.time() > deadline:
            raise RuntimeError(
                f"Timed out after {timeout:.0f}s waiting for state.json "
                f"under {cache_dir}.  Is the data pipeline running?"
            )
        elapsed = time.time() - started
        if elapsed - last_logged >= log_every:
            logger.info(
                "Still waiting for state.json (%.0fs elapsed) …", elapsed,
            )
            last_logged = elapsed


def _compute_cache_invalidation(
    prev_payload: dict | None,
    cur_payload: dict,
) -> tuple[set[int] | None, bool]:
    """Decide what to invalidate in the tile cache after a state.json poll.

    Returns ``(timestamps, full_clear)``:
    - ``(None, True)``       — clear the whole cache (signature changed; rare).
    - ``(ts_set, False)``    — call ``invalidate_timestamp(ts)`` for each ``ts``.

    The "signature" is the tuple of (store_name, reference_time) for every
    store with a ``reference_time`` field, plus (ecmwf_grid, sorted(timesteps))
    to catch hourly timestep slides that keep ``reference_time`` unchanged.
    On any signature change → full clear, because cached geometry may have
    sampled stale NWP content.

    Targeted invalidation covers: radar frame evictions + version bumps
    (merges), and all nowcast timestamps (content regenerates every cycle —
    the 5 overlapping timestamps have stale content; full-set invalidation
    preserves correctness without per-frame versioning on NowcastStore).
    """
    if prev_payload is None:
        # First poll — nothing to diff against; full clear (matches the
        # pre-diff unconditional cache.clear() on the first refresh).
        return None, True

    cur_stores = cur_payload.get("stores", {})
    prev_stores = prev_payload.get("stores", {})
    store_names = set(cur_stores) | set(prev_stores)

    def _signature(stores: dict) -> tuple:
        items: list[tuple] = []
        for name in store_names:
            items.append((name, stores.get(name, {}).get("reference_time")))
            if name == "ecmwf_grid":
                # Hourly timestep slides re-key the same unix timestamps
                # with different content while ``reference_time`` stays
                # stable.  JSON coerces int keys to strings, but only the
                # sorted key *set* is compared, so string keys are fine.
                ts_keys = sorted(stores.get(name, {}).get("timesteps", {}).keys())
                items.append(("ecmwf_grid_timesteps", tuple(ts_keys)))
        return tuple(sorted(items))

    if _signature(cur_stores) != _signature(prev_stores):
        return None, True

    invalidate: set[int] = set()

    # FrameStore content versions: a timestamp removed (evicted) or bumped
    # (region merge) means its cached geometry is stale.  Keys arrive as
    # strings (JSON coercion); convert to int on read.
    prev_versions = {
        int(k): v
        for k, v in prev_stores.get("frame_store", {}).get("frame_versions", {}).items()
    }
    cur_versions = {
        int(k): v
        for k, v in cur_stores.get("frame_store", {}).get("frame_versions", {}).items()
    }
    for ts in prev_versions:
        if ts not in cur_versions:
            invalidate.add(ts)
        elif cur_versions[ts] != prev_versions[ts]:
            invalidate.add(ts)

    # Nowcast: the sliding forecast window persists most timestamps across
    # cycles, but their content is regenerated every cycle — treat every
    # timestamp in either snapshot as stale.
    prev_nowcast = {
        int(f["timestamp"])
        for f in prev_stores.get("nowcast_store", {}).get("frames", [])
    }
    cur_nowcast = {
        int(f["timestamp"])
        for f in cur_stores.get("nowcast_store", {}).get("frames", [])
    }
    invalidate |= prev_nowcast | cur_nowcast

    return invalidate, False


def _maintain_shared_tiles(store, full_clear: bool, ts_set: set[int] | None) -> None:
    """Shared-store maintenance after a state refresh (runs off the loop).

    Mirrors the in-memory tile-cache invalidation: a signature change
    (full clear) sweeps every published file because cached geometry may
    have sampled stale NWP content; otherwise only the invalidated
    timestamps' entries are removed (versioned keys already make stale
    content unreachable - this just reclaims the space).  The full clear
    sweeps final files only (``sweep_final_files``), keeping the tree and
    any in-flight publishes: a concurrent publisher's ``.tmp`` survives
    and its os.replace lands a current-version entry, while every
    published file is removed so stale-NWP content is still fully
    reclaimed.  ``prune`` (the full on-disk scan / budget enforcement) is
    owned by the pipeline process, which prunes once per fetch cycle;
    render workers only do correctness invalidation here.
    """
    if full_clear:
        store.sweep_final_files()
    else:
        for ts in ts_set:
            store.invalidate_timestamp(ts)


def _drop_absent_stores(stores: dict, refreshed: list[str]) -> None:
    """Null stores absent from the first state snapshot.

    Drops genuinely-disabled providers (e.g. ICON-EU in a CONUS-only
    deployment) so the NWP chain and routes don't dispatch to empty grids.
    Infrastructure stores that are constructed unconditionally and are
    always shipped once the pipeline supports them - ``frame_store``,
    ``precip_mask``, and ``nowcast_store`` - are exempt: a stale/legacy
    first snapshot (e.g. an in-place upgrade from a build that predates a
    store) must not permanently kill the store, because
    :func:`apply_state` skips ``None`` stores and the poller could never
    bring it back.  The store repopulates in place via ``__setstate__``
    on the next poll once the pipeline ships it.
    """
    keep = {"frame_store", "precip_mask", "nowcast_store"}
    for name in list(stores.keys()):
        if name not in refreshed and name not in keep:
            stores[name] = None


def _maybe_resurrect_precip_mask(
    stores: dict, payload: dict, cache_dir,
) -> bool:
    """Heal a render worker whose precip mask was nulled at boot.

    Workers that started against a pre-``precip_mask`` ``state.json`` had
    their :class:`PrecipMaskStore` dropped by the boot-time drop loop and,
    because :func:`apply_state` skips ``None`` stores, could never recover
    it - so the Tier 2 empty-tile gate stayed dead for the life of the
    process.  This re-instantiates the store from the current snapshot so
    the gate self-heals on the first poll after the pipeline ships masks,
    without a process restart.

    Returns ``True`` if the store was resurrected this call.  Idempotent:
    a no-op once the store is live, or while the snapshot still lacks the
    ``precip_mask`` entry.
    """
    if stores.get("precip_mask") is not None:
        return False
    snap = payload.get("stores", {}).get("precip_mask")
    if snap is None:
        return False
    store = PrecipMaskStore(cache_dir=cache_dir)
    store.__setstate__(snap)
    stores["precip_mask"] = store
    return True


def _maybe_resurrect_nowcast_store(
    stores: dict, payload: dict, cache_dir,
) -> bool:
    """Heal a render worker whose nowcast store was nulled at boot.

    Workers that started against a ``state.json`` lacking a
    ``nowcast_store`` entry had their :class:`NowcastStore` dropped by the
    boot-time drop loop and, because :func:`apply_state` skips ``None``
    stores, could never recover it - so nowcast tiles stayed empty for the
    life of the process.  This re-instantiates the store from the current
    snapshot so it self-heals on the first poll after the pipeline ships
    the entry, without a process restart.  ``cleanup_tmp=False`` skips the
    constructor's ``*.tmp`` sweep so a resurrected store can't unlink a
    ``.dat.tmp`` the pipeline is concurrently writing.

    Returns ``True`` if the store was resurrected this call.  Idempotent:
    a no-op once the store is live, or while the snapshot still lacks the
    ``nowcast_store`` entry.
    """
    if stores.get("nowcast_store") is not None:
        return False
    snap = payload.get("stores", {}).get("nowcast_store")
    if snap is None:
        return False
    store = NowcastStore(cache_dir=cache_dir, cleanup_tmp=False)
    store.__setstate__(snap)
    stores["nowcast_store"] = store
    return True


@asynccontextmanager
async def _render_only_lifespan(app: FastAPI):
    """Lifespan for tile-server workers in the multi-worker split.

    Pulls all radar / NWP / satellite data from the snapshot the data
    pipeline writes, and refreshes it in place every time
    ``state.json``'s mtime advances.  No fetcher, no NWP HTTP clients,
    no nowcast computation — just rendering.
    """
    if not settings.cache_dir:
        raise RuntimeError(
            "LIBREWXR_RENDER_ONLY=1 requires LIBREWXR_CACHE_DIR to be set "
            "(it's the shared volume the pipeline writes state.json into)."
        )
    from pathlib import Path
    cache_dir = Path(settings.cache_dir)

    await _wait_for_state(cache_dir, settings.state_wait_timeout)

    # Empty stores; __setstate__ wires up cache_dir and reopens memmaps.
    # We construct the same superset the pipeline can dump (via the
    # provider walk) so apply_state picks up whichever entries are
    # present in the snapshot.  Sources disabled by config produce no
    # contribution and so won't be loaded even if the snapshot includes
    # them — keep settings in sync between pipeline and render workers.
    store = FrameStore(max_frames=settings.max_frames, cache_dir=cache_dir)
    cache = TileCache(max_mb=settings.tile_cache_mb)
    # Shared on-disk encoded-tile store: one worker's encode serves all
    # workers (plain past-frame tiles only; see routes.radar_tile).  The
    # content-versioned keys make stale entries unreachable between fetch
    # cycles, so the poller only does correctness invalidation on
    # signature changes (see _maintain_shared_tiles; the pipeline process
    # owns budget pruning, once per fetch cycle).
    # Auto = 2048 MB for render workers; 0 or negative disables.
    shared_tiles = None
    mb = settings.shared_tile_store_mb
    mb = 2048 if mb is None else mb
    if mb > 0:
        shared_tiles = SharedTileStore(cache_dir, max_mb=mb)
        logger.info("Shared tile store: %d MB budget under %s", mb, cache_dir)
    nwp_contribs = collect_nwp_contributions(settings, cache_dir)
    nwp_grids_by_slug: dict[str, object] = {
        nwp_grid_slug(c): c.instance for c in nwp_contribs
    }
    satellite_contribs = collect_satellite_contributions(settings, cache_dir)
    satellite_grids_by_slug: dict[str, object] = {
        satellite_source_slug(c): c.instance for c in satellite_contribs
    }
    nowcast_store = (
        # The pipeline owns the shared nowcast dir and may be mid-write;
        # a worker boot must never delete its in-flight tmp files (the
        # stale-tmp sweep stays the pipeline's job at its own boot).
        NowcastStore(cache_dir=cache_dir, cleanup_tmp=False)
        if (settings.nowcast_enabled or settings.arrow_flow_enabled)
        else None
    )
    storm_cell_store = (
        # The pipeline owns the shared storm-cells dir and may be
        # mid-write; a worker boot must never delete its in-flight tmp
        # files (the stale-tmp sweep stays the pipeline's job at its own
        # boot).
        StormCellStore(cache_dir=cache_dir, cleanup_tmp=False)
        if settings.storm_cells_enabled
        else None
    )
    alerts_store = AlertsStore() if settings.alerts_enabled else None

    # Per-timestamp global precip mask, built by the pipeline and
    # snapshotted into state.json.  Re-mmaps the mask files read-only;
    # render workers query it instead of probing the NWP chain.
    precip_mask_store = PrecipMaskStore(cache_dir=cache_dir)

    stores: dict[str, object | None] = {
        "frame_store": store,
        **nwp_grids_by_slug,
        **satellite_grids_by_slug,
        "nowcast_store": nowcast_store,
        "storm_cell_store": storm_cell_store,
        "alerts_store": alerts_store,
        "precip_mask": precip_mask_store,
    }

    payload = load_state(cache_dir)
    if payload is None:
        raise RuntimeError(
            f"state.json disappeared between mtime check and load — "
            f"is something else writing to {cache_dir}?"
        )
    refreshed = apply_state(payload, stores)
    logger.debug(
        "Render-only worker loaded snapshot: %s",
        ", ".join(refreshed) if refreshed else "(empty)",
    )

    # Stores that didn't appear in the snapshot are useless (e.g. ICON-EU
    # in a CONUS-only deployment) - drop the references so the NWP chain
    # and routes don't dispatch to empty grids.  Infrastructure stores
    # that are constructed unconditionally (frame_store, precip_mask) are
    # exempt: a stale/legacy first snapshot must not permanently kill the
    # precip mask, since apply_state skips None stores and the poller
    # could not otherwise bring it back.
    _drop_absent_stores(stores, refreshed)
    # Rebuild the slug → grid dict from the (post-drop) stores so the
    # routes / chain only see grids that actually loaded from disk.
    nwp_grids_by_slug = {
        slug: stores[slug] for slug in nwp_grids_by_slug if stores[slug] is not None
    }
    satellite_grids_by_slug = {
        slug: stores[slug]
        for slug in satellite_grids_by_slug
        if stores[slug] is not None
    }
    ecmwf_grid = nwp_grids_by_slug.get("ecmwf_grid")
    nowcast_store = stores["nowcast_store"]
    storm_cell_store = stores["storm_cell_store"]
    alerts_store = stores["alerts_store"]

    # The enabled set includes every always-on contribution region (the
    # coarse global observed tier stays fetchable/renderable even under a
    # narrow region spec).
    enabled = enabled_regions_with_always_on(settings)
    station_map, range_overrides, coverage_polygons = collect_radar_coverage_metadata(settings)
    # Prefer the persisted masks (read-only memmap) when the pipeline has
    # already saved a set built from identical parameters; otherwise build
    # exactly as before and persist so the next worker/boot gets a hit.
    # A miss never blocks boot on the pipeline: it just falls back to
    # building in-process.
    if not load_masks(
        cache_dir, enabled, station_map, range_overrides, coverage_polygons,
    ):
        build_coverage_masks(
            station_map,
            range_overrides=range_overrides,
            coverage_polygons=coverage_polygons,
        )
        build_feather_masks()
        # Keep the task referenced for the worker's lifetime so it can't
        # be garbage-collected mid-write (tens of MB of file I/O).
        _hold_mask_save_task(
            app,
            persist_masks_in_background(
                cache_dir, enabled, station_map, range_overrides,
                coverage_polygons,
            ),
        )

    # Chain order mirrors ``collect_nwp_contributions`` (sorted by
    # priority) — only include grids that survived the snapshot drop
    # above.  Built from the sorted ``nwp_contribs`` rather than the
    # dict so chain order stays stable independent of dict iteration.
    chain_sources = [
        c.instance for c in nwp_contribs
        if nwp_grid_slug(c) in nwp_grids_by_slug
    ]
    nwp_chain = NWPChain(chain_sources)
    logger.debug(
        "Render-only NWP chain: [%s]",
        ", ".join(s.name for s in nwp_chain.sources),
    )

    # 16 render workers x small tiles would oversubscribe the 48-thread host at OpenCV's default hardware-concurrency pool; in multi mode each worker only does per-tile blurs so 2 threads is ample.
    cv2.setNumThreads(2)
    pool_size = settings.warmer_threads or max((os.cpu_count() or 4) - 1, 1)
    request_executor = ThreadPoolExecutor(max_workers=pool_size)
    # Separate present pool: the cheap ``present_tile`` tail (colorize,
    # encode) runs here so it never queues behind long geometry computes
    # on the shared default executor during a cold-tile burst.  Half the
    # compute pool, floored at 2 - presents are short-lived, computes are
    # the bottleneck.
    present_executor = ThreadPoolExecutor(max_workers=max(2, pool_size // 2))
    # Dedicated pool for shared-tile-store I/O + state-snapshot apply so
    # they never queue behind geometry computes on the default executor.
    io_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='tile-io')
    asyncio.get_running_loop().set_default_executor(request_executor)
    routes.present_executor = present_executor
    routes.io_executor = io_executor

    mem_limit = detect_memory_limit_mb(settings.memory_limit_mb)
    monitor = MemoryMonitor(
        tile_cache=cache,
        coord_cache_clear_fn=_clear_coord_caches,
        memory_limit_mb=mem_limit,
        check_interval=settings.memory_pressure_check_interval,
    )

    tile_request_tracker = (
        TileRequestTracker(
            min_zoom=settings.tile_tracking_min_zoom,
            max_entries=settings.tile_tracking_max_entries,
        )
        if settings.tile_tracking_enabled
        else None
    )

    routes.frame_store = store
    routes.tile_cache = cache
    # Shared encoded-tile store for the multi-worker fleet; None when
    # disabled (0/negative budget) — the route's shared-store paths are
    # then a complete no-op.
    routes.shared_tile_store = shared_tiles
    routes.nwp_grids = nwp_grids_by_slug
    routes.ecmwf_grid = ecmwf_grid
    routes.nwp_chain = nwp_chain
    routes.satellite_grids = satellite_grids_by_slug
    routes.tile_warmer = None
    routes.nowcast_store = nowcast_store
    routes.storm_cell_store = storm_cell_store
    routes.tile_request_tracker = tile_request_tracker
    routes.start_time = time.time()
    routes.enabled_regions = enabled
    routes.radar_cache = None
    routes.radar_fetcher = None
    # Alerts ride the master_state snapshot — pipeline owns the WMO ingest,
    # render workers just read alerts_store via apply_state.  alerts_fetcher
    # stays None here (no duplicate fetching), and alerts_enabled tracks
    # whether the snapshot actually included an alerts_store entry.
    routes.alerts_store = alerts_store
    routes.alerts_fetcher = None
    routes.alerts_enabled = alerts_store is not None
    # The precip mask rides the snapshot.  The store is constructed
    # unconditionally and exempt from the boot-time drop loop above, so
    # routes.precip_mask is always a live PrecipMaskStore here.  When the
    # first snapshot lacks mask data (e.g. an in-place upgrade from a
    # pre-mask build) the store starts empty and returns conservative
    # True on every query (Tier 2 gate off -> Tier 1 fallback); the
    # poller's apply_state then populates it via __setstate__ once the
    # pipeline ships masks, and _maybe_resurrect_precip_mask is a safety
    # net for any future path that nulls the store mid-run.
    routes.precip_mask = stores["precip_mask"]
    # Cluster /health: the monitor's cgroup anon/file/shmem split backs
    # the ``cluster.memory.container`` block.
    routes.memory_monitor = monitor

    last_mtime = state_mtime(cache_dir)
    # Seed the diff from the boot payload so the first poll skips unchanged stores.
    last_payload: dict | None = payload
    poller_stop = asyncio.Event()
    pulse_stop = asyncio.Event()

    async def _poll_state() -> None:
        nonlocal last_mtime, last_payload
        while not poller_stop.is_set():
            try:
                # Jitter the poll cadence (uniform over [0.5, 1.5] x
                # interval) so the 16 render workers don't hit state.json
                # in lockstep on every pipeline cycle.
                await asyncio.wait_for(
                    poller_stop.wait(),
                    timeout=settings.state_poll_interval * (0.5 + random.random()),
                )
                return
            except asyncio.TimeoutError:
                pass
            mtime = state_mtime(cache_dir)
            if mtime is None or mtime == last_mtime:
                continue
            try:
                # load_state (json.loads) + apply_state (memmap re-opens)
                # run in a worker thread — see _load_and_apply_state for
                # the thread-safety basis.  The diff-based cache
                # invalidation stays on the loop: it compares payload
                # content, not in-memory store state, so skipping
                # __setstate__ for unchanged stores never affects it.
                payload, refreshed = await asyncio.get_running_loop().run_in_executor(
                    io_executor, _load_and_apply_state, cache_dir, stores, last_payload,
                )
                if payload is None:
                    continue
                last_mtime = mtime
                logger.debug(
                    "Render worker refreshed: %s", ", ".join(refreshed),
                )

                # Heal a precip mask that was permanently nulled at boot
                # (e.g. the worker started against a pre-mask legacy
                # state.json).  Self-heals on the first poll after the
                # pipeline ships a mask entry; no restart needed.
                if _maybe_resurrect_precip_mask(stores, payload, cache_dir):
                    routes.precip_mask = stores["precip_mask"]
                    logger.info("Precip mask resurrected from state snapshot")

                # Heal a nowcast store that was permanently nulled at boot
                # (e.g. the worker started against a legacy state.json
                # that predates the nowcast entry).  The rebind below is
                # essential - routes.nowcast_store was bound once at
                # lifespan setup and stays None otherwise.
                if _maybe_resurrect_nowcast_store(stores, payload, cache_dir):
                    routes.nowcast_store = stores["nowcast_store"]
                    logger.info("Nowcast store resurrected from state snapshot")

                # Diff-based cache invalidation: preserve cached geometry
                # for radar timestamps whose content didn't change between
                # snapshots.  Full clear only when a NWP content signature
                # changed (IFS ref_time, IFS timestep set, or any NWP
                # store's reference_time).
                ts_to_invalidate, full_clear = _compute_cache_invalidation(
                    last_payload, payload,
                )
                if full_clear:
                    cache.clear()
                else:
                    for ts in ts_to_invalidate:
                        cache.invalidate_timestamp(ts)
                if shared_tiles is not None:
                    # Same invalidation semantics for the shared on-disk
                    # encoded-tile store.  The store ops are file I/O, so
                    # the whole maintenance pass runs off the event loop.
                    await asyncio.get_running_loop().run_in_executor(
                        io_executor,
                        _maintain_shared_tiles,
                        shared_tiles, full_clear, ts_to_invalidate,
                    )
                last_payload = payload
            except Exception:
                logger.exception("Failed to refresh state from %s", cache_dir)

    poller_task = asyncio.create_task(_poll_state())
    # Cluster /health pulse: this worker's share of the shared cache-dir
    # aggregation.  cache_dir is guaranteed non-empty here (render-only
    # requires it), but keep the guard for symmetry with the single-mode
    # lifespan.
    pulse_task = (
        asyncio.create_task(_worker_pulse_loop(pulse_stop, cache_dir))
        if settings.cache_dir
        else None
    )
    await monitor.start()

    # Pre-warm coordinate caches as a BACKGROUND task so the worker starts
    # serving immediately: on slow storage (ZFS/HDD, cold page cache) an
    # eager warm held boot for 14-26 minutes.  Coordinate wrappers handle
    # unwarmed entries gracefully (compute on demand + publish to the shared
    # on-disk coord store), so readiness never depends on the warm.
    # Mirrors the single-mode lifespan call; here the compute pool
    # (request_executor) plays the single-mode warmer's role.  The
    # cold-probe/jitter logic lives inside the warm task — it only dedupes
    # the publish stampede when a warm is actually enabled.  Disabled when
    # the effective zoom is not positive (multi-mode default; see
    # config._MODE_DEFAULTS), which is also the shipped default here.
    warm_task = None
    if settings.warm_coord_zoom > 0:
        warm_task = asyncio.create_task(
            _warm_coord_caches_background(
                request_executor, enabled, settings.warm_coord_zoom, jitter=True,
            )
        )
        logger.info(
            "Coordinate cache warm running in background (zoom %d)",
            settings.warm_coord_zoom,
        )

    try:
        logger.info(
            "Render-only worker ready (cache_dir=%s, regions=%s, tile_cache=%d MB)",
            cache_dir, ", ".join(enabled), settings.tile_cache_mb,
        )

        yield
    finally:
        poller_stop.set()
        pulse_stop.set()
        try:
            await poller_task
        except Exception:
            logger.exception("Poller shutdown error")
        if pulse_task is not None:
            try:
                await pulse_task
            except Exception:
                logger.exception("Pulse loop shutdown error")
        await _cancel_coord_warm_task(warm_task)
        await monitor.stop()
        request_executor.shutdown(wait=False)
        present_executor.shutdown(wait=False)
        io_executor.shutdown(wait=False)
        # Unwire the routes handle so a stale reference to a shut-down pool
        # can never be scheduled against (single mode always keeps None).
        routes.present_executor = None
        routes.io_executor = None
        cache.clear()
        store.cleanup()
        if nowcast_store is not None:
            nowcast_store.cleanup()
        if storm_cell_store is not None:
            storm_cell_store.cleanup()
        logger.info("Render-only worker shutdown complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.render_only:
        async with _render_only_lifespan(app):
            yield
        return

    store = FrameStore(max_frames=settings.max_frames)
    cache = TileCache(max_mb=settings.tile_cache_mb)
    from pathlib import Path
    nwp_cache_dir = Path(settings.cache_dir) if settings.cache_dir else None
    # Walk the auto-discovered NWP providers under ``librewxr.sources``;
    # each returns a contribution (or ``None`` when its config flag is
    # off).  Chain order is set by ``NWPContribution.priority``: HRRR
    # (10) → HRRR-Alaska (11) → HRDPS (20) → JMA MSM (20) → AROME
    # Antilles (25) → AROME Guyane (26) → AROME Indien (27) → AROME
    # Ncaled (28) → AROME Polyn (29) → DMI DINI (30) → ICON-EU (35) →
    # WRF-SMN (40) → IFS (1000 — global catch-all).  NOAA RRQPE used to
    # lead the chain at priority 5; it is now the global *observed*
    # radar region (``sources/world/rrqpe``) and flows through the
    # FrameStore / radar compositor instead of the NWP chain.
    nwp_contribs = collect_nwp_contributions(settings, nwp_cache_dir)
    nwp_grids_by_slug: dict[str, object] = {
        nwp_grid_slug(c): c.instance for c in nwp_contribs
    }
    # IFS is still special-cased by the radar tile arrow path and the
    # tile warmer; pull it out by slug for those consumers.  Everything
    # else flows through ``nwp_chain`` / ``nwp_grids_by_slug``.
    ecmwf_grid = nwp_grids_by_slug.get("ecmwf_grid")
    nwp_chain = NWPChain([c.instance for c in nwp_contribs])
    logger.info("NWP chain: [%s]", ", ".join(s.name for s in nwp_chain.sources))
    satellite_contribs = collect_satellite_contributions(settings, nwp_cache_dir)
    satellite_grids_by_slug: dict[str, object] = {
        satellite_source_slug(c): c.instance for c in satellite_contribs
    }
    if satellite_contribs:
        logger.info(
            "Satellite chain: [%s]",
            ", ".join(c.name for c in satellite_contribs),
        )
    # The enabled set includes every always-on contribution region (the
    # coarse global observed tier stays fetchable/renderable even under a
    # narrow region spec).
    enabled = enabled_regions_with_always_on(settings)

    # Precompute radar station coverage masks used by the ECMWF fallback
    # to distinguish "outside radar range" from "clear sky within range".
    # Each radar provider contributes its own per-region station list +
    # range override; the registry walk merges them based on the active
    # settings (e.g. NA source = MRMS pulls in NEXRAD + Canadian; NA
    # source = IEM pulls NEXRAD only).
    station_map, range_overrides, coverage_polygons = collect_radar_coverage_metadata(settings)
    # Prefer the persisted masks (read-only memmap) when a previous boot —
    # single-mode itself or a co-located multi-mode pipeline sharing the
    # dir — has already saved a set built from identical parameters,
    # mirroring the render-only workers.  Otherwise build exactly as
    # before and persist so the next boot gets a hit.  Gated on cache_dir
    # being set: non-persistent deployments keep current behaviour (always
    # build in-process, never save).  A persisted-mask miss never blocks
    # boot — it just falls back to building in-process.
    masks_loaded = nwp_cache_dir is not None and load_masks(
        nwp_cache_dir, enabled, station_map, range_overrides,
        coverage_polygons,
    )
    if not masks_loaded:
        build_coverage_masks(
            station_map,
            range_overrides=range_overrides,
            coverage_polygons=coverage_polygons,
        )
        build_feather_masks()
        if nwp_cache_dir is not None:
            # Saved in a background thread so startup isn't held up by the
            # tens-of-MB write; keep the task referenced for the process
            # lifetime so it can't be garbage-collected mid-write.
            _hold_mask_save_task(
                app,
                persist_masks_in_background(
                    nwp_cache_dir, enabled, station_map, range_overrides,
                    coverage_polygons,
                ),
            )

    # Nowcast store and generator.  Constructed whenever nowcast is on
    # OR arrow_flow is on — the latter reuses NowcastGenerator's
    # Phase A (optical flow) to populate the arrow overlay's flow
    # vectors without running the extrapolation phase, so arrows
    # show real storm motion even with nowcast disabled.
    nowcast_store = None
    nowcast_generator = None
    if settings.nowcast_enabled or settings.arrow_flow_enabled:
        nowcast_store = NowcastStore()
        # External nowcast contributions are only relevant to the
        # extrapolation path; skip the fetch when nowcast is off.
        nowcast_contribs = (
            collect_nowcast_contributions(settings)
            if settings.nowcast_enabled
            else []
        )
        nowcast_generator = NowcastGenerator(
            store, nowcast_store, cache=cache,
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

    # Storm-cell detection store + generator.  Constructed whenever
    # storm_cells is on -- the detection runs each cycle after nowcast
    # generation so it can reuse the just-computed optical flow.
    storm_cell_store = None
    storm_cell_generator = None
    if settings.storm_cells_enabled:
        storm_cell_store = StormCellStore()
        storm_cell_generator = StormCellGenerator(
            store, storm_cell_store, nowcast_store=nowcast_store,
        )
        logger.info("Storm-cell detection enabled (min_dbz=%d, min_area=%.1f km^2)",
                     settings.storm_cells_min_dbz, settings.storm_cells_min_area_km2)
    else:
        logger.info("Storm-cell detection: disabled (LIBREWXR_STORM_CELLS_ENABLED=false)")

    # Separate thread pools for direct requests and background warming.
    # Direct requests get their own pool so they are never queued behind
    # warming tasks.  The warmer gets an equal-sized pool so it can use
    # all cores when no requests are active.  Brief over-subscription
    # when both are active is handled well by the OS scheduler.
    # OpenCV's thread pool is intentionally left at its default here —
    # single mode shares one process between rendering and nowcast
    # generation, and the Farneback optical-flow work wants several
    # threads.
    pool_size = settings.warmer_threads or max((os.cpu_count() or 4) - 1, 1)
    request_executor = ThreadPoolExecutor(max_workers=pool_size)
    warmer_executor = ThreadPoolExecutor(max_workers=pool_size)
    asyncio.get_running_loop().set_default_executor(request_executor)

    warmer = TileWarmer(
        store, cache,
        executor=warmer_executor,
        enabled_regions=enabled,
        nowcast_store=nowcast_store,
        ecmwf_grid=ecmwf_grid,
        nwp_chain=nwp_chain,
    )

    # Memory pressure monitor
    mem_limit = detect_memory_limit_mb(settings.memory_limit_mb)
    monitor = MemoryMonitor(
        tile_cache=cache,
        coord_cache_clear_fn=_clear_coord_caches,
        memory_limit_mb=mem_limit,
        check_interval=settings.memory_pressure_check_interval,
    )

    tile_request_tracker = (
        TileRequestTracker(
            min_zoom=settings.tile_tracking_min_zoom,
            max_entries=settings.tile_tracking_max_entries,
        )
        if settings.tile_tracking_enabled
        else None
    )

    # --- WMO Alerts subsystem ---
    alerts_store = None
    alerts_fetcher = None
    if settings.alerts_enabled:
        alerts_cache = Path(settings.cache_dir) if settings.cache_dir else None
        if alerts_cache is None and settings.alerts_cache_dir:
            alerts_cache = Path(settings.alerts_cache_dir)

        alerts_store = AlertsStore()
        alerts_fetcher = WMOAlertsFetcher(
            store=alerts_store,
            cache_dir=str(alerts_cache) if alerts_cache else None,
            interval=settings.alerts_fetch_interval,
            concurrency=settings.alerts_concurrency,
        )
        routes.alerts_store = alerts_store
        routes.alerts_fetcher = alerts_fetcher
        routes.alerts_enabled = True
        await alerts_fetcher.start()
        logger.info(
            "Alerts: WMO ingest started (interval=%ds)",
            settings.alerts_fetch_interval,
        )
    else:
        routes.alerts_enabled = False
        logger.info("Alerts: disabled (LIBREWXR_ALERTS_ENABLED=false)")

    # Wire up the shared state
    routes.frame_store = store
    routes.tile_cache = cache
    # Single mode has one process — the per-worker in-memory cache is
    # enough, so the shared store stays off.  The explicit None also
    # guards module reuse across test runs (no leaked multi-mode store).
    routes.shared_tile_store = None
    routes.nwp_grids = nwp_grids_by_slug
    routes.ecmwf_grid = ecmwf_grid
    routes.nwp_chain = nwp_chain
    routes.satellite_grids = satellite_grids_by_slug
    routes.tile_warmer = warmer
    routes.nowcast_store = nowcast_store
    routes.storm_cell_store = storm_cell_store
    routes.tile_request_tracker = tile_request_tracker
    routes.start_time = time.time()
    routes.enabled_regions = enabled
    # Cluster /health: the monitor's cgroup anon/file/shmem split backs
    # the ``cluster.memory.container`` block.
    routes.memory_monitor = monitor

    radar_cache = None
    if settings.cache_dir:
        from pathlib import Path

        from librewxr.data.radar_cache import RadarFrameCache
        from librewxr.data.regions import REGIONS

        radar_cache = RadarFrameCache(Path(settings.cache_dir))
        regions_by_name = {name: REGIONS[name] for name in enabled}
        restored = radar_cache.load_frames(regions_by_name)
        if restored:
            for frame in restored:
                await store.add_frame(frame)
            logger.info(
                "Restored %d radar frame(s) from disk cache (%d → %d)",
                len(restored),
                restored[0].timestamp,
                restored[-1].timestamp,
            )

    # Single-mode state.json dump: mirrors data_pipeline.py:218-232 so a
    # stdio MCP transport (``python -m librewxr.mcp``) can run alongside
    # a single-mode server and read the snapshot.  Gated on
    # ``LIBREWXR_CACHE_DIR`` -- without a cache dir there's nowhere to
    # write and ``dump_state`` would raise.  Only dumps in single mode;
    # multi mode's pipeline owns the snapshot.
    on_cycle_complete = None
    if settings.cache_dir:
        from pathlib import Path
        state_cache_dir = Path(settings.cache_dir)
        state_stores: dict[str, object] = {
            "frame_store": store,
            **nwp_grids_by_slug,
            **satellite_grids_by_slug,
            "nowcast_store": nowcast_store,
            "storm_cell_store": storm_cell_store,
            "alerts_store": alerts_store,
        }

        async def on_cycle_complete() -> None:
            try:
                dump_state(state_stores, state_cache_dir)
            except Exception:
                logger.exception("Failed to dump state snapshot (single mode)")
            # Single mode owns coord-store maintenance (render workers never
            # prune).  Invoked like dump_state above; the helper never raises.
            prune_shared_coord_store()

    fetcher = RadarFetcher(
        store, cache,
        nwp_contributions=nwp_contribs,
        satellite_contributions=satellite_contribs,
        nowcast_generator=nowcast_generator,
        storm_cell_generator=storm_cell_generator,
        warmer=warmer,
        radar_cache=radar_cache,
        on_cycle_complete=on_cycle_complete,
    )
    routes.radar_cache = radar_cache
    routes.radar_fetcher = fetcher
    logger.info(
        "Starting LibreWXR (public_url=%s, max_zoom=%d, regions=%s, "
        "tile_cache=%d MB, memory_limit=%d MB, nowcast=%s, "
        "arrow_flow=%s, alerts=%s, cache_dir=%s)",
        settings.public_url,
        settings.max_zoom,
        ", ".join(enabled),
        settings.tile_cache_mb,
        mem_limit,
        f"{settings.nowcast_frames} frames" if settings.nowcast_enabled else "off",
        (
            f"on (target_dim={settings.arrow_flow_target_dim})"
            if settings.arrow_flow_enabled and not settings.nowcast_enabled
            else "on" if settings.arrow_flow_enabled else "off"
        ),
        "enabled" if settings.alerts_enabled else "off",
        settings.cache_dir or "(none)",
    )
    await fetcher.start()
    await monitor.start()

    # Cluster /health pulse: this process publishes its /health payload to
    # the shared cache dir so any worker can aggregate the whole cluster.
    # Gated on cache_dir (no shared volume = nothing to publish into).
    pulse_stop = asyncio.Event()
    pulse_task = None
    if settings.cache_dir:
        pulse_task = asyncio.create_task(
            _worker_pulse_loop(pulse_stop, Path(settings.cache_dir))
        )

    # Pre-warm coordinate caches as a BACKGROUND task so boot finishes
    # immediately and the warm proceeds alongside serving (same slow-storage
    # rationale as the render-only lifespan; single mode has no sibling
    # workers to de-synchronise against, so jitter=False).  A warm failure
    # is swallowed inside the task — coordinate wrappers compute on demand
    # via the store either way.  Disabled when the effective zoom is not
    # positive (set a negative LIBREWXR_WARM_COORD_ZOOM to turn it off).
    warm_task = None
    if settings.warm_coord_zoom > 0:
        warm_task = asyncio.create_task(
            _warm_coord_caches_background(
                warmer_executor, enabled, settings.warm_coord_zoom, jitter=False,
            )
        )
        logger.info(
            "Coordinate cache warm running in background (zoom %d)",
            settings.warm_coord_zoom,
        )

    yield

    pulse_stop.set()
    if pulse_task is not None:
        try:
            await pulse_task
        except Exception:
            logger.exception("Pulse loop shutdown error")
    await _cancel_coord_warm_task(warm_task)
    await monitor.stop()
    await fetcher.stop()
    if alerts_fetcher is not None:
        await alerts_fetcher.close()
    warmer.shutdown()
    warmer_executor.shutdown(wait=False)
    request_executor.shutdown(wait=False)
    if nowcast_store is not None:
        nowcast_store.cleanup()
    if storm_cell_store is not None:
        storm_cell_store.cleanup()
    cache.clear()
    store.cleanup()
    logger.info("LibreWXR shutdown complete")


# --- MCP HTTP transport ------------------------------------------------
# Build the FastMCP HTTP app once at module load, gated on the [mcp]
# extra being importable AND ``LIBREWXR_MCP_ENABLED``.  The MCP app's
# lifespan is combined with ``lifespan`` via ``combine_lifespans`` so
# its session manager starts AFTER LibreWXR's stores are wired (single
# mode + multi render-only both flow through the one ``lifespan``
# function, so a single combine call covers both modes -- the R2
# refinement from the build plan).  If the [mcp] extra is missing or
# the build throws, MCP is silently disabled and the app boots lean.
mcp_app = None
combined_lifespan = lifespan
if settings.mcp_enabled:
    try:
        from fastmcp.utilities.lifespan import combine_lifespans

        from librewxr.mcp.server import build_mcp_http_app

        mcp_app = build_mcp_http_app()
        combined_lifespan = combine_lifespans(lifespan, mcp_app.lifespan)
        logger.debug("MCP HTTP transport built; will mount at %s", settings.mcp_path)
    except Exception:
        logger.exception(
            "MCP HTTP app build failed; MCP transport disabled. "
            "Install with `pip install -e '.[mcp]'` to enable."
        )

app = FastAPI(title="LibreWXR", version="0.1.0", lifespan=combined_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(routes.router)

if mcp_app is not None:
    app.mount(settings.mcp_path, mcp_app)
    routes.mcp_mounted = True
    routes.mcp_path = settings.mcp_path
    # Matches the tool names registered by _register_tools in
    # librewxr/mcp/server.py.  Update this list when a new tool is added.
    routes.mcp_tools = ["get_precip_nowcast", "get_active_alerts", "get_storm_cells"]
    logger.debug("MCP HTTP transport mounted at %s", settings.mcp_path)

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Log requests to non-existent endpoints (path only, no client info)."""
    if exc.status_code == 404 and exc.detail == "Not Found":
        logger.warning("404 Not Found: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


def _describe_worker_exit(exitcode: int) -> str:
    """Human-readable reason for a render worker exit code."""
    if exitcode < 0:
        signum = -exitcode
        try:
            signame = signal.Signals(signum).name
        except ValueError:
            signame = f"signal {signum}"
        if signum == signal.SIGKILL:
            return f"killed by {signame} (exit code {exitcode}) - OOM-killed or externally killed"
        return f"killed by {signame} (exit code {exitcode})"
    return f"exit status {exitcode}"


def main():
    import uvicorn
    # Optional direct TLS: only enabled when both cert and key are set.
    # Otherwise serve plain HTTP (TLS handled by a reverse proxy / tunnel).
    ssl_kwargs = {}
    if settings.ssl_certfile and settings.ssl_keyfile:
        ssl_kwargs = {
            "ssl_certfile": settings.ssl_certfile,
            "ssl_keyfile": settings.ssl_keyfile,
        }
    if settings.workers > 1:
        # uvicorn logs worker deaths at INFO without the exit code, and the
        # rotating file handler only records WARNING+, so nothing durable
        # captures why a render worker died. Swap in a supervisor subclass
        # that logs the reaped worker's exit code/signal at WARNING.
        # Guarded for uvicorn versions whose supervisor lacks this method.
        # NOTE: ``import uvicorn.main`` binds the click Command (uvicorn's
        # __init__ re-exports ``main``), so the module - and its
        # call-time ``Multiprocess`` global - is reached via importlib.
        import importlib

        uvicorn_main = importlib.import_module("uvicorn.main")
        base_supervisor = uvicorn_main.Multiprocess
        if hasattr(base_supervisor, "keep_subprocess_alive"):

            class _DeathLoggingMultiprocess(base_supervisor):
                """Multiprocess supervisor that logs worker death reasons."""

                def keep_subprocess_alive(self) -> None:
                    before = [
                        (p, p.process.pid, p.exitcode) for p in self.processes
                    ]
                    super().keep_subprocess_alive()
                    for process, pid, prior_exitcode in before:
                        if prior_exitcode is not None:
                            continue
                        exitcode = process.exitcode
                        if exitcode is None or exitcode == 3:
                            # Still alive this tick, or a STARTUP_FAILURE that
                            # uvicorn already reports at ERROR before stopping.
                            continue
                        logger.warning(
                            "Render worker [pid=%d] died: %s; respawning a replacement",
                            pid,
                            _describe_worker_exit(exitcode),
                        )

            uvicorn_main.Multiprocess = _DeathLoggingMultiprocess
    uvicorn.run(
        "librewxr.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info",
        access_log=False,
        # Don't let uvicorn install its own handlers/format — its loggers
        # propagate to our shared Rich-tagged root handler instead.
        log_config=None,
        # Trust X-Forwarded-Proto/Host from any peer: LibreWXR is documented
        # as a behind-reverse-proxy deployment (cloudflared tunnel / nginx
        # on the Docker network, not localhost), and forwarded headers only
        # affect generated URLs (307 redirect Location, advertised URLs) --
        # no auth/rate-limit decisions key off the client IP, so trusting
        # all peers is safe here.  Without this uvicorn ignores
        # X-Forwarded-Proto unless the peer IP is a trusted proxy, and the
        # cloudflared container's Docker-network IP is not, so redirects
        # advertised http:// behind an https tunnel.
        proxy_headers=True,
        forwarded_allow_ips="*",
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
