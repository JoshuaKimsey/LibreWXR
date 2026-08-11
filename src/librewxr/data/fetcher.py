# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import inspect
import logging
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

import numpy as np

from librewxr.config import settings
from librewxr.memory import release_memory
from librewxr.data.regions import REGIONS, RegionDef
# Cross-source policy stays in this file (the discovery walker
# populates ``self._sources`` but blending and fallback still belong
# here).  MRMS contributes ``MRMS_EXTENTS["USCOMP"]`` for the CACOMP
# blend mask; IEM/MSC are reached as fallback/blend partners via direct
# instantiation when MRMS is the primary NA source.
from librewxr.sources.regional.north_america.canada.radar.msc_canada import (
    MSCCanadaSource,
)
from librewxr.sources.regional.north_america.usa.radar.iem import IEMSource
from librewxr.sources.regional.north_america.usa.radar.mrms import (
    MRMS_EXTENTS,
    MRMSCompositeSource,
    MRMSSource,
)
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.sources import collect_radar_contributions
from librewxr.sources._base import NWPContribution, SatelliteContribution
from librewxr.tiles.cache import TileCache

logger = logging.getLogger(__name__)


class RadarFetcher:
    """Background task that periodically fetches radar frames."""

    # Regions that MRMS can serve (each has its own regional product).
    _MRMS_REGIONS = {
        "USCOMP", "CACOMP", "AKCOMP", "HICOMP", "PRCOMP", "GUCOMP"
    }

    # When a region's fetch returns no data, fill the new frame from the
    # most recent prior frame in the store rather than leaving the region
    # absent (which causes the renderer to fall through to NWP fill, a
    # jarring visual flicker for what's usually a transient upstream
    # blip).  Bounded staleness: a region absent for more than this many
    # consecutive fetch intervals drops out instead of carrying forward
    # indefinitely.  At 10-min cadence, 2 intervals = up to 20 min of
    # stale data — enough to bridge typical publication delays without
    # masking a genuine multi-cycle outage.
    _CARRY_FORWARD_MAX_INTERVALS = 2

    # A carried-forward region is a stale copy standing in for a product
    # the source failed to serve.  Several sources can still serve the
    # missed slot from a short archive window afterwards (Météo-France's
    # DPPaquetRadar packet holds ~15 min), so carried slots stay eligible
    # for re-fetch this many intervals back before being written off.
    # Without the re-fetch, the stale copy permanently masks the slot:
    # the store reports the region present, every later cycle skips it,
    # and the animation keeps a frozen duplicate frame.
    _CARRY_FORWARD_REFETCH_MAX_INTERVALS = 3

    # Debounce window for on_cycle_complete triggers (initial backfill,
    # end of cycle, frame merges, satellite background fetches).  Multiple
    # triggers within the window collapse into a single trailing-edge
    # invocation so the full state.json serialisation runs at most once per
    # burst instead of up to 3-4x per fetch cycle.  5 s still coalesces the
    # per-cycle trigger burst into one invocation while cutting render-worker
    # visibility latency from ~30 s to ~5 s.
    _CYCLE_COMPLETE_DEBOUNCE_S = 5.0

    def __init__(
        self,
        store: FrameStore,
        cache: TileCache,
        nwp_contributions: list[NWPContribution] | None = None,
        satellite_contributions: list[SatelliteContribution] | None = None,
        nowcast_generator=None,
        storm_cell_generator=None,
        warmer=None,
        radar_cache=None,
        on_cycle_complete: Callable[[], Awaitable[None] | None] | None = None,
    ):
        self._store = store
        self._cache = cache
        # Every enabled NWP source flows through ``nwp_contributions``;
        # the dispatch loop in ``_fetch_auxiliary_grids`` and the
        # shutdown loop in ``close()`` iterate this list, so adding a
        # new NWP source needs no edits here.  ``inspect.signature`` is
        # used at call time to decide whether to pass
        # ``history_seconds``/``horizon_seconds`` — IFS's fetch takes
        # neither, every other grid takes both.
        self._nwp_contributions: list[NWPContribution] = list(
            nwp_contributions or []
        )
        # Same dispatch pattern as nwp_contributions: every active
        # satellite source (one per channel, e.g. GMGSI LW + VIS)
        # flows through this list and gets a per-cycle background
        # fetch.  Adding a new satellite source needs zero edits here.
        self._satellite_contributions: list[SatelliteContribution] = list(
            satellite_contributions or []
        )
        self._nowcast_generator = nowcast_generator
        self._storm_cell_generator = storm_cell_generator
        self._warmer = warmer
        self._radar_cache = radar_cache
        self._on_cycle_complete = on_cycle_complete
        # Coalescer state for the cycle-complete dump (see
        # _fire_cycle_complete): triggers within one debounce window
        # collapse into a single trailing-edge invocation.
        self._cycle_complete_debounce = self._CYCLE_COMPLETE_DEBOUNCE_S
        self._cycle_complete_task: asyncio.Task | None = None
        self._cycle_complete_dirty = False
        # NWP-phase wall time from the last auxiliary-grid fetch, reported
        # in the per-cycle phase breakdown logged by ``_fetch_all_frames``
        # and ``_fetch_initial``.  Defaults to 0.0 before the first cycle.
        self._last_nwp_phase_s = 0.0
        self._closed = False
        self._task: asyncio.Task | None = None
        self._satellite_tasks: dict[str, asyncio.Task] = {}
        # Provenance of carry-forward fills: ts -> region names whose
        # store entry is a stale copy, still awaiting the real product.
        self._carried_regions: dict[int, set[str]] = {}
        self._enabled_regions = [
            REGIONS[name] for name in settings.get_enabled_regions()
        ]

        self._na_source = settings.na_source
        self._ca_source = settings.ca_source

        # All radar source wiring now flows through the discovery
        # registry under ``librewxr.sources``.  Each source package
        # owns a ``radar_provider`` function that reads ``settings``
        # and returns a contribution (or ``None`` to opt out).  The
        # loop below applies the contribution to every enabled region
        # it covers; ``setdefault`` lets the first provider to claim
        # a region keep it (currently no two providers contest the
        # same region, but the guard is cheap and keeps order
        # deterministic).
        self._sources: dict[
            str,
            IEMSource | MRMSCompositeSource | MSCCanadaSource,
        ] = {}
        enabled_names = {r.name for r in self._enabled_regions}
        for contribution in collect_radar_contributions(settings):
            for region in contribution.regions:
                if region.name in enabled_names:
                    self._sources.setdefault(region.name, contribution.instance)

        # A provider returning ``None`` (e.g. ``mmd_enabled=False``) leaves
        # its regions out of ``self._sources``.  Drop those from the working
        # set so the fetch loop never lands on a region with no source.
        # Legacy-elif-managed regions always have a source, so this only
        # affects regions that have been migrated to the discovery path.
        dropped = [
            r.name for r in self._enabled_regions if r.name not in self._sources
        ]
        if dropped:
            logger.info(
                "Skipping regions with disabled providers: %s",
                ", ".join(sorted(dropped)),
            )
            self._enabled_regions = [
                r for r in self._enabled_regions if r.name in self._sources
            ]

        # MSC blending for CACOMP: only in ``ca_source=mrms_with_msc_blend``
        # mode.  In that mode MRMS owns the standalone
        # ``self._sources["CACOMP"]`` slot (via the MRMS provider's
        # ``regions`` list including CACOMP), so the blend partner is a
        # separately-managed MSC instance.  In ``ca_source=mrms`` mode no
        # MSC is fetched; in ``ca_source=msc`` mode MSC is the primary in
        # ``self._sources["CACOMP"]`` and no blend partner is needed.
        self._cacomp_msc_source: MSCCanadaSource | None = None
        if self._ca_source == "mrms_with_msc_blend" and any(
            r.name == "CACOMP" for r in self._enabled_regions
        ):
            self._cacomp_msc_source = MSCCanadaSource(
                settings.msc_canada_base_url
            )

        # IEM fallback for all US-group MRMS regions: only in mrms_fallback.
        # Gated on ``na_source`` (US-side concern) — independent of
        # ``ca_source``.
        self._iem_fallback: IEMSource | None = None
        if self._na_source == "mrms_fallback" and any(
            r.name in self._MRMS_REGIONS and r.group == "US"
            for r in self._enabled_regions
        ):
            self._iem_fallback = IEMSource(settings.iem_base_url)

        # Tracks last-known availability of MSC Canada blending so we can
        # log on state change instead of per blend call (backfill would
        # otherwise spam an identical WARNING for every historical frame).
        # None = not yet observed; True/False = last seen state.
        self._cacomp_msc_available: bool | None = None

    async def start(self) -> None:
        """Start the background fetch loop.

        Fetches auxiliary grids and the latest radar frame immediately so
        the server can start serving tiles within seconds.  Historical
        frames are backfilled in a background task.
        """
        region_names = [r.name for r in self._enabled_regions]
        logger.info("Fetching regions: %s", ", ".join(region_names))
        await self._fetch_initial()
        self._task = asyncio.create_task(self._backfill_then_loop())
        logger.info("Radar fetcher started (backfill running in background)")

    async def stop(self) -> None:
        # No new cycle-complete dumps may be scheduled once shutdown
        # begins, and any windowed dump still pending must be cancelled so
        # no orphaned timer task outlives the fetcher.  The dump task is
        # cancelled before the main loop task because the loop may be
        # blocked awaiting it inside _fire_cycle_complete.
        self._closed = True
        pending_dump = getattr(self, "_cycle_complete_task", None)
        if pending_dump is not None and not pending_dump.done():
            pending_dump.cancel()
            try:
                await pending_dump
            except asyncio.CancelledError:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Close all unique sources
        closed: set[int] = set()
        for source in self._sources.values():
            if id(source) not in closed:
                await source.close()
                closed.add(id(source))
        if self._iem_fallback and id(self._iem_fallback) not in closed:
            await self._iem_fallback.close()
            closed.add(id(self._iem_fallback))
        if self._cacomp_msc_source and id(self._cacomp_msc_source) not in closed:
            await self._cacomp_msc_source.close()
            closed.add(id(self._cacomp_msc_source))
        for contrib in self._nwp_contributions:
            try:
                await contrib.instance.close()
            except Exception:
                logger.exception("Error closing %s", contrib.name)
        for contrib in self._satellite_contributions:
            try:
                await contrib.instance.close()
            except Exception:
                logger.exception("Error closing %s", contrib.name)
        logger.info("Radar fetcher stopped")

    async def _backfill_then_loop(self) -> None:
        """Backfill historical frames, then enter the regular refresh loop.

        The loop sleeps until the next clock-aligned boundary (e.g. the next
        :x0 minute mark when fetch_interval=600) so that frame timestamps are
        always on clean multiples regardless of when the server started.
        """
        cycle_start = time.time()
        logger.debug("─── fetch cycle start (initial backfill) ───")
        nowcast_s = 0.0
        cells_s = 0.0
        try:
            await self._fetch_all_frames()
            # Per-stage timing for the cycle-complete log: nowcast and
            # storm-cell generation are the post-fetch compute stages
            # whose cost dominates the cycle duration.
            _nc_start = time.monotonic()
            await self._run_nowcast()
            nowcast_s = time.monotonic() - _nc_start
            _sc_start = time.monotonic()
            await self._run_storm_cells()
            cells_s = time.monotonic() - _sc_start
            # Fire-and-forget: the dump runs at the trailing edge of the
            # debounce window in the background, so the fetch loop is not
            # stalled and the imminent cycle-end trigger folds into this
            # same window.
            asyncio.create_task(self._fire_cycle_complete("initial-backfill"))
            if self._warmer is not None and settings.warm_overview_zoom >= 0:
                await self._warmer.warm_latest()
            self._schedule_warm()
        except Exception:
            logger.exception("Error in initial backfill")
        logger.info(
            "─── fetch cycle complete in %.1fs (initial backfill) ─── "
            "(nowcast=%.1fs cells=%.1fs)",
            time.time() - cycle_start, nowcast_s, cells_s,
        )
        release_memory()

        interval = settings.fetch_interval
        while True:
            now = time.time()
            next_boundary = (int(now // interval) + 1) * interval
            await asyncio.sleep(max(next_boundary - now, 1.0))
            cycle_start = time.time()
            boundary_iso = datetime.fromtimestamp(
                next_boundary, tz=timezone.utc,
            ).strftime("%Y-%m-%d %H:%M UTC")
            logger.debug("─── fetch cycle start (boundary %s) ───", boundary_iso)
            nowcast_s = 0.0
            cells_s = 0.0
            try:
                await self._fetch_all_frames()
                _nc_start = time.monotonic()
                await self._run_nowcast()
                nowcast_s = time.monotonic() - _nc_start
                _sc_start = time.monotonic()
                await self._run_storm_cells()
                cells_s = time.monotonic() - _sc_start
                # Fire-and-forget (see initial-backfill site): the dump
                # lands at the trailing edge of the window without stalling
                # the loop, and any satellite completion inside the window
                # is folded into the same invocation.
                asyncio.create_task(self._fire_cycle_complete("end-of-cycle"))
                self._schedule_warm()
            except Exception:
                logger.exception("Error in fetch loop")
            logger.info(
                "─── fetch cycle complete in %.1fs (boundary %s) ─── "
                "(nowcast=%.1fs cells=%.1fs)",
                time.time() - cycle_start, boundary_iso, nowcast_s, cells_s,
            )
            release_memory()

    async def _fire_cycle_complete(self, reason: str = "cycle") -> None:
        """Coalesced invocation of the on_cycle_complete hook.

        Every trigger site (end of cycle, frame merges, satellite
        background fetches) funnels through here.  Multiple triggers
        within ``_cycle_complete_debounce`` seconds collapse into a single
        trailing-edge invocation: the first trigger becomes the burst
        leader and waits out the window, and later triggers in the window
        are folded in — the dump reads the live stores at fire time, so
        their state is captured by the one invocation.  A trigger that
        arrives after the window closes starts a fresh burst, so a
        satellite that completes after the cycle-end dump still produces
        a dump within one debounce window of its completion instead of
        waiting for the next cycle.  An end-of-cycle trigger is never
        debounced away: it either leads a burst (whose dump is guaranteed
        to run) or is captured by the burst currently pending.

        Trigger sites either await this directly (the satellite background
        task, which must see the dump complete before it returns) or wrap
        it in ``asyncio.create_task`` (cycle-end / merge / backfill sites,
        so the fetch loop is never stalled by a window).  Either way the
        leader's dump is guaranteed to run and the caller blocks for at
        most one window.  The on_cycle_complete callback interface is
        unchanged.
        """
        if self._on_cycle_complete is None:
            return
        self._init_cycle_complete_state()
        if self._closed:
            return
        task = self._cycle_complete_task
        if task is not None and not task.done():
            # A windowed dump is already pending — fold this trigger into
            # it instead of serialising the snapshot again.
            self._cycle_complete_dirty = True
            logger.debug(
                "on_cycle_complete trigger (%s) coalesced into pending dump",
                reason,
            )
            return
        self._cycle_complete_task = asyncio.create_task(
            self._cycle_complete_runner(reason)
        )
        try:
            await self._cycle_complete_task
        finally:
            # The runner may have replaced the attribute with a follow-up
            # window (a trigger landed while the invocation was running) —
            # only clear it when the task we awaited really finished.
            if (
                self._cycle_complete_task is not None
                and self._cycle_complete_task.done()
            ):
                self._cycle_complete_task = None

    async def _cycle_complete_runner(self, reason: str) -> None:
        """Trailing-edge window for one trigger burst, then invoke the hook.

        Sleeps out the debounce window so burst members coalesce, then
        invokes the callback exactly once — the live-state read at that
        point captures every trigger that joined the burst.  A trigger
        that lands while the invocation is running may have been missed by
        the live read, so a follow-up window is scheduled for it.
        """
        try:
            await asyncio.sleep(self._cycle_complete_debounce)
        except asyncio.CancelledError:
            raise
        # Triggers folded into this burst are captured by the dump about
        # to run; reset the latch before invoking so only triggers that
        # arrive during the invocation can re-arm it.
        self._cycle_complete_dirty = False
        await self._invoke_on_cycle_complete(reason)
        if self._cycle_complete_dirty:
            self._cycle_complete_dirty = False
            self._cycle_complete_task = asyncio.create_task(
                self._cycle_complete_runner(reason)
            )

    async def _invoke_on_cycle_complete(self, reason: str) -> None:
        """Invoke the on_cycle_complete hook exactly once; never propagate failure.

        The hook may be a coroutine function (both call sites — main.py
        and data_pipeline.py — use ``async def``) or a plain sync
        function.  A sync hook's serialisation would block the loop, so it
        runs in a worker thread.  A failed dump must never kill the fetch
        loop — render workers simply read the previous snapshot.
        """
        hook = self._on_cycle_complete
        try:
            if inspect.iscoroutinefunction(hook):
                await hook()
            else:
                await asyncio.to_thread(hook)
        except Exception:
            logger.exception("on_cycle_complete hook failed (%s)", reason)

    def _init_cycle_complete_state(self) -> None:
        """Lazily initialise coalescer state.

        Production fetchers get it in ``__init__``; fetchers built via
        ``__new__`` in tests bypass that, so initialise on first use.  Test
        fetchers use a tiny debounce window so the coalesced invocation
        path is exercised without making each awaited call sleep the full
        production window.
        """
        if "_cycle_complete_task" in self.__dict__:
            return
        self._cycle_complete_debounce = 0.05
        self._cycle_complete_task = None
        self._cycle_complete_dirty = False
        self._closed = False

    def _schedule_warm(self) -> None:
        """Re-trigger overview warming for any previously-requested categories.

        Only starts warm_overview if at least one frame type (past or
        nowcast) has been triggered by a user request.  Skips if a
        previous warm pass is still running.
        """
        if self._warmer is None or settings.warm_overview_zoom < 0:
            return
        self._warmer.schedule_warm()

    async def _fetch_initial(self) -> None:
        """Quick startup: fetch auxiliary grids and latest radar frame only."""
        # Time the NWP-grid phase.  The NWP gather overlaps the radar fetch
        # the same way ``_fetch_all_frames`` does: the auxiliary grids are
        # independent of the radar path, so the task runs concurrently and
        # the finally block below reaps it before either return path exits.
        # The per-cycle phase breakdown is logged here (see below), so the
        # timing lives in this caller rather than in ``_fetch_timestamps``.
        _nwp_start = time.monotonic()
        nwp_task = asyncio.create_task(self._fetch_auxiliary_grids())
        try:
            interval = settings.fetch_interval
            now_rounded = int(time.time() // interval) * interval
            radar_s, post_s, write_s = await self._fetch_timestamps(
                [(now_rounded, "live", 0)]
            )
        finally:
            # Capture any in-flight radar exception BEFORE the await: if
            # the NWP task also raised, re-raising here would silently
            # REPLACE the radar exception (finally semantics).  The radar
            # path is the primary data source, so the NWP failure is
            # logged and the radar exception wins; when the radar path
            # succeeded, the NWP exception re-raises exactly as the old
            # sequential await did.
            radar_exc = sys.exc_info()[0]
            try:
                await nwp_task
            except Exception:
                if radar_exc is not None:
                    logger.exception(
                        "NWP auxiliary fetch failed (radar path also failed)"
                    )
                else:
                    raise
            self._last_nwp_phase_s = time.monotonic() - _nwp_start
        logger.info(
            "Fetch phases: nwp=%.1fs radar=%.1fs post=%.1fs write=%.1fs",
            self._last_nwp_phase_s, radar_s, post_s, write_s,
        )
        await self._run_nowcast()
        await self._run_storm_cells()

    async def _run_nowcast(self) -> None:
        """Trigger nowcast generation if enabled."""
        if self._nowcast_generator is not None:
            try:
                await self._nowcast_generator.generate()
            except Exception:
                logger.exception("Nowcast generation failed")

    async def _run_storm_cells(self) -> None:
        """Trigger storm-cell detection if enabled."""
        if self._storm_cell_generator is not None:
            try:
                await self._storm_cell_generator.generate()
            except Exception:
                logger.exception("Storm-cell detection failed")

    async def _fetch_auxiliary_grids(self) -> None:
        """Fetch every enabled NWP grid and kick off background satellite fetches.

        NWP grids hit independent S3 / HTTPS endpoints with no shared
        state, so they fan out under a small semaphore (sized by
        ``settings.nwp_fetch_concurrency``).  Wall-clock time drops from
        ``sum(fetch_i)`` to roughly ``max(fetch_i)`` × ceil(N / cap),
        bounded by the slowest single source.  The cap keeps peak RAM
        sane — each grid holds tens-to-hundreds of MB during decode.

        Cloud stays detached: it can take 40 s+ per ``.om`` file and we
        don't want to gate the radar cycle on it.
        """
        nwp_horizon = settings.nowcast_frames * settings.fetch_interval
        nwp_history = settings.max_frames * settings.fetch_interval

        # Build (label, grid, kwargs, failure-message) tuples for every
        # active NWP contribution.  ``inspect.signature`` decides whether
        # to pass ``history_seconds``/``horizon_seconds`` — IFS doesn't
        # accept either, every other grid does.  Each entry runs under
        # the semaphore independently so a slow source never holds back
        # another finishing.
        grid_specs: list[tuple[str, object, dict, str]] = []
        for contrib in self._nwp_contributions:
            grid = contrib.instance
            try:
                sig = inspect.signature(grid.fetch)
            except (TypeError, ValueError):
                sig = None
            kwargs: dict = {}
            if sig is not None and "history_seconds" in sig.parameters:
                kwargs["history_seconds"] = nwp_history
            if sig is not None and "horizon_seconds" in sig.parameters:
                kwargs["horizon_seconds"] = nwp_horizon
            grid_specs.append((
                contrib.name,
                grid,
                kwargs,
                f"{contrib.name} fetch failed, NWP layer may be stale",
            ))

        if grid_specs:
            cap = max(settings.nwp_fetch_concurrency, 1)
            semaphore = asyncio.Semaphore(cap)

            async def _run(label: str, grid, kwargs: dict, fail_msg: str) -> None:
                async with semaphore:
                    started = time.time()
                    try:
                        await grid.fetch(**kwargs)
                        logger.info(
                            "%s fetch finished in %.1fs", label, time.time() - started,
                        )
                    except Exception:
                        logger.warning(fail_msg)

            await asyncio.gather(*[
                _run(label, grid, kwargs, msg)
                for label, grid, kwargs, msg in grid_specs
            ])

        # Satellite sources run as background tasks — one per
        # contribution.  Per-channel detachment keeps GMGSI LW from
        # gating on a slow GMGSI VIS fetch (or vice versa).  Each
        # fetch is small (~7.5 MB) but goes through a high-latency
        # S3 endpoint, so detaching from the main cycle prevents
        # network jitter from stalling radar/NWP work.
        from librewxr.sources import satellite_source_slug

        for contrib in self._satellite_contributions:
            slug = satellite_source_slug(contrib)
            existing = self._satellite_tasks.get(slug)
            if existing is not None and not existing.done():
                logger.debug("%s fetch still running, skipping", contrib.name)
                continue
            self._satellite_tasks[slug] = asyncio.create_task(
                self._fetch_satellite_background(contrib),
            )

    async def _fetch_satellite_background(
        self, contrib: SatelliteContribution,
    ) -> None:
        """Fetch one satellite source detached from the main cycle.

        One background task per satellite source, firing the
        cycle-complete hook on success so render-only workers pick up
        new frames mid-cycle.  A failed fetch is warned and dropped;
        the next cycle retries.

        The fetch runs under ``satellite_fetch_timeout``.  The scheduler
        skips a channel while its previous task is still pending, so a
        fetch that hangs (instead of failing) would otherwise freeze the
        channel forever with no log above DEBUG.  On expiry the task
        completes, the skip-gate opens, and the next cycle retries.  Note
        the cancellation only abandons the coroutine — a worker thread
        blocked inside a syscall lingers until its socket dies, which the
        source's client-side timeouts keep bounded.
        """
        try:
            new_frames = await asyncio.wait_for(
                contrib.instance.fetch(), timeout=settings.satellite_fetch_timeout,
            )
        except TimeoutError:
            logger.warning(
                "%s fetch timed out after %.0fs, satellite layer may be stale",
                contrib.name, settings.satellite_fetch_timeout,
            )
            release_memory()
            return
        except Exception:
            logger.warning(
                "%s fetch failed, satellite layer may be stale", contrib.name,
            )
            release_memory()
            return
        release_memory()
        if new_frames:
            await self._fire_cycle_complete("satellite")

    async def _fetch_all_frames(self) -> None:
        """Fetch frames for all enabled regions to fill the store.

        Timestamps are aligned to ``fetch_interval`` boundaries (e.g. every
        10 minutes on the :x0 mark) so frames land on clean clock positions
        regardless of when the server was started.

        IEM's live endpoint serves the last 12 five-minute composites
        (indices 0–11, covering 0–55 min ago).  At 10-min spacing, frames
        0–50 min ago map to live indices 0, 2, 4, 6, 8, 10.  Older frames
        fall back to the archive endpoint.
        """
        # Time the NWP-grid phase.  The NWP gather now overlaps the radar
        # fetch: the auxiliary grids are independent of the radar path, so
        # the task runs concurrently and the finally block below reaps it
        # before either return path exits.  The quick-start
        # ``_fetch_initial`` path overlaps the same way (the attribute
        # defaults to 0.0 before the first cycle).
        _nwp_start = time.monotonic()
        nwp_task = asyncio.create_task(self._fetch_auxiliary_grids())
        try:
            interval = settings.fetch_interval
            interval_min = interval // 60
            now_rounded = int(time.time() // interval) * interval

            # Skip timestamps that already have all enabled regions.
            # Incomplete frames (e.g. a region fetch failed transiently) are
            # re-fetched so the missing regions can be merged in.
            existing_frames = await self._store.get_region_keys()
            enabled_names = {r.name for r in self._enabled_regions}

            # Prune carry-forward provenance for slots the source archive can
            # no longer serve (or that left the store) — past that point the
            # stale copy is final and retrying is wasted bandwidth.
            refetch_cutoff = (
                now_rounded - self._CARRY_FORWARD_REFETCH_MAX_INTERVALS * interval
            )
            for carried_ts in list(self._carried_regions):
                if carried_ts < refetch_cutoff or carried_ts not in existing_frames:
                    del self._carried_regions[carried_ts]

            ts_and_sources: list[tuple[int, str, int | datetime]] = []

            for i in range(settings.max_frames):
                minutes_ago = i * interval_min
                ts = now_rounded - i * interval

                # Carried-forward regions are stale copies; while their slot
                # is inside the recovery window they count as missing so the
                # real product can replace the copy (store merge + tile-cache
                # invalidation handle the swap).
                carried = self._carried_regions.get(ts, set())
                if (
                    ts in existing_frames
                    and enabled_names <= existing_frames[ts] - carried
                ):
                    continue

                # IEM live endpoint covers 0-55 min ago
                if minutes_ago <= 55:
                    ts_and_sources.append((ts, "live", minutes_ago))
                else:
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    ts_and_sources.append((ts, "archive", dt))

            if not ts_and_sources:
                logger.debug("All radar frames up to date, nothing to fetch")
                return

            radar_s, post_s, write_s = await self._fetch_timestamps(
                ts_and_sources, skip_regions=existing_frames,
            )
        finally:
            # Both the normal path and the early-return path land here, so
            # the NWP gather always completes before this method returns.
            # Capture any in-flight radar exception BEFORE the await: if
            # the NWP task also raised, re-raising here would silently
            # REPLACE the radar exception (finally semantics).  The radar
            # path is the primary data source, so the NWP failure is
            # logged and the radar exception wins; when the radar path
            # succeeded, the NWP exception re-raises exactly as the old
            # sequential await did.
            radar_exc = sys.exc_info()[0]
            try:
                await nwp_task
            except Exception:
                if radar_exc is not None:
                    logger.exception(
                        "NWP auxiliary fetch failed (radar path also failed)"
                    )
                else:
                    raise
            self._last_nwp_phase_s = time.monotonic() - _nwp_start

        # Per-stage cycle breakdown.  Timing is guaranteed on this path
        # (the finally block above always awaits the NWP task), so no
        # ``getattr`` fallback is needed.
        logger.info(
            "Fetch phases: nwp=%.1fs radar=%.1fs post=%.1fs write=%.1fs",
            self._last_nwp_phase_s, radar_s, post_s, write_s,
        )

    async def _fetch_timestamps(
        self,
        ts_and_sources: list[tuple[int, str, int | datetime]],
        skip_regions: dict[int, set[str]] | None = None,
    ) -> tuple[float, float, float]:
        """Fetch enabled regions for the given timestamps.

        Args:
            skip_regions: Optional mapping of timestamp -> region names to
                skip (already present in the store).  Only missing regions
                are fetched, saving bandwidth on retries for incomplete
                frames.  Regions recorded in ``_carried_regions`` are
                fetched despite being present — their store entry is a
                carry-forward copy awaiting the real product.

        Returns:
            ``(radar_s, post_s, write_s)``: wall-clock seconds spent in the
            radar gather, per-result post-processing, and store write
            phases.  The per-cycle phase log lives in the callers, which
            own the NWP timing now that the NWP fetch overlaps the radar
            fetch, so this method only measures and returns.
        """
        # For each timestamp, fetch regions in parallel (skipping any
        # already present from a previous partial fetch).
        sem = asyncio.Semaphore(settings.radar_fetch_concurrency)
        tasks = []
        task_meta: list[tuple[int, RegionDef, str, int | datetime]] = []

        for ts, source_type, source_arg in ts_and_sources:
            have = skip_regions.get(ts, set()) if skip_regions else set()
            carried = self._carried_regions.get(ts, set())
            for region in self._enabled_regions:
                if region.name in have and region.name not in carried:
                    continue
                source = self._sources[region.name]
                if source_type == "live":
                    tasks.append(_bounded_fetch(sem, source.fetch_frame(region, source_arg)))
                else:
                    tasks.append(_bounded_fetch(sem, source.fetch_archive_frame(region, source_arg)))
                task_meta.append((ts, region, source_type, source_arg))

        radar_start = time.monotonic()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        radar_s = time.monotonic() - radar_start

        # Group results by timestamp and build frames.  Pre-seed every
        # ts with an empty dict so the carry-forward pass below can fire
        # even when *every* region failed for a given timestamp — that's
        # the case we most want to bridge with prior data.
        frames_by_ts: dict[int, dict[str, np.ndarray]] = {
            ts: {} for ts, _, _ in ts_and_sources
        }

        # Per-result post-processing (fallback, CACOMP blend, despeckle)
        # touches only its own result, so all finalization coroutines fan
        # out under one gather instead of serialising each result's
        # awaits.  Exception results keep their inline warning + skip;
        # gather preserves the order of the coroutines passed, so the
        # merge pass below zips back 1:1 in the original result order.
        post_start = time.monotonic()
        finalize_coros: list[Awaitable[np.ndarray | None]] = []
        finalize_meta: list[tuple[int, RegionDef, str, int | datetime]] = []
        for (ts, region, source_type, source_arg), result in zip(task_meta, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Failed to fetch %s for ts=%d: %s", region.name, ts, result
                )
                continue
            finalize_coros.append(
                self._finalize_result(region, ts, source_type, source_arg, result)
            )
            finalize_meta.append((ts, region, source_type, source_arg))

        finalized = await asyncio.gather(*finalize_coros)

        for (ts, region, source_type, source_arg), result in zip(finalize_meta, finalized):
            if result is None:
                # Silent drop — nothing to merge for this slot.
                continue

            frames_by_ts[ts][region.name] = result

            # Real data landed — retire any carry-forward provenance so
            # later cycles stop re-fetching this slot.
            carried_here = self._carried_regions.get(ts)
            if carried_here is not None:
                carried_here.discard(region.name)
                if not carried_here:
                    del self._carried_regions[ts]
        post_s = time.monotonic() - post_start

        # Store frames in chronological order so carry-forward lookback
        # sees the freshest data (a backfill cycle that fetches several
        # timestamps at once needs the older frames stored before the
        # newer ones look back for missing regions).
        added = 0
        any_merged = False
        enabled_names = {r.name for r in self._enabled_regions}
        interval = settings.fetch_interval

        write_start = time.monotonic()
        for ts in sorted(frames_by_ts.keys()):
            regions_data = frames_by_ts[ts]

            # Carry-forward: any enabled region missing from both this
            # cycle's fetch AND the existing store frame at ts gets
            # filled from the most recent prior frame within
            # _CARRY_FORWARD_MAX_INTERVALS.  The .copy() detaches from
            # the prior frame's memmap so eviction of the source frame
            # later can't invalidate the carried data.
            already_have = (skip_regions or {}).get(ts, set())
            missing = enabled_names - set(regions_data.keys()) - already_have
            for lookback in range(1, self._CARRY_FORWARD_MAX_INTERVALS + 1):
                if not missing:
                    break
                prev_ts = ts - lookback * interval
                prev_frame = await self._store.get_frame(prev_ts)
                if prev_frame is None:
                    continue
                for name in list(missing):
                    if name in prev_frame.regions:
                        regions_data[name] = np.asarray(
                            prev_frame.regions[name]
                        ).copy()
                        stale_min = (lookback * interval) // 60
                        logger.info(
                            "%s: carry-forward into ts=%d from %d (%d min stale)",
                            name, ts, prev_ts, stale_min,
                        )
                        # Record provenance so later cycles re-fetch this
                        # slot while the source archive can still serve it.
                        self._carried_regions.setdefault(ts, set()).add(name)
                        missing.discard(name)

            if not regions_data:
                # Nothing to add — every enabled region was either
                # already in the store or absent with no carry-forward
                # source.  Skip the empty-frame write entirely.
                continue

            frame = RadarFrame(timestamp=ts, regions=regions_data)
            evicted_ts, merged = await self._store.add_frame(frame)
            if evicted_ts is not None:
                self._cache.invalidate_timestamp(evicted_ts)
            if merged:
                # Region data was merged into an existing frame — flush
                # stale tiles that were rendered without the new regions.
                self._cache.invalidate_timestamp(ts)
                any_merged = True
            if self._radar_cache is not None and not self._store._persistent:
                try:
                    # In persistent mode the FrameStore's own memmaps
                    # (<cache_dir>/radar/<ts>_<region>.dat) already persist
                    # every frame; RadarFrameCache.write_frame would write a
                    # second, identical copy of each region array into the
                    # same directory.  Only a non-persistent store (temp
                    # memmaps, e.g. single mode) needs the radar cache's
                    # copy to survive restarts.  Synchronous memmap writes
                    # run in a worker thread.
                    await asyncio.to_thread(self._radar_cache.write_frame, frame)
                except Exception:
                    logger.exception("Failed to persist radar frame %d", ts)
            added += 1
        write_s = time.monotonic() - write_start

        # Merges only invalidate the pipeline's local cache, which the
        # render workers don't share.  Dump state.json so the workers poll
        # the mtime change and flush their own tile caches without waiting
        # for the end-of-cycle snapshot.  Fire-and-forget: the cycle-end
        # trigger follows within the debounce window, so the two collapse
        # into a single trailing-edge dump that captures both.
        if any_merged:
            asyncio.create_task(self._fire_cycle_complete("frame-merge"))

        if self._radar_cache is not None and frames_by_ts:
            try:
                active_ts = await self._store.get_timestamps()
                self._radar_cache.cleanup(active_ts)
                regions_by_name = {r.name: r for r in self._enabled_regions}
                self._radar_cache.save_metadata(regions_by_name, active_ts)
            except Exception:
                logger.exception("Failed to update radar cache metadata")

        count = await self._store.frame_count()
        region_summary = ", ".join(
            f"{r.name}" for r in self._enabled_regions
        )
        logger.info(
            "Fetch complete: %d frames added, %d total in store (%s)",
            added, count, region_summary,
        )
        # Return the per-stage timings; the callers log the phase breakdown
        # (they own the NWP timing, which now overlaps the radar fetch).
        return radar_s, post_s, write_s

    async def _finalize_result(
        self,
        region: RegionDef,
        ts: int,
        source_type: str,
        source_arg: int | datetime,
        result: np.ndarray | None,
    ) -> np.ndarray | None:
        """Run one fetch result's post-processing: fallback, blend, despeckle.

        Extracted from the result loop in ``_fetch_timestamps`` so every
        result's per-result awaits (cross-source fallback, CACOMP blend,
        worker-thread despeckle) overlap under a single gather instead of
        serialising result after result.  The logic is byte-identical to
        the old inline pass: returns the final array, or ``None`` when the
        result was a silent drop (the caller then skips the slot).
        """
        if result is None:
            # MRMS fallback: if MRMS returned None, try IEM for US
            # regions (when na_source=mrms_fallback) or MSC standalone
            # for CACOMP (when ca_source=mrms_with_msc_blend).
            fallback_result = await self._try_fallback(region, ts, source_type, source_arg)  # noqa: E501
            if fallback_result is not None:
                result = fallback_result
            else:
                # Silent-None drops on the live slot are the visible-to-
                # user case — they create a partial or missing region in
                # the most-recent frame, which produces masking artifacts
                # and degrades the nowcast input.  Archive misses are
                # normal (sources have finite history) and stay silent.
                if source_type == "live":
                    logger.warning(
                        "%s: source returned no data for live ts=%d "
                        "(silent drop — region absent from this frame)",
                        region.name, ts,
                    )
                return None
        else:
            # CACOMP blending: MRMS+MSC blend mode only.
            if (
                self._ca_source == "mrms_with_msc_blend"
                and region.name == "CACOMP"
                and self._cacomp_msc_source is not None
            ):
                result = await self._blend_cacomp(result, region, ts, source_type, source_arg)  # noqa: E501

        if settings.despeckle_min_neighbors > 0:
            # Full-region numpy convolution (USCOMP ~63 MB) — run in
            # a worker thread.
            result = await asyncio.to_thread(
                _despeckle, result, settings.despeckle_min_neighbors,
            )

        return result

    async def _try_fallback(
        self,
        region: RegionDef,
        ts: int,
        source_type: str,
        source_arg: int | datetime,
    ) -> np.ndarray | None:
        """Cross-source fallback when the primary returned None.

        US-side: in ``na_source=mrms_fallback`` mode, fall back from
        MRMS to IEM for any US-group region.

        Canada-side: in ``ca_source=mrms_with_msc_blend`` mode, fall back
        from MRMS to MSC Canada standalone for CACOMP.  In ``ca_source=msc``
        mode MSC is already the primary so no fallback is needed; in
        ``ca_source=mrms`` mode MSC isn't fetched at all and the failure
        is just propagated (IFS fills the gap).
        """
        # US-group MRMS regions fall back to IEM (gated by na_source).
        if (
            self._na_source == "mrms_fallback"
            and region.name in self._MRMS_REGIONS
            and region.group == "US"
            and self._iem_fallback is not None
        ):
            logger.info("MRMS failed for %s, falling back to IEM", region.name)
            try:
                if source_type == "live":
                    return await self._iem_fallback.fetch_frame(region, source_arg)
                else:
                    return await self._iem_fallback.fetch_archive_frame(region, source_arg)
            except Exception:
                logger.exception("IEM fallback also failed for %s", region.name)
                return None

        # CACOMP MRMS-failure fallback to MSC standalone (gated by ca_source).
        if (
            self._ca_source == "mrms_with_msc_blend"
            and region.name == "CACOMP"
            and self._cacomp_msc_source is not None
        ):
            logger.info("MRMS failed for CACOMP, falling back to MSC standalone")
            try:
                if source_type == "live":
                    return await self._cacomp_msc_source.fetch_frame(region, source_arg)
                else:
                    return await self._cacomp_msc_source.fetch_archive_frame(region, source_arg)
            except Exception:
                logger.exception("MSC fallback also failed for CACOMP")
                return None

        return None

    async def _blend_cacomp(
        self,
        mrms_data: np.ndarray,
        region: RegionDef,
        ts: int,
        source_type: str,
        source_arg: int | datetime,
    ) -> np.ndarray:
        """Blend MRMS and MSC Canada data for CACOMP.

        MRMS data takes priority within its extent (20-55°N, -130 to -60°W).
        MSC fills gaps: north of 55°N, east of -60°W, west of -130°W.
        """
        # Fetch MSC Canada data for the same timestamp
        msc_data: np.ndarray | None = None
        if self._cacomp_msc_source is not None:
            try:
                if source_type == "live":
                    msc_data = await self._cacomp_msc_source.fetch_frame(region, source_arg)
                else:
                    msc_data = await self._cacomp_msc_source.fetch_archive_frame(region, source_arg)
            except Exception:
                logger.exception("MSC Canada fetch failed during CACOMP blend")

        if msc_data is None:
            if self._cacomp_msc_available is not False:
                logger.warning("CACOMP: MSC data unavailable, using MRMS only")
                self._cacomp_msc_available = False
            return mrms_data

        if self._cacomp_msc_available is False:
            logger.info("CACOMP: MSC data available again, resuming blend")
        self._cacomp_msc_available = True

        # Mask construction + blending is full-region numpy work (tens of
        # MB of temporaries for CACOMP) — run it in a worker thread.
        return await asyncio.to_thread(
            _blend_cacomp_arrays, mrms_data, msc_data, region,
        )


async def _bounded_fetch(sem: asyncio.Semaphore, coro: Awaitable) -> object:
    """Run a single fetch coroutine under the radar fetch-concurrency semaphore.

    Wraps an already-created (but not yet scheduled) coroutine object so
    the gather in ``_fetch_timestamps`` can hand each task a semaphore
    slot without starting any fetch ahead of the gather.
    """
    async with sem:
        return await coro


def _blend_cacomp_arrays(
    mrms_data: np.ndarray,
    msc_data: np.ndarray,
    region: RegionDef,
) -> np.ndarray:
    """Blend MRMS and MSC Canada arrays for CACOMP (pure numpy).

    Runs in a worker thread from ``RadarFetcher._blend_cacomp`` so the
    full-region mask build and boolean-indexed assignment never block
    the event loop.
    """
    # Build the MRMS extent mask at the region's pixel resolution.
    # Lats go north-to-south (row 0 = northernmost pixel) to match
    # the data array convention used by the renderer.
    ps_y = region._ps_y
    north_center = region.north - ps_y / 2
    south_center = region.south + ps_y / 2
    lats = np.linspace(north_center, south_center, region.height)
    lons = np.arange(region.west, region.east, region.pixel_size)

    mrms_south, mrms_north, mrms_west, mrms_east = MRMS_EXTENTS["USCOMP"]
    mrms_extent_mask = (
        (lats[:, None] >= mrms_south) &
        (lats[:, None] <= mrms_north) &
        (lons[None, :] >= mrms_west) &
        (lons[None, :] <= mrms_east)
    )

    # Blend: start with MSC, overlay MRMS where available
    result = msc_data.copy()
    # Only overlay MRMS where it has actual data (non-zero) within its extent
    mrms_has_data = (mrms_data > 0) & mrms_extent_mask
    result[mrms_has_data] = mrms_data[mrms_has_data]

    # Outside MRMS extent, keep MSC as-is (including zeros for no data)
    # Inside MRMS extent but where MRMS has no data (value=0), also keep
    # MSC if it has data — but only if MSC data exists.
    mrms_extent_nodata = mrms_extent_mask & (mrms_data == 0) & (msc_data > 0)
    result[mrms_extent_nodata] = msc_data[mrms_extent_nodata]

    return result


def _despeckle(data: np.ndarray, min_neighbors: int) -> np.ndarray:
    """Remove isolated pixels (ground clutter / AP artifacts).

    Uses padded slicing instead of np.roll for ~2.4x speedup on large
    arrays.  Slicing also avoids the wrap-around artifact that np.roll
    produces at array edges.
    """
    mask = data > 0
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    count = np.zeros((h, w), dtype=np.int8)
    for dr in range(3):
        for dc in range(3):
            if dr == 1 and dc == 1:
                continue
            count += padded[dr:dr + h, dc:dc + w]

    result = data.copy()
    result[mask & (count < min_neighbors)] = 0
    return result
