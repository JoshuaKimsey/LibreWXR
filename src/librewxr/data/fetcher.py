# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

import numpy as np

from librewxr.config import settings
from librewxr.memory import release_memory
from librewxr.data.arome_antilles_grid import AROMEAntillesGrid
from librewxr.data.wrf_smn_grid import WRFSMNGrid
from librewxr.data.cloud_grid import CloudGrid
from librewxr.data.dmi_dini_grid import DMIDiniGrid
from librewxr.data.ecmwf_grid import ECMWFGrid
from librewxr.data.hrdps_grid import HRDPSGrid
from librewxr.data.hrrr_alaska_grid import HRRRAlaskaGrid
from librewxr.data.hrrr_grid import HRRRGrid
from librewxr.data.icon_eu_grid import ICONEUGrid
from librewxr.data.regions import REGIONS, RegionDef
from librewxr.data.sources import (
    CWASource,
    IEMSource,
    MARNSource,
    MMDSource,
    MRMS_EXTENTS,
    MRMS_PRODUCTS,
    MRMSSource,
    MSCCanadaSource,
    OperaSource,
)
from librewxr.data.store import FrameStore, RadarFrame
from librewxr.tiles.cache import TileCache

logger = logging.getLogger(__name__)


class RadarFetcher:
    """Background task that periodically fetches radar frames."""

    # Regions that MRMS can serve (each has its own regional product).
    _MRMS_REGIONS = {
        "USCOMP", "CACOMP", "AKCOMP", "HICOMP", "PRCOMP", "GUCOMP"
    }

    def __init__(
        self,
        store: FrameStore,
        cache: TileCache,
        ecmwf_grid: ECMWFGrid | None = None,
        hrrr_grid: HRRRGrid | None = None,
        hrrr_alaska_grid: HRRRAlaskaGrid | None = None,
        hrdps_grid: HRDPSGrid | None = None,
        arome_antilles_grid: AROMEAntillesGrid | None = None,
        wrf_smn_grid: WRFSMNGrid | None = None,
        icon_eu_grid: ICONEUGrid | None = None,
        dmi_dini_grid: DMIDiniGrid | None = None,
        cloud_grid: CloudGrid | None = None,
        nowcast_generator=None,
        warmer=None,
        radar_cache=None,
        on_cycle_complete: Callable[[], Awaitable[None] | None] | None = None,
    ):
        self._store = store
        self._cache = cache
        self._ecmwf_grid = ecmwf_grid
        self._hrrr_grid = hrrr_grid
        self._hrrr_alaska_grid = hrrr_alaska_grid
        self._hrdps_grid = hrdps_grid
        self._arome_antilles_grid = arome_antilles_grid
        self._wrf_smn_grid = wrf_smn_grid
        self._icon_eu_grid = icon_eu_grid
        self._dmi_dini_grid = dmi_dini_grid
        self._cloud_grid = cloud_grid
        self._nowcast_generator = nowcast_generator
        self._warmer = warmer
        self._radar_cache = radar_cache
        self._on_cycle_complete = on_cycle_complete
        self._task: asyncio.Task | None = None
        self._cloud_task: asyncio.Task | None = None
        self._enabled_regions = [
            REGIONS[name] for name in settings.get_enabled_regions()
            # Honour mmd_enabled (default True) — when off, drop the MET
            # Malaysia regions even if the user's region spec is a group
            # alias (e.g. SOUTHEAST_ASIA, ALL) that would otherwise pull
            # them in.
            if not (
                name in ("MYPENINSULAR", "MYEAST")
                and not settings.mmd_enabled
            )
        ]

        self._na_source = settings.na_source

        # Build a source for each enabled region based on its group
        # and the na_source setting.
        #
        # Group → source class dispatch (kept alphabetical; extend by
        # adding a new ``elif region.group == "X":`` branch below):
        #   CANADA           → MSCCanadaSource
        #   CENTRAL_AMERICA  → MARNSource
        #   EUROPE           → OperaSource
        #   SOUTHEAST_ASIA   → MMDSource (MYPENINSULAR, MYEAST)
        #   TAIWAN           → CWASource
        #   US               → MRMSSource (when na_source uses mrms) or IEMSource
        self._sources: dict[
            str,
            CWASource | IEMSource | MARNSource | MMDSource | MRMSSource | MSCCanadaSource | OperaSource,
        ] = {}
        canada_source: MSCCanadaSource | None = None
        cwa_source: CWASource | None = None
        marn_source: MARNSource | None = None
        mmd_source: MMDSource | None = None
        iem_source: IEMSource | None = None
        opera_source: OperaSource | None = None
        # Keyed by MRMS product path so regions sharing a product (e.g.
        # USCOMP and CACOMP both use the bare CONUS path) share one
        # MRMSSource — one HTTP client, one directory cache, one GRIB2
        # download per fetch cycle.
        mrms_sources: dict[str, MRMSSource] = {}

        use_mrms = self._na_source in ("mrms", "mrms_fallback")
        for region in self._enabled_regions:
            # MRMS routes by region.name (one product path per region),
            # so it's checked first as an override for US-group regions.
            if use_mrms and region.name in self._MRMS_REGIONS:
                product = MRMS_PRODUCTS[region.name]
                if product not in mrms_sources:
                    mrms_sources[product] = MRMSSource(
                        settings.mrms_base_url, region_name=region.name
                    )
                self._sources[region.name] = mrms_sources[product]
            elif region.group == "CANADA":
                if canada_source is None:
                    canada_source = MSCCanadaSource(settings.msc_canada_base_url)
                self._sources[region.name] = canada_source
            elif region.group == "CENTRAL_AMERICA":
                if marn_source is None:
                    marn_source = MARNSource(settings.marn_base_url)
                self._sources[region.name] = marn_source
            elif region.group == "EUROPE":
                if opera_source is None:
                    opera_source = OperaSource(settings.opera_base_url)
                self._sources[region.name] = opera_source
            elif region.group == "SOUTHEAST_ASIA":
                # MYPENINSULAR + MYEAST both ride on MET Malaysia's
                # combined GIF (peer regions, one HTTP fetch per cycle
                # shared across both).  The ``mmd_enabled`` toggle is
                # enforced upstream in ``self._enabled_regions``.
                if mmd_source is None:
                    mmd_source = MMDSource(
                        settings.mmd_base_url,
                        publish_lag_sec=settings.mmd_publish_lag_sec,
                    )
                self._sources[region.name] = mmd_source
            elif region.group == "TAIWAN":
                if cwa_source is None:
                    cwa_source = CWASource(settings.cwa_base_url)
                self._sources[region.name] = cwa_source
            else:
                # Default for US-group regions when not using MRMS.
                if iem_source is None:
                    iem_source = IEMSource(settings.iem_base_url)
                self._sources[region.name] = iem_source

        # MSC blending for CACOMP: only in mrms_fallback mode.
        self._cacomp_msc_source: MSCCanadaSource | None = None
        if self._na_source == "mrms_fallback" and any(
            r.name == "CACOMP" for r in self._enabled_regions
        ):
            if canada_source is None:
                canada_source = MSCCanadaSource(settings.msc_canada_base_url)
            self._cacomp_msc_source = canada_source

        # IEM fallback for all US-group MRMS regions: only in mrms_fallback.
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
        if self._ecmwf_grid:
            await self._ecmwf_grid.close()
        if self._hrrr_grid:
            await self._hrrr_grid.close()
        if self._hrrr_alaska_grid:
            await self._hrrr_alaska_grid.close()
        if self._hrdps_grid:
            await self._hrdps_grid.close()
        if self._arome_antilles_grid:
            await self._arome_antilles_grid.close()
        if self._wrf_smn_grid:
            await self._wrf_smn_grid.close()
        if self._icon_eu_grid:
            await self._icon_eu_grid.close()
        if self._dmi_dini_grid:
            await self._dmi_dini_grid.close()
        logger.info("Radar fetcher stopped")

    async def _backfill_then_loop(self) -> None:
        """Backfill historical frames, then enter the regular refresh loop.

        The loop sleeps until the next clock-aligned boundary (e.g. the next
        :x0 minute mark when fetch_interval=600) so that frame timestamps are
        always on clean multiples regardless of when the server started.
        """
        cycle_start = time.time()
        logger.info("─── fetch cycle start (initial backfill) ───")
        try:
            await self._fetch_all_frames()
            await self._run_nowcast()
            await self._fire_cycle_complete()
            if self._warmer is not None and settings.warm_overview_zoom >= 0:
                await self._warmer.warm_latest()
            self._schedule_warm()
        except Exception:
            logger.exception("Error in initial backfill")
        logger.info(
            "─── fetch cycle complete in %.1fs (initial backfill) ───",
            time.time() - cycle_start,
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
            logger.info("─── fetch cycle start (boundary %s) ───", boundary_iso)
            try:
                await self._fetch_all_frames()
                await self._run_nowcast()
                await self._fire_cycle_complete()
                self._schedule_warm()
            except Exception:
                logger.exception("Error in fetch loop")
            logger.info(
                "─── fetch cycle complete in %.1fs (boundary %s) ───",
                time.time() - cycle_start, boundary_iso,
            )
            release_memory()

    async def _fire_cycle_complete(self) -> None:
        """Invoke the on_cycle_complete hook if set; never propagate failure.

        The data-pipeline process uses this to dump a cross-process state
        snapshot.  A failed dump must never kill the fetcher loop — render
        workers will simply read the previous snapshot.
        """
        if self._on_cycle_complete is None:
            return
        try:
            result = self._on_cycle_complete()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("on_cycle_complete hook failed")

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
        await self._fetch_auxiliary_grids()

        interval = settings.fetch_interval
        now_rounded = int(time.time() // interval) * interval
        await self._fetch_timestamps([(now_rounded, "live", 0)])
        await self._run_nowcast()

    async def _run_nowcast(self) -> None:
        """Trigger nowcast generation if enabled."""
        if self._nowcast_generator is not None:
            try:
                await self._nowcast_generator.generate()
            except Exception:
                logger.exception("Nowcast generation failed")

    async def _fetch_auxiliary_grids(self) -> None:
        """Fetch every enabled NWP grid and kick off background cloud fetch.

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

        # (label, grid, kwargs, failure-message) tuples.  Each entry runs
        # under the semaphore independently, so a slow source never holds
        # back another finishing.
        grid_specs: list[tuple[str, object, dict, str]] = []
        if self._ecmwf_grid is not None:
            grid_specs.append((
                "ECMWF IFS", self._ecmwf_grid, {},
                "ECMWF IFS fetch failed, global fallback may be stale",
            ))
        if self._hrrr_grid is not None:
            grid_specs.append((
                "HRRR", self._hrrr_grid,
                {"history_seconds": nwp_history, "horizon_seconds": nwp_horizon},
                "HRRR fetch failed, CONUS NWP layer may be stale",
            ))
        if self._hrrr_alaska_grid is not None:
            grid_specs.append((
                "HRRR-Alaska", self._hrrr_alaska_grid,
                {"history_seconds": nwp_history, "horizon_seconds": nwp_horizon},
                "HRRR-Alaska fetch failed, AK NWP layer may be stale",
            ))
        if self._hrdps_grid is not None:
            grid_specs.append((
                "HRDPS", self._hrdps_grid,
                {"history_seconds": nwp_history, "horizon_seconds": nwp_horizon},
                "HRDPS fetch failed, Canada NWP layer may be stale",
            ))
        if self._arome_antilles_grid is not None:
            grid_specs.append((
                "AROME Antilles", self._arome_antilles_grid,
                {"history_seconds": nwp_history, "horizon_seconds": nwp_horizon},
                "AROME Antilles fetch failed, Caribbean NWP layer may be stale",
            ))
        if self._wrf_smn_grid is not None:
            grid_specs.append((
                "WRF-SMN", self._wrf_smn_grid,
                {"history_seconds": nwp_history, "horizon_seconds": nwp_horizon},
                "WRF-SMN fetch failed, S. America NWP layer may be stale",
            ))
        if self._icon_eu_grid is not None:
            grid_specs.append((
                "ICON-EU", self._icon_eu_grid,
                {"history_seconds": nwp_history, "horizon_seconds": nwp_horizon},
                "ICON-EU fetch failed, EU NWP layer may be stale",
            ))
        if self._dmi_dini_grid is not None:
            grid_specs.append((
                "DMI DINI", self._dmi_dini_grid,
                {"history_seconds": nwp_history, "horizon_seconds": nwp_horizon},
                "DMI DINI fetch failed, NW Europe NWP layer may be stale",
            ))

        if grid_specs:
            cap = max(settings.nwp_fetch_concurrency, 1)
            semaphore = asyncio.Semaphore(cap)

            async def _run(label: str, grid, kwargs: dict, fail_msg: str) -> None:
                async with semaphore:
                    started = time.time()
                    try:
                        await grid.fetch(**kwargs)
                        logger.debug(
                            "%s fetch finished in %.1fs", label, time.time() - started,
                        )
                    except Exception:
                        logger.warning(fail_msg)

            await asyncio.gather(*[
                _run(label, grid, kwargs, msg)
                for label, grid, kwargs, msg in grid_specs
            ])

        # Cloud data loads in the background — never blocks radar startup.
        # Skip if a previous fetch is still running (downloading .om files
        # takes ~40s each).  With disk caching, already-cached timestamps
        # are free to check, so there's no cost to running every cycle.
        if self._cloud_grid is not None:
            already_running = (
                self._cloud_task is not None and not self._cloud_task.done()
            )
            if already_running:
                logger.debug("Cloud fetch still running, skipping")
            else:
                self._cloud_task = asyncio.create_task(
                    self._fetch_cloud_background()
                )

    async def _fetch_cloud_background(self) -> None:
        """Fetch cloud cover data without blocking the main fetch cycle.

        Cloud is the slowest auxiliary fetch (~40 s per .om file), so it
        runs detached and the main cycle dumps ``state.json`` without
        waiting.  We re-fire the cycle-complete hook here on success so
        render-only workers pick up the new cloud grid mid-cycle instead
        of waiting for the next :x0 boundary.
        """
        try:
            await self._cloud_grid.fetch()
        except Exception:
            logger.warning("Cloud cover fetch failed, satellite layer may be stale")
            release_memory()
            return
        release_memory()
        await self._fire_cycle_complete()

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
        await self._fetch_auxiliary_grids()

        interval = settings.fetch_interval
        interval_min = interval // 60
        now_rounded = int(time.time() // interval) * interval

        # Skip timestamps that already have all enabled regions.
        # Incomplete frames (e.g. a region fetch failed transiently) are
        # re-fetched so the missing regions can be merged in.
        existing_frames = await self._store.get_region_keys()
        enabled_names = {r.name for r in self._enabled_regions}

        ts_and_sources: list[tuple[int, str, int | datetime]] = []

        for i in range(settings.max_frames):
            minutes_ago = i * interval_min
            ts = now_rounded - i * interval

            if ts in existing_frames and enabled_names <= existing_frames[ts]:
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

        await self._fetch_timestamps(ts_and_sources, skip_regions=existing_frames)

    async def _fetch_timestamps(
        self,
        ts_and_sources: list[tuple[int, str, int | datetime]],
        skip_regions: dict[int, set[str]] | None = None,
    ) -> None:
        """Fetch enabled regions for the given timestamps.

        Args:
            skip_regions: Optional mapping of timestamp -> region names to
                skip (already present in the store).  Only missing regions
                are fetched, saving bandwidth on retries for incomplete frames.
        """
        # For each timestamp, fetch regions in parallel (skipping any
        # already present from a previous partial fetch).
        tasks = []
        task_meta: list[tuple[int, RegionDef]] = []

        for ts, source_type, source_arg in ts_and_sources:
            have = skip_regions.get(ts, set()) if skip_regions else set()
            for region in self._enabled_regions:
                if region.name in have:
                    continue
                source = self._sources[region.name]
                if source_type == "live":
                    tasks.append(source.fetch_frame(region, source_arg))
                else:
                    tasks.append(source.fetch_archive_frame(region, source_arg))
                task_meta.append((ts, region))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Group results by timestamp and build frames
        frames_by_ts: dict[int, dict[str, np.ndarray]] = {}
        for (ts, region), result in zip(task_meta, results):
            if isinstance(result, Exception):
                logger.warning(
                    "Failed to fetch %s for ts=%d: %s", region.name, ts, result
                )
                continue
            if result is None:
                # MRMS fallback: if MRMS returned None, try IEM for USCOMP
                # or MSC standalone for CACOMP.
                fallback_result = await self._try_fallback(region, ts, source_type, source_arg)  # noqa: E501
                if fallback_result is not None:
                    result = fallback_result
                else:
                    continue
            else:
                # CACOMP blending: only in mrms_fallback mode.
                if (
                    self._na_source == "mrms_fallback"
                    and region.name == "CACOMP"
                    and self._cacomp_msc_source is not None
                ):
                    result = await self._blend_cacomp(result, region, ts, source_type, source_arg)  # noqa: E501

            if settings.despeckle_min_neighbors > 0:
                result = _despeckle(result, settings.despeckle_min_neighbors)

            if ts not in frames_by_ts:
                frames_by_ts[ts] = {}
            frames_by_ts[ts][region.name] = result

        # Store frames
        added = 0
        for ts, regions_data in frames_by_ts.items():
            frame = RadarFrame(timestamp=ts, regions=regions_data)
            evicted_ts, merged = await self._store.add_frame(frame)
            if evicted_ts is not None:
                self._cache.invalidate_timestamp(evicted_ts)
            if merged:
                # Region data was merged into an existing frame — flush
                # stale tiles that were rendered without the new regions.
                self._cache.invalidate_timestamp(ts)
            if self._radar_cache is not None:
                try:
                    self._radar_cache.write_frame(frame)
                except Exception:
                    logger.exception("Failed to persist radar frame %d", ts)
            added += 1

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

    async def _try_fallback(
        self,
        region: RegionDef,
        ts: int,
        source_type: str,
        source_arg: int | datetime,
    ) -> np.ndarray | None:
        """Fall back to IEM if MRMS fails for any US-group region, or MSC for CACOMP."""
        if self._na_source != "mrms_fallback":
            return None

        # US-group MRMS regions fall back to IEM
        if (
            region.name in self._MRMS_REGIONS
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

        # CACOMP fallback: use MSC Canada standalone (no MRMS blending)
        if region.name == "CACOMP" and self._cacomp_msc_source is not None:
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
