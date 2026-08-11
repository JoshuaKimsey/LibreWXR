# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import functools
import json
import logging
import os
import pathlib
import time

import psutil

from fastapi import APIRouter, HTTPException, Path, Query, Request, Response

from datetime import datetime

from librewxr.api.models import (
    AlertProperties,
    AlertsResponse,
    ColorScheme,
    GeoJSONFeature,
    RadarData,
    RadarTimestamp,
    SatelliteData,
    WeatherMapsResponse,
)
from librewxr.api.conditional import compute_etag, conditional_response
from librewxr.colors.schemes import SCHEME_NAMES
from librewxr.config import settings
from librewxr.data.store import FrameStore
from librewxr.data.worker_pulse import read_worker_pulses
from librewxr.mcp.discovery import build_ai_catalog
from librewxr.memory import detect_memory_limit_mb
from librewxr.tiles.cache import CachedRender, TileCache
from librewxr.tiles.coordinates import coord_cache_bytes, coord_cache_stats
from librewxr.tiles.renderer import (
    compute_tile_geometry,
    present_tile,
    render_coverage_tile,
)
from librewxr.tiles.request_tracker import TileRequestTracker
from librewxr.tiles.satellite_renderer import (
    render_gmgsi_composite_tile,
    render_gmgsi_tile,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# These get set by main.py during startup
frame_store: FrameStore | None = None
tile_cache: TileCache | None = None
# All NWP grids live in a single dict keyed by slug
# (``hrrr_grid``, ``arome_antilles_grid``, ``ecmwf_grid``, etc.) —
# generated from ``NWPContribution.name`` via ``nwp_grid_slug``.  The
# ``/health`` endpoint iterates this dict so adding a new NWP source
# requires no edits here.  ``ecmwf_grid`` is also bound as an attribute
# below for the radar tile arrow path that still treats IFS specially.
nwp_grids: dict[str, object] = {}
ecmwf_grid = None  # ECMWFGrid | None — special-cased by /v2/radar arrows
nwp_chain = None  # NWPChain | None
precip_mask = None  # PrecipMaskStore | None — set by main.py (multi mode only)
# GMGSI satellite sources keyed by slug (gmgsi_lw_grid, gmgsi_vis_grid).
# Routes index by slug so the /health endpoint and tile dispatcher
# auto-pick up new channels without per-source plumbing.
satellite_grids: dict[str, object] = {}
tile_warmer = None  # TileWarmer | None
nowcast_store = None  # NowcastStore | None
storm_cell_store = None  # StormCellStore | None
radar_cache = None  # RadarFrameCache | None
radar_fetcher = None  # RadarFetcher | None
tile_request_tracker: TileRequestTracker | None = None
start_time: float = 0.0
enabled_regions: list[str] | None = None
# Memory monitor — set by main.py in both lifespans.  Provides the cgroup
# anon/file/shmem split for the /health ``cluster`` section.
memory_monitor = None  # MemoryMonitor | None

# Tile present pool - set by main.py.  Multi-mode render workers get a
# dedicated executor for the cheap ``present_tile`` tail (colorize/encode)
# so those jobs never queue behind long geometry computes on the shared
# default executor under a cold-tile burst.  Single mode leaves this None
# and the tile endpoints fall back to ``asyncio.to_thread`` (the loop
# default executor), byte-identical to the pre-split behaviour.
present_executor = None  # ThreadPoolExecutor | None

# Shared-store I/O pool - set by main.py in multi mode only.  Dedicated
# executor for shared-tile-store reads/publishes so they never queue
# behind geometry computes on the default executor.  Single mode leaves
# this None and the shared-store call sites fall back to
# ``asyncio.to_thread`` (the loop default executor).
io_executor = None  # ThreadPoolExecutor | None

# Shared on-disk encoded-tile store - set by main.py in multi mode only.
# A ``radar_tile`` hit skips frame fetch + geometry compute + present
# entirely; plain present misses publish their fresh encode for the other
# workers.
shared_tile_store = None  # SharedTileStore | None

# Holds fire-and-forget shared-store publish tasks so they can't be GC'd
# mid-flight; each task discards itself on completion.
_pending_shared_publishes: set = set()

# Latest-timestamp TTL cache for the radar tile hot path:
# (monotonic time, timestamp list).  ``radar_tile`` only needs the latest
# frame to pick the Cache-Control ``max_age`` bucket (300 s vs 7200 s), so
# re-querying the store lock on every request is wasted contention; 5 s of
# staleness is immaterial next to those buckets.
_latest_ts_cache: tuple[float, list[int]] | None = None
_LATEST_TS_TTL = 5.0

# WMO alerts — set by main.py during startup
alerts_store = None  # AlertsStore | None
alerts_fetcher = None  # WMOAlertsFetcher | None
alerts_enabled: bool = False

# MCP server — set by main.py during startup (only when settings.mcp_enabled
# is True AND the [mcp] extra successfully imported + mounted).  ``mcp_mounted``
# distinguishes "config asked for MCP but the build/import failed" (False)
# from "MCP endpoint is live and answering" (True).
mcp_mounted: bool = False
mcp_path: str = "/mcp"
mcp_tools: list[str] = []


def _nwp_grid_health_blocks() -> dict[str, dict]:
    """Build per-grid ``/health`` blocks for every entry in ``nwp_grids``.

    IFS reports a different shape (``reference_time`` + ``timesteps``)
    than the chain-source grids (``latest_run`` + ``frames``).  Detect
    by attribute presence rather than slug — keeps the shape stable if
    a future provider adopts either pattern.
    """
    blocks: dict[str, dict] = {}
    for slug, grid in nwp_grids.items():
        if grid is None:
            blocks[slug] = {"enabled": False, "loaded": False}
            continue
        if hasattr(grid, "reference_time") and hasattr(grid, "timestep_count"):
            blocks[slug] = {
                "loaded": getattr(grid, "data", None) is not None,
                "reference_time": grid.reference_time,
                "timesteps": grid.timestep_count,
            }
        else:
            blocks[slug] = {
                "enabled": True,
                "loaded": grid.has_data(),
                "latest_run": grid.latest_run_iso,
                "frames": grid.frame_count,
            }
    return blocks


def _avg_ms(total_ns: int, count: int) -> float:
    """Mean latency in milliseconds from ns totals; 0.0 when empty."""
    if count == 0:
        return 0.0
    return round(total_ns / count / 1e6, 2)


def collect_worker_pulse() -> dict:
    """Compact per-process payload for the cluster worker-pulse files.

    Every field is derived from the module-level singletons with None
    guards — render-only mode leaves several unset (``radar_cache``,
    ``radar_fetcher``, ``alerts_fetcher``, ``tile_warmer``).  The payload
    is deliberately small (< 2 KB) so a /health scan of 16 tiny JSON
    files stays cheap.
    """
    payload = {
        "pid": os.getpid(),
        "written_at": int(time.time()),
        "rss_bytes": psutil.Process().memory_info().rss,
    }

    if tile_cache is not None:
        payload["tile_cache"] = {
            "entries": tile_cache.size,
            "total_bytes": tile_cache.total_bytes,
            "max_bytes": tile_cache.max_bytes,
        }

    coord: dict = {"caches": {}}
    try:
        coord_stats = coord_cache_stats()
    except Exception:
        coord_stats = None
    if coord_stats is not None:
        for name, info in coord_stats.get("caches", {}).items():
            coord["caches"][name] = {
                "entries": info["entries"],
                "hits": info["hits"],
                "misses": info["misses"],
            }
        coord["store"] = None
        store_stats = coord_stats.get("store")
        if store_stats is not None:
            coord["store"] = {
                "hits": store_stats["hits"],
                "misses": store_stats["misses"],
                "publishes": store_stats["publishes"],
            }
    payload["coord"] = coord

    requests = {"enabled": False}
    if tile_request_tracker is not None:
        try:
            tracker_stats = tile_request_tracker.stats()
        except Exception:
            tracker_stats = None
        if tracker_stats is not None:
            requests = {
                "enabled": True,
                "total_requests": tracker_stats["total_requests"],
                "hot_tiles": tracker_stats["hot_tiles"],
                "fast_path_total": tracker_stats["fast_path"]["total"],
                "cache_hits": tracker_stats["cache"]["hits"],
                "cache_misses": tracker_stats["cache"]["misses"],
            }
    payload["requests"] = requests

    # Tile-latency accumulators, additive across workers: ns totals and
    # stage counts.  Old pulses that predate these fields are tolerated
    # by the aggregator via .get(..., 0).
    if tile_request_tracker is not None:
        try:
            lat = tile_request_tracker.latency_snapshot()
        except Exception:
            lat = None
        if lat is not None:
            payload["tile_latency"] = {
                "request_ns_total": lat["request_ns_total"],
                "request_count": lat["request_count"],
                "compute_ns_total": lat["compute_ns_total"],
                "compute_count": lat["compute_count"],
                "present_ns_total": lat["present_ns_total"],
                "present_count": lat["present_count"],
            }

    return payload


def _cluster_health_section() -> dict:
    """Aggregate the live worker pulses into the /health ``cluster`` block.

    Reads the pid-unique pulse files the worker pulse loops write under
    ``<cache_dir>/workers/`` (mtime-filtered, no locks — see
    ``librewxr.data.worker_pulse``), then unions in THIS process's live
    payload by pid so a worker reports even before its first pulse write
    lands on disk.  Per-process counters (RSS, tile-cache bytes, tracker
    counts, coord-cache hits) are summed across workers; the coord store
    ``entries``/``bytes`` describe the single global on-disk store and
    come from this worker's live stats instead.

    Every read here is a tiny file scan; the caller wraps this in
    try/except so a scan failure degrades the section to None rather
    than breaking /health.
    """
    cache_dir = (
        pathlib.Path(settings.cache_dir) if settings.cache_dir else None
    )
    pulses = read_worker_pulses(cache_dir) if cache_dir is not None else []
    by_pid: dict[int, dict] = {}
    for pulse in pulses:
        if isinstance(pulse, dict) and isinstance(pulse.get("pid"), int):
            by_pid[pulse["pid"]] = pulse
    # The live payload is strictly fresher than any on-disk file this
    # process left behind, so it wins the pid-keyed union.
    by_pid[os.getpid()] = collect_worker_pulse()
    pulses = list(by_pid.values())

    rss_values = [
        pulse["rss_bytes"] for pulse in pulses if pulse.get("rss_bytes")
    ]
    workers_rss_mb = {
        "sum": round(sum(rss_values) / (1024 * 1024), 1),
        "min": round(min(rss_values) / (1024 * 1024), 1),
        "max": round(max(rss_values) / (1024 * 1024), 1),
    }
    memory_block = {
        # cgroup split is only meaningful inside a container; None there.
        "container": (
            memory_monitor.cgroup_memory_mb if memory_monitor is not None else None
        ),
        "workers_rss_mb": workers_rss_mb,
    }

    tile_entries = sum(
        pulse["tile_cache"]["entries"] for pulse in pulses if pulse.get("tile_cache")
    )
    tile_bytes = sum(
        pulse["tile_cache"]["total_bytes"] for pulse in pulses if pulse.get("tile_cache")
    )
    tile_cache_block = {
        "entries": tile_entries,
        "used_mb": round(tile_bytes / (1024 * 1024), 1),
    }

    # Per-cache counters sum across workers; hit_ratio is recomputed from
    # the SUMS (a per-worker ratio averaged arithmetically would weight
    # idle workers as strongly as busy ones).
    cache_sums: dict[str, dict] = {}
    for pulse in pulses:
        for name, info in pulse.get("coord", {}).get("caches", {}).items():
            agg = cache_sums.setdefault(
                name, {"entries": 0, "hits": 0, "misses": 0},
            )
            agg["entries"] += info["entries"]
            agg["hits"] += info["hits"]
            agg["misses"] += info["misses"]
    for name, agg in cache_sums.items():
        total = agg["hits"] + agg["misses"]
        agg["hit_ratio"] = round(agg["hits"] / total, 3) if total else None
    coord_block = {"caches": cache_sums, "store": None}

    # Shared store: hits/misses/publishes are per-process counters and sum
    # across workers, but entries/bytes are a scan of the ONE global on-disk
    # store — every worker sees the same values, so summing would over-count.
    # They come from this worker's live stats instead.
    store_sums = {"hits": 0, "misses": 0, "publishes": 0}
    for pulse in pulses:
        store_stats = pulse.get("coord", {}).get("store")
        if store_stats:
            store_sums["hits"] += store_stats["hits"]
            store_sums["misses"] += store_stats["misses"]
            store_sums["publishes"] += store_stats["publishes"]
    try:
        live_store = coord_cache_stats().get("store")
    except Exception:
        live_store = None
    if live_store is not None:
        coord_block["store"] = {
            **store_sums,
            "entries": live_store["entries"],
            "bytes": live_store["bytes"],
        }

    # Tracked tile counts: hot_tiles is summed and can double-count a tile
    # that several workers all served — it's a cross-worker activity proxy,
    # not a distinct-tile count.  hit_rate is recomputed from the summed
    # hits/misses (mirroring the per-worker format: 0.0 when idle).
    requests_block = {
        "total_requests": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "fast_path_total": 0,
        "hot_tiles": 0,
    }
    for pulse in pulses:
        req = pulse.get("requests") or {}
        if not req.get("enabled"):
            continue
        for key in requests_block:
            requests_block[key] += req.get(key, 0)
    hits = requests_block["cache_hits"]
    misses = requests_block["cache_misses"]
    requests_block["hit_rate"] = (
        hits / (hits + misses) if (hits + misses) > 0 else 0.0
    )

    # Tile-latency sums: additive ns totals/counts across workers; the
    # cluster-wide averages are recomputed from the SUMS (mirroring the
    # hit_rate recomputation above — an arithmetic mean of per-worker
    # averages would weight idle workers as strongly as busy ones).
    # Pulses written before this field existed are tolerated via
    # .get(..., 0).
    lat_sums = {
        "request_ns_total": 0,
        "request_count": 0,
        "compute_ns_total": 0,
        "compute_count": 0,
        "present_ns_total": 0,
        "present_count": 0,
    }
    for pulse in pulses:
        lat = pulse.get("tile_latency") or {}
        for key in lat_sums:
            lat_sums[key] += lat.get(key, 0)
    tile_latency_block = {
        "avg_request_ms": _avg_ms(
            lat_sums["request_ns_total"], lat_sums["request_count"],
        ),
        "avg_compute_ms": _avg_ms(
            lat_sums["compute_ns_total"], lat_sums["compute_count"],
        ),
        "avg_present_ms": _avg_ms(
            lat_sums["present_ns_total"], lat_sums["present_count"],
        ),
    }

    return {
        "workers_reporting": len(pulses),
        "memory": memory_block,
        "tile_cache": tile_cache_block,
        "coord": coord_block,
        "requests": requests_block,
        "tile_latency": tile_latency_block,
    }


@router.get("/.well-known/ai-catalog.json")
async def ai_catalog() -> Response:
    """AI Catalog (proposal) entry pointing at the MCP server card.

    Self-description directory entry that resolves to the SEP-2127
    (draft) server card at ``<mcp_path>/server-card``.  Draft proposal,
    not a ratified standard.  404s when MCP is disabled by config or the
    HTTP transport failed to mount (``mcp_mounted`` False).  CORS is
    handled by the parent app's CORSMiddleware.
    """
    if not settings.mcp_enabled or not mcp_mounted:
        raise HTTPException(status_code=404, detail="MCP not available")
    return Response(
        content=json.dumps(build_ai_catalog()),
        media_type="application/ai-catalog+json",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/health")
async def health():
    """Health and status endpoint."""
    now = int(time.time())
    uptime = now - int(start_time)
    mem_limit_mb = detect_memory_limit_mb(settings.memory_limit_mb)
    rss_bytes = psutil.Process().memory_info().rss
    rss_mb = rss_bytes / (1024 * 1024)
    ram_usage = round(rss_mb / mem_limit_mb * 100, 1)
    frame_count = await frame_store.frame_count()
    timestamps = await frame_store.get_timestamps()
    latest_ts = max(timestamps) if timestamps else None
    oldest_ts = min(timestamps) if timestamps else None

    # Per-region frame counts catch silent regional failures: if OPERA
    # falls behind while USCOMP keeps fetching, the totals diverge here.
    region_keys = await frame_store.get_region_keys()
    per_region_counts: dict[str, int] = {}
    for names in region_keys.values():
        for name in names:
            per_region_counts[name] = per_region_counts.get(name, 0) + 1
    for name in (enabled_regions or []):
        per_region_counts.setdefault(name, 0)

    # Per-component memory breakdown.  Every NWP grid is iterated from
    # ``nwp_grids``; the per-slug byte counts are folded into both
    # ``tracked_bytes`` and the ``breakdown`` dict below so adding a new
    # NWP source requires no edits here.
    radar_bytes = frame_store.data_bytes
    tile_cache_bytes = tile_cache.total_bytes
    nwp_bytes_by_slug: dict[str, int] = {
        slug: (grid.data_bytes if grid is not None else 0)
        for slug, grid in nwp_grids.items()
    }
    nowcast_bytes = nowcast_store.data_bytes if nowcast_store else 0
    satellite_bytes = sum(
        grid.data_bytes
        for grid in satellite_grids.values()
        if grid is not None
    )
    coord_stats = coord_cache_stats()
    store_stats = coord_stats.get("store")
    if store_stats is not None:
        # Store-backed: entries are shared read-only memmap pages, not private
        # heap - report the on-disk footprint separately, contribute 0 to RSS
        # reconciliation.
        coord_bytes = 0
    else:
        coord_bytes = coord_cache_bytes()
    tracked_bytes = (
        radar_bytes + tile_cache_bytes + sum(nwp_bytes_by_slug.values())
        + nowcast_bytes + satellite_bytes + coord_bytes
    )
    other_bytes = max(0, rss_bytes - tracked_bytes)

    breakdown = {
        "radar_frames_mb": round(radar_bytes / (1024 * 1024), 1),
        "tile_cache_mb": round(tile_cache_bytes / (1024 * 1024), 1),
    }
    for slug, nbytes in nwp_bytes_by_slug.items():
        breakdown[f"{slug}_mb"] = round(nbytes / (1024 * 1024), 1)
    breakdown.update({
        "nowcast_mb": round(nowcast_bytes / (1024 * 1024), 1),
        "satellite_mb": round(satellite_bytes / (1024 * 1024), 1),
        "coord_caches_mb": round(coord_bytes / (1024 * 1024), 1),
        "coord_store_mb": (
            round(store_stats["bytes"] / (1024 * 1024), 1)
            if store_stats else 0.0
        ),
        "coord_store_entries": store_stats["entries"] if store_stats else 0,
        "other_mb": round(other_bytes / (1024 * 1024), 1),
    })

    # Split the tile cache into its four entry kinds: satellite render
    # entries (``"sat"``-prefixed keys), geometry entries (int timestamp +
    # 6-element viewport key), present render entries (int timestamp +
    # 9-element viewport/visual key), and overlay present entries (int
    # timestamp + 9-element viewport/visual key + 2-element style suffix,
    # nowcast frames only).  Each kind is reported with its own count and
    # byte total.
    cache_kind_geometry = 0
    cache_kind_geometry_bytes = 0
    cache_kind_present = 0
    cache_kind_present_bytes = 0
    cache_kind_overlay = 0
    cache_kind_overlay_bytes = 0
    cache_kind_satellite = 0
    cache_kind_satellite_bytes = 0
    for key, size in tile_cache.entries():
        if key[0] == "sat":
            cache_kind_satellite += 1
            cache_kind_satellite_bytes += size
        elif key and isinstance(key[0], int) and len(key) == 7:
            cache_kind_geometry += 1
            cache_kind_geometry_bytes += size
        elif key and isinstance(key[0], int) and len(key) == 10:
            cache_kind_present += 1
            cache_kind_present_bytes += size
        elif key and isinstance(key[0], int) and len(key) == 12:
            cache_kind_overlay += 1
            cache_kind_overlay_bytes += size

    # Cluster-wide aggregation: lock-free scan of the tiny per-worker pulse
    # files under the shared cache dir, unioned with this worker's live
    # payload.  Degrades to None on any failure — never an exception.
    try:
        cluster = _cluster_health_section()
    except Exception:
        logger.exception("Failed to assemble cluster health section")
        cluster = None

    return {
        "status": "ok" if frame_count > 0 else "degraded",
        "uptime_seconds": uptime,
        "cluster": cluster,
        "memory": {
            "resident_mb": round(rss_mb, 1),
            "limit_mb": round(mem_limit_mb, 1),
            "usage_pct": ram_usage,
            "breakdown": breakdown,
        },
        "frames": {
            "count": frame_count,
            "max": settings.max_frames,
            "latest": latest_ts,
            "oldest": oldest_ts,
            "latest_age_seconds": now - latest_ts if latest_ts else None,
            "per_region": per_region_counts,
        },
        "tile_cache": {
            "entries": tile_cache.size,
            "used_mb": round(tile_cache.total_bytes / (1024 * 1024), 1),
            "max_mb": settings.tile_cache_mb,
            "geometry_entries": cache_kind_geometry,
            "geometry_bytes": cache_kind_geometry_bytes,
            "present_entries": cache_kind_present,
            "present_bytes": cache_kind_present_bytes,
            "overlay_entries": cache_kind_overlay,
            "overlay_bytes": cache_kind_overlay_bytes,
            "satellite_entries": cache_kind_satellite,
            "satellite_bytes": cache_kind_satellite_bytes,
        },
        **_nwp_grid_health_blocks(),
        "nwp_chain": {
            "sources": [s.name for s in nwp_chain.sources] if nwp_chain else [],
        },
        "nowcast": {
            "enabled": settings.nowcast_enabled,
            "arrow_flow_enabled": settings.arrow_flow_enabled,
            "arrow_flow_target_dim": settings.arrow_flow_target_dim,
            "arrow_nwp_flow_resolution_deg": settings.arrow_nwp_flow_resolution_deg,
            "flows": len(await nowcast_store.get_flows() or {}) if nowcast_store else 0,
            "nwp_flow": await nowcast_store.get_nwp_flow() is not None if nowcast_store else False,
            "frames": await nowcast_store.get_timestamps() if nowcast_store else [],
            "count": len(await nowcast_store.get_timestamps()) if nowcast_store else 0,
        },
        "satellite": {
            "enabled": settings.satellite_enabled,
            "channels": {
                slug: {
                    "loaded": grid is not None and bool(grid.timestamps),
                    "frames": len(grid.timestamps) if grid is not None else 0,
                    "latest": (
                        grid.timestamps[-1]
                        if grid is not None and grid.timestamps
                        else None
                    ),
                }
                for slug, grid in satellite_grids.items()
            },
        },
        "enabled_regions": enabled_regions or [],
        "sources": {
            "na_source": settings.na_source,
            "ca_source": settings.ca_source,
            # CACOMP MSC blending state: True/False once observed,
            # None if blending isn't configured for this region set.
            "cacomp_msc_blending": (
                radar_fetcher._cacomp_msc_available
                if radar_fetcher is not None
                and radar_fetcher._cacomp_msc_source is not None
                else None
            ),
        },
        "radar_cache": (
            {"enabled": True, **radar_cache.stats()}
            if radar_cache is not None
            else {"enabled": False}
        ),
        "coord_caches": coord_cache_stats(),
        "tile_requests": (
            {"enabled": True, **tile_request_tracker.stats()}
            if tile_request_tracker is not None
            else {"enabled": False}
        ),
        "alerts": {
            "enabled": alerts_enabled,
            "count": alerts_store.count if alerts_store is not None else 0,
            "last_updated": int(alerts_store.last_updated) if alerts_store is not None else 0,
            "ingest_ok": alerts_store.fetch_success if alerts_store is not None else False,
        } if alerts_enabled else {"enabled": False},
        "mcp": {
            "enabled": settings.mcp_enabled,
            "mounted": mcp_mounted,
            "path": mcp_path,
            "tools": list(mcp_tools),
        } if settings.mcp_enabled else {"enabled": False},
        "storm_cells": {
            "enabled": settings.storm_cells_enabled,
            "count": storm_cell_store.total_count if storm_cell_store is not None else 0,
            "last_updated": int(storm_cell_store.last_updated) if storm_cell_store is not None else 0,
            "per_region": await storm_cell_store.get_counts() if storm_cell_store is not None else {},
        } if settings.storm_cells_enabled else {"enabled": False},
    }


def _content_type(ext: str) -> str:
    return "image/webp" if ext == "webp" else "image/png"


@router.get("/public/weather-maps.json")
async def weather_maps() -> WeatherMapsResponse:
    """Rain Viewer-compatible metadata endpoint."""
    timestamps = await frame_store.get_timestamps()
    host = settings.public_url.rstrip("/")

    past = [
        RadarTimestamp(time=ts, path=f"/v2/radar/{ts}")
        for ts in sorted(timestamps)
    ]

    nowcast = []
    if nowcast_store is not None:
        nc_timestamps = await nowcast_store.get_timestamps()
        nowcast = [
            RadarTimestamp(time=ts, path=f"/v2/radar/{ts}")
            for ts in nc_timestamps
        ]

    infrared = []
    # Catalog timestamps come from GMGSI LW since LW is the always-on
    # 24/7 baseline (VIS only carries the daytime half of the day).
    # When the satellite layer is disabled or unloaded the array is
    # empty and the tile endpoint returns 503.
    gmgsi_lw = satellite_grids.get("gmgsi_lw_grid") if satellite_grids else None
    if gmgsi_lw is not None and gmgsi_lw.timestamps:
        infrared = [
            RadarTimestamp(time=ts, path=f"/v2/satellite/{ts}")
            for ts in gmgsi_lw.timestamps
        ]

    color_schemes = [
        ColorScheme(id=sid, name=name)
        for sid, name in SCHEME_NAMES.items()
    ]

    return WeatherMapsResponse(
        version="2.0",
        generated=int(time.time()),
        host=host,
        radar=RadarData(past=past, nowcast=nowcast, colorSchemes=color_schemes),
        satellite=SatelliteData(infrared=infrared),
    )


async def _latest_timestamps_cached() -> list[int]:
    """Latest radar timestamps with a 5 s TTL.

    The tile hot path only needs the latest frame to pick the Cache-Control
    ``max_age`` bucket, so re-querying the store lock on every request is
    wasted contention.  Degrades to ``[]`` when the store isn't wired
    (``frame_store`` None -> no frames -> ``latest_ts`` None -> 300 s
    bucket), matching the single-mode no-store behaviour.
    """
    global _latest_ts_cache
    now = time.monotonic()
    if _latest_ts_cache is not None and now - _latest_ts_cache[0] < _LATEST_TS_TTL:
        return _latest_ts_cache[1]
    if frame_store is None:
        _latest_ts_cache = (now, [])
        return []
    timestamps = await frame_store.get_timestamps()
    _latest_ts_cache = (now, timestamps)
    return timestamps


def _present_and_hash(geom, **kwargs) -> tuple[bytes, str]:
    """Run ``present_tile`` and hash the result into an ETag.

    Kept together so both run off the event loop: the SHA-256 of a tile
    is a per-request cost that would otherwise stall every miss on the
    loop.
    """
    tile_bytes = present_tile(geom, **kwargs)
    return tile_bytes, compute_etag(tile_bytes)


def _shared_tile_key(timestamp, version, z, x, y, tile_size, smooth, snow, color, ext) -> str:
    """Shared-store key for an encoded radar tile.

    Folds every input that determines the encoded bytes (including the
    frame's content version) so a merge/eviction or config change re-keys
    the tile instead of serving stale bytes.
    """
    return (
        f"{timestamp}-v{version}-{z}-{x}-{y}-{tile_size}-"
        f"{int(smooth)}{int(snow)}-{color}-{ext}-q{settings.webp_quality}"
    )


def _shared_get_and_hash(store, key: str):
    """Read a shared tile and hash it off the event loop; None on miss."""
    data = store.get(key)
    if data is None:
        return None
    return data, compute_etag(data)


async def _present_tile_async(geom, **kwargs) -> tuple[bytes, str]:
    """Run ``present_tile`` + ETag hash off the event loop.

    Multi-mode render workers get a dedicated present pool
    (``routes.present_executor``) so cheap colorize/encode jobs never queue
    behind long geometry computes on the shared default executor.  Single
    mode leaves ``present_executor`` None and falls back to
    ``asyncio.to_thread`` - byte-identical to the pre-split behaviour.
    """
    if present_executor is not None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            present_executor,
            functools.partial(_present_and_hash, geom, **kwargs),
        )
    return await asyncio.to_thread(_present_and_hash, geom, **kwargs)


@router.get("/v2/radar/{timestamp}/{size}/{z}/{x}/{y}/{color}/{smooth_snow}.{ext}")
async def radar_tile(
    request: Request,
    timestamp: int,
    size: int = Path(ge=256, le=512),
    z: int = Path(ge=0),
    x: int = Path(ge=0),
    y: int = Path(ge=0),
    color: int = Path(ge=0, le=255),
    smooth_snow: str = Path(pattern=r"^\d+_\d+$"),
    ext: str = Path(pattern=r"^(png|webp)$"),
    arrows: str = Query(default=""),
    cells: str = Query(default=""),
) -> Response:
    """Rain Viewer-compatible tile endpoint."""
    t0 = time.perf_counter_ns()
    logger.debug("Tile request: z=%d x=%d y=%d color=%d smooth_snow=%s ext=%s", z, x, y, color, smooth_snow, ext)
    if z > settings.max_zoom:
        raise HTTPException(status_code=400, detail=f"Zoom {z} exceeds max {settings.max_zoom}")

    max_tiles = 2**z
    if x >= max_tiles or y >= max_tiles:
        raise HTTPException(status_code=400, detail="Tile coordinates out of range")

    parts = smooth_snow.split("_")
    smooth = parts[0] == "1"
    snow = parts[1] == "1" if len(parts) > 1 else False

    tile_size = 512 if size >= 512 else 256

    arrow_style = ""
    if arrows in ("1", "true", "light"):
        arrow_style = "light"
    elif arrows == "dark":
        arrow_style = "dark"

    cell_style = ""
    if cells in ("1", "true", "light"):
        cell_style = "light"
    elif cells == "dark":
        cell_style = "dark"

    # Plain tile: no overlays requested.  Computed here (before the frame
    # fetch) because the shared-store lookup only serves plain tiles and
    # needs the flag for its guard.
    is_plain = not arrows and not cells

    # Geometry cache: keyed only on inputs that affect the sampled values
    # (radar source + viewport + smoothing + snow-mask presence).  Color
    # scheme, output format, and arrow style apply per-request in
    # ``present_tile`` so a single cached entry serves every visual
    # variant of the same viewport.
    geom_key = (timestamp, z, x, y, tile_size, smooth, snow)
    geom = tile_cache.get(geom_key)

    # Present-stage cache key: one entry per visual variant of the same
    # geometry.  Stores the encoded bytes plus the ETag so a present
    # cache hit skips both ``present_tile`` and the ETag hash.  Hoisted
    # here (before the shared-store lookup) so the early-serve fast path
    # below can use it without recomputing the tuple.
    present_key = (
        timestamp, z, x, y, tile_size, smooth, snow,
        color, ext, settings.webp_quality,
    )

    # Geometry-stage cache outcome: this is the meaningful hit/miss for
    # "is the fast path helping".  Batched with the (z, x, y) counter so
    # one lock acquisition covers both.
    if tile_request_tracker is not None:
        tile_request_tracker.record_request(z, x, y, cache_hit=geom is not None)

    # Early-serve fast path: a plain request whose geometry is already in
    # memory and whose encoded bytes are present-cached skips the shared-
    # store lookup, frame fetch, geometry compute, and present entirely.
    # The ``geom is not None`` gate is required: it keeps the nowcast-
    # timestamp edge case on the existing path (with a geom hit,
    # ``need_frame`` is False and the warmer hook below resolves
    # ``frame_type`` via ``_latest_timestamps_cached``).
    present_cache_hit = False
    if is_plain and geom is not None:
        cached = tile_cache.get(present_key)
        if isinstance(cached, CachedRender):
            tile_bytes = cached.data
            etag = cached.etag
            present_cache_hit = True

    # ``need_frame``/``is_nowcast`` live above the branch because the
    # warmer hook below runs on every path; on a shared hit (past frames
    # only) ``is_nowcast`` stays False, which is exactly what a plain
    # cached-hit request resolves to.
    is_nowcast = False
    need_frame = geom is None or bool(arrow_style) or bool(cell_style)
    compute_ns = None
    present_ns = None
    if not present_cache_hit:
        # Shared-store lookup: plain past-frame tiles only.  A hit here still
        # counts as a geometry miss in ``record_request`` above (accepted -
        # it avoided the compute, not the lookup).  The key folds the frame's
        # content version so a merge/eviction re-keys the tile.
        shared_hit = None
        if shared_tile_store is not None and is_plain and frame_store is not None:
            version = frame_store.frame_version(timestamp)
            if version is not None:  # past frames only; nowcast ts has no version
                shared_key = _shared_tile_key(timestamp, version, z, x, y, tile_size, smooth, snow, color, ext)
                if io_executor is not None:
                    loop = asyncio.get_running_loop()
                    shared_hit = await loop.run_in_executor(
                        io_executor, _shared_get_and_hash, shared_tile_store, shared_key,
                    )
                else:
                    shared_hit = await asyncio.to_thread(_shared_get_and_hash, shared_tile_store, shared_key)

        if shared_hit is not None:
            # Shared hit: the published bytes (and ETag) are byte-identical to
            # a fresh render, so skip frame fetch, geometry compute, overlays,
            # and present entirely.  Prime the in-memory present cache so
            # same-worker repeats hit RAM instead of the shared volume.  The
            # warmer hook below stays reachable from every path; in practice it
            # never fires here because the shared store is only wired in multi
            # mode, where ``tile_warmer`` is None.
            tile_bytes, etag = shared_hit
            tile_cache.put(present_key, CachedRender(data=tile_bytes, etag=etag))
        else:
            # We need the radar frame whenever geometry must be computed AND
            # whenever an overlay is requested: arrows need live frame data +
            # flow fields, and cells need ``frame.regions`` to decide which
            # regions actually carry data on this tile (without it
            # ``_draw_storm_cells`` sees an empty region list and draws
            # nothing).  Skip the fetch on pure cache hits without overlays -
            # that's the hot path Merry Sky-style clients exercise.
            frame = None
            nowcast_blend = None
            if need_frame:
                frame = await frame_store.get_frame(timestamp)
                if frame is None and nowcast_store is not None:
                    nc_frame, nowcast_blend = await nowcast_store.get_frame(timestamp)
                    if nc_frame is not None:
                        frame = nc_frame
                        is_nowcast = True
                if frame is None:
                    raise HTTPException(status_code=404, detail="Frame not found")

            if geom is None:
                compute_start = time.perf_counter_ns()
                geom = await asyncio.to_thread(
                    compute_tile_geometry,
                    frame_regions=frame.regions,
                    z=z, x=x, y=y,
                    tile_size=tile_size,
                    smooth=smooth,
                    snow=snow,
                    nwp_chain=nwp_chain,
                    enabled_regions=enabled_regions,
                    frame_timestamp=timestamp,
                    nowcast_blend=nowcast_blend,
                    precip_mask=precip_mask,
                )
                compute_ns = time.perf_counter_ns() - compute_start
                tile_cache.put(geom_key, geom)
                # Only fire on the cold-compute path: a fast-path label here means
                # this request actually paid for the empty-tile work (cache hits
                # of a previously-computed transparent geometry are already counted
                # by ``record_request`` above, not a fast-path firing now).
                if tile_request_tracker is not None and geom.fast_path is not None:
                    tile_request_tracker.record_fast_path(geom.fast_path)

            flow_regions = None
            nwp_flow = None
            if arrow_style:
                if nowcast_store is not None:
                    flow_regions = await nowcast_store.get_flows() or None
                    nwp_flow = await nowcast_store.get_nwp_flow()

            cells_by_region = None
            cell_counts = None
            if cell_style and storm_cell_store is not None:
                # Only show cells on the frame the detection actually ran on --
                # showing current-detected cells on past or nowcast frames is
                # misleading (the cells represent "what storms are detected RIGHT
                # NOW", not historical positions).
                if timestamp == storm_cell_store.detected_at_timestamp:
                    cells_by_region = await storm_cell_store.get_cells() or None
                    cell_counts = await storm_cell_store.get_counts() or None

            # Effective overlay styles as actually passed to ``present_tile``:
            # an arrows/cells request degrades to plain when no flow or cell
            # data is available for this request.
            eff_arrow = arrow_style if (flow_regions or nwp_flow is not None) else ""
            eff_cells = cell_style if cells_by_region else ""

            if is_plain or not (eff_arrow or eff_cells):
                # An overlay request with no flow/cell data available also lands
                # here - it falls through to the exact plain present path (same
                # present_key, same cache entry) rather than creating a duplicate.
                cached = tile_cache.get(present_key)
                if isinstance(cached, CachedRender):
                    tile_bytes = cached.data
                    etag = cached.etag
                else:
                    present_start = time.perf_counter_ns()
                    tile_bytes, etag = await _present_tile_async(
                        geom,
                        color_scheme=color,
                        fmt=ext,
                        arrow_style=eff_arrow,
                        flow_regions=flow_regions,
                        frame_regions=frame.regions if frame is not None else None,
                        enabled_regions=enabled_regions,
                        nwp_flow=nwp_flow,
                        nwp_chain=nwp_chain,
                        frame_timestamp=timestamp,
                        z=z, x=x, y=y,
                        cell_style=eff_cells,
                        cells_by_region=cells_by_region,
                        cell_counts=cell_counts,
                    )
                    present_ns = time.perf_counter_ns() - present_start
                    tile_cache.put(present_key, CachedRender(data=tile_bytes, etag=etag))
                    # Publish the fresh encode to the shared store for the other
                    # workers, fire-and-forget so the response never waits on the
                    # shared-volume write; the set holds references so tasks
                    # can't be GC'd mid-flight.  Never fires for nowcast tiles
                    # (their timestamp has no frame version).  A version bump
                    # between lookup and publish writes a stale entry, but the
                    # render worker's poller detects the bump within one poll
                    # interval (~1 s) and ``invalidate_timestamp`` sweeps every
                    # key for that timestamp, so the stale window is bounded by
                    # the poll cadence (pruning covers orphaned entries after
                    # eviction).
                    if shared_tile_store is not None and frame_store is not None:
                        version = frame_store.frame_version(timestamp)
                        if version is not None:
                            key = _shared_tile_key(timestamp, version, z, x, y, tile_size, smooth, snow, color, ext)
                            if io_executor is not None:
                                loop = asyncio.get_running_loop()
                                task = asyncio.ensure_future(
                                    loop.run_in_executor(io_executor, shared_tile_store.publish, key, tile_bytes)
                                )
                            else:
                                task = asyncio.ensure_future(
                                    asyncio.to_thread(shared_tile_store.publish, key, tile_bytes)
                                )
                            _pending_shared_publishes.add(task)
                            task.add_done_callback(_pending_shared_publishes.discard)
            else:
                # Overlay present cache (nowcast frames only): the render worker's
                # state poller invalidates every nowcast timestamp each cycle, so
                # a cached overlay entry is never served with flows older than one
                # fetch cycle.  Past frames are deliberately NOT cached here -
                # their timestamp survives for 2 hours while flows regenerate
                # every cycle, which would pin arrows/cells to stale flow fields;
                # those requests re-render the cheap present tail per request (the
                # pre-change behaviour).
                overlay_key = present_key + (eff_arrow, eff_cells)
                cached = tile_cache.get(overlay_key) if is_nowcast else None
                if isinstance(cached, CachedRender):
                    tile_bytes = cached.data
                    etag = cached.etag
                else:
                    present_start = time.perf_counter_ns()
                    tile_bytes, etag = await _present_tile_async(
                        geom,
                        color_scheme=color,
                        fmt=ext,
                        arrow_style=eff_arrow,
                        flow_regions=flow_regions,
                        frame_regions=frame.regions if frame is not None else None,
                        enabled_regions=enabled_regions,
                        nwp_flow=nwp_flow,
                        nwp_chain=nwp_chain,
                        frame_timestamp=timestamp,
                        z=z, x=x, y=y,
                        cell_style=eff_cells,
                        cells_by_region=cells_by_region,
                        cell_counts=cell_counts,
                    )
                    present_ns = time.perf_counter_ns() - present_start
                    if is_nowcast:
                        tile_cache.put(overlay_key, CachedRender(data=tile_bytes, etag=etag))

    if tile_warmer is not None:
        # When the cache hit short-circuited the frame fetch, we still
        # need a frame_type for the warmer.  Cheap lookup against the
        # in-memory timestamp list.
        if not need_frame:
            past_timestamps = await _latest_timestamps_cached()
            is_nowcast = timestamp not in past_timestamps
        asyncio.ensure_future(
            tile_warmer.warm(
                triggered_timestamp=timestamp,
                z=z, x=x, y=y,
                tile_size=tile_size,
                smooth=smooth,
                snow=snow,
                frame_type="nowcast" if is_nowcast else "past",
            )
        )

    # Historical frames are immutable once backfill is complete — cache them
    # for their full 2-hour lifetime.  Latest and nowcast frames still evolve.
    timestamps = await _latest_timestamps_cached()
    latest_ts = max(timestamps) if timestamps else None
    max_age = 7200 if (latest_ts is not None and timestamp < latest_ts) else 300

    # Request latency: the request is always counted; compute/present only
    # when that stage actually ran (None on cache hits).
    if tile_request_tracker is not None:
        tile_request_tracker.record_latency(
            time.perf_counter_ns() - t0, compute_ns, present_ns,
        )

    return conditional_response(
        request=request,
        body=tile_bytes,
        etag=etag,
        content_type=_content_type(ext),
        max_age=max_age,
    )


@router.get("/v2/coverage/0/{size}/{z}/{x}/{y}/0/0_0.png")
async def coverage_tile(
    request: Request,
    size: int = Path(ge=256, le=512),
    z: int = Path(ge=0),
    x: int = Path(ge=0),
    y: int = Path(ge=0),
) -> Response:
    """Coverage tile showing where radar data exists."""
    if z > settings.max_zoom:
        raise HTTPException(status_code=400, detail=f"Zoom {z} exceeds max {settings.max_zoom}")

    max_tiles = 2**z
    if x >= max_tiles or y >= max_tiles:
        raise HTTPException(status_code=400, detail="Tile coordinates out of range")

    tile_size = 512 if size >= 512 else 256

    frame = await frame_store.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No radar data available")

    # Cache the encoded coverage tile bytes keyed on the content that
    # determines them: the latest frame's timestamp + viewport.  The
    # endpoint is always PNG (fixed URL), so the format is constant and
    # needs no key element.  The "cov" namespace prefix mirrors the
    # satellite endpoint's "sat" keys - it can't collide with the radar
    # geometry/present keys (which start with the int timestamp), and it
    # deliberately sits outside ``invalidate_timestamp`` (which sweeps
    # by ``key[0] == timestamp``): coverage tracks only the latest frame,
    # so a new frame re-keys the entries and the old ones age out through
    # the LRU, same as satellite.  ``enabled_regions`` is fixed at
    # startup, so it's not in the key (the radar geometry path treats it
    # the same way).
    cache_key = ("cov", frame.timestamp, z, x, y, tile_size)
    cached = tile_cache.get(cache_key)
    if cached is not None:
        if isinstance(cached, CachedRender):
            tile_bytes = cached.data
            etag = cached.etag
        else:
            # Legacy raw-bytes entry (pre-ETag cache format).
            tile_bytes = cached
            etag = compute_etag(cached)
        return conditional_response(
            request=request,
            body=tile_bytes,
            etag=etag,
            content_type="image/png",
            max_age=300,
        )

    tile_bytes = await asyncio.to_thread(
        render_coverage_tile,
        frame_regions=frame.regions,
        z=z, x=x, y=y,
        tile_size=tile_size,
        enabled_regions=enabled_regions,
    )

    etag = compute_etag(tile_bytes)
    tile_cache.put(cache_key, CachedRender(data=tile_bytes, etag=etag))

    return conditional_response(
        request=request,
        body=tile_bytes,
        etag=etag,
        content_type="image/png",
        max_age=300,
    )


@router.get("/v2/satellite/{timestamp}/{size}/{z}/{x}/{y}/0/0_0.{ext}")
async def satellite_tile(
    request: Request,
    timestamp: int,
    size: int = Path(ge=256, le=512),
    z: int = Path(ge=0),
    x: int = Path(ge=0),
    y: int = Path(ge=0),
    ext: str = Path(pattern=r"^(png|webp)$"),
) -> Response:
    """Real satellite imagery tile, backed by NOAA GMGSI.

    Backing renderer is picked per request: the VIS-over-LW composite
    when both channels have ingested frames (the production path during
    Phase 2+), or the stand-alone LW renderer when only longwave IR is
    loaded.  When the satellite layer is disabled or neither channel has
    any frames yet, returns 503.
    """
    if z > settings.max_zoom:
        raise HTTPException(status_code=400, detail=f"Zoom {z} exceeds max {settings.max_zoom}")

    max_tiles = 2**z
    if x >= max_tiles or y >= max_tiles:
        raise HTTPException(status_code=400, detail="Tile coordinates out of range")

    tile_size = 512 if size >= 512 else 256

    # Backing selection: composite when both channels loaded, LW-only
    # otherwise, 503 if nothing's ready.
    gmgsi_lw = satellite_grids.get("gmgsi_lw_grid") if satellite_grids else None
    gmgsi_vis = satellite_grids.get("gmgsi_vis_grid") if satellite_grids else None
    has_lw = gmgsi_lw is not None and bool(gmgsi_lw.timestamps)
    has_vis = gmgsi_vis is not None and bool(gmgsi_vis.timestamps)

    if has_lw and has_vis:
        backing = "gmgsi_composite"
    elif has_lw:
        backing = "gmgsi_lw"
    else:
        raise HTTPException(status_code=503, detail="Satellite data not available")

    # Older-than-latest frames are immutable; give them a long max-age.
    # Computed before the lookup so cache hits get the same semantics.
    sat_timestamps = gmgsi_lw.timestamps
    latest_sat_ts = max(sat_timestamps) if sat_timestamps else None
    max_age = 7200 if (latest_sat_ts is not None and timestamp < latest_sat_ts) else 300

    # Distinct cache keys per backing so a runtime swap (e.g. VIS ingest
    # catching up after restart) doesn't serve stale composites.
    cache_key = ("sat", backing, timestamp, z, x, y, tile_size, ext)
    cached = tile_cache.get(cache_key)
    if cached is not None:
        if isinstance(cached, CachedRender):
            tile_bytes = cached.data
            etag = cached.etag
        else:
            # Legacy raw-bytes entry (pre-ETag cache format).
            tile_bytes = cached
            etag = compute_etag(cached)
        return conditional_response(
            request=request,
            body=tile_bytes,
            etag=etag,
            content_type=_content_type(ext),
            max_age=max_age,
        )

    if backing == "gmgsi_composite":
        tile_bytes = await asyncio.to_thread(
            render_gmgsi_composite_tile,
            lw_source=gmgsi_lw,
            vis_source=gmgsi_vis,
            z=z, x=x, y=y,
            tile_size=tile_size,
            timestamp=timestamp,
            fmt=ext,
        )
    else:
        tile_bytes = await asyncio.to_thread(
            render_gmgsi_tile,
            source=gmgsi_lw,
            z=z, x=x, y=y,
            tile_size=tile_size,
            timestamp=timestamp,
            fmt=ext,
        )
    etag = compute_etag(tile_bytes)

    tile_cache.put(cache_key, CachedRender(data=tile_bytes, etag=etag))

    return conditional_response(
        request=request,
        body=tile_bytes,
        etag=etag,
        content_type=_content_type(ext),
        max_age=max_age,
    )


# ---------------------------------------------------------------------------
# Alert helpers
# ---------------------------------------------------------------------------

def _parse_cap_time(value: str) -> int | None:
    """Parse CAP ISO 8601 time string to Unix epoch."""
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(value).timestamp())
    except (ValueError, TypeError):
        return None


def _alert_not_expired(alert, now_utc: int) -> bool:
    """Check if alert has not expired. Returns True for alerts without expires field."""
    expires = _parse_cap_time(alert.expires)
    return expires is None or expires > now_utc


@router.get("/v2/alerts", response_model=AlertsResponse)
async def get_alerts(
    lat: float | None = Query(None, ge=-90, le=90, description="Latitude for point lookup"),
    lon: float | None = Query(None, ge=-180, le=180, description="Longitude for point lookup"),
    bbox: str | None = Query(None, description="Bounding box: west,south,east,north"),
    simplify: float = Query(1000.0, ge=0, description="Polygon simplification tolerance in meters (0=off)"),
):
    """Weather alerts as GeoJSON FeatureCollection.

    - No params: all active alerts worldwide.
    - lat+lon: alerts containing that point.  Zone-based alerts (e.g.
      Tornado Watches) are resolved to zone polygons at ingest, so every
      alert is visible in point lookups without any per-request NWS query.
    - bbox: alerts intersecting the bounding box (polygon-only).
    """
    if not alerts_enabled or alerts_store is None:
        raise HTTPException(status_code=503, detail="Alerts not available")

    alerts = alerts_store.alerts

    # Filter by point
    if lat is not None and lon is not None:
        from shapely.geometry import Point
        point = Point(lon, lat)
        alerts = [a for a in alerts if a.polygon is not None and a.polygon.intersects(point)]
    # Filter by bbox
    elif bbox is not None:
        parts = bbox.split(",")
        if len(parts) != 4:
            raise HTTPException(status_code=400, detail="bbox must be: west,south,east,north")
        try:
            w, s, e, n = map(float, parts)
        except ValueError:
            raise HTTPException(status_code=400, detail="bbox values must be numeric")
        if w < -180 or e > 180 or s < -90 or n > 90 or w > e or s > n:
            raise HTTPException(status_code=400, detail="bbox values out of range")
        from shapely.geometry import box
        bbox_poly = box(w, s, e, n)
        alerts = [a for a in alerts if a.polygon is not None and a.polygon.intersects(bbox_poly)]

    # Expiry filter
    now_utc = int(time.time())
    alerts = [a for a in alerts if _alert_not_expired(a, now_utc)]

    # Build GeoJSON features from WMO alerts
    deg_per_meter = simplify / 111_000.0 if simplify > 0 else 0.0
    from shapely.geometry import mapping
    features: list[GeoJSONFeature] = []
    seen_uris: set[str] = set()

    for alert in alerts:
        geom = alert.polygon
        if deg_per_meter > 0 and geom is not None:
            geom = geom.simplify(deg_per_meter, preserve_topology=True)

        uri = alert.url
        if uri in seen_uris:
            continue
        seen_uris.add(uri)

        features.append(
            GeoJSONFeature(
                type="Feature",
                properties=AlertProperties(
                    title=alert.event,
                    severity=alert.severity,
                    time=_parse_cap_time(alert.effective),
                    expires=_parse_cap_time(alert.expires),
                    description=alert.description,
                    regions=[alert.area_desc] if alert.area_desc else [],
                    uri=uri,
                ),
                geometry=mapping(geom) if geom is not None else None,
            )
        )

    return AlertsResponse(type="FeatureCollection", features=features)
