# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import logging
import time
import psutil

from fastapi import APIRouter, HTTPException, Path, Query, Response

from librewxr.api.models import (
    ColorScheme,
    RadarData,
    RadarTimestamp,
    SatelliteData,
    WeatherMapsResponse,
)
from librewxr.colors.schemes import SCHEME_NAMES
from librewxr.config import settings
from librewxr.data.store import FrameStore
from librewxr.memory import detect_memory_limit_mb
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import coord_cache_bytes, coord_cache_stats
from librewxr.tiles.renderer import render_coverage_tile, render_tile
from librewxr.tiles.request_tracker import TileRequestTracker
from librewxr.tiles.satellite_renderer import render_satellite_tile

logger = logging.getLogger(__name__)

router = APIRouter()

# These get set by main.py during startup
frame_store: FrameStore | None = None
tile_cache: TileCache | None = None
ecmwf_grid = None  # ECMWFGrid | None
hrrr_grid = None  # HRRRGrid | None
icon_eu_grid = None  # ICONEUGrid | None
nwp_chain = None  # NWPChain | None
cloud_grid = None  # CloudGrid | None
tile_warmer = None  # TileWarmer | None
nowcast_store = None  # NowcastStore | None
radar_cache = None  # RadarFrameCache | None
radar_fetcher = None  # RadarFetcher | None
tile_request_tracker: TileRequestTracker | None = None
start_time: float = 0.0
enabled_regions: list[str] | None = None


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

    # Per-component memory breakdown
    radar_bytes = frame_store.data_bytes
    tile_cache_bytes = tile_cache.total_bytes
    ecmwf_bytes = ecmwf_grid.data_bytes if ecmwf_grid else 0
    hrrr_bytes = hrrr_grid.data_bytes if hrrr_grid else 0
    icon_eu_bytes = icon_eu_grid.data_bytes if icon_eu_grid else 0
    nowcast_bytes = nowcast_store.data_bytes if nowcast_store else 0
    satellite_bytes = cloud_grid.data_bytes if cloud_grid else 0
    coord_bytes = coord_cache_bytes()
    tracked_bytes = (
        radar_bytes + tile_cache_bytes + ecmwf_bytes + hrrr_bytes
        + icon_eu_bytes + nowcast_bytes + satellite_bytes + coord_bytes
    )
    other_bytes = max(0, rss_bytes - tracked_bytes)

    return {
        "status": "ok" if frame_count > 0 else "degraded",
        "uptime_seconds": uptime,
        "memory": {
            "resident_mb": round(rss_mb, 1),
            "limit_mb": round(mem_limit_mb, 1),
            "usage_pct": ram_usage,
            "breakdown": {
                "radar_frames_mb": round(radar_bytes / (1024 * 1024), 1),
                "tile_cache_mb": round(tile_cache_bytes / (1024 * 1024), 1),
                "ecmwf_grid_mb": round(ecmwf_bytes / (1024 * 1024), 1),
                "hrrr_grid_mb": round(hrrr_bytes / (1024 * 1024), 1),
                "icon_eu_grid_mb": round(icon_eu_bytes / (1024 * 1024), 1),
                "nowcast_mb": round(nowcast_bytes / (1024 * 1024), 1),
                "satellite_mb": round(satellite_bytes / (1024 * 1024), 1),
                "coord_caches_mb": round(coord_bytes / (1024 * 1024), 1),
                "other_mb": round(other_bytes / (1024 * 1024), 1),
            },
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
        },
        "ecmwf_grid": {
            "loaded": ecmwf_grid is not None and ecmwf_grid.data is not None,
            "reference_time": ecmwf_grid.reference_time if ecmwf_grid else None,
            "timesteps": ecmwf_grid.timestep_count if ecmwf_grid else 0,
        },
        "hrrr_grid": {
            "enabled": hrrr_grid is not None,
            "loaded": hrrr_grid is not None and hrrr_grid.has_data(),
            "latest_run": hrrr_grid.latest_run_iso if hrrr_grid else None,
            "frames": hrrr_grid.frame_count if hrrr_grid else 0,
        },
        "icon_eu_grid": {
            "enabled": icon_eu_grid is not None,
            "loaded": icon_eu_grid is not None and icon_eu_grid.has_data(),
            "latest_run": icon_eu_grid.latest_run_iso if icon_eu_grid else None,
            "frames": icon_eu_grid.frame_count if icon_eu_grid else 0,
        },
        "nwp_chain": {
            "sources": [s.name for s in nwp_chain.sources] if nwp_chain else [],
        },
        "nowcast": {
            "enabled": settings.nowcast_enabled,
            "frames": await nowcast_store.get_timestamps() if nowcast_store else [],
            "count": len(await nowcast_store.get_timestamps()) if nowcast_store else 0,
        },
        "satellite": {
            "enabled": settings.satellite_enabled,
            "loaded": cloud_grid is not None and cloud_grid.loaded,
            "reference_time": cloud_grid.reference_time if cloud_grid else None,
            "timesteps": cloud_grid.timestep_count if cloud_grid else 0,
        },
        "enabled_regions": enabled_regions or [],
        "sources": {
            "na_source": settings.na_source,
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
    if cloud_grid is not None and cloud_grid.loaded:
        infrared = [
            RadarTimestamp(time=ts, path=f"/v2/satellite/{ts}")
            for ts in cloud_grid.timestamps
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


@router.get("/v2/radar/{timestamp}/{size}/{z}/{x}/{y}/{color}/{smooth_snow}.{ext}")
async def radar_tile(
    timestamp: int,
    size: int = Path(ge=256, le=512),
    z: int = Path(ge=0),
    x: int = Path(ge=0),
    y: int = Path(ge=0),
    color: int = Path(ge=0, le=255),
    smooth_snow: str = Path(pattern=r"^\d+_\d+$"),
    ext: str = Path(pattern=r"^(png|webp)$"),
    arrows: str = Query(default=""),
) -> Response:
    """Rain Viewer-compatible tile endpoint."""
    logger.debug("Tile request: z=%d x=%d y=%d color=%d smooth_snow=%s ext=%s", z, x, y, color, smooth_snow, ext)
    if z > settings.max_zoom:
        raise HTTPException(status_code=400, detail=f"Zoom {z} exceeds max {settings.max_zoom}")

    max_tiles = 2**z
    if x >= max_tiles or y >= max_tiles:
        raise HTTPException(status_code=400, detail="Tile coordinates out of range")

    if tile_request_tracker is not None:
        tile_request_tracker.record(z, x, y)

    parts = smooth_snow.split("_")
    smooth = parts[0] == "1"
    snow = parts[1] == "1" if len(parts) > 1 else False

    tile_size = 512 if size >= 512 else 256

    arrow_style = ""
    if arrows in ("1", "true", "light"):
        arrow_style = "light"
    elif arrows == "dark":
        arrow_style = "dark"

    cache_key = (timestamp, z, x, y, tile_size, color, smooth, snow, ext, arrow_style)
    cached = tile_cache.get(cache_key)
    if cached is not None:
        return Response(
            content=cached,
            media_type=_content_type(ext),
            headers={"Cache-Control": "public, max-age=300"},
        )

    frame = await frame_store.get_frame(timestamp)
    nowcast_blend = None
    if frame is None and nowcast_store is not None:
        nc_frame, nowcast_blend = await nowcast_store.get_frame(timestamp)
        if nc_frame is not None:
            frame = nc_frame
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found")

    flow_regions = None
    ecmwf_flow = None
    if arrow_style:
        if nowcast_store is not None:
            flow_regions = await nowcast_store.get_flows() or None
        if ecmwf_grid is not None and ecmwf_grid.flow is not None:
            ecmwf_flow = ecmwf_grid.flow

    tile_bytes = await asyncio.to_thread(
        render_tile,
        frame_regions=frame.regions,
        z=z, x=x, y=y,
        tile_size=tile_size,
        color_scheme=color,
        smooth=smooth,
        snow=snow,
        fmt=ext,
        ecmwf_grid=ecmwf_grid,
        nwp_chain=nwp_chain,
        enabled_regions=enabled_regions,
        frame_timestamp=timestamp,
        nowcast_blend=nowcast_blend,
        flow_regions=flow_regions,
        ecmwf_flow=ecmwf_flow,
        arrow_style=arrow_style,
    )

    tile_cache.put(cache_key, tile_bytes)

    if tile_warmer is not None:
        asyncio.ensure_future(
            tile_warmer.warm(
                triggered_timestamp=timestamp,
                z=z, x=x, y=y,
                tile_size=tile_size,
                color=color,
                smooth=smooth,
                snow=snow,
                ext=ext,
                ecmwf_grid=ecmwf_grid,
                nwp_chain=nwp_chain,
            )
        )

    # Historical frames are immutable once backfill is complete — cache them
    # for their full 2-hour lifetime.  Latest and nowcast frames still evolve.
    timestamps = await frame_store.get_timestamps()
    latest_ts = max(timestamps) if timestamps else None
    max_age = 7200 if (latest_ts is not None and timestamp < latest_ts) else 300

    return Response(
        content=tile_bytes,
        media_type=_content_type(ext),
        headers={"Cache-Control": f"public, max-age={max_age}"},
    )


@router.get("/v2/coverage/0/{size}/{z}/{x}/{y}/0/0_0.png")
async def coverage_tile(
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

    tile_bytes = await asyncio.to_thread(
        render_coverage_tile,
        frame_regions=frame.regions,
        z=z, x=x, y=y,
        tile_size=tile_size,
        enabled_regions=enabled_regions,
    )

    return Response(
        content=tile_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=300"},
    )


@router.get("/v2/satellite/{timestamp}/{size}/{z}/{x}/{y}/0/0_0.{ext}")
async def satellite_tile(
    timestamp: int,
    size: int = Path(ge=256, le=512),
    z: int = Path(ge=0),
    x: int = Path(ge=0),
    y: int = Path(ge=0),
    ext: str = Path(pattern=r"^(png|webp)$"),
) -> Response:
    """Satellite-like cloud cover tile from IFS cloud data."""
    if cloud_grid is None or not cloud_grid.loaded:
        raise HTTPException(status_code=503, detail="Satellite data not available")

    if z > settings.max_zoom:
        raise HTTPException(status_code=400, detail=f"Zoom {z} exceeds max {settings.max_zoom}")

    max_tiles = 2**z
    if x >= max_tiles or y >= max_tiles:
        raise HTTPException(status_code=400, detail="Tile coordinates out of range")

    tile_size = 512 if size >= 512 else 256

    cache_key = ("sat", timestamp, z, x, y, tile_size, ext)
    cached = tile_cache.get(cache_key)
    if cached is not None:
        return Response(
            content=cached,
            media_type=_content_type(ext),
            headers={"Cache-Control": "public, max-age=300"},
        )

    tile_bytes = await asyncio.to_thread(
        render_satellite_tile,
        cloud_grid=cloud_grid,
        z=z, x=x, y=y,
        tile_size=tile_size,
        timestamp=timestamp,
        fmt=ext,
    )

    tile_cache.put(cache_key, tile_bytes)

    # Satellite frames are all historical once loaded (IFS model-run based)
    satellite_timestamps = cloud_grid.timestamps if cloud_grid else []
    latest_sat_ts = max(satellite_timestamps) if satellite_timestamps else None
    max_age = 7200 if (latest_sat_ts is not None and timestamp < latest_sat_ts) else 300

    return Response(
        content=tile_bytes,
        media_type=_content_type(ext),
        headers={"Cache-Control": f"public, max-age={max_age}"},
    )
