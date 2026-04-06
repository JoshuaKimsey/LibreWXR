# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import logging
import time
import psutil

from fastapi import APIRouter, HTTPException, Path, Response

from librewxr.api.models import (
    RadarData,
    RadarTimestamp,
    SatelliteData,
    WeatherMapsResponse,
)
from librewxr.config import settings
from librewxr.data.store import FrameStore
from librewxr.tiles.cache import TileCache
from librewxr.tiles.renderer import render_coverage_tile, render_tile

logger = logging.getLogger(__name__)

router = APIRouter()

# These get set by main.py during startup
frame_store: FrameStore | None = None
tile_cache: TileCache | None = None
ecmwf_grid = None  # ECMWFGrid | None
tile_warmer = None  # TileWarmer | None
start_time: float = 0.0
enabled_regions: list[str] | None = None


@router.get("/health")
async def health():
    """Health and status endpoint."""
    now = int(time.time())
    uptime = now - int(start_time)
    ram = psutil.virtual_memory()
    ram_usage = ram.percent
    ram_used = round(ram.used / 1e9, 2)
    frame_count = await frame_store.frame_count()
    timestamps = await frame_store.get_timestamps()
    latest_ts = max(timestamps) if timestamps else None
    oldest_ts = min(timestamps) if timestamps else None

    return {
        "status": "ok" if frame_count > 0 else "degraded",
        "uptime_seconds": uptime,
        "RAM Usage (%)": ram_usage,
        "RAM Used (GB)": ram_used,
        "frames": {
            "count": frame_count,
            "max": settings.max_frames,
            "latest": latest_ts,
            "oldest": oldest_ts,
            "latest_age_seconds": now - latest_ts if latest_ts else None,
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
        "enabled_regions": enabled_regions or [],
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

    return WeatherMapsResponse(
        version="2.0",
        generated=int(time.time()),
        host=host,
        radar=RadarData(past=past, nowcast=[]),
        satellite=SatelliteData(infrared=[]),
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
) -> Response:
    """Rain Viewer-compatible tile endpoint."""
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

    cache_key = (timestamp, z, x, y, tile_size, color, smooth, snow, ext)
    cached = tile_cache.get(cache_key)
    if cached is not None:
        return Response(
            content=cached,
            media_type=_content_type(ext),
            headers={"Cache-Control": "public, max-age=300"},
        )

    frame = await frame_store.get_frame(timestamp)
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found")

    tile_bytes = render_tile(
        frame_regions=frame.regions,
        z=z, x=x, y=y,
        tile_size=tile_size,
        color_scheme=color,
        smooth=smooth,
        snow=snow,
        fmt=ext,
        ecmwf_grid=ecmwf_grid,
        enabled_regions=enabled_regions,
        frame_timestamp=timestamp,
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
            )
        )

    return Response(
        content=tile_bytes,
        media_type=_content_type(ext),
        headers={"Cache-Control": "public, max-age=300"},
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

    tile_bytes = render_coverage_tile(
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
