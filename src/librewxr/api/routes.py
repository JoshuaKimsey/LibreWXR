# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
import asyncio
import httpx
import logging
import time
import psutil

from fastapi import APIRouter, HTTPException, Path, Query, Response

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
from librewxr.colors.schemes import SCHEME_NAMES
from librewxr.config import settings
from librewxr.data.store import FrameStore
from librewxr.memory import detect_memory_limit_mb
from librewxr.tiles.cache import TileCache
from librewxr.tiles.coordinates import coord_cache_bytes, coord_cache_stats
from librewxr.tiles.renderer import render_coverage_tile, render_tile
from librewxr.tiles.request_tracker import TileRequestTracker
from librewxr.tiles.satellite_renderer import render_gmgsi_tile, render_satellite_tile

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
cloud_grid = None  # CloudGrid | None — IFS-derived synthetic satellite (Phase 1.5 deletes)
# GMGSI satellite sources keyed by slug (gmgsi_lw_grid, gmgsi_vis_grid).
# When non-empty, the satellite tile endpoint serves from these instead
# of the legacy IFS-derived cloud_grid path.
satellite_grids: dict[str, object] = {}
tile_warmer = None  # TileWarmer | None
nowcast_store = None  # NowcastStore | None
radar_cache = None  # RadarFrameCache | None
radar_fetcher = None  # RadarFetcher | None
tile_request_tracker: TileRequestTracker | None = None
start_time: float = 0.0
enabled_regions: list[str] | None = None

# WMO alerts — set by main.py during startup
alerts_store = None  # AlertsStore | None
alerts_fetcher = None  # WMOAlertsFetcher | None
alerts_enabled: bool = False

# NWS point-lookup cache: {(lat, lon): (timestamp, list[GeoJSONFeature])}
_nws_point_cache: dict[tuple[float, float], tuple[float, list[GeoJSONFeature]]] = {}
_NWS_CACHE_TTL = 300  # 5 minutes
_NWS_API_URL = "https://api.weather.gov/alerts/active"


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
    satellite_bytes = cloud_grid.data_bytes if cloud_grid else 0
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
        "other_mb": round(other_bytes / (1024 * 1024), 1),
    })

    return {
        "status": "ok" if frame_count > 0 else "degraded",
        "uptime_seconds": uptime,
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
        },
        **_nwp_grid_health_blocks(),
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
    # GMGSI takes precedence: when LW frames are loaded, the catalog
    # timestamps come from GMGSI and the endpoint serves real satellite
    # imagery.  Falls back to the IFS-derived synthetic cloud timestamps
    # only when GMGSI is disabled / unloaded — that fallback path is
    # slated for deletion in Phase 1.5.
    gmgsi_lw = satellite_grids.get("gmgsi_lw_grid") if satellite_grids else None
    if gmgsi_lw is not None and gmgsi_lw.timestamps:
        infrared = [
            RadarTimestamp(time=ts, path=f"/v2/satellite/{ts}")
            for ts in gmgsi_lw.timestamps
        ]
    elif cloud_grid is not None and cloud_grid.loaded:
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
    is_nowcast = False
    if frame is None and nowcast_store is not None:
        nc_frame, nowcast_blend = await nowcast_store.get_frame(timestamp)
        if nc_frame is not None:
            frame = nc_frame
            is_nowcast = True
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
                frame_type="nowcast" if is_nowcast else "past",
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
    """Real satellite imagery tile (GMGSI) — IFS-derived fallback if GMGSI off.

    GMGSI is the canonical backing in Phase 1+: the LW (longwave IR)
    channel covers the populated globe and renders 24/7.  Phase 2 will
    swap this for the VIS-over-LW composite.  The fallback to the
    IFS-derived synthetic renderer exists only until Phase 1.5 deletes
    that path entirely; do not rely on it as a long-term behaviour.
    """
    if z > settings.max_zoom:
        raise HTTPException(status_code=400, detail=f"Zoom {z} exceeds max {settings.max_zoom}")

    max_tiles = 2**z
    if x >= max_tiles or y >= max_tiles:
        raise HTTPException(status_code=400, detail="Tile coordinates out of range")

    tile_size = 512 if size >= 512 else 256

    # Pick backing source: GMGSI when available, else IFS-derived fallback.
    gmgsi_lw = satellite_grids.get("gmgsi_lw_grid") if satellite_grids else None
    use_gmgsi = gmgsi_lw is not None and bool(gmgsi_lw.timestamps)
    if not use_gmgsi and (cloud_grid is None or not cloud_grid.loaded):
        raise HTTPException(status_code=503, detail="Satellite data not available")

    # Distinct cache keys so the GMGSI and IFS-derived tiles never
    # collide if the toggle ever flips at runtime.
    backing = "gmgsi" if use_gmgsi else "ifs"
    cache_key = ("sat", backing, timestamp, z, x, y, tile_size, ext)
    cached = tile_cache.get(cache_key)
    if cached is not None:
        return Response(
            content=cached,
            media_type=_content_type(ext),
            headers={"Cache-Control": "public, max-age=300"},
        )

    if use_gmgsi:
        tile_bytes = await asyncio.to_thread(
            render_gmgsi_tile,
            source=gmgsi_lw,
            z=z, x=x, y=y,
            tile_size=tile_size,
            timestamp=timestamp,
            fmt=ext,
        )
        sat_timestamps = gmgsi_lw.timestamps
    else:
        tile_bytes = await asyncio.to_thread(
            render_satellite_tile,
            cloud_grid=cloud_grid,
            z=z, x=x, y=y,
            tile_size=tile_size,
            timestamp=timestamp,
            fmt=ext,
        )
        sat_timestamps = cloud_grid.timestamps if cloud_grid else []

    tile_cache.put(cache_key, tile_bytes)

    # Older-than-latest frames are immutable; give them a long max-age.
    latest_sat_ts = max(sat_timestamps) if sat_timestamps else None
    max_age = 7200 if (latest_sat_ts is not None and timestamp < latest_sat_ts) else 300

    return Response(
        content=tile_bytes,
        media_type=_content_type(ext),
        headers={"Cache-Control": f"public, max-age={max_age}"},
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


async def _fetch_nws_point_alerts(lat: float, lon: float) -> list[GeoJSONFeature]:
    """Fetch NWS alerts for a specific lat/lon via the NWS point endpoint.

    The NWS API returns GeoJSON with polygon geometry for all alert types,
    including Tornado Watches which lack polygons in the global feed.
    Results are cached for 5 minutes.
    """
    cache_key = (round(lat, 4), round(lon, 4))
    now = time.time()
    cached = _nws_point_cache.get(cache_key)
    if cached is not None:
        ts, features = cached
        if now - ts < _NWS_CACHE_TTL:
            return features

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{_NWS_API_URL}?point={lat},{lon}",
                headers={"User-Agent": "(LibreWXR, librewxr@localhost)"},
            )
        if resp.status_code != 200:
            logger.debug("NWS point API returned %d for %s,%s", resp.status_code, lat, lon)
            return []
        data = resp.json()
    except Exception as exc:
        logger.debug("NWS point API error for %s,%s: %s", lat, lon, exc)
        return []

    features: list[GeoJSONFeature] = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry")

        # Skip cancelled/test
        status = props.get("status", "").lower()
        msg_type = props.get("messageType", "").lower()
        if status == "cancel" or msg_type == "test":
            continue

        # Use headline > event > description for title
        headline = props.get("headline", "") or ""
        event = props.get("event", "") or ""
        description = props.get("description", "") or ""
        title = headline or event or ""
        desc = description or headline or ""

        features.append(
            GeoJSONFeature(
                type="Feature",
                properties=AlertProperties(
                    title=title,
                    severity=props.get("severity", "Unknown"),
                    time=_parse_cap_time(props.get("effective", "")),
                    expires=_parse_cap_time(props.get("expires", "")),
                    description=desc,
                    regions=[props.get("areaDesc", "")] if props.get("areaDesc") else [],
                    uri=props.get("id", "") or feature.get("id", ""),
                ),
                geometry=geom,
            )
        )

    _nws_point_cache[cache_key] = (now, features)
    logger.debug("NWS point API: %d alerts cached for %s,%s", len(features), lat, lon)
    return features


@router.get("/v2/alerts", response_model=AlertsResponse)
async def get_alerts(
    lat: float | None = Query(None, ge=-90, le=90, description="Latitude for point lookup"),
    lon: float | None = Query(None, ge=-180, le=180, description="Longitude for point lookup"),
    bbox: str | None = Query(None, description="Bounding box: west,south,east,north"),
    simplify: float = Query(1000.0, ge=0, description="Polygon simplification tolerance in meters (0=off)"),
):
    """Weather alerts as GeoJSON FeatureCollection.

    - No params: all active alerts worldwide.
    - lat+lon: alerts containing that point.  For US locations, also queries
      the NWS point endpoint to include alerts (e.g. Tornado Watches) that
      lack polygon geometry in the global feed.
    - bbox: alerts intersecting the bounding box (polygon-only).
    """
    if not alerts_enabled or alerts_store is None:
        raise HTTPException(status_code=503, detail="Alerts not available")

    alerts = alerts_store.alerts
    nws_point_features: list[GeoJSONFeature] = []

    # Filter by point
    if lat is not None and lon is not None:
        from shapely.geometry import Point
        point = Point(lon, lat)
        alerts = [a for a in alerts if a.polygon is not None and a.polygon.intersects(point)]
        # For US points, also fetch NWS point-specific alerts (with geometry)
        if (-130 <= lon <= -60) and (20 <= lat <= 55):
            nws_point_features = await _fetch_nws_point_alerts(lat, lon)
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

    # Merge NWS point features, deduplicating by URI
    for feat in nws_point_features:
        uri = feat.properties.uri
        if uri and uri not in seen_uris:
            seen_uris.add(uri)
            features.append(feat)

    return AlertsResponse(type="FeatureCollection", features=features)
