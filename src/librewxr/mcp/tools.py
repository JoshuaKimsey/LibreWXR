# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""MCP pure tool functions for precipitation nowcast, weather alerts,
and storm-cell queries.

These functions accept stores as explicit arguments so that both the HTTP
transport (reading ``routes.*`` globals) and the stdio transport (building
its own stores) can call them identically.  No module-level mutable state.
"""

import logging
import math
import time

import numpy as np

from librewxr.api.models import AlertsResponse
from librewxr.data.regions import REGIONS
from librewxr.mcp.alerts_query import alerts_within_radius
from librewxr.mcp.sampling import (
    dbz_to_rate_mmh,
    decode_dbz,
    resolve_region_for_point,
    sample_nowcast_at_point,
)
from librewxr.mcp.storm_cells import cell_pixel_to_latlon

logger = logging.getLogger(__name__)


async def get_precip_nowcast(
    nwp_chain,
    nowcast_store,
    enabled_regions,
    lat: float,
    lon: float,
    minutes: int = 60,
) -> list[dict]:
    """Sample nowcast frames at a point, with NWP fallback.

    Parameters
    ----------
    nwp_chain : NWPChain | None
        The NWP chain (None when NWP is disabled by config).
    nowcast_store : NowcastStore | None
        The nowcast store (None when nowcast is disabled by config).
    enabled_regions : list[str]
        Enabled radar region names (e.g. ``["USCOMP", "CACOMP", ...]``).
    lat : float
        Query latitude in degrees.
    lon : float
        Query longitude in degrees.
    minutes : int
        Forecast horizon in minutes (1-60; passed through as-is).

    Returns
    -------
    list[dict]
        One dict per future nowcast frame whose ``minutes_offset <= minutes``.
        Returns ``[]`` when *nowcast_store* is None (nowcast disabled).

    Notes
    -----
    Returns future nowcast frames only (the latest observed radar frame at
    t=0 is NOT included -- that lives in *frame_store*, not *nowcast_store*).
    Falls back to *nwp_chain* for points outside radar coverage; if NWP also
    has no data at the frame's timestamp, the frame is returned with
    ``source='none'``.  Never raises.
    """
    if nowcast_store is None:
        return []

    now = int(time.time())
    timestamps = await nowcast_store.get_timestamps()

    frames: list[dict] = []

    for ts in timestamps:
        minutes_offset = max(0, int(round((ts - now) / 60.0)))
        if minutes_offset > minutes:
            continue

        frame, blend_weight = await nowcast_store.get_frame(ts)
        if frame is None:
            continue

        dbz = None
        source = "none"
        coverage = "out_of_range"
        frame_bw = 0.0

        # ---- Radar path ---------------------------------------------------
        region = resolve_region_for_point(lat, lon, enabled_regions)
        if region is not None:
            sample_dbz, sample_bw, sample_cov = sample_nowcast_at_point(
                region.name, lat, lon, frame,
            )
            if sample_cov == "in_range":
                dbz = sample_dbz
                source = "radar"
                coverage = "in_range"
                frame_bw = float(sample_bw)

        # ---- NWP path (point is outside radar coverage or off-frame) ------
        radar_missed = region is None or (
            region is not None and sample_cov == "out_of_range"  # noqa: F821  # set in radar path
        )
        if radar_missed and nwp_chain is not None:
            # Check whether any registered source can answer for this ts.
            if any(src.has_data_at(ts) for src in nwp_chain.sources):
                try:
                    pixels = nwp_chain.sample(
                        np.array([lat], dtype=np.float32),
                        np.array([lon], dtype=np.float32),
                        timestamp=ts,
                    )
                    dbz = decode_dbz(int(pixels[0]))
                    source = "nwp"
                    coverage = "in_range"
                    frame_bw = 0.0
                except Exception:
                    logger.warning(
                        "NWP sample failed at lat=%s lon=%s ts=%s",
                        lat,
                        lon,
                        ts,
                        exc_info=True,
                    )
                    dbz = None
                    source = "none"
                    coverage = "out_of_range"
                    frame_bw = 0.0

        rate_mmh = dbz_to_rate_mmh(dbz)

        frames.append(
            {
                "time": int(ts),
                "minutes_offset": minutes_offset,
                "dbz": dbz,
                "rate_mmh": rate_mmh,
                "source": source,
                "blend_weight": frame_bw,
                "coverage": coverage,
            }
        )

    return frames


async def get_active_alerts(
    alerts_store,
    alerts_enabled: bool,
    lat: float,
    lon: float,
    radius_km: float = 25.0,
    severity: str | None = None,
) -> AlertsResponse:
    """Query active weather alerts near a point.

    A thin gate over :func:`alerts_within_radius` that enforces the
    degraded-empty contract: when alerts are disabled or the store is
    unavailable, returns an empty ``FeatureCollection`` without raising.

    Alerts come from the merged WMO + NWS store; US zone-based alerts
    (e.g. Tornado Watches) are resolved to zone polygons at ingest, so
    no query-time NWS calls are made.

    Parameters
    ----------
    alerts_store : AlertsStore | None
        The global alerts store (None when alerts are disabled by config).
    alerts_enabled : bool
        The global alerts-enabled flag.
    lat : float
        Query latitude in degrees.
    lon : float
        Query longitude in degrees.
    radius_km : float
        Search radius in kilometres (default 25.0).
    severity : str | None
        If set, only include alerts with this exact severity string.

    Returns
    -------
    AlertsResponse
        A GeoJSON FeatureCollection.  Returns an empty collection when alerts
        are disabled or the store is unavailable.  Never raises.
    """
    if not alerts_enabled or alerts_store is None:
        return AlertsResponse(type="FeatureCollection", features=[])

    return await alerts_within_radius(alerts_store, lat, lon, radius_km, severity)


async def get_storm_cells(
    storm_cell_store,
    lat: float,
    lon: float,
    radius_km: float = 100.0,
) -> list[dict]:
    """Query detected storm cells within a radius of a geographic point.

    Returns a list of cell dicts, each with lat/lon centroid, area, max
    dBZ, and motion vector (speed + heading).  Cells are filtered to
    those within ``radius_km`` of (lat, lon) using the equirectangular
    cos(lat) approximation.  Returns an empty list when storm-cell
    detection is disabled or no cells are within range; never raises.

    Parameters
    ----------
    storm_cell_store : StormCellStore | None
        The global storm-cell store (None when detection is disabled).
    lat : float
        Query latitude in degrees.
    lon : float
        Query longitude in degrees.
    radius_km : float
        Search radius in kilometres (default 100.0 -- wider than the
        alerts default because storm cells are sparser than alerts).

    Returns
    -------
    list[dict]
        One dict per detected cell within radius.  Each dict has:
        ``lat``, ``lon``, ``area_km2``, ``max_dbz``, ``motion_speed_kmh``,
        ``motion_heading_deg``, ``region``.
    """
    if storm_cell_store is None:
        return []

    cells_by_region = await storm_cell_store.get_cells()
    counts = await storm_cell_store.get_counts()
    if not cells_by_region or not counts:
        return []

    # Equirectangular cos(lat) distance -- same formula as alerts_query.py.
    cos_lat = math.cos(math.radians(lat))
    deg_to_km_lat = 111.0
    deg_to_km_lon = 111.0 * cos_lat
    radius_km_sq = radius_km * radius_km

    results: list[dict] = []
    for region_name, cells in cells_by_region.items():
        count = counts.get(region_name, 0)
        if count == 0:
            continue

        region = REGIONS.get(region_name)
        if region is None:
            continue

        for i in range(count):
            cell = cells[i]
            cr = float(cell["centroid_row"])
            cc = float(cell["centroid_col"])

            # Convert pixel coords to lat/lon.
            cell_lat, cell_lon = cell_pixel_to_latlon(region, cr, cc)

            # Radius filter: equirectangular distance.
            dlat_km = (cell_lat - lat) * deg_to_km_lat
            dlon_km = (cell_lon - lon) * deg_to_km_lon
            d2 = dlat_km * dlat_km + dlon_km * dlon_km
            if d2 > radius_km_sq:
                continue

            speed = float(cell["motion_speed_kmh"])
            heading = float(cell["motion_heading_deg"])

            results.append({
                "lat": cell_lat,
                "lon": cell_lon,
                "area_km2": float(cell["area_km2"]),
                "max_dbz": float(cell["max_dbz"]),
                "motion_speed_kmh": speed if not math.isnan(speed) else None,
                "motion_heading_deg": heading if not math.isnan(heading) else None,
                "region": region_name,
            })

    return results
