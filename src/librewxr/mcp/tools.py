# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""MCP pure tool functions for precipitation nowcast and weather alerts.

These functions accept stores as explicit arguments so that both the HTTP
transport (reading ``routes.*`` globals) and the stdio transport (building
its own stores) can call them identically.  No module-level mutable state.
"""

import logging
import time

import numpy as np

from librewxr.api.models import AlertsResponse
from librewxr.mcp.alerts_query import alerts_within_radius
from librewxr.mcp.sampling import (
    dbz_to_rate_mmh,
    decode_dbz,
    resolve_region_for_point,
    sample_nowcast_at_point,
)

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
