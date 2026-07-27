# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""MCP tool: query active weather alerts by radius/severity, enriched with NWS point alerts for US locations.

Uses an equirectangular cos(lat) approximation for the km-plane reprojection:
    x_km = (px - lon) * 111.0 * cos(radians(lat))
    y_km = (py - lat) * 111.0
"""

import logging
import math
import time

from shapely.geometry import Point, Polygon, mapping

from librewxr.api.models import AlertsResponse, GeoJSONFeature, AlertProperties
from librewxr.api.routes import _alert_not_expired, _fetch_nws_point_alerts, _parse_cap_time

logger = logging.getLogger(__name__)


async def alerts_within_radius(
    alerts_store,
    lat: float,
    lon: float,
    radius_km: float = 25.0,
    severity: str | None = None,
) -> AlertsResponse:
    """Filter alerts by radius from (lat, lon), optional severity, and enrich with NWS point alerts for US locations.

    Parameters
    ----------
    alerts_store : AlertsStore | None
        The global alerts store (may be None when alerts are disabled).
    lat : float
        Query latitude in degrees.
    lon : float
        Query longitude in degrees.
    radius_km : float, optional
        Search radius in kilometres (default 25.0).
    severity : str | None, optional
        If set, only include alerts with this exact severity string.

    Returns
    -------
    AlertsResponse
        A GeoJSON FeatureCollection. Returns an empty collection on degraded input
        (no store, no alerts in radius, NWS API failure). Never raises.
    """
    # Degraded empty: no alerts store available
    if alerts_store is None:
        return AlertsResponse(type="FeatureCollection", features=[])

    alerts = alerts_store.alerts
    now_utc = int(time.time())

    features: list[GeoJSONFeature] = []
    seen_uris: set[str] = set()

    # Precompute constants for the equirectangular km-plane reprojection.
    cos_lat = math.cos(math.radians(lat))
    deg_to_km_lat = 111.0
    deg_to_km_lon = 111.0 * cos_lat
    query_point = Point(lon, lat)
    circle = Point(0.0, 0.0).buffer(radius_km)

    for alert in alerts:
        # Expiry filter
        if not _alert_not_expired(alert, now_utc):
            continue

        # Severity filter
        if severity is not None and alert.severity != severity:
            continue

        # Need a polygon to do geometry-based filtering
        if alert.polygon is None:
            continue

        # Radius filter: reproject polygon exterior coordinates to a local
        # km-plane centred on (lon, lat), then test intersection with a
        # circle of the given radius.
        reprojected_coords = [
            ((px - lon) * deg_to_km_lon, (py - lat) * deg_to_km_lat)
            for px, py in alert.polygon.exterior.coords
        ]
        reprojected_poly = Polygon(reprojected_coords)

        within_radius = (
            circle.intersects(reprojected_poly)
            or alert.polygon.contains(query_point)
        )

        if not within_radius:
            continue

        # Dedup by URI (mirrors routes.py:694-701)
        uri = alert.url
        if uri in seen_uris:
            continue
        seen_uris.add(uri)

        # Build GeoJSONFeature mirroring the field mapping at routes.py:704-718
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
                geometry=mapping(alert.polygon) if alert.polygon is not None else None,
            )
        )

    # US NWS enrichment: for US-lat-lon points, fetch point-specific NWS
    # alerts (which may have polygons not present in the global WMO feed).
    if -130.0 <= lon <= -60.0 and 20.0 <= lat <= 55.0:
        try:
            nws_features = await _fetch_nws_point_alerts(lat, lon)
            for nws_feature in nws_features:
                nws_uri = nws_feature.properties.uri
                if nws_uri and nws_uri not in seen_uris:
                    seen_uris.add(nws_uri)
                    features.append(nws_feature)
        except Exception:
            logger.warning(
                "NWS point alerts fetch failed for %s,%s; continuing with WMO alerts only",
                lat,
                lon,
            )

    return AlertsResponse(type="FeatureCollection", features=features)
