# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""MCP tool: query active weather alerts by radius/severity from the merged
WMO + NWS store.  US zone-based alerts (e.g. Tornado Watches) are resolved
to zone polygons at ingest, so no query-time NWS calls exist.

Uses an equirectangular cos(lat) approximation for the km-plane reprojection:
    x_km = (px - lon) * 111.0 * cos(radians(lat))
    y_km = (py - lat) * 111.0
"""

import math
import time

from shapely.geometry import Point, Polygon, MultiPolygon, mapping

from librewxr.api.models import AlertsResponse, GeoJSONFeature, AlertProperties
from librewxr.api.routes import _alert_not_expired, _parse_cap_time


def _reproject_to_km_plane(
    geometry, lon: float, lat: float,
    deg_to_km_lon: float, deg_to_km_lat: float,
):
    """Reproject a shapely Polygon or MultiPolygon to a km-plane centred on (lon, lat).

    Uses the equirectangular cos(lat) approximation (matches the formula at
    ``librewxr.data.coverage``:67-72): x_km = (px - lon) * 111 * cos(lat),
    y_km = (py - lat) * 111.  Returns a Polygon (input was Polygon) or a
    MultiPolygon (input was MultiPolygon) in km-plane coordinates suitable
    for intersection testing against ``Point(0, 0).buffer(radius_km)``.
    """
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon([
            Polygon([
                ((px - lon) * deg_to_km_lon, (py - lat) * deg_to_km_lat)
                for px, py in sub.exterior.coords
            ])
            for sub in geometry.geoms
        ])
    return Polygon([
        ((px - lon) * deg_to_km_lon, (py - lat) * deg_to_km_lat)
        for px, py in geometry.exterior.coords
    ])


async def alerts_within_radius(
    alerts_store,
    lat: float,
    lon: float,
    radius_km: float = 25.0,
    severity: str | None = None,
) -> AlertsResponse:
    """Filter alerts by radius from (lat, lon) and optional severity.

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
        (no store, no alerts in radius). Never raises.
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

        # Radius filter: reproject the alert's polygon (which may be a
        # MultiPolygon for alerts spanning disjoint areas, e.g. multi-
        # county warnings) to a local km-plane centred on (lon, lat),
        # then test intersection with a circle of the given radius.
        reprojected = _reproject_to_km_plane(
            alert.polygon, lon, lat, deg_to_km_lon, deg_to_km_lat,
        )

        within_radius = (
            circle.intersects(reprojected)
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

    return AlertsResponse(type="FeatureCollection", features=features)
