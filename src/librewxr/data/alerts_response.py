# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

"""Alerts GeoJSON response construction, shared by every consumer.

This lives in the data layer (not ``api.routes``) so the alerts artifact writer
can package the response without importing the API module, while the HTTP route,
the WebSocket ``alerts`` topic and the MCP radius tool keep sharing one
implementation.  ``api.routes`` re-exports ``build_alerts_response`` and the two
CAP-time helpers, so existing imports keep working unchanged.
"""

import time
from datetime import datetime

from librewxr.api.models import AlertProperties, AlertsResponse, GeoJSONFeature
from librewxr.data.alerts_store import AlertEntry, ensure_valid_polygon


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


def build_alerts_response(
    alerts: list[AlertEntry], *, simplify: float = 1000.0,
) -> AlertsResponse:
    """Build the alerts GeoJSON payload from an (already geo-filtered) list.

    Shared by the ``/v2/alerts`` route and the WebSocket ``alerts`` topic.
    Sync on purpose: the shapely union / simplify pass is the expensive
    part and callers run this in a worker thread.
    """
    # Expiry filter
    now_utc = int(time.time())
    alerts = [a for a in alerts if _alert_not_expired(a, now_utc)]

    # Build GeoJSON features from WMO alerts
    deg_per_meter = simplify / 111_000.0 if simplify > 0 else 0.0
    # Imported lazily on purpose: the attribute lookup happens at call time, so
    # tests can monkeypatch ``shapely.ops.unary_union`` -- and the shapely import
    # cost stays off module import.
    from shapely.geometry import mapping
    from shapely.ops import unary_union
    from shapely.errors import GEOSException
    features: list[GeoJSONFeature] = []

    # Multi-area CAP alerts (Bulgaria, Romania, France, ...) arrive as many
    # AlertEntry objects sharing one url, each with its own region polygon.
    # Group by url and union the group's polygons so one feature carries the
    # full footprint instead of only the first region's polygon.
    by_uri: dict[str, list] = {}
    for alert in alerts:
        by_uri.setdefault(alert.url, []).append(alert)

    for group in by_uri.values():
        first = group[0]
        # Repair at the request boundary: an invalid polygon reaching the
        # union raises GEOSException (TopologyException) and 500s the whole
        # endpoint. The store enforces validity at insert/snapshot load;
        # this guards any geometry that slips past those chokepoints.
        polygons = [
            p for p in (
                ensure_valid_polygon(a.polygon)
                for a in group
                if a.polygon is not None
            )
            if p is not None
        ]
        geom = None
        if polygons:
            try:
                geom = unary_union(polygons)
            except GEOSException:
                geom = polygons[0]
            if deg_per_meter > 0 and geom is not None:
                try:
                    geom = geom.simplify(deg_per_meter, preserve_topology=True)
                except GEOSException:
                    pass  # ship unsimplified rather than fail the request

        regions: list[str] = []
        for a in group:
            if a.area_desc and a.area_desc not in regions:
                regions.append(a.area_desc)

        features.append(
            GeoJSONFeature(
                type="Feature",
                properties=AlertProperties(
                    title=first.event,
                    severity=first.severity,
                    time=_parse_cap_time(first.effective),
                    expires=_parse_cap_time(first.expires),
                    description=first.description,
                    regions=regions,
                    uri=first.url,
                ),
                geometry=mapping(geom) if geom is not None else None,
            )
        )

    return AlertsResponse(type="FeatureCollection", features=features)
