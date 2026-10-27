# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

import threading
import time
from dataclasses import dataclass
from typing import Optional

from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.validation import make_valid


@dataclass
class AlertEntry:
    """Parsed CAP alert with optional polygon geometry."""
    source_id: str
    event: str
    description: str
    severity: str
    effective: str
    expires: str
    area_desc: str
    url: str
    polygon: Optional[Polygon] = None


def _as_polygonal(geom) -> Optional[Polygon]:
    """Normalize a geometry to a non-empty Polygon/MultiPolygon, or None.

    Flattens GeometryCollections to their polygonal members (the shape
    ``make_valid`` returns for a self-intersection that separates into two
    lobes) and reports any non-polygonal or empty result as None.
    """
    if isinstance(geom, Polygon):
        parts: list[Polygon] = [] if geom.is_empty else [geom]
    elif isinstance(geom, MultiPolygon):
        parts = [part for part in geom.geoms if not part.is_empty]
    elif getattr(geom, "geom_type", "") == "GeometryCollection":
        parts = []
        for member in geom.geoms:
            if isinstance(member, Polygon):
                if not member.is_empty:
                    parts.append(member)
            elif isinstance(member, MultiPolygon):
                parts.extend(part for part in member.geoms if not part.is_empty)
    else:
        return None
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return MultiPolygon(parts)


def ensure_valid_polygon(geom: Optional[Polygon]) -> Optional[Polygon]:
    """Return a valid Polygon/MultiPolygon, repairing invalid geometry.

    ``None`` in -> ``None`` out, and an already-valid geometry is returned
    unchanged (same object, no copy) so the common ingest path stays
    allocation-free.  Invalid geometry is repaired with ``make_valid``; a
    GeometryCollection repair result keeps only its polygonal members
    (0 usable parts -> None, 1 -> that part, more -> MultiPolygon).  If the
    repair is non-polygonal (collapsed to a line/point) or empty the result is
    None, and geometry still invalid after ``make_valid`` gets one last
    ``buffer(0)`` attempt (the table-repair convention in
    ``data/alert_geocodes.py``).  Never raises: an unrecoverable polygon is
    reported as None, which callers treat as the supported "alert without
    geometry" case.
    """
    if geom is None:
        return None
    try:
        if geom.is_valid:
            return geom
        repaired = _as_polygonal(make_valid(geom))
        if repaired is None:
            return None
        if repaired.is_valid:
            return repaired
        fixed = _as_polygonal(repaired.buffer(0))
        if fixed is None or not fixed.is_valid:
            return None
        return fixed
    except GEOSException:
        return None


class AlertsStore:
    """Thread-safe in-memory store of active weather alerts."""

    def __init__(self):
        self._lock = threading.RLock()
        self._alerts: list[AlertEntry] = []
        self.last_updated: float = 0.0
        self._fetch_success: bool = False

    @property
    def alerts(self) -> list[AlertEntry]:
        with self._lock:
            return list(self._alerts)

    def replace_all(
        self, alerts: list[AlertEntry], *, fetch_success: bool = True
    ) -> None:
        with self._lock:
            # Store invariant: every polygon is a valid Polygon/MultiPolygon
            # (or None).  Ingest repairs at construction; this catches anything
            # that slips past it, since invalid geometry raises
            # GEOSException at the request-time unary_union.
            for a in alerts:
                if a.polygon is not None:
                    a.polygon = ensure_valid_polygon(a.polygon)
            self._alerts = alerts
            self.last_updated = time.time()
            self._fetch_success = fetch_success

    def mark_failed(self) -> None:
        with self._lock:
            self._fetch_success = False

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._alerts)

    @property
    def fetch_success(self) -> bool:
        with self._lock:
            return self._fetch_success

    def __getstate__(self) -> dict:
        """Serialize state for the multi-worker snapshot.

        Polygons are emitted as GeoJSON via shapely's ``__geo_interface__``
        so the whole payload is JSON-friendly.  Render-only workers
        rebuild Polygon objects in ``__setstate__``.
        """
        with self._lock:
            return {
                "last_updated": self.last_updated,
                "fetch_success": self._fetch_success,
                "alerts": [
                    {
                        "source_id": a.source_id,
                        "event": a.event,
                        "description": a.description,
                        "severity": a.severity,
                        "effective": a.effective,
                        "expires": a.expires,
                        "area_desc": a.area_desc,
                        "url": a.url,
                        "polygon": (
                            a.polygon.__geo_interface__ if a.polygon is not None else None
                        ),
                    }
                    for a in self._alerts
                ],
            }

    def __setstate__(self, state: dict) -> None:
        """Restore from a snapshot written by ``__getstate__``."""
        if not hasattr(self, "_lock"):
            # __setstate__ may run before __init__ (pickle compat).
            self._lock = threading.RLock()
        with self._lock:
            self._alerts = [
                AlertEntry(
                    source_id=a["source_id"],
                    event=a["event"],
                    description=a["description"],
                    severity=a["severity"],
                    effective=a["effective"],
                    expires=a["expires"],
                    area_desc=a["area_desc"],
                    url=a["url"],
                    # Repair on load too: a snapshot written by an older
                    # revision may carry geometry the current ingest would
                    # have fixed, and a render worker must never hold an
                    # invalid polygon (it 500s the alerts endpoint).
                    polygon=ensure_valid_polygon(
                        shape(a["polygon"]) if a.get("polygon") else None
                    ),
                )
                for a in state.get("alerts", [])
            ]
            self.last_updated = state.get("last_updated", 0.0)
            self._fetch_success = state.get("fetch_success", False)
