# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey

import threading
import time
from dataclasses import dataclass
from typing import Optional

from shapely.geometry import Polygon, shape


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
                    polygon=shape(a["polygon"]) if a.get("polygon") else None,
                )
                for a in state.get("alerts", [])
            ]
            self.last_updated = state.get("last_updated", 0.0)
            self._fetch_success = state.get("fetch_success", False)
